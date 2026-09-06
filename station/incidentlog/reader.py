"""Read an incident log back: filter, summarise, replay.

This is the after-action tool. It answers the questions a review actually
asks -- what did the model output, when, with what confidence, under which
weights, and which frame of the recording does each box belong to -- and it is
deliberately careful about the question it cannot answer.

**Honesty rule for everything printed here.** A stretch of frames with empty
results is reported as exactly that: frames on which this model returned
nothing. It is never summarised as a quiet period, a safe interval, or an
absence of fire. The model fails toward silence -- thin smoke on bright sky,
smouldering without flame, fire under canopy, fire at night -- so an empty
result carries no information about the scene, and a review tool that phrased
it as reassurance would launder a null result into a finding. See
``station/core/safety.py``.

Robustness: a station gets its battery pulled mid-flight, so the last line of
``detections.jsonl`` may be a partial write. That case is expected and
handled -- the truncated tail is dropped and reported. A corrupt line in the
*middle* of the file is different: it means records were lost, so it is
counted and shown in the summary rather than silently skipped, because an
audit over a log with holes has to know the holes are there.

CLI::

    python -m station.incidentlog.reader INCIDENT_DIR --class fire --min-conf 0.5 --summary
    python -m station.incidentlog.reader INCIDENT_DIR --since 120 --until 180 --video
    python -m station.incidentlog.reader INCIDENT_DIR --json > filtered.jsonl
"""

from __future__ import annotations

import argparse
import bisect
import heapq
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from station.core.types import (
    MSG_DETECTIONS,
    WIRE_VERSION,
    Detection,
    FrameDetections,
    ModelInfo,
    parse_message,
)
from station.incidentlog.writer import (
    DETECTIONS_FILENAME,
    EVENTS_FILENAME,
    META_FILENAME,
    VIDEO_SUBDIR,
)
from station.incidentlog.recorder import FRAME_INDEX_FILENAME, MANIFEST_FILENAME

__all__ = [
    "IncidentLogError",
    "ReadStats",
    "ClassSummary",
    "IncidentSummary",
    "VideoPosition",
    "IncidentLogReader",
    "format_frame",
    "main",
]

log = logging.getLogger(__name__)

#: Printed under every summary. The single most important sentence in the
#: whole subsystem: it is what stops a reviewer reading a column of zeroes as
#: a column of good news.
EMPTY_RESULT_CAVEAT = (
    "An empty result means this model returned nothing on that frame. It is not a\n"
    "measurement of the scene and it is not evidence that there was nothing there:\n"
    "thin smoke on bright sky, smouldering without flame, fire under canopy and fire\n"
    "at night all produce empty results. These counts describe the model's output,\n"
    "not the ground. Review the recorded video."
)


class IncidentLogError(RuntimeError):
    """The incident log cannot be read as what it claims to be."""


@dataclass(slots=True)
class ReadStats:
    """What happened while parsing the file. Reported, never hidden."""

    lines: int = 0
    frames: int = 0
    #: Lines that were not parseable JSON, or not a detections record, in the
    #: middle of the file. Each one is a lost frame record.
    corrupt: int = 0
    #: Well-formed records that were not ``type: detections``.
    other_records: int = 0
    #: True when the final line was incomplete -- the normal signature of a
    #: process killed mid-write, and not a cause for alarm on its own.
    truncated_tail: bool = False

    @property
    def clean(self) -> bool:
        """``True`` when every record in the file was read successfully."""
        return self.corrupt == 0 and not self.truncated_tail


@dataclass(slots=True)
class ClassSummary:
    """Per-class totals over the frames a summary covered."""

    cls: str
    detections: int = 0
    frames: int = 0
    conf_min: float = 1.0
    conf_max: float = 0.0
    conf_sum: float = 0.0
    tracks: set[int] = field(default_factory=set)
    first_pts: float | None = None
    last_pts: float | None = None
    max_conf_pts: float | None = None

    @property
    def conf_mean(self) -> float:
        """Mean confidence across every detection of this class."""
        return self.conf_sum / self.detections if self.detections else 0.0

    def observe(self, det: Detection, pts: float) -> None:
        """Fold one detection into the totals."""
        self.detections += 1
        self.conf_sum += det.conf
        self.conf_min = min(self.conf_min, det.conf)
        if det.conf > self.conf_max:
            self.conf_max = det.conf
            self.max_conf_pts = pts
        if det.track_id is not None:
            self.tracks.add(int(det.track_id))
        if self.first_pts is None:
            self.first_pts = pts
        self.last_pts = pts


@dataclass(slots=True)
class IncidentSummary:
    """After-action summary of one incident log.

    Every field describes **model output**. None of them describes the scene.
    """

    incident_id: str
    station_name: str
    source_uri: str
    model: ModelInfo | None
    started_at: str | None
    ended_at: str | None
    closed_cleanly: bool
    filter_description: str
    #: True when a class/confidence/track filter narrowed the detections, in
    #: which case ``frames_empty`` counts frames with nothing *matching the
    #: filter* -- a different and much weaker statement than "the model
    #: returned nothing", and the text output distinguishes them.
    detection_filter: bool = False
    frames_logged: int = 0
    frames_in_window: int = 0
    frames_with_detections: int = 0
    frames_empty: int = 0
    detections_total: int = 0
    classes: dict[str, ClassSummary] = field(default_factory=dict)
    first_pts: float | None = None
    last_pts: float | None = None
    first_wall_time: str | None = None
    last_wall_time: str | None = None
    #: Largest pts gap between consecutive logged frames, and where it starts.
    #: A gap far above the inference interval means the pipeline was not
    #: producing results then -- which is a fact about the station, and the
    #: reviewer must not read that stretch as anything else at all.
    max_gap_s: float = 0.0
    max_gap_at_pts: float | None = None
    mean_interval_s: float | None = None
    #: Longest unbroken run of consecutive logged frames carrying detections.
    longest_run_frames: int = 0
    longest_run_pts: tuple[float, float] | None = None
    inference_ms_mean: float | None = None
    inference_ms_max: float | None = None
    read: ReadStats = field(default_factory=ReadStats)
    video: dict[str, Any] | None = None

    @property
    def pts_span_s(self) -> float:
        """Media seconds between the first and last logged frame."""
        if self.first_pts is None or self.last_pts is None:
            return 0.0
        return self.last_pts - self.first_pts

    @property
    def gap_is_notable(self) -> bool:
        """Whether the largest gap is worth reporting as a gap.

        At a steady 10 results/s the largest interval between consecutive
        records is one inference period, which is the system working. Only a
        stretch several times longer than the norm means the pipeline actually
        stopped producing results, and only that is worth an operator's
        attention.
        """
        if self.max_gap_at_pts is None or not self.mean_interval_s:
            return False
        return self.max_gap_s >= max(2.0 * self.mean_interval_s, 0.5)

    @property
    def tracks_total(self) -> int:
        """Distinct tracks across all classes."""
        return sum(len(c.tracks) for c in self.classes.values())

    def as_dict(self) -> dict[str, Any]:
        """JSON-ready form, for feeding a report generator."""
        return {
            "incident_id": self.incident_id,
            "station_name": self.station_name,
            "source_uri": self.source_uri,
            "model": self.model.to_wire() if self.model else None,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "closed_cleanly": self.closed_cleanly,
            "filter": self.filter_description,
            "frames_logged": self.frames_logged,
            "frames_in_window": self.frames_in_window,
            "frames_with_detections": self.frames_with_detections,
            "frames_empty": self.frames_empty,
            "detections_total": self.detections_total,
            "tracks_total": self.tracks_total,
            "first_pts": self.first_pts,
            "last_pts": self.last_pts,
            "pts_span_s": round(self.pts_span_s, 3),
            "mean_interval_s": self.mean_interval_s,
            "max_gap_s": round(self.max_gap_s, 3),
            "max_gap_at_pts": self.max_gap_at_pts,
            "gap_is_notable": self.gap_is_notable,
            "detection_filter": self.detection_filter,
            "longest_run_frames": self.longest_run_frames,
            "longest_run_pts": list(self.longest_run_pts) if self.longest_run_pts else None,
            "inference_ms_mean": self.inference_ms_mean,
            "inference_ms_max": self.inference_ms_max,
            "classes": {
                name: {
                    "detections": c.detections,
                    "frames": c.frames,
                    "conf_min": round(c.conf_min, 3) if c.detections else None,
                    "conf_max": round(c.conf_max, 3) if c.detections else None,
                    "conf_mean": round(c.conf_mean, 3) if c.detections else None,
                    "tracks": len(c.tracks),
                    "first_pts": c.first_pts,
                    "last_pts": c.last_pts,
                    "max_conf_pts": c.max_conf_pts,
                }
                for name, c in sorted(self.classes.items())
            },
            "read": {
                "lines": self.read.lines,
                "frames": self.read.frames,
                "corrupt": self.read.corrupt,
                "other_records": self.read.other_records,
                "truncated_tail": self.read.truncated_tail,
            },
            "video": self.video,
            "caveat": EMPTY_RESULT_CAVEAT.replace("\n", " "),
        }

    def to_text(self) -> str:
        """Render the operator-facing summary.

        Returns:
            A multi-line report. Wording is constrained by
            ``station/core/safety.py``: counts are attributed to the model, and
            an absence of detections is never phrased as an absence of fire.
        """
        model = "unknown"
        if self.model is not None:
            bits = [f"{self.model.name} {self.model.version}"]
            if self.model.weights_sha:
                bits.append(f"weights {self.model.weights_sha}")
            if self.model.conf_threshold is not None:
                bits.append(f"conf>={self.model.conf_threshold}")
            if self.model.imgsz:
                bits.append(f"imgsz {self.model.imgsz}")
            model = " | ".join(bits)

        lines: list[str] = [
            f"Incident {self.incident_id}",
            f"  station          : {self.station_name or '-'}",
            f"  source           : {self.source_uri or '-'}",
            f"  model            : {model}",
            f"  log              : {'closed cleanly' if self.closed_cleanly else 'ENDED WITHOUT CLOSING (station stopped abruptly?)'}",
            f"  wall time        : {self.first_wall_time or self.started_at or '-'} -> {self.last_wall_time or self.ended_at or '-'}",
        ]
        if self.first_pts is not None:
            lines.append(
                f"  media pts        : {self.first_pts:.3f} -> {self.last_pts:.3f} s "
                f"({self.pts_span_s:.1f} s spanned)"
            )
        if self.filter_description:
            lines.append(f"  filter           : {self.filter_description}")
        lines.append(f"  frames logged    : {self.frames_logged}")
        if self.frames_in_window != self.frames_logged:
            lines.append(f"  frames in window : {self.frames_in_window}")
        share = (100.0 * self.frames_empty / self.frames_in_window) if self.frames_in_window else 0.0
        empty_label = "with nothing matching the filter" if self.detection_filter else "with an empty result"
        lines.append(
            f"  model output     : {self.frames_with_detections} frame(s) with detections, "
            f"{self.frames_empty} {empty_label} ({share:.1f}%)"
        )
        if self.mean_interval_s:
            lines.append(
                f"  logging interval : {self.mean_interval_s:.3f} s mean "
                f"({1.0 / self.mean_interval_s:.2f} results/s)"
            )
        if self.gap_is_notable and self.max_gap_at_pts is not None:
            # Only surfaced when it is genuinely anomalous -- see gap_is_notable.
            lines.append(
                f"  longest gap      : {self.max_gap_s:.3f} s of media with no logged result at all, "
                f"starting at pts {self.max_gap_at_pts:.3f}. The pipeline produced nothing across "
                f"that stretch, so the log says nothing whatsoever about it"
            )
        if self.inference_ms_mean is not None:
            lines.append(
                f"  inference        : {self.inference_ms_mean:.1f} ms mean, {self.inference_ms_max:.1f} ms max"
            )

        lines.append(f"  detections       : {self.detections_total} across {self.tracks_total} track(s)")
        for name, c in sorted(self.classes.items()):
            if not c.detections:
                continue
            lines.append(
                f"    {name:<6}: {c.detections} on {c.frames} frame(s), "
                f"conf {c.conf_min:.2f}-{c.conf_max:.2f} (mean {c.conf_mean:.2f}), "
                f"{len(c.tracks)} track(s), pts {c.first_pts:.3f}-{c.last_pts:.3f}, "
                f"peak conf at pts {c.max_conf_pts:.3f}"
            )
        if self.longest_run_frames and self.longest_run_pts:
            lines.append(
                f"  longest run      : {self.longest_run_frames} consecutive frame(s) with detections, "
                f"pts {self.longest_run_pts[0]:.3f} -> {self.longest_run_pts[1]:.3f}"
            )
        if self.video:
            segs = self.video.get("segments") or []
            lines.append(
                f"  video            : {len(segs)} segment(s), {self.video.get('frames_written', 0)} frames "
                f"via {self.video.get('backend', '?')}"
                + ("" if self.video.get("pts_exact") else " (pts approximate -- replay via the frame index)")
            )
        else:
            lines.append("  video            : none recorded -- detections cannot be checked against frames")

        if not self.read.clean:
            lines.append(
                f"  READ WARNINGS    : {self.read.corrupt} unreadable record(s)"
                + (", truncated final line" if self.read.truncated_tail else "")
                + " -- this log has holes and any audit over it is incomplete"
            )
        lines.append("")
        lines.append(EMPTY_RESULT_CAVEAT)
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class VideoPosition:
    """Where in the recording a given media pts lives."""

    segment: int
    path: Path
    #: Seconds to seek to inside ``path``.
    offset_s: float
    #: ``True`` when the mapping came from a real timestamp rather than from
    #: frame counting on a uniform grid.
    exact: bool
    frame_id: int | None = None
    frame_index: int | None = None

    def describe(self) -> str:
        """One-line, paste-into-a-player form."""
        kind = "exact" if self.exact else "approx"
        return f"{self.path.name}@{self.offset_s:.3f}s ({kind})"


def _parse_wall(value: str | None) -> datetime | None:
    """Parse an RFC 3339 stamp, tolerating the ``Z`` suffix.

    Args:
        value: A wall-time string from the log or the command line.

    Returns:
        A timezone-aware datetime, or ``None`` if it is absent or unparseable.
    """
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


@dataclass(frozen=True, slots=True)
class _Filter:
    """Frame- and detection-level selection criteria."""

    classes: tuple[str, ...] = ()
    min_conf: float | None = None
    min_persisted: int | None = None
    track_id: int | None = None
    start_pts: float | None = None
    end_pts: float | None = None
    start_wall: datetime | None = None
    end_wall: datetime | None = None

    @property
    def selects_detections(self) -> bool:
        """``True`` when the filter narrows detections within a frame."""
        return bool(self.classes) or self.min_conf is not None or self.min_persisted is not None or self.track_id is not None

    def in_window(self, frame: FrameDetections) -> bool:
        """Whether the frame falls inside the time window."""
        if self.start_pts is not None and frame.pts < self.start_pts:
            return False
        if self.end_pts is not None and frame.pts > self.end_pts:
            return False
        if self.start_wall is not None or self.end_wall is not None:
            when = _parse_wall(frame.wall_time)
            if when is None:
                return False  # cannot place it in time; excluded rather than guessed
            if self.start_wall is not None and when < self.start_wall:
                return False
            if self.end_wall is not None and when > self.end_wall:
                return False
        return True

    def keep(self, det: Detection) -> bool:
        """Whether one detection survives the detection-level criteria."""
        if self.classes and det.cls not in self.classes:
            return False
        if self.min_conf is not None and det.conf < self.min_conf:
            return False
        if self.min_persisted is not None and det.persisted < self.min_persisted:
            return False
        if self.track_id is not None and det.track_id != self.track_id:
            return False
        return True

    def apply(self, frame: FrameDetections) -> FrameDetections:
        """Return ``frame`` with only the detections that pass the filter."""
        if not self.selects_detections:
            return frame
        kept = tuple(d for d in frame.detections if self.keep(d))
        if len(kept) == len(frame.detections):
            return frame
        # dataclasses.replace would work, but FrameDetections is frozen+slots
        # and this keeps every field explicit at the one place it is rebuilt.
        return FrameDetections(
            frame_id=frame.frame_id,
            pts=frame.pts,
            wall_time=frame.wall_time,
            detections=kept,
            model=frame.model,
            inference_ms=frame.inference_ms,
            rtp_ts=frame.rtp_ts,
            source_id=frame.source_id,
        )

    def describe(self) -> str:
        """Human-readable rendering, for the summary header."""
        bits: list[str] = []
        if self.classes:
            bits.append("class in {" + ", ".join(self.classes) + "}")
        if self.min_conf is not None:
            bits.append(f"conf >= {self.min_conf}")
        if self.min_persisted is not None:
            bits.append(f"persisted >= {self.min_persisted}")
        if self.track_id is not None:
            bits.append(f"track == {self.track_id}")
        if self.start_pts is not None:
            bits.append(f"pts >= {self.start_pts}")
        if self.end_pts is not None:
            bits.append(f"pts <= {self.end_pts}")
        if self.start_wall is not None:
            bits.append(f"wall >= {self.start_wall.isoformat()}")
        if self.end_wall is not None:
            bits.append(f"wall <= {self.end_wall.isoformat()}")
        return "; ".join(bits)


def format_frame(frame: FrameDetections, *, position: VideoPosition | None = None) -> str:
    """Render one frame record as a single operator-readable line.

    Args:
        frame: The frame record, already filtered.
        position: Optional video position to append, for scrubbing.

    Returns:
        One line. Frames with an empty result render as ``(empty result)`` --
        never as a reassuring phrase.
    """
    head = f"pts {frame.pts:9.3f}  frame {frame.frame_id:<8} {frame.wall_time or '-'}"
    if not frame.detections:
        body = "(empty result)"
    else:
        parts = []
        for det in frame.detections:
            box = ",".join(f"{v:.3f}" for v in det.box.as_tuple())
            track = f" track={det.track_id}" if det.track_id is not None else ""
            parts.append(f"{det.cls} {det.conf:.2f} [{box}]{track} seen={det.persisted}")
        body = "; ".join(parts)
    tail = f"  video {position.describe()}" if position is not None else ""
    return f"{head}  {body}{tail}"


class IncidentLogReader:
    """Reads one incident directory written by :mod:`station.incidentlog.writer`.

    Example:
        >>> reader = IncidentLogReader("incidents/20260906-142231Z-flight-3")  # doctest: +SKIP
        >>> print(reader.summarise(classes=("fire",), min_conf=0.5).to_text())  # doctest: +SKIP
    """

    def __init__(self, incident_dir: str | Path) -> None:
        """Open an incident directory.

        Args:
            incident_dir: The directory containing ``detections.jsonl``. The
                path to ``detections.jsonl`` itself is also accepted, since
                that is what shell completion tends to produce.

        Raises:
            IncidentLogError: The directory or the detections file is missing.
        """
        path = Path(incident_dir)
        if path.is_file() and path.name == DETECTIONS_FILENAME:
            path = path.parent
        if not path.is_dir():
            raise IncidentLogError(f"{path} is not an incident directory")
        self.dir = path
        self.detections_path = path / DETECTIONS_FILENAME
        if not self.detections_path.is_file():
            raise IncidentLogError(
                f"{path} contains no {DETECTIONS_FILENAME}; it is not an incident log"
            )
        self.meta: dict[str, Any] = self._load_meta()
        self._video: dict[str, Any] | None = None
        self._video_loaded = False
        self._index: list[tuple[float, int, int, float, int]] | None = None
        self.last_read: ReadStats = ReadStats()

    # ------------------------------------------------------------------
    # metadata
    # ------------------------------------------------------------------

    def _load_meta(self) -> dict[str, Any]:
        """Load ``meta.json``, tolerating its absence.

        A missing or unreadable meta file is survivable -- every record in
        ``detections.jsonl`` carries its own model info and timestamps -- so it
        degrades to a warning rather than refusing to open the log at all.
        """
        path = self.dir / META_FILENAME
        if not path.is_file():
            log.warning("%s has no %s; reporting from the records alone", self.dir, META_FILENAME)
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("%s is unreadable (%s); reporting from the records alone", path, exc)
            return {}

    @property
    def incident_id(self) -> str:
        """Incident id from the metadata, falling back to the directory name."""
        return str(self.meta.get("incident_id") or self.dir.name)

    @property
    def station_name(self) -> str:
        """Station that wrote the log."""
        return str(self.meta.get("station_name") or "")

    @property
    def source_uri(self) -> str:
        """Video source, as recorded (with credentials redacted by the writer)."""
        return str(self.meta.get("source_uri") or "")

    @property
    def config(self) -> dict[str, Any] | None:
        """The config snapshot taken when the incident started."""
        cfg = self.meta.get("config")
        return cfg if isinstance(cfg, dict) else None

    @property
    def closed_cleanly(self) -> bool:
        """Whether the writer finished normally. ``False`` implies a crash."""
        return bool(self.meta.get("closed_cleanly"))

    @property
    def model(self) -> ModelInfo | None:
        """Model identity from the metadata, if it was recorded there."""
        raw = self.meta.get("model")
        return ModelInfo.from_wire(raw) if isinstance(raw, dict) else None

    # ------------------------------------------------------------------
    # iteration
    # ------------------------------------------------------------------

    def _records(self, stats: ReadStats) -> Iterator[FrameDetections]:
        """Yield every detections record, counting what could not be read.

        Raises:
            IncidentLogError: A record announces a wire version this build does
                not speak. Refused rather than skipped, for the same reason the
                tablet refuses it: partially-understood records would look like
                a sparser log rather than an incompatible one.
        """
        pending_unterminated: str | None = None
        with open(self.detections_path, "r", encoding="utf-8", errors="replace") as handle:
            for raw in handle:
                stats.lines += 1
                terminated = raw.endswith("\n")
                text = raw.strip()
                if not text:
                    continue
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError:
                    if not terminated:
                        # Expected shape of a killed process: the last write did
                        # not complete. Everything before it is still valid.
                        pending_unterminated = text
                        stats.truncated_tail = True
                        continue
                    stats.corrupt += 1
                    log.warning("%s line %d is not valid JSON; record lost", self.detections_path, stats.lines)
                    continue
                if payload.get("v") != WIRE_VERSION:
                    raise IncidentLogError(
                        f"{self.detections_path} line {stats.lines} has wire version "
                        f"{payload.get('v')!r}; this build speaks v{WIRE_VERSION}. Read it with a "
                        "matching build rather than trusting a partial parse."
                    )
                if payload.get("type") != MSG_DETECTIONS:
                    stats.other_records += 1
                    continue
                try:
                    message = parse_message(payload)
                except (ValueError, KeyError, TypeError) as exc:
                    stats.corrupt += 1
                    log.warning("%s line %d is malformed (%s); record lost", self.detections_path, stats.lines, exc)
                    continue
                if not isinstance(message, FrameDetections):
                    stats.other_records += 1
                    continue
                stats.frames += 1
                yield message
        if pending_unterminated is not None:
            log.warning(
                "%s ends with an incomplete record (%d bytes); the station was stopped mid-write",
                self.detections_path,
                len(pending_unterminated),
            )

    def frames(
        self,
        *,
        classes: Sequence[str] | None = None,
        min_conf: float | None = None,
        min_persisted: int | None = None,
        track_id: int | None = None,
        start_pts: float | None = None,
        end_pts: float | None = None,
        start_wall: str | datetime | None = None,
        end_wall: str | datetime | None = None,
        keep_empty: bool = True,
    ) -> Iterator[FrameDetections]:
        """Iterate frame records in file order, optionally filtered.

        Args:
            classes: Keep only detections of these classes.
            min_conf: Keep only detections at or above this confidence.
            min_persisted: Keep only detections seen in at least this many of
                the last ``m`` frames.
            track_id: Keep only detections belonging to this track.
            start_pts: Earliest media pts to yield.
            end_pts: Latest media pts to yield.
            start_wall: Earliest wall time, RFC 3339 or a datetime.
            end_wall: Latest wall time.
            keep_empty: Yield frames whose (filtered) detection list is empty.
                Defaults to ``True``: the empty frames are the part of the log
                that a false-negative audit is actually about, so dropping
                them has to be an explicit choice by the caller.

        Yields:
            :class:`~station.core.types.FrameDetections`, with detections
            narrowed to those that passed the filter.

        Raises:
            IncidentLogError: The log announces an unsupported wire version.
        """
        flt = self._build_filter(
            classes=classes,
            min_conf=min_conf,
            min_persisted=min_persisted,
            track_id=track_id,
            start_pts=start_pts,
            end_pts=end_pts,
            start_wall=start_wall,
            end_wall=end_wall,
        )
        stats = ReadStats()
        self.last_read = stats
        for frame in self._records(stats):
            if not flt.in_window(frame):
                continue
            selected = flt.apply(frame)
            if not selected.detections and not keep_empty:
                continue
            yield selected

    def __iter__(self) -> Iterator[FrameDetections]:
        """Iterate every record, unfiltered, empty frames included."""
        return self.frames()

    def detections(self, **filters: Any) -> Iterator[tuple[FrameDetections, Detection]]:
        """Iterate individual detections with the frame each came from.

        Args:
            **filters: As :meth:`frames`.

        Yields:
            ``(frame, detection)`` pairs, in file order.
        """
        filters.setdefault("keep_empty", False)
        for frame in self.frames(**filters):
            for det in frame.detections:
                yield frame, det

    def replay(
        self,
        *,
        window: int = 256,
        **filters: Any,
    ) -> Iterator[FrameDetections]:
        """Yield frames in strict pts order, for scrubbing beside the video.

        The log is normally already in pts order, but a station that
        reconnected, or a future writer with a worker pool, could emit a
        record slightly late. Replay must not hand a scrubber a backwards
        timestamp -- that is what makes an overlay land on the wrong frame --
        so records pass through a small reordering heap first.

        Args:
            window: How many records to hold while reordering. Bounded so a
                twelve-hour log does not have to fit in memory.
            **filters: As :meth:`frames`.

        Yields:
            Frames ordered by ``pts``, ascending.
        """
        heap: list[tuple[float, int, FrameDetections]] = []
        seq = 0
        for frame in self.frames(**filters):
            heapq.heappush(heap, (frame.pts, seq, frame))
            seq += 1
            if len(heap) > max(1, window):
                yield heapq.heappop(heap)[2]
        while heap:
            yield heapq.heappop(heap)[2]

    def events(self) -> Iterator[dict[str, Any]]:
        """Iterate ``events.jsonl``: pipeline status and operator notes.

        Yields:
            One decoded event mapping per line. Missing file yields nothing --
            older or shorter incidents may have no events at all.
        """
        path = self.dir / EVENTS_FILENAME
        if not path.is_file():
            return
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            for raw in handle:
                text = raw.strip()
                if not text:
                    continue
                try:
                    yield json.loads(text)
                except json.JSONDecodeError:
                    continue  # truncated tail of the event log; nothing to salvage

    # ------------------------------------------------------------------
    # summary
    # ------------------------------------------------------------------

    def summarise(
        self,
        *,
        classes: Sequence[str] | None = None,
        min_conf: float | None = None,
        min_persisted: int | None = None,
        track_id: int | None = None,
        start_pts: float | None = None,
        end_pts: float | None = None,
        start_wall: str | datetime | None = None,
        end_wall: str | datetime | None = None,
    ) -> IncidentSummary:
        """Summarise the log in one pass.

        Args:
            classes: Restrict the detection counts to these classes.
            min_conf: Restrict the detection counts to this confidence and up.
            min_persisted: Restrict to detections with at least this much
                temporal persistence.
            track_id: Restrict to one track.
            start_pts: Window start on the media timeline.
            end_pts: Window end on the media timeline.
            start_wall: Window start on the wall clock.
            end_wall: Window end on the wall clock.

        Returns:
            An :class:`IncidentSummary`. ``frames_empty`` counts frames whose
            result was empty *after* the filter, which is not the same as
            frames the model returned nothing on -- the summary text says so.
        """
        flt = self._build_filter(
            classes=classes,
            min_conf=min_conf,
            min_persisted=min_persisted,
            track_id=track_id,
            start_pts=start_pts,
            end_pts=end_pts,
            start_wall=start_wall,
            end_wall=end_wall,
        )
        stats = ReadStats()
        self.last_read = stats
        summary = IncidentSummary(
            incident_id=self.incident_id,
            station_name=self.station_name,
            source_uri=self.source_uri,
            model=self.model,
            started_at=self.meta.get("started_at"),
            ended_at=self.meta.get("ended_at"),
            closed_cleanly=self.closed_cleanly,
            filter_description=flt.describe(),
            detection_filter=flt.selects_detections,
            read=stats,
            video=self.video_manifest(),
        )

        prev_pts: float | None = None
        interval_sum = 0.0
        interval_count = 0
        inference_sum = 0.0
        inference_count = 0
        run_frames = 0
        run_start: float | None = None
        model_seen: ModelInfo | None = None

        for frame in self._records(stats):
            summary.frames_logged += 1
            if not flt.in_window(frame):
                continue
            summary.frames_in_window += 1
            if model_seen is None and frame.model is not None:
                model_seen = frame.model
            if summary.first_pts is None:
                summary.first_pts = frame.pts
                summary.first_wall_time = frame.wall_time or None
            summary.last_pts = frame.pts
            if frame.wall_time:
                summary.last_wall_time = frame.wall_time

            if prev_pts is not None:
                delta = frame.pts - prev_pts
                if delta > 0:
                    interval_sum += delta
                    interval_count += 1
                    if delta > summary.max_gap_s:
                        summary.max_gap_s = delta
                        summary.max_gap_at_pts = prev_pts
            prev_pts = frame.pts

            if frame.inference_ms is not None:
                inference_sum += float(frame.inference_ms)
                inference_count += 1
                summary.inference_ms_max = max(summary.inference_ms_max or 0.0, float(frame.inference_ms))

            kept = [d for d in frame.detections if flt.keep(d)]
            if kept:
                summary.frames_with_detections += 1
                summary.detections_total += len(kept)
                for det in kept:
                    entry = summary.classes.get(det.cls)
                    if entry is None:
                        entry = summary.classes[det.cls] = ClassSummary(cls=det.cls)
                    entry.observe(det, frame.pts)
                for name in {d.cls for d in kept}:
                    summary.classes[name].frames += 1
                if run_frames == 0:
                    run_start = frame.pts
                run_frames += 1
                if run_frames > summary.longest_run_frames:
                    summary.longest_run_frames = run_frames
                    summary.longest_run_pts = (run_start or frame.pts, frame.pts)
            else:
                summary.frames_empty += 1
                run_frames = 0
                run_start = None

        if summary.model is None:
            summary.model = model_seen
        if interval_count:
            summary.mean_interval_s = interval_sum / interval_count
        if inference_count:
            summary.inference_ms_mean = inference_sum / inference_count
        return summary

    # ------------------------------------------------------------------
    # video correspondence
    # ------------------------------------------------------------------

    def video_manifest(self) -> dict[str, Any] | None:
        """Load ``video/segments.json``, or the copy inside ``meta.json``.

        Returns:
            The recorder's manifest, or ``None`` when no video was recorded.
        """
        if self._video_loaded:
            return self._video
        self._video_loaded = True
        path = self.dir / VIDEO_SUBDIR / MANIFEST_FILENAME
        if path.is_file():
            try:
                self._video = json.loads(path.read_text(encoding="utf-8"))
                return self._video
            except (OSError, json.JSONDecodeError) as exc:
                log.warning("%s is unreadable (%s)", path, exc)
        meta_video = self.meta.get("video")
        self._video = meta_video if isinstance(meta_video, dict) else None
        return self._video

    def _frame_index(self) -> list[tuple[float, int, int, float, int]]:
        """Load ``video/frames.jsonl`` as ``(pts, segment, i, offset, frame_id)``.

        Loaded whole and sorted, which is fine for a review tool on a laptop:
        an hour at 30 fps is about 108k rows. Memory is traded for an exact
        pts -> frame mapping that does not depend on the muxer having kept the
        timestamps we asked it for.
        """
        if self._index is not None:
            return self._index
        index: list[tuple[float, int, int, float, int]] = []
        path = self.dir / VIDEO_SUBDIR / FRAME_INDEX_FILENAME
        if path.is_file():
            with open(path, "r", encoding="utf-8", errors="replace") as handle:
                for raw in handle:
                    text = raw.strip()
                    if not text:
                        continue
                    try:
                        row = json.loads(text)
                        index.append(
                            (float(row["pts"]), int(row["s"]), int(row["i"]), float(row["t"]), int(row["f"]))
                        )
                    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                        continue  # truncated tail or a damaged row; the rest still maps
            index.sort()
        self._index = index
        return index

    def video_position(self, pts: float) -> VideoPosition | None:
        """Locate the recorded frame a media pts belongs to.

        This is the join that makes the log checkable: given the pts of a
        detection, it returns the segment file and the offset to seek to.

        Args:
            pts: Media presentation timestamp, as logged.

        Returns:
            A :class:`VideoPosition`, or ``None`` when there is no recording
            or the pts falls outside it.
        """
        manifest = self.video_manifest()
        if not manifest:
            return None
        segments = manifest.get("segments") or []
        if not segments:
            return None
        video_dir = self.dir / str(manifest.get("dir") or VIDEO_SUBDIR)
        fps = float(manifest.get("fps") or 0.0) or None
        exact_pts = bool(manifest.get("pts_exact"))

        index = self._frame_index()
        if index:
            # Nearest recorded frame at or before pts; the recorder samples at
            # the source rate and inference at a lower one, so an exact hit is
            # normal but not guaranteed.
            slot = bisect.bisect_right(index, (pts, 10**9, 10**9, 1e18, 10**9)) - 1
            if slot < 0:
                slot = 0
            frame_pts, seg_index, frame_i, offset, frame_id = index[slot]
            if seg_index < len(segments):
                seg = segments[seg_index]
                # On the OpenCV path the file has no timestamps at all: frame i
                # is at i/fps, full stop. Using the pts offset there would
                # accumulate every dropped frame as drift.
                if not exact_pts and fps:
                    offset = frame_i / fps
                return VideoPosition(
                    segment=seg_index,
                    path=video_dir / str(seg.get("file")),
                    offset_s=max(0.0, offset),
                    exact=exact_pts and abs(frame_pts - pts) < 1e-6,
                    frame_id=frame_id,
                    frame_index=frame_i,
                )

        for seg in segments:
            start = float(seg.get("start_pts", 0.0))
            end = seg.get("end_pts")
            end_pts = float(end) if end is not None else float("inf")
            if start <= pts <= end_pts:
                return VideoPosition(
                    segment=int(seg.get("index", 0)),
                    path=video_dir / str(seg.get("file")),
                    offset_s=max(0.0, pts - start),
                    exact=exact_pts,
                )
        return None

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    @staticmethod
    def _build_filter(
        *,
        classes: Sequence[str] | None,
        min_conf: float | None,
        min_persisted: int | None,
        track_id: int | None,
        start_pts: float | None,
        end_pts: float | None,
        start_wall: str | datetime | None,
        end_wall: str | datetime | None,
    ) -> _Filter:
        """Normalise keyword filters into a :class:`_Filter`."""
        def wall(value: str | datetime | None) -> datetime | None:
            if value is None or isinstance(value, datetime):
                return value
            parsed = _parse_wall(value)
            if parsed is None:
                raise IncidentLogError(f"cannot parse wall time {value!r}; expected RFC 3339")
            return parsed

        return _Filter(
            classes=tuple(classes or ()),
            min_conf=min_conf,
            min_persisted=min_persisted,
            track_id=track_id,
            start_pts=start_pts,
            end_pts=end_pts,
            start_wall=wall(start_wall),
            end_wall=wall(end_wall),
        )


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for ``python -m station.incidentlog.reader``."""
    parser = argparse.ArgumentParser(
        prog="python -m station.incidentlog.reader",
        description=(
            "Read back a wildfire-watch incident log. Reports what the model output, "
            "frame by frame. It cannot report what was in the scene."
        ),
        epilog=EMPTY_RESULT_CAVEAT.replace("\n", " "),
    )
    parser.add_argument("incident_dir", help="incident directory (or its detections.jsonl)")
    parser.add_argument(
        "--class",
        dest="classes",
        action="append",
        metavar="NAME",
        help="keep only this class; repeatable (e.g. --class fire --class smoke)",
    )
    parser.add_argument("--min-conf", type=float, metavar="C", help="keep only detections with conf >= C")
    parser.add_argument(
        "--min-persisted", type=int, metavar="N", help="keep only detections seen in >= N of the last m frames"
    )
    parser.add_argument("--track", type=int, metavar="ID", help="keep only detections on track ID")
    parser.add_argument("--since", type=float, metavar="PTS", help="start of the media-time window, seconds")
    parser.add_argument("--until", type=float, metavar="PTS", help="end of the media-time window, seconds")
    parser.add_argument("--after", metavar="RFC3339", help="start of the wall-clock window")
    parser.add_argument("--before", metavar="RFC3339", help="end of the wall-clock window")
    parser.add_argument("--summary", action="store_true", help="print the after-action summary")
    parser.add_argument("--json", action="store_true", help="emit matching records as JSONL on stdout")
    parser.add_argument(
        "--empty",
        action="store_true",
        help="also list frames whose result was empty (they are always counted in the summary)",
    )
    parser.add_argument("--replay", action="store_true", help="emit in strict pts order rather than file order")
    parser.add_argument("--video", action="store_true", help="append the recorded video position to each line")
    parser.add_argument("--events", action="store_true", help="print the pipeline event log instead of detections")
    parser.add_argument("--limit", type=int, metavar="N", help="stop after N printed records")
    parser.add_argument("-v", "--verbose", action="store_true", help="show warnings about damaged records")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point.

    Args:
        argv: Arguments, defaulting to ``sys.argv[1:]``.

    Returns:
        Process exit status: 0 on success, 2 when the log cannot be read.
    """
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(message)s",
    )
    try:
        reader = IncidentLogReader(args.incident_dir)
    except IncidentLogError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    filters: dict[str, Any] = {
        "classes": args.classes,
        "min_conf": args.min_conf,
        "min_persisted": args.min_persisted,
        "track_id": args.track,
        "start_pts": args.since,
        "end_pts": args.until,
        "start_wall": args.after,
        "end_wall": args.before,
    }

    try:
        if args.events:
            for event in reader.events():
                print(json.dumps(event, separators=(",", ":")))
            return 0

        # Listing is the default action; --summary adds the report. When only
        # --summary is asked for, the per-frame listing is suppressed so the
        # report is not buried under ten thousand lines.
        listing = not args.summary or args.json
        printed = 0
        if listing:
            stream: Iterable[FrameDetections]
            stream = reader.replay(**filters) if args.replay else reader.frames(**filters)
            for frame in stream:
                if not frame.detections and not args.empty:
                    continue
                if args.json:
                    print(frame.to_json())
                else:
                    position = reader.video_position(frame.pts) if args.video else None
                    print(format_frame(frame, position=position))
                printed += 1
                if args.limit is not None and printed >= args.limit:
                    break
            if not args.json and not args.summary:
                stats = reader.last_read
                # Always state the denominator. "12 frames had detections" on
                # its own invites the reader to supply their own meaning for
                # the other 9,000.
                print(
                    f"\n{printed} record(s) printed from {stats.frames} frame(s) in the log."
                    + (f" {stats.corrupt} damaged record(s)." if stats.corrupt else "")
                )
                print(EMPTY_RESULT_CAVEAT)

        if args.summary:
            if listing:
                print()
            print(reader.summarise(**filters).to_text())
    except IncidentLogError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except BrokenPipeError:  # piping into head is normal usage
        return 0
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
