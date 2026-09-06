"""Append-only incident log: every frame the model looked at, as it saw it.

This is the subsystem that makes invariant 4 real. Every inference result is
written here -- *including* the empty ones. A log that recorded only frames
with detections would be a highlight reel of what the model happened to catch,
which is precisely the wrong artefact: the question an after-action review has
to answer is "what did the model do on the frames where a human later found
fire", and that question is unanswerable if the quiet frames were never
written down. It is also how the false-negative validation set in
``docs/VALIDATION.md`` accumulates: real flights, real conditions, with the
model's actual output beside them.

Nothing in this file is allowed to take the video feed down. Logging is a
forensic nicety in the moment and a legal/engineering necessity afterwards,
but an operator standing on a fire line watching a tablet needs the video more
than they need the last line of a JSONL file. Every failure path here
therefore degrades to a warning and a counter -- see :meth:`_degrade` -- and
the writer keeps trying. The trade-off is deliberate and it points one way:
a lost log line is recoverable, a dropped feed during an incident is not.

Crash-safety, because a station in the field gets its battery pulled:

* ``detections.jsonl`` is append-only, one complete ``write()`` per record,
  fsynced every ``IncidentLogConfig.flush_every`` frames (default: every
  frame). A killed process leaves a file whose last line may be truncated and
  whose earlier lines are all intact and parseable;
  :class:`~station.incidentlog.reader.IncidentLogReader` tolerates exactly
  that shape.
* ``meta.json`` is written at open, refreshed periodically, and rewritten at
  close, always via a temp file plus :func:`os.replace`, so it is never
  observed half-written. ``closed_cleanly`` in the final copy is what tells a
  reader whether the counters can be trusted or have to be recounted.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import re
import socket
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import Any, Iterable, Mapping

from station.core.config import Config, IncidentLogConfig
from station.core.types import (
    WIRE_VERSION,
    FrameDetections,
    ModelInfo,
    PipelineStatus,
    utc_now_iso,
)

__all__ = [
    "DETECTIONS_FILENAME",
    "META_FILENAME",
    "EVENTS_FILENAME",
    "VIDEO_SUBDIR",
    "EMPTY_RESULT_NOTE",
    "IncidentCounters",
    "IncidentLogWriter",
    "new_incident_id",
    "slugify",
    "redact_uri",
]

log = logging.getLogger(__name__)

DETECTIONS_FILENAME = "detections.jsonl"
META_FILENAME = "meta.json"
EVENTS_FILENAME = "events.jsonl"
VIDEO_SUBDIR = "video"

#: Copied verbatim into every ``meta.json``. Incident logs outlive the people
#: who made them and get read by someone with no context; the file has to
#: carry its own reading instructions, or a stretch of empty results will be
#: misread as a stretch of nothing happening.
EMPTY_RESULT_NOTE = (
    "detections.jsonl contains one record per inferred frame, including frames "
    "where the model returned nothing. An empty 'detections' list means only "
    "that this model returned nothing on that frame -- it is not a measurement "
    "of the scene, and it is not evidence that the scene was empty. Thin smoke "
    "on bright sky, smouldering without flame, fire under canopy and fire at "
    "night all produce empty results. Read the recorded video."
)

_SLUG_STRIP = re.compile(r"[^a-z0-9]+")
#: ``scheme://user:password@host`` -- the password is what we blank.
_URI_CREDENTIALS = re.compile(r"^(?P<head>[a-zA-Z][\w+.-]*://[^/@:]+):(?P<pw>[^/@]*)@")


def slugify(text: str, *, max_len: int = 40) -> str:
    """Reduce arbitrary text to a filesystem-safe incident-id fragment.

    Args:
        text: Free text -- a station name, a source URI, an operator's label.
        max_len: Truncation length for the result.

    Returns:
        Lowercase ``a-z0-9-`` with no leading or trailing dashes, possibly
        empty if ``text`` contained nothing usable.
    """
    return _SLUG_STRIP.sub("-", text.strip().lower()).strip("-")[:max_len].strip("-")


def new_incident_id(*, when: datetime | None = None, slug: str = "") -> str:
    """Build a sortable incident id: ``YYYYmmdd-HHMMSSZ`` plus an optional slug.

    UTC and lexicographically sortable so that ``ls incidents/`` is in
    chronological order on any station in any timezone -- crews and aircraft
    cross timezones, and an incident directory that sorts wrong is one that
    gets grabbed wrong at 3am.

    Args:
        when: Timestamp to name the incident after. Defaults to now, UTC.
        slug: Optional trailing label; slugified defensively.

    Returns:
        The incident id, e.g. ``20260906-142231Z-flight-3``.
    """
    moment = (when or datetime.now(timezone.utc)).astimezone(timezone.utc)
    stamp = moment.strftime("%Y%m%d-%H%M%SZ")
    tail = slugify(slug)
    return f"{stamp}-{tail}" if tail else stamp


def redact_uri(uri: str) -> str:
    """Blank the password in a URI, keeping everything else readable.

    Incident logs get copied around -- onto a USB stick, into an after-action
    report, attached to an email. The drone downlink's RTSP credentials must
    not travel with them. The host, port and path survive because they are
    what makes the log identifiable.

    Args:
        uri: Any URI, possibly containing ``user:password@``.

    Returns:
        The URI with the password replaced by ``***``. Unchanged if there is
        no password to redact.
    """
    return _URI_CREDENTIALS.sub(lambda m: f"{m.group('head')}:***@", uri or "")


@dataclass(slots=True)
class IncidentCounters:
    """Running totals for ``meta.json``.

    ``frames_empty`` is a first-class counter and not an afterthought: the
    ratio of empty to non-empty frames is the headline number of any
    false-negative audit, and the audit needs the denominator.
    """

    frames_logged: int = 0
    #: Frames whose result was an empty list. Not "quiet frames", not "clear
    #: frames" -- frames on which this model returned nothing.
    frames_empty: int = 0
    frames_with_detections: int = 0
    detections_total: int = 0
    per_class: dict[str, int] = field(default_factory=dict)
    events_logged: int = 0
    #: Records the writer failed to persist. Non-zero means this log has holes
    #: and any audit over it is incomplete; the reader says so out loud.
    write_errors: int = 0
    first_pts: float | None = None
    last_pts: float | None = None
    first_wall_time: str | None = None
    last_wall_time: str | None = None
    bytes_written: int = 0

    def observe(self, frame: FrameDetections, nbytes: int) -> None:
        """Fold one successfully written frame record into the totals."""
        self.frames_logged += 1
        self.bytes_written += nbytes
        count = len(frame.detections)
        if count:
            self.frames_with_detections += 1
            self.detections_total += count
            for det in frame.detections:
                self.per_class[det.cls] = self.per_class.get(det.cls, 0) + 1
        else:
            self.frames_empty += 1
        if self.first_pts is None:
            self.first_pts = frame.pts
            self.first_wall_time = frame.wall_time or None
        self.last_pts = frame.pts
        if frame.wall_time:
            self.last_wall_time = frame.wall_time

    def as_dict(self) -> dict[str, Any]:
        """JSON-ready snapshot of the counters."""
        return dataclasses.asdict(self)


def _sanitise(value: Any) -> Any:
    """Coerce an arbitrary object into something ``json.dump`` will accept."""
    return json.loads(json.dumps(value, default=str))


def _snapshot_config(cfg: Config | Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Freeze the running configuration into the incident metadata.

    Without the config snapshot an incident log cannot be interpreted: the
    confidence threshold, the N-of-M window and the inference rate all change
    what "the model returned nothing" means on any given frame.
    """
    if cfg is None:
        return None
    raw = dataclasses.asdict(cfg) if dataclasses.is_dataclass(cfg) else dict(cfg)
    snap = _sanitise(raw)
    source = snap.get("source")
    if isinstance(source, dict) and source.get("uri"):
        source["uri"] = redact_uri(str(source["uri"]))
    return snap


class IncidentLogWriter:
    """Writes one incident directory: detections, events and metadata.

    Layout under ``<root>/<incident-id>/``::

        detections.jsonl   one FrameDetections.to_wire() per line, every frame
        events.jsonl       pipeline status/state changes (created on demand)
        meta.json          station, config, model, source, times, counters
        video/             segments written by station.incidentlog.recorder

    Use it as a context manager, or call :meth:`start` and :meth:`close`
    yourself. Every public method is safe to call on a writer whose disk has
    filled up, whose directory has been deleted underneath it, or which was
    configured off entirely -- the caller does not branch on any of that.

    Example:
        >>> from station.core.config import Config
        >>> import tempfile
        >>> tmp = tempfile.mkdtemp()
        >>> cfg = Config()
        >>> cfg.incident_log.dir = tmp
        >>> with IncidentLogWriter.from_config(cfg, slug="doctest") as wr:
        ...     ok = wr.write(FrameDetections(frame_id=0, pts=0.0))
        >>> ok
        True
    """

    def __init__(
        self,
        incident_dir: str | Path,
        *,
        station_name: str = "",
        source_uri: str = "",
        model: ModelInfo | None = None,
        config_snapshot: Config | Mapping[str, Any] | None = None,
        flush_every: int = 1,
        enabled: bool = True,
        meta_refresh_s: float = 30.0,
        extra_meta: Mapping[str, Any] | None = None,
    ) -> None:
        """Prepare a writer. No filesystem work happens until :meth:`start`.

        Args:
            incident_dir: Directory for this incident. Created on start; a
                collision is resolved by appending ``-2``, ``-3``, ... so two
                stations (or a restart within the same second) never
                interleave records into one file.
            station_name: Operator-facing station identity, into ``meta.json``.
            source_uri: The video source. Stored redacted.
            model: What is running. May be omitted and adopted from the first
                frame that carries :class:`~station.core.types.ModelInfo`.
            config_snapshot: The whole :class:`~station.core.config.Config`,
                or any mapping. Frozen into ``meta.json``.
            flush_every: fsync cadence in frames; ``<= 1`` fsyncs every frame.
            enabled: When ``False`` every method is a no-op that returns
                ``False``. The pipeline calls the same methods either way.
            meta_refresh_s: Seconds between metadata refreshes while running,
                so a crashed log still has roughly-right counters. ``<= 0``
                disables refreshing (meta is still written at open and close).
            extra_meta: Anything else worth recording, merged into
                ``meta.json`` under ``extra``.
        """
        self._requested_dir = Path(incident_dir)
        self._dir: Path | None = None
        self.station_name = station_name
        self.source_uri = redact_uri(source_uri)
        self._raw_source_uri = source_uri
        self.model = model
        self.enabled = bool(enabled)
        self.flush_every = max(1, int(flush_every))
        self.meta_refresh_s = float(meta_refresh_s)
        self._config_snapshot = _snapshot_config(config_snapshot)
        self._extra_meta = dict(extra_meta or {})

        self.counters = IncidentCounters()
        self.started_at: str | None = None
        self.ended_at: str | None = None
        self._started_monotonic: float | None = None
        self._fh: Any = None
        self._events_fh: Any = None
        self._since_flush = 0
        self._last_meta_refresh = 0.0
        self._closed = False
        self._healthy = True
        self._error_streak = 0
        self._video_manifest: dict[str, Any] | None = None
        self._notes: list[str] = []

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        cfg: Config,
        *,
        model: ModelInfo | None = None,
        slug: str | None = None,
        incident_id: str | None = None,
        root: str | Path | None = None,
        extra_meta: Mapping[str, Any] | None = None,
    ) -> "IncidentLogWriter":
        """Build a writer from the station config.

        Args:
            cfg: The loaded station configuration. ``incident_log.dir``,
                ``incident_log.flush_every`` and ``incident_log.enabled`` are
                taken from it, and the whole thing is snapshotted into
                ``meta.json``.
            model: Model identity, if already known at startup.
            slug: Label appended to the generated incident id. Defaults to a
                slug of the source URI, which is what an operator recognises.
            incident_id: Use this exact id instead of generating one.
            root: Override ``cfg.incident_log.dir``.
            extra_meta: Merged into ``meta.json`` under ``extra``.

        Returns:
            An unstarted :class:`IncidentLogWriter`.
        """
        log_cfg: IncidentLogConfig = cfg.incident_log
        base = Path(root) if root is not None else Path(log_cfg.dir)
        # The URI, not the station name, is the useful slug: one station runs
        # many flights and the source is what distinguishes them.
        label = slug if slug is not None else slugify(redact_uri(cfg.source.uri) or cfg.source.type)
        ident = incident_id or new_incident_id(slug=label)
        return cls(
            base / ident,
            station_name=cfg.station_name,
            source_uri=cfg.source.uri,
            model=model,
            config_snapshot=cfg,
            flush_every=log_cfg.flush_every,
            enabled=log_cfg.enabled,
            extra_meta=extra_meta,
        )

    # ------------------------------------------------------------------
    # properties
    # ------------------------------------------------------------------

    @property
    def dir(self) -> Path | None:
        """The incident directory, or ``None`` before :meth:`start`."""
        return self._dir

    @property
    def incident_id(self) -> str:
        """Directory name of this incident (the requested one until started)."""
        return (self._dir or self._requested_dir).name

    @property
    def detections_path(self) -> Path | None:
        """Path of ``detections.jsonl``, or ``None`` before :meth:`start`."""
        return None if self._dir is None else self._dir / DETECTIONS_FILENAME

    @property
    def video_dir(self) -> Path | None:
        """Directory the recorder should write segments into."""
        return None if self._dir is None else self._dir / VIDEO_SUBDIR

    @property
    def healthy(self) -> bool:
        """``False`` once any write has failed. The log then has holes."""
        return self._healthy

    @property
    def closed(self) -> bool:
        """``True`` after :meth:`close`."""
        return self._closed

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def start(self) -> "IncidentLogWriter":
        """Create the incident directory and open the log. Idempotent.

        Returns:
            ``self``, so it chains.
        """
        if not self.enabled or self._dir is not None or self._closed:
            return self
        try:
            self._dir = self._make_dir(self._requested_dir)
            self._fh = open(self._dir / DETECTIONS_FILENAME, "a", encoding="utf-8", newline="\n")
            self.started_at = utc_now_iso()
            self._started_monotonic = time.monotonic()
            self._write_meta()
            self._last_meta_refresh = time.monotonic()
            log.info("incident log open: %s", self._dir)
        except Exception as exc:  # noqa: BLE001 - see module docstring
            self._degrade("open incident log", exc)
            self.enabled = False  # nothing to retry against; stop trying quietly
        return self

    def _make_dir(self, wanted: Path) -> Path:
        """Create ``wanted``, or ``wanted-2``, ``-3``... if it already exists.

        Never appends into an existing incident directory. Two sessions
        sharing one ``detections.jsonl`` would produce a file with two
        interleaved, separately-based pts timelines, and pts is the only key
        that ties a detection to the frame it came from.
        """
        wanted.parent.mkdir(parents=True, exist_ok=True)
        candidate = wanted
        suffix = 1
        while True:
            try:
                candidate.mkdir()
                return candidate
            except FileExistsError:
                suffix += 1
                candidate = wanted.with_name(f"{wanted.name}-{suffix}")
                if suffix > 999:
                    raise

    def __enter__(self) -> "IncidentLogWriter":
        return self.start()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        # An exception on the way out is itself worth logging: it is the most
        # likely explanation for why the record stops where it does.
        if exc is not None:
            self.note(f"session ended with {exc_type.__name__ if exc_type else 'error'}: {exc}")
        self.close()

    def close(self) -> None:
        """Flush, fsync, finalise ``meta.json``. Safe to call twice."""
        if self._closed:
            return
        self._closed = True
        if self._dir is None:
            return
        self.ended_at = utc_now_iso()
        for handle in (self._fh, self._events_fh):
            if handle is None:
                continue
            try:
                handle.flush()
                os.fsync(handle.fileno())
            except Exception as exc:  # noqa: BLE001
                self._degrade("flush incident log", exc)
            finally:
                try:
                    handle.close()
                except Exception:  # noqa: BLE001 - already closing down
                    pass
        self._fh = None
        self._events_fh = None
        self._write_meta(final=True)
        log.info(
            "incident log closed: %s (%d frames, %d with detections, %d empty, %d write errors)",
            self._dir,
            self.counters.frames_logged,
            self.counters.frames_with_detections,
            self.counters.frames_empty,
            self.counters.write_errors,
        )

    # ------------------------------------------------------------------
    # writing
    # ------------------------------------------------------------------

    def write(self, frame: FrameDetections) -> bool:
        """Append one frame result, detections or not.

        Args:
            frame: The post-temporal-filter result for one frame. Frames with
                an empty ``detections`` tuple are written exactly like any
                other; that is the point of the file.

        Returns:
            ``True`` if the record reached the file, ``False`` if logging is
            off or the write failed. Never raises: a logging fault must not
            propagate into the frame loop and take the video down.
        """
        if not self.enabled or self._closed:
            return False
        if self._dir is None:
            self.start()
            if self._dir is None:
                return False
        try:
            if self.model is None and frame.model is not None:
                # The runner knows the weights hash; adopt it the first time we
                # see it so meta.json identifies the model even when the
                # pipeline could not supply it at startup.
                self.model = frame.model
            line = json.dumps(frame.to_wire(), separators=(",", ":")) + "\n"
            # One write() call per record: a partially written line can then
            # only ever be the final line of the file, which the reader knows
            # how to discard. Interleaved half-records would be unreadable.
            self._fh.write(line)
            self.counters.observe(frame, len(line))
            self._since_flush += 1
            if self._since_flush >= self.flush_every:
                self._sync()
            self._maybe_refresh_meta()
            return True
        except Exception as exc:  # noqa: BLE001
            self._degrade("write detection record", exc)
            return False

    def write_many(self, frames: Iterable[FrameDetections]) -> int:
        """Append several frame results.

        Args:
            frames: The results to append, in order.

        Returns:
            How many records reached the file.
        """
        return sum(1 for frame in frames if self.write(frame))

    def log_status(self, status: PipelineStatus) -> bool:
        """Record a pipeline heartbeat or state change in ``events.jsonl``.

        A gap in ``detections.jsonl`` is ambiguous on its own: the pipeline may
        have stalled, or the source may have dropped, or inference may simply
        have been rate-limited. The event stream is what disambiguates it
        during review, so a reviewer never has to guess whether a quiet stretch
        was a quiet stretch or a dead pipeline.

        Args:
            status: The status object as sent on the data channel.

        Returns:
            ``True`` if the event was written.
        """
        return self.log_event("status", **status.to_wire())

    def log_event(self, kind: str, **fields: Any) -> bool:
        """Record an arbitrary structured event in ``events.jsonl``.

        Args:
            kind: Short event tag, e.g. ``"status"``, ``"source"``, ``"note"``.
            **fields: JSON-serialisable payload merged into the record.

        Returns:
            ``True`` if the event was written.
        """
        if not self.enabled or self._closed:
            return False
        if self._dir is None:
            self.start()
            if self._dir is None:
                return False
        try:
            if self._events_fh is None:
                self._events_fh = open(self._dir / EVENTS_FILENAME, "a", encoding="utf-8", newline="\n")
            record = {"wall_time": utc_now_iso(), "kind": kind}
            record.update(fields)
            self._events_fh.write(json.dumps(_sanitise(record), separators=(",", ":")) + "\n")
            self._events_fh.flush()
            os.fsync(self._events_fh.fileno())
            self.counters.events_logged += 1
            return True
        except Exception as exc:  # noqa: BLE001
            self._degrade("write event record", exc)
            return False

    def note(self, text: str) -> None:
        """Attach an operator-facing note to ``meta.json`` and the event log.

        Args:
            text: Free text, e.g. why the session ended.
        """
        self._notes.append(f"{utc_now_iso()} {text}")
        self.log_event("note", text=text)

    def set_video_manifest(self, manifest: Mapping[str, Any] | None) -> None:
        """Record where the video for this incident lives.

        Args:
            manifest: :meth:`station.incidentlog.recorder.VideoRecorder.manifest`
                output, or ``None`` if recording was off or failed. Written
                into ``meta.json`` so a reviewer can find the frames the
                detections refer to without knowing the recorder's layout.
        """
        self._video_manifest = _sanitise(dict(manifest)) if manifest is not None else None

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _sync(self) -> None:
        """flush + fsync the detections file, resetting the cadence counter."""
        self._since_flush = 0
        self._fh.flush()
        os.fsync(self._fh.fileno())

    def _maybe_refresh_meta(self) -> None:
        """Rewrite ``meta.json`` if the refresh interval has elapsed.

        Cheap insurance: if the station is killed mid-incident, the metadata on
        disk is at most ``meta_refresh_s`` out of date instead of being the
        empty snapshot taken at startup. The counters are still marked with
        ``counters_as_of`` and ``closed_cleanly: false`` so nobody mistakes
        them for final.
        """
        if self.meta_refresh_s <= 0:
            return
        now = time.monotonic()
        if now - self._last_meta_refresh < self.meta_refresh_s:
            return
        self._last_meta_refresh = now
        self._write_meta()

    def _meta(self, *, final: bool) -> dict[str, Any]:
        """Assemble the ``meta.json`` payload."""
        uptime = None
        if self._started_monotonic is not None:
            uptime = round(time.monotonic() - self._started_monotonic, 3)
        meta: dict[str, Any] = {
            "incident_id": self.incident_id,
            "wire_version": WIRE_VERSION,
            "station_name": self.station_name,
            "source_uri": self.source_uri,
            "started_at": self.started_at,
            "ended_at": self.ended_at if final else None,
            "closed_cleanly": bool(final),
            "duration_s": uptime,
            "model": self.model.to_wire() if self.model is not None else None,
            "config": self._config_snapshot,
            "counters": self.counters.as_dict(),
            "counters_as_of": utc_now_iso(),
            "files": {
                "detections": DETECTIONS_FILENAME,
                "events": EVENTS_FILENAME if self.counters.events_logged else None,
                "video_dir": VIDEO_SUBDIR,
            },
            "video": self._video_manifest,
            "writer": {
                "host": socket.gethostname(),
                "pid": os.getpid(),
                "python": sys.version.split()[0],
                "flush_every": self.flush_every,
                "healthy": self._healthy,
            },
            "notes": list(self._notes),
            "reading_this_log": EMPTY_RESULT_NOTE,
        }
        if self._extra_meta:
            meta["extra"] = _sanitise(self._extra_meta)
        return meta

    def _write_meta(self, *, final: bool = False) -> None:
        """Atomically replace ``meta.json``.

        temp file -> fsync -> rename -> fsync directory. A reader that opens
        ``meta.json`` at any instant sees either the previous complete copy or
        the new complete copy, never a truncated one.
        """
        if self._dir is None:
            return
        target = self._dir / META_FILENAME
        tmp = self._dir / f".{META_FILENAME}.tmp"
        try:
            payload = json.dumps(self._meta(final=final), indent=2, sort_keys=False, default=str)
            with open(tmp, "w", encoding="utf-8") as handle:
                handle.write(payload + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, target)
            self._fsync_dir(self._dir)
        except Exception as exc:  # noqa: BLE001
            self._degrade("write incident metadata", exc)

    @staticmethod
    def _fsync_dir(path: Path) -> None:
        """fsync a directory so a rename into it survives power loss."""
        try:
            fd = os.open(path, os.O_RDONLY)
        except OSError:
            return  # some filesystems will not open a directory; not fatal
        try:
            os.fsync(fd)
        except OSError:
            pass
        finally:
            os.close(fd)

    def _degrade(self, what: str, exc: BaseException) -> None:
        """Absorb a logging failure: count it, warn about it, carry on.

        Rate-limited on purpose. A full disk fails on every single frame, and
        a warning per frame would bury the operator-facing log -- which is
        also where the pipeline reports that it has stalled, the message that
        actually matters on a fire line.
        """
        self._healthy = False
        self.counters.write_errors += 1
        self._error_streak += 1
        if self._error_streak == 1 or self._error_streak % 100 == 0:
            log.warning(
                "incident log: failed to %s (%s: %s); the video feed is unaffected, "
                "but this log now has %d missing record(s)",
                what,
                type(exc).__name__,
                exc,
                self.counters.write_errors,
            )
