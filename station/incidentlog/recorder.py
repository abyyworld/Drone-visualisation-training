"""Record the source video beside the log, on the same pts timeline.

``detections.jsonl`` says "at pts 137.4667 the model put a box here". That
sentence is only checkable if the frame at pts 137.4667 still exists. Recording
the video the station actually decoded -- not a re-encode of something else,
not a screen capture with the boxes burned in -- is what turns the incident log
from a list of assertions into evidence, and it is what supplies the
false-negative validation set in ``docs/VALIDATION.md``: real flights where a
human, reviewing afterwards, can point at a frame the model said nothing about.

**The pts correspondence is the whole point of this module.** Two mechanisms
maintain it, and they are belt and braces on purpose:

1. Each segment records the media pts of its first frame, and frames are
   encoded with timestamps relative to that, at 1/90000 s resolution. Seeking
   to ``pts - segment.start_pts`` in the file lands on the right frame.
2. ``video/frames.jsonl`` records, for every recorded frame, its ``frame_id``,
   its media ``pts`` and its position within the segment. This survives
   encoders and muxers that quietly re-time the stream, and it is the only
   mechanism available on the OpenCV fallback path, which has no concept of a
   presentation timestamp at all and lays frames on a uniform grid.

Boxes are never drawn into these pixels. The recording is the unannotated
frames; the detections stay in JSON beside them. Burned-in boxes cannot be
switched off, cannot be re-thresholded, and cannot be disagreed with by the
person doing the review.

PyAV and OpenCV are imported inside :meth:`VideoRecorder._open_segment`, never
at module import time, so the pure-logic modules and their tests still import
on a laptop with no codecs installed.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from station.core.config import Config
from station.core.types import utc_now_iso

if TYPE_CHECKING:  # pragma: no cover - typing only, keeps numpy/ingest optional
    from station.ingest.base import Frame

__all__ = [
    "MANIFEST_FILENAME",
    "FRAME_INDEX_FILENAME",
    "MANIFEST_VERSION",
    "PTS_TIME_BASE",
    "Segment",
    "RecordableFrame",
    "VideoRecorder",
]

log = logging.getLogger(__name__)

MANIFEST_FILENAME = "segments.json"
FRAME_INDEX_FILENAME = "frames.jsonl"
MANIFEST_VERSION = 1

#: Encoding timestamp resolution, in ticks per second. 90 kHz is the RTP clock
#: rate for video and is far finer than any frame interval, so a decoded pts
#: round-trips through the container without collapsing two frames onto one
#: timestamp.
PTS_TIME_BASE = Fraction(1, 90000)

#: os.path.getsize per frame would be a syscall per frame for no benefit; a
#: segment cannot meaningfully overshoot its size cap in 30 frames.
_SIZE_CHECK_EVERY = 30

#: Frame-index lines are flushed in batches. The index is a convenience for
#: replay, not the safety-critical record -- detections.jsonl is -- so it does
#: not earn an fsync per frame at source frame rate.
_INDEX_FLUSH_EVERY = 60


@runtime_checkable
class RecordableFrame(Protocol):
    """The subset of :class:`station.ingest.base.Frame` this module needs.

    Structural rather than imported so the recorder does not drag numpy and
    the ingest package into anything that merely wants to read a manifest.
    """

    image: Any
    frame_id: int
    pts: float
    width: int
    height: int


@dataclass(slots=True)
class Segment:
    """One recorded file and the slice of the media timeline it holds."""

    index: int
    file: str
    #: Media pts of this segment's first frame. ``offset within file =
    #: detection.pts - start_pts`` -- the join between log and video.
    start_pts: float
    end_pts: float | None = None
    frames: int = 0
    bytes: int = 0
    started_at: str = field(default_factory=utc_now_iso)
    ended_at: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """JSON-ready form for the manifest."""
        return {
            "index": self.index,
            "file": self.file,
            "start_pts": round(self.start_pts, 4),
            "end_pts": None if self.end_pts is None else round(self.end_pts, 4),
            "duration_s": None if self.end_pts is None else round(self.end_pts - self.start_pts, 4),
            "frames": self.frames,
            "bytes": self.bytes,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
        }


class VideoRecorder:
    """Writes the decoded source video into ``<incident>/video/``.

    Segments rotate by size or by duration so that a long deployment produces
    files that can be copied off a station one at a time, and so that a crash
    costs at most the tail of one segment rather than the whole flight.

    Like :class:`~station.incidentlog.writer.IncidentLogWriter`, this never
    raises into the frame loop. If the encoder is missing, the disk fills, or
    a muxer falls over mid-flight, recording stops, :attr:`healthy` goes
    ``False``, a warning is logged once, and the pipeline keeps streaming
    video to the tablets and keeps writing detections. Losing the recording is
    bad; losing the live feed during an incident is unacceptable.

    Example:
        >>> rec = VideoRecorder("incidents/x", fps=30.0, enabled=False)
        >>> rec.write(object())  # disabled: no-op, no import, no exception
        False
    """

    def __init__(
        self,
        incident_dir: str | Path,
        *,
        fps: float = 30.0,
        subdir: str = "video",
        basename: str = "segment",
        container: str = "mp4",
        codec: str = "libx264",
        bitrate: int | None = None,
        options: dict[str, str] | None = None,
        max_segment_bytes: int = 512 * 1024 * 1024,
        max_segment_seconds: float = 600.0,
        backend: str = "auto",
        write_frame_index: bool = True,
        enabled: bool = True,
    ) -> None:
        """Configure a recorder. Nothing is opened until the first frame.

        Args:
            incident_dir: The incident directory; video goes in ``subdir``
                under it, beside ``detections.jsonl``.
            fps: Nominal source rate. Used as the encoder's declared frame
                rate, and as the *only* timing information on the OpenCV
                fallback path.
            subdir: Sub-directory name for the video files.
            basename: Segment filename stem.
            container: Container extension, e.g. ``mp4``, ``mkv``.
            codec: Encoder name for PyAV, e.g. ``libx264``, ``h264_nvenc``.
            bitrate: Target bitrate in bits/s, or ``None`` for the encoder
                default.
            options: Extra encoder options passed to PyAV, e.g.
                ``{"crf": "23", "preset": "veryfast"}``.
            max_segment_bytes: Rotate once a segment file exceeds this.
            max_segment_seconds: Rotate once a segment spans this much media
                time. Rotation happens on a frame boundary, and the new
                segment records its own ``start_pts``, so the correspondence
                with the log survives it.
            backend: ``"auto"``, ``"pyav"`` or ``"opencv"``.
            write_frame_index: Write ``frames.jsonl``. Leave on: it is the
                mechanism that makes replay exact when the muxer re-times the
                stream, and mandatory for the OpenCV path.
            enabled: ``False`` makes every method a no-op returning ``False``.
        """
        self.incident_dir = Path(incident_dir)
        self.video_dir = self.incident_dir / subdir
        self.fps = float(fps) if fps and fps > 0 else 30.0
        self.basename = basename
        self.container = container.lstrip(".")
        self.codec = codec
        self.bitrate = bitrate
        self.options = dict(options or {})
        self.max_segment_bytes = int(max_segment_bytes)
        self.max_segment_seconds = float(max_segment_seconds)
        self.backend_choice = backend.strip().lower()
        self.write_frame_index = bool(write_frame_index)
        self.enabled = bool(enabled)

        self.segments: list[Segment] = []
        self.frames_written = 0
        self.frames_dropped = 0
        self.pts_adjusted = 0
        self.width = 0
        self.height = 0
        self.backend = "none"
        self.notes: list[str] = []

        self._healthy = True
        self._closed = False
        self._encoder: Any = None  # av container, or cv2.VideoWriter
        self._stream: Any = None  # av stream; None on the OpenCV path
        self._segment: Segment | None = None
        self._segment_path: Path | None = None
        self._segment_frame_index = 0
        self._last_tick: int | None = None
        self._index_fh: Any = None
        self._index_since_flush = 0
        self._frames_since_size_check = 0
        self._pts_exact = False

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        cfg: Config,
        incident_dir: str | Path,
        *,
        fps: float | None = None,
        **kwargs: Any,
    ) -> "VideoRecorder":
        """Build a recorder from the station config.

        Args:
            cfg: Loaded station configuration; ``incident_log.record_video``
                and ``incident_log.enabled`` decide whether it records at all.
            incident_dir: The incident directory created by the writer.
            fps: Source rate, from ``SourceInfo.output_fps`` when known.
                Falls back to ``source.target_fps`` and then to 30.
            **kwargs: Passed through to :meth:`__init__`.

        Returns:
            A configured, unopened :class:`VideoRecorder`.
        """
        rate = fps or cfg.source.target_fps or 30.0
        return cls(
            incident_dir,
            fps=rate,
            enabled=bool(cfg.incident_log.enabled and cfg.incident_log.record_video),
            **kwargs,
        )

    # ------------------------------------------------------------------
    # properties
    # ------------------------------------------------------------------

    @property
    def healthy(self) -> bool:
        """``False`` once recording has failed and stopped."""
        return self._healthy

    @property
    def recording(self) -> bool:
        """``True`` while a segment is open."""
        return self._encoder is not None

    @property
    def current_segment(self) -> Segment | None:
        """The segment being written, if any."""
        return self._segment

    # ------------------------------------------------------------------
    # writing
    # ------------------------------------------------------------------

    def write(self, frame: RecordableFrame) -> bool:
        """Encode one frame into the current segment, opening one if needed.

        Args:
            frame: A decoded frame carrying ``image`` (HxWx3 BGR uint8),
                ``pts`` in media seconds, and ``frame_id``.

        Returns:
            ``True`` when the frame was handed to the encoder. ``False`` when
            recording is off or has failed. Never raises.
        """
        if not self.enabled or self._closed or not self._healthy:
            return False
        try:
            if self._encoder is None:
                self._open_segment(frame)
            elif self._should_rotate(frame):
                self._close_segment()
                self._open_segment(frame)
            self._encode(frame)
            self._note_frame(frame)
            self.frames_written += 1
            return True
        except Exception as exc:  # noqa: BLE001 - see class docstring
            self._degrade("record frame", exc)
            return False

    def _should_rotate(self, frame: RecordableFrame) -> bool:
        """Decide whether this frame starts a new segment."""
        seg = self._segment
        if seg is None:
            return True
        if self.max_segment_seconds > 0 and float(frame.pts) - seg.start_pts >= self.max_segment_seconds:
            return True
        self._frames_since_size_check += 1
        if self.max_segment_bytes > 0 and self._frames_since_size_check >= _SIZE_CHECK_EVERY:
            self._frames_since_size_check = 0
            try:
                if self._segment_path is not None and self._segment_path.stat().st_size >= self.max_segment_bytes:
                    return True
            except OSError:
                pass  # the size cap is a convenience; failing to stat is not fatal
        return False

    def _open_segment(self, frame: RecordableFrame) -> None:
        """Open a new segment file starting at ``frame``'s pts.

        Raises:
            RuntimeError: No usable backend, or the file could not be opened.
                Caught by :meth:`write`, which degrades rather than propagates.
        """
        self.video_dir.mkdir(parents=True, exist_ok=True)
        height, width = int(frame.image.shape[0]), int(frame.image.shape[1])
        if self.width and (width, height) != (self.width, self.height):
            # A mid-flight resolution change (source reconnect at a different
            # profile) starts a fresh segment rather than being scaled: the
            # normalised boxes in the log stay valid either way, and silently
            # rescaling frames would falsify what the model actually saw.
            self._note(f"source resolution changed {self.width}x{self.height} -> {width}x{height}")
        self.width, self.height = width, height
        index = len(self.segments)
        name = f"{self.basename}-{index:03d}.{self.container}"
        path = self.video_dir / name
        backend, encoder, stream, pts_exact = self._make_encoder(path)
        self.backend = backend
        self._pts_exact = pts_exact
        self._encoder = encoder
        self._stream = stream
        self._segment_path = path
        self._segment_frame_index = 0
        self._last_tick = None
        self._frames_since_size_check = 0
        self._segment = Segment(index=index, file=name, start_pts=float(frame.pts))
        self.segments.append(self._segment)
        log.info(
            "recording %s from pts %.3f (%dx%d, %s, %.3g fps)",
            path,
            self._segment.start_pts,
            width,
            height,
            backend,
            self.fps,
        )

    def _make_encoder(self, path: Path) -> tuple[str, Any, Any, bool]:
        """Open an encoder for ``path`` with whichever backend is available.

        PyAV first: it is the only one of the two that can carry a real
        presentation timestamp into the file, which is what makes seeking by
        ``pts - start_pts`` land on the right frame.

        Returns:
            ``(backend_name, encoder, stream_or_None, pts_exact)``.

        Raises:
            RuntimeError: Nothing could open the file.
        """
        errors: list[str] = []
        want = self.backend_choice
        if want in ("auto", "pyav"):
            try:
                return self._open_pyav(path)
            except Exception as exc:  # noqa: BLE001 - fall through to OpenCV
                errors.append(f"PyAV: {type(exc).__name__}: {exc}")
        if want in ("auto", "opencv"):
            try:
                return self._open_opencv(path)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"OpenCV: {type(exc).__name__}: {exc}")
        detail = "; ".join(errors) or f"unknown backend {self.backend_choice!r}"
        raise RuntimeError(
            f"cannot record video to {path}: {detail}. "
            "Install PyAV ('pip install av') for exact-timestamp recording, or "
            "set incident_log.record_video: false to log detections only."
        )

    def _open_pyav(self, path: Path) -> tuple[str, Any, Any, bool]:
        """Open a PyAV container and video stream. Lazy import, by design."""
        import av  # noqa: PLC0415 - deferred: the pure-logic modules must import without it

        container = av.open(str(path), mode="w")
        try:
            stream = container.add_stream(self.codec, rate=Fraction(self.fps).limit_denominator(1000))
            stream.width = self.width
            stream.height = self.height
            stream.pix_fmt = "yuv420p"
            if self.bitrate:
                stream.bit_rate = int(self.bitrate)
            if self.options:
                stream.options = dict(self.options)
            # Ask for a 90 kHz timebase so sub-frame pts survives encoding. Not
            # every encoder honours the request, which is exactly why
            # frames.jsonl exists as the authoritative index.
            pts_exact = True
            try:
                stream.time_base = PTS_TIME_BASE
                stream.codec_context.time_base = PTS_TIME_BASE
            except Exception:  # noqa: BLE001
                pts_exact = False
                self._note("encoder refused a 90 kHz timebase; use frames.jsonl for exact replay")
        except Exception:
            container.close()
            raise
        return "PyAV", container, stream, pts_exact

    def _open_opencv(self, path: Path) -> tuple[str, Any, Any, bool]:
        """Open a cv2.VideoWriter. Lazy import, by design.

        The fallback path. ``cv2.VideoWriter`` has no timestamp concept at
        all: it lays frames on a uniform 1/fps grid, so a variable-rate source
        or a dropped frame shifts everything after it. ``pts_exact`` is
        therefore ``False`` and replay must go through ``frames.jsonl``.
        """
        import cv2  # noqa: PLC0415 - deferred, same reason as PyAV

        fourcc_for = {"mp4": "mp4v", "mkv": "mp4v", "avi": "MJPG", "mov": "mp4v"}
        fourcc = cv2.VideoWriter_fourcc(*fourcc_for.get(self.container, "mp4v"))
        writer = cv2.VideoWriter(str(path), fourcc, self.fps, (self.width, self.height))
        if not writer.isOpened():
            raise RuntimeError(f"cv2.VideoWriter would not open {path}")
        self._note(
            "recorded with OpenCV: frames are on a uniform 1/fps grid, so seek by "
            "the frames.jsonl index rather than by pts arithmetic"
        )
        return "OpenCV", writer, None, False

    def _encode(self, frame: RecordableFrame) -> None:
        """Hand one frame to the open encoder."""
        seg = self._segment
        assert seg is not None and self._encoder is not None  # guarded by write()
        if self._stream is not None:
            import av  # noqa: PLC0415 - deferred; already imported by _open_pyav

            picture = av.VideoFrame.from_ndarray(frame.image, format="bgr24")
            tick = int(round((float(frame.pts) - seg.start_pts) / float(PTS_TIME_BASE)))
            if self._last_tick is not None and tick <= self._last_tick:
                # Encoders reject non-monotonic pts outright. Ingest guarantees
                # strictly increasing pts, so this only fires if something
                # upstream broke that guarantee; nudging by one tick keeps the
                # recording alive and the discrepancy is counted and reported.
                tick = self._last_tick + 1
                self.pts_adjusted += 1
            self._last_tick = tick
            picture.pts = tick
            picture.time_base = PTS_TIME_BASE
            for packet in self._stream.encode(picture):
                self._encoder.mux(packet)
        else:
            self._encoder.write(frame.image)
        seg.frames += 1
        seg.end_pts = float(frame.pts)

    def _note_frame(self, frame: RecordableFrame) -> None:
        """Append this frame to ``frames.jsonl``.

        The index, not the container, is what a reviewer trusts: it maps the
        exact ``frame_id``/``pts`` written into ``detections.jsonl`` onto a
        segment and a frame position inside it.
        """
        if not self.write_frame_index or self._segment is None:
            return
        seg = self._segment
        if self._index_fh is None:
            self._index_fh = open(self.video_dir / FRAME_INDEX_FILENAME, "a", encoding="utf-8", newline="\n")
        record = {
            "f": int(frame.frame_id),
            "pts": round(float(frame.pts), 4),
            "s": seg.index,
            "i": self._segment_frame_index,
            # Media offset within the segment file: what you seek to.
            "t": round(float(frame.pts) - seg.start_pts, 4),
        }
        self._index_fh.write(json.dumps(record, separators=(",", ":")) + "\n")
        self._segment_frame_index += 1
        self._index_since_flush += 1
        if self._index_since_flush >= _INDEX_FLUSH_EVERY:
            self._index_since_flush = 0
            self._index_fh.flush()

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def _close_segment(self) -> None:
        """Flush the encoder, close the file, finalise the segment record."""
        seg, encoder, stream = self._segment, self._encoder, self._stream
        self._segment = None
        self._encoder = None
        self._stream = None
        if encoder is None:
            return
        try:
            if stream is not None:
                for packet in stream.encode():  # flush the encoder's delay queue
                    encoder.mux(packet)
                encoder.close()
            else:
                encoder.release()  # cv2.VideoWriter
        except Exception as exc:  # noqa: BLE001
            self._degrade("close video segment", exc)
        if seg is not None:
            seg.ended_at = utc_now_iso()
            try:
                if self._segment_path is not None:
                    seg.bytes = self._segment_path.stat().st_size
            except OSError:
                pass
        self._segment_path = None
        self.write_manifest()

    def close(self) -> None:
        """Stop recording and write the final manifest. Safe to call twice."""
        if self._closed:
            return
        self._closed = True
        self._close_segment()
        if self._index_fh is not None:
            try:
                self._index_fh.flush()
                os.fsync(self._index_fh.fileno())
                self._index_fh.close()
            except Exception as exc:  # noqa: BLE001
                self._degrade("close frame index", exc)
            self._index_fh = None
        self.write_manifest()

    def __enter__(self) -> "VideoRecorder":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # ------------------------------------------------------------------
    # manifest
    # ------------------------------------------------------------------

    def manifest(self) -> dict[str, Any]:
        """Describe the recording: segments, timing and how to replay it.

        Returns:
            A JSON-ready mapping. Hand it to
            :meth:`station.incidentlog.writer.IncidentLogWriter.set_video_manifest`
            so ``meta.json`` points at the frames the detections refer to.
        """
        return {
            "manifest_version": MANIFEST_VERSION,
            "dir": self.video_dir.name,
            "backend": self.backend,
            "codec": self.codec if self.backend == "PyAV" else "mjpeg/mp4v",
            "container": self.container,
            "fps": round(self.fps, 4),
            "width": self.width,
            "height": self.height,
            #: True when segment pts arithmetic is trustworthy; when False the
            #: frame index is the only exact mapping.
            "pts_exact": self._pts_exact,
            "time_base": f"{PTS_TIME_BASE.numerator}/{PTS_TIME_BASE.denominator}",
            "frame_index": FRAME_INDEX_FILENAME if self.write_frame_index else None,
            "frames_written": self.frames_written,
            "frames_dropped": self.frames_dropped,
            "pts_adjusted": self.pts_adjusted,
            "healthy": self._healthy,
            "segments": [seg.as_dict() for seg in self.segments],
            "notes": list(self.notes),
        }

    def write_manifest(self) -> None:
        """Atomically write ``video/segments.json``.

        Rewritten on every rotation as well as at close, so a station that
        loses power mid-flight still leaves a manifest describing every
        completed segment.
        """
        if not self.enabled or not self.segments:
            return
        target = self.video_dir / MANIFEST_FILENAME
        tmp = self.video_dir / f".{MANIFEST_FILENAME}.tmp"
        try:
            self.video_dir.mkdir(parents=True, exist_ok=True)
            with open(tmp, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(self.manifest(), indent=2) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, target)
        except Exception as exc:  # noqa: BLE001
            self._degrade("write video manifest", exc)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _note(self, text: str) -> None:
        """Record a caveat about this recording, once."""
        if text not in self.notes:
            self.notes.append(text)
            log.info("recorder: %s", text)

    def _degrade(self, what: str, exc: BaseException) -> None:
        """Stop recording, loudly but harmlessly.

        Unlike the detection log, a broken recording is not worth retrying
        per-frame: a half-open container usually stays broken, and retrying
        would burn CPU that the inference stage needs. Recording stops, the
        detection log carries on, and the operator's video feed never notices.
        """
        if self._healthy:
            log.warning(
                "video recording stopped: failed to %s (%s: %s). Detection logging and "
                "the live feed are unaffected; this incident will have detections "
                "without the frames they were computed from.",
                what,
                type(exc).__name__,
                exc,
            )
        self._healthy = False
        self._note(f"recording stopped after failure to {what}: {type(exc).__name__}: {exc}")
        self._encoder = None
        self._stream = None
