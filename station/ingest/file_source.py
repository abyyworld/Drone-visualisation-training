"""Decode a recorded video file.

The most important source in the repository, because it is the one that makes
everything else testable. A recorded flight replayed through this source
exercises the identical code path a live drone does -- same Frame, same pts
semantics, same decimation -- so a regression in the temporal filter or the
overlay can be reproduced on a laptop in a room with no drone in it, over and
over, deterministically.

Backends, in order of preference:

* **PyAV**, because it hands over the container's real presentation
  timestamps. A recording is not necessarily constant-rate: dropped frames on
  the original radio link, a variable-frame-rate phone capture, or a
  concatenated file all produce timing that ``frame_index / fps`` gets wrong.
  Replaying such a file with invented uniform timestamps would quietly make
  the overlay-synchronisation code look better than it is.
* **OpenCV**, when PyAV is not installed. It exposes ``CAP_PROP_POS_MSEC``,
  which is a real timestamp when the backend supports it -- and a constant
  zero when it does not, which this layer detects and reports rather than
  believing.

Only if neither backend produces timestamps does pts fall back to
``frame_index / fps``, and :attr:`SourceInfo.pts_origin` then says
``frame_index`` and a note explains it. That admission is the point: a
downstream component that cares about real capture timing can see that it is
not getting any.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from station.core.config import SourceConfig

from station.ingest.base import (
    FrameSource,
    backend_failure,
    IngestError,
    MissingDependencyError,
    OpenCVReader,
    PtsOrigin,
    PyAVReader,
    SourceConfigError,
    SourceUnavailableError,
)

__all__ = ["FileSource"]

BACKENDS = ("auto", "pyav", "opencv")


class FileSource(FrameSource):
    """Frames decoded from a video file on disk.

    Args:
        cfg: Source configuration. ``uri`` is the path; ``loop`` replays the
            file forever (development and demos only); ``target_fps`` decimates
            exactly as it does for a live source.
        backend: ``"auto"`` (PyAV, then OpenCV), ``"pyav"`` or ``"opencv"``.
            Pinning a backend is for tests that need to exercise one of them.

    Raises:
        SourceConfigError: no path was given, or it does not exist.
    """

    def __init__(self, cfg: SourceConfig, *, backend: str = "auto") -> None:
        if backend not in BACKENDS:
            raise SourceConfigError(f"unknown file backend {backend!r}; expected one of {BACKENDS}")
        if not cfg.uri:
            raise SourceConfigError("source.uri must be the path to a video file for source.type=file")
        self.path = Path(cfg.uri).expanduser()
        if not self.path.exists():
            raise SourceConfigError(f"video file not found: {self.path}")
        if self.path.is_dir():
            raise SourceConfigError(f"source.uri is a directory, expected a video file: {self.path}")
        super().__init__(cfg, source_type="file", uri=str(cfg.uri))
        self._backend_choice = backend
        self._reader: Any = None
        self._loops = 0
        self._frames_this_pass = 0

    @property
    def loops_completed(self) -> int:
        """How many times the file has been replayed from the start."""
        return self._loops

    def _open_impl(self) -> None:
        self._reader = self._connect()
        self._frames_this_pass = 0
        self._set_info(
            width=self._reader.width,
            height=self._reader.height,
            fps=self._reader.fps,
            backend=self._reader.backend,
            is_live=False,
        )
        # Assume the container has timestamps; the first decoded frame settles
        # it and _observe_first_frame publishes whichever turned out to be true.
        self._init_clock(PtsOrigin.CONTAINER, self._reader.fps)

    def _connect(self) -> Any:
        """Open the best available decoder for this file."""
        errors: list[IngestError] = []
        if self._backend_choice in ("auto", "pyav"):
            reader = PyAVReader(str(self.path))
            try:
                reader.open()
                return reader
            except MissingDependencyError as exc:
                if self._backend_choice == "pyav":
                    raise
                errors.append(exc)
            except SourceUnavailableError as exc:
                if self._backend_choice == "pyav":
                    raise
                # A file PyAV cannot demux is worth mentioning even when
                # OpenCV goes on to manage it: it usually means an unusual
                # container that will behave differently in other ways too.
                errors.append(exc)
        if self._backend_choice in ("auto", "opencv"):
            reader = OpenCVReader(str(self.path), use_timestamps=True)
            try:
                reader.open()
                if errors:
                    self._add_note("PyAV unavailable for this file; decoding with OpenCV")
                return reader
            except (MissingDependencyError, SourceUnavailableError) as exc:
                errors.append(exc)
        raise backend_failure(str(self.path), errors)

    def _read_raw(self) -> tuple[np.ndarray, float | None] | None:
        while True:
            item = None
            if self._reader is not None:
                try:
                    item = self._reader.next()
                except IngestError as exc:
                    # A decode error part-way through a file is terminal: unlike
                    # a radio link, retrying a corrupt byte range gets the same
                    # bytes back. Record it and end the source cleanly so the
                    # pipeline reports a stopped state rather than hanging.
                    self._stats.decode_errors += 1
                    self._stats.last_error = str(exc)
                    self._add_note(f"decode stopped early: {exc}")
                    return None
            if item is not None:
                self._frames_this_pass += 1
                return item
            if not self.cfg.loop or self._closing:
                return None
            if self._frames_this_pass == 0:
                # The file decoded to nothing on this pass. Looping would spin
                # at full speed forever, which on a field laptop looks exactly
                # like a working pipeline while producing no frames at all.
                self._add_note("loop stopped: the file yielded no frames")
                return None
            self._rewind()

    def _rewind(self) -> None:
        """Restart the file, continuing the media timeline across the seam.

        pts must not jump back to zero at the loop point. The tablet matches
        detections to frames by pts, so a rewinding timeline would let boxes
        from the end of the previous pass match frames at the start of the next
        one -- a real box drawn over the wrong scene, which is the failure this
        system exists to avoid.
        """
        rewound = False
        seek = getattr(self._reader, "seek_start", None)
        if seek is not None:
            try:
                seek()
                rewound = True
            except IngestError:
                rewound = False
        if not rewound:
            if self._reader is not None:
                self._reader.close()
            self._reader = self._connect()
        self._loops += 1
        self._frames_this_pass = 0
        clock = self._clock
        if clock is not None:
            # One nominal frame period across the seam: a loop is a fiction of
            # continuous playback, not a real gap in time like a dropout.
            clock.discontinuity(clock.nominal_step)
        self._decimator.reset()

    def _close_impl(self) -> None:
        reader, self._reader = self._reader, None
        if reader is not None:
            try:
                reader.close()
            except Exception:  # pragma: no cover
                pass
