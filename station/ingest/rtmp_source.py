"""RTMP ingest -- the relay path.

Some ground-station controllers push RTMP rather than serving RTSP, and some
deployments put a small media server (MediaMTX, nginx-rtmp) between the
aircraft and the laptop so that several consumers can share one downlink. In
both cases the station is an RTMP *client* pulling from a URL, and everything
downstream is identical to the RTSP path.

Worth knowing about this path, because it changes what the tablet can do:
a relay generally cannot tell us the sender's RTP timestamps, so tier-1
overlay alignment (``docs/CONTRACT.md``) is unavailable and the tablet falls
back to the pts-offset estimator. That is a supported, labelled mode -- but it
is a reason to prefer the direct RTSP path when the choice exists.

RTMP is TCP, so there is no transport to choose and no packet loss to design
around; the latency work is all in refusing to buffer. ``rtmp_live=live`` is
the load-bearing option: without it FFmpeg treats the stream as a recording,
which adds seek logic and buffering to something that is neither.
"""

from __future__ import annotations

from typing import Any

from station.core.config import SourceConfig

from station.ingest.base import (
    FrameReader,
    MissingDependencyError,
    OpenCVReader,
    PyAVReader,
    ReconnectingSource,
    SourceConfigError,
    SourceUnavailableError,
)

__all__ = ["RtmpSource", "DEFAULT_RTMP_URI", "LOW_LATENCY_OPTIONS"]

#: A local relay is the usual arrangement; override it in config.
DEFAULT_RTMP_URI = "rtmp://127.0.0.1:1935/live/drone"

BACKENDS = ("auto", "pyav", "opencv")

LOW_LATENCY_OPTIONS: dict[str, str] = {
    # Tell FFmpeg this is a live stream, not a file it may seek in.
    "rtmp_live": "live",
    # Client-side buffer in milliseconds. The default is 3000, which is three
    # seconds of a moving aerial scene held back before anyone sees it.
    "rtmp_buffer": "100",
    # I/O timeout in microseconds, so a dead relay surfaces as a reconnect
    # instead of a blocked read thread.
    "rw_timeout": "5000000",
    "fflags": "nobuffer",
    "flags": "low_delay",
}

_OPENCV_FFMPEG_OPTIONS = "rtmp_live;live|rtmp_buffer;100|fflags;nobuffer|flags;low_delay"


class RtmpSource(ReconnectingSource):
    """A live RTMP stream, reconnecting on its own for as long as it is open.

    Args:
        cfg: Source configuration. ``uri`` defaults to
            :data:`DEFAULT_RTMP_URI` when empty.
        backend: ``"auto"`` (PyAV, then OpenCV), ``"pyav"`` or ``"opencv"``.
        options: FFmpeg options merged over :data:`LOW_LATENCY_OPTIONS`.
        max_reconnect_attempts: ``None`` (the default) keeps trying forever.
            A publisher that has not started yet looks exactly like one that
            has stopped, and waiting for it is usually the right behaviour.
        max_backoff_s: Ceiling on the exponential backoff.
        open_timeout_s: Seconds to wait for the connection to establish.
        read_timeout_s: Seconds a single read may block before reconnecting.

    Raises:
        SourceConfigError: an unknown backend, or a URI that is not RTMP.
    """

    def __init__(
        self,
        cfg: SourceConfig,
        *,
        backend: str = "auto",
        options: dict[str, str] | None = None,
        max_reconnect_attempts: int | None = None,
        max_backoff_s: float = 15.0,
        open_timeout_s: float = 5.0,
        read_timeout_s: float = 5.0,
    ) -> None:
        if backend not in BACKENDS:
            raise SourceConfigError(f"unknown rtmp backend {backend!r}; expected one of {BACKENDS}")
        uri = cfg.uri or DEFAULT_RTMP_URI
        if not uri.startswith(("rtmp://", "rtmps://", "rtmpt://", "rtmpe://")):
            raise SourceConfigError(f"source.type=rtmp needs an rtmp:// URI, got {uri!r}")
        super().__init__(
            cfg,
            source_type="rtmp",
            uri=uri,
            max_reconnect_attempts=max_reconnect_attempts,
            max_backoff_s=max_backoff_s,
        )
        self._backend_choice = backend
        self._options = {**LOW_LATENCY_OPTIONS, **(options or {})}
        self._open_timeout = open_timeout_s
        self._read_timeout = read_timeout_s

    def _connect(self) -> FrameReader:
        """Open the stream with whichever backend is available."""
        errors: list[str] = []
        if self._backend_choice in ("auto", "pyav"):
            reader: Any = PyAVReader(
                self._info.uri,
                options=self._options,
                open_timeout_s=self._open_timeout,
                read_timeout_s=self._read_timeout,
                thread_type="SLICE",
            )
            try:
                reader.open()
                return reader
            except MissingDependencyError as exc:
                if self._backend_choice == "pyav":
                    raise
                errors.append(str(exc))
            except SourceUnavailableError as exc:
                if self._backend_choice == "pyav":
                    raise
                errors.append(str(exc))
        if self._backend_choice in ("auto", "opencv"):
            reader = OpenCVReader(
                self._info.uri,
                api_preference="CAP_FFMPEG",
                ffmpeg_options=_OPENCV_FFMPEG_OPTIONS,
                use_timestamps=True,
                buffer_size=1,
            )
            try:
                reader.open()
                if errors:
                    self._add_note("PyAV unavailable; RTMP decoded with OpenCV")
                return reader
            except (MissingDependencyError, SourceUnavailableError) as exc:
                errors.append(str(exc))
        raise SourceUnavailableError(
            f"could not open {self._info.uri}: " + "; ".join(errors or ["no backend available"])
        )
