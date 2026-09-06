"""RTSP ingest -- the drone path.

The ground-station controller (a SIYI MK15 handset, see ``docs/HARDWARE.md``)
publishes the downlink as RTSP on its own WiFi network. The station laptop is
a client on that network and pulls the stream from
``rtsp://192.168.144.25:8554/main.264``.

Two properties of this path drive everything in this module.

**Latency is safety-relevant.** The overlay says "look here" about a scene the
operator is watching live. Every buffered frame between the sensor and the
canvas is time during which the box describes somewhere the aircraft has
already flown past. So: TCP transport (a lost UDP packet costs a corrupt
frame and a decoder resync, which is worse than the retransmit), no
reordering queue, no demuxer buffering, and low-delay decoding. The cost is
a slightly higher chance of a stalled read, which the reconnect logic handles.

**The link drops.** Not as an exception -- routinely, whenever the aircraft
puts a ridge or its own airframe between the antennas. The pipeline has to
come back by itself, because nobody is standing at the laptop; they are
looking at the sky. Reconnection is therefore automatic, backed off (per
``SourceConfig.reconnect_s``), unbounded by default, and it advances the media
timeline by the real duration of the outage so the incident log records what
was actually missed rather than pretending the frames were contiguous.
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

__all__ = ["RtspSource", "DEFAULT_RTSP_URI", "LOW_LATENCY_OPTIONS"]

#: SIYI MK15 ground unit, main stream. The default because it is what is in
#: the flight case; any other RTSP URI works just as well.
DEFAULT_RTSP_URI = "rtsp://192.168.144.25:8554/main.264"

BACKENDS = ("auto", "pyav", "opencv")

#: FFmpeg demuxer/decoder options, applied to every RTSP connection.
#:
#: ``timeout`` and ``stimeout`` are the same socket timeout under two names --
#: FFmpeg renamed it, and which one the installed build honours depends on its
#: version. Passing both is deliberate: an unrecognised option is a warning,
#: whereas no timeout at all means a dead link blocks the read thread forever
#: and the pipeline never notices it should reconnect. That is the failure
#: this module exists to prevent, so it is worth one spurious warning.
LOW_LATENCY_OPTIONS: dict[str, str] = {
    "rtsp_transport": "tcp",
    "rtsp_flags": "prefer_tcp",
    "timeout": "5000000",       # microseconds
    "stimeout": "5000000",      # microseconds, older FFmpeg spelling
    "max_delay": "200000",      # 200 ms of demuxer reordering tolerance
    "reorder_queue_size": "0",  # TCP already delivers in order
    "buffer_size": "262144",
    "fflags": "nobuffer",
    "flags": "low_delay",
}

#: The same intent expressed for OpenCV, which takes FFmpeg options only
#: through an environment variable in ``key;value|key;value`` form.
_OPENCV_FFMPEG_OPTIONS = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|stimeout;5000000"


class RtspSource(ReconnectingSource):
    """A live RTSP stream, reconnecting on its own for as long as it is open.

    Args:
        cfg: Source configuration. ``uri`` defaults to
            :data:`DEFAULT_RTSP_URI` when empty; ``reconnect_s`` is the first
            backoff delay; ``target_fps`` decimates by pts.
        backend: ``"auto"`` (PyAV, then OpenCV), ``"pyav"`` or ``"opencv"``.
        options: FFmpeg options merged over :data:`LOW_LATENCY_OPTIONS`. Use
            this to raise a timeout on a marginal link, not to re-enable
            buffering.
        max_reconnect_attempts: ``None`` (the default) means keep trying for
            as long as the source is open. Bound it only in tests, or where a
            supervisor above this layer will restart the pipeline.
        max_backoff_s: Ceiling on the exponential backoff. Kept short because
            the link comes back suddenly and the operator should not be
            waiting out a long sleep when it does.
        open_timeout_s: Seconds to wait for the connection to establish.
        read_timeout_s: Seconds a single read may block before the stream is
            treated as dead and reconnected.

    Raises:
        SourceConfigError: an unknown backend was requested.
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
            raise SourceConfigError(f"unknown rtsp backend {backend!r}; expected one of {BACKENDS}")
        uri = cfg.uri or DEFAULT_RTSP_URI
        if not uri.startswith(("rtsp://", "rtsps://")):
            raise SourceConfigError(
                f"source.type=rtsp needs an rtsp:// or rtsps:// URI, got {uri!r}"
            )
        super().__init__(
            cfg,
            source_type="rtsp",
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
                # Frame-threaded H.264 decoding holds frames back to fill its
                # pipeline, which adds latency on a live feed. Slice threading
                # gives most of the speed with none of the delay.
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
                # RTSP over FFmpeg does report a usable POS_MSEC on most
                # builds; OpenCVReader detects and disables it if it turns out
                # to be a constant, so trying costs nothing.
                use_timestamps=True,
                buffer_size=1,
            )
            try:
                reader.open()
                if errors:
                    self._add_note("PyAV unavailable; RTSP decoded with OpenCV")
                return reader
            except (MissingDependencyError, SourceUnavailableError) as exc:
                errors.append(str(exc))
        raise SourceUnavailableError(
            f"could not open {self._info.uri}: " + "; ".join(errors or ["no backend available"])
        )
