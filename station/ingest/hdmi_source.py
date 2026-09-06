"""HDMI capture-card ingest -- the universal fallback.

Every ground-station controller has an HDMI output, whatever it does or does
not do over IP. A USB capture card turns that output into a UVC webcam, and
this source turns the webcam into :class:`~station.ingest.base.Frame` objects.
It is the path that works when the controller speaks a proprietary protocol,
when the RTSP service is not exposed, or when a firmware update has moved
things around an hour before a deployment.

Two things are different here from a network stream, and both are handled
below.

**There are no timestamps.** A capture device delivers pixels, not a
container: nothing in the pipe carries a presentation time. pts therefore
comes from frame arrival on the station's monotonic clock, and
:attr:`SourceInfo.pts_origin` reports ``wall_clock`` so nothing downstream
mistakes it for capture timing. Arrival time is the better fiction of the two
available: unlike ``frame_index / fps`` it at least reflects a frame that was
late or a frame that never came.

**The pixel format has to be set before the frame size.** The classic UVC
trap: ask a 1080p card for 1920x1080 without first selecting MJPG and the
driver quietly keeps raw YUY2, which does not fit through USB 2 at 30 fps, so
it gives you 5 fps instead. The stream works, the overlay works, and the
system samples the world six times less often than the operator believes.
:class:`~station.ingest.base.OpenCVReader` applies properties in the order
given, and this module gives FOURCC first.

The captured picture is whatever the controller draws on its screen, which
usually means the drone's own telemetry overlay is burned into the pixels.
That is unavoidable on this path and harmless -- but note that it is the
*controller's* overlay in the video, not ours: this system's boxes never
touch the pixels, they travel as JSON and are drawn on a canvas on the tablet.
"""

from __future__ import annotations

import sys
from typing import Any
from urllib.parse import parse_qsl, urlsplit

from station.core.config import SourceConfig

from station.ingest.base import (
    FrameReader,
    MissingDependencyError,
    OpenCVReader,
    PtsOrigin,
    PyAVReader,
    ReconnectingSource,
    SourceConfigError,
    SourceUnavailableError,
)

__all__ = ["HdmiSource", "DEFAULT_HDMI_URI"]

#: First capture device on the machine.
DEFAULT_HDMI_URI = "hdmi://0"

BACKENDS = ("auto", "opencv", "pyav")

_VALID_PARAMS = ("width", "height", "fps", "fourcc", "api")

#: Platform capture backends, for both OpenCV and FFmpeg.
_PLATFORM = {
    "linux": ("CAP_V4L2", "v4l2"),
    "darwin": ("CAP_AVFOUNDATION", "avfoundation"),
    "win32": ("CAP_DSHOW", "dshow"),
}


def _parse_device(uri: str) -> tuple[str | int, dict[str, str]]:
    """Split a device URI into an OpenCV target and its parameters.

    Accepts ``0``, ``hdmi://0``, ``/dev/video1``,
    ``/dev/video0?width=1920&height=1080&fourcc=MJPG``, and on Windows a
    DirectShow device name.

    Returns:
        ``(target, params)`` where ``target`` is an ``int`` device index or a
        device path/name.

    Raises:
        SourceConfigError: an unknown query parameter.
    """
    text = uri or DEFAULT_HDMI_URI
    if text.startswith("hdmi://"):
        text = text[len("hdmi://") :]
    elif text.startswith("hdmi:"):
        text = text[len("hdmi:") :]
    split = urlsplit(text)
    query = split.query or (text.split("?", 1)[1] if "?" in text else "")
    body = text.split("?", 1)[0]
    params: dict[str, str] = {}
    for key, value in parse_qsl(query, keep_blank_values=True):
        if key not in _VALID_PARAMS:
            raise SourceConfigError(
                f"unknown hdmi parameter {key!r} in {uri!r}; valid parameters are {list(_VALID_PARAMS)}"
            )
        params[key] = value
    target: str | int = int(body) if body.isdigit() else body
    if isinstance(target, str) and not target:
        target = 0
    return target, params


class HdmiSource(ReconnectingSource):
    """Frames from a USB/HDMI capture device.

    Args:
        cfg: Source configuration. ``uri`` names the device (index, path, or
            name) and may carry ``width``, ``height``, ``fps``, ``fourcc`` and
            ``api`` as query parameters. ``reconnect_s`` governs retries after
            a cable is pulled.
        backend: ``"auto"`` (OpenCV, then PyAV), ``"opencv"`` or ``"pyav"``.
            OpenCV first here -- the opposite of the network sources -- because
            it exposes the pixel-format and buffer controls a capture card
            needs, and there are no container timestamps for PyAV to win with.
        max_reconnect_attempts: ``None`` (the default) keeps retrying, which is
            what an unplugged cable warrants.
        max_backoff_s: Ceiling on the exponential backoff.

    Raises:
        SourceConfigError: an unknown backend or query parameter.
    """

    def __init__(
        self,
        cfg: SourceConfig,
        *,
        backend: str = "auto",
        max_reconnect_attempts: int | None = None,
        max_backoff_s: float = 15.0,
    ) -> None:
        if backend not in BACKENDS:
            raise SourceConfigError(f"unknown hdmi backend {backend!r}; expected one of {BACKENDS}")
        target, params = _parse_device(cfg.uri)
        super().__init__(
            cfg,
            source_type="hdmi",
            uri=cfg.uri or DEFAULT_HDMI_URI,
            max_reconnect_attempts=max_reconnect_attempts,
            max_backoff_s=max_backoff_s,
        )
        self.device: str | int = target
        self._params = params
        self._backend_choice = backend

    def _pts_mode(self) -> str:
        """Arrival time is the only clock a capture device offers."""
        return PtsOrigin.WALL_CLOCK

    def _capture_properties(self) -> list[tuple[str, float]]:
        """OpenCV properties to apply, in the order they must be applied."""
        props: list[tuple[str, float]] = []
        fourcc = self._params.get("fourcc")
        if fourcc:
            if len(fourcc) != 4:
                raise SourceConfigError(f"hdmi fourcc must be 4 characters, got {fourcc!r}")
            # FOURCC first: see the module docstring. Setting it after the
            # frame size silently gets you the old format at a reduced rate.
            props.append(("CAP_PROP_FOURCC", _fourcc_value(fourcc)))
        for key, prop in (("width", "CAP_PROP_FRAME_WIDTH"), ("height", "CAP_PROP_FRAME_HEIGHT"), ("fps", "CAP_PROP_FPS")):
            value = self._params.get(key)
            if value:
                try:
                    props.append((prop, float(value)))
                except ValueError as exc:
                    raise SourceConfigError(f"hdmi {key} must be a number, got {value!r}") from exc
        return props

    def _connect(self) -> FrameReader:
        """Open the capture device with whichever backend is available."""
        errors: list[str] = []
        api_name, ffmpeg_format = _PLATFORM.get(sys.platform, (None, None))
        api = self._params.get("api") or api_name
        if self._backend_choice in ("auto", "opencv"):
            reader: Any = OpenCVReader(
                self.device,
                api_preference=api,
                # A capture device's POS_MSEC is a frame counter at best and a
                # constant zero at worst; asking for it would only feed the pts
                # clock noise it then has to repair.
                use_timestamps=False,
                buffer_size=1,
                properties=self._capture_properties(),
            )
            try:
                reader.open()
                return reader
            except MissingDependencyError as exc:
                if self._backend_choice == "opencv":
                    raise
                errors.append(str(exc))
            except SourceUnavailableError as exc:
                if self._backend_choice == "opencv":
                    raise
                errors.append(str(exc))
        if self._backend_choice in ("auto", "pyav"):
            if ffmpeg_format is None:
                errors.append(f"no FFmpeg capture format known for platform {sys.platform!r}")
            else:
                reader = PyAVReader(
                    self._ffmpeg_device(ffmpeg_format),
                    container_format=ffmpeg_format,
                    options=self._ffmpeg_options(),
                    thread_type="SLICE",
                )
                try:
                    reader.open()
                    if errors:
                        self._add_note("OpenCV unavailable; capture device opened with PyAV")
                    return reader
                except (MissingDependencyError, SourceUnavailableError) as exc:
                    errors.append(str(exc))
        raise SourceUnavailableError(
            f"could not open capture device {self.device!r}: "
            + "; ".join(errors or ["no backend available"])
        )

    def _ffmpeg_device(self, ffmpeg_format: str) -> str:
        """Device string in the form the platform's FFmpeg input expects."""
        if ffmpeg_format == "v4l2":
            return self.device if isinstance(self.device, str) else f"/dev/video{self.device}"
        if ffmpeg_format == "avfoundation":
            return f"{self.device}:none"  # video index, no audio
        # DirectShow addresses devices by name, never by index.
        if isinstance(self.device, int):
            raise SourceUnavailableError(
                "DirectShow needs a device name rather than an index; set source.uri to the "
                "capture card's name, e.g. 'USB3 HDMI Video'"
            )
        return f"video={self.device}"

    def _ffmpeg_options(self) -> dict[str, str]:
        """Capture options for the FFmpeg path, mirroring the OpenCV ones."""
        options: dict[str, str] = {"fflags": "nobuffer", "flags": "low_delay"}
        width, height = self._params.get("width"), self._params.get("height")
        if width and height:
            options["video_size"] = f"{width}x{height}"
        if self._params.get("fps"):
            options["framerate"] = str(self._params["fps"])
        fourcc = self._params.get("fourcc")
        if fourcc:
            # v4l2 calls it input_format and wants a codec name, not a FOURCC.
            options["input_format"] = "mjpeg" if fourcc.upper() in ("MJPG", "MJPEG") else fourcc.lower()
        return options


def _fourcc_value(fourcc: str) -> float:
    """Pack a four-character code the way ``cv2.VideoWriter_fourcc`` does.

    Open-coded rather than imported so that constructing an
    :class:`HdmiSource` -- and reading its configuration back in a test --
    does not require OpenCV to be installed.
    """
    code = 0
    for i, ch in enumerate(fourcc):
        code |= (ord(ch) & 0xFF) << (8 * i)
    return float(code)
