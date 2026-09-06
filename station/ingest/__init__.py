"""Swappable video ingest: one interface, five sources.

``open_source(cfg)`` is the only entry point the pipeline needs. Everything
behind it -- a recorded file, the drone's RTSP downlink, an RTMP relay, an
HDMI capture card, or a generated test pattern -- produces the same
:class:`~station.ingest.base.Frame` with the same pts semantics, so the code
that runs inference, filters detections and streams to tablets neither knows
nor cares which one is attached.

That seam is why this system can be worked on at all. A recorded flight
replayed through :class:`~station.ingest.file_source.FileSource` exercises the
production path with no aircraft, no radio and no flying; the
:class:`~station.ingest.synthetic_source.SyntheticSource` goes further and
needs no video file either, generating frames whose ground-truth geometry is
known exactly, which is what makes the temporal filter and the overlay
testable rather than merely observable.

Example:
    >>> from station.core.config import SourceConfig
    >>> from station.ingest import open_source
    >>> cfg = SourceConfig(type="synthetic", uri="synthetic:?fps=10", target_fps=5.0)
    >>> with open_source(cfg) as source:
    ...     for frame in source:
    ...         break
    >>> frame.pts, frame.frame_id
    (0.0, 0)

Import cost: this package pulls in numpy and nothing else. PyAV, OpenCV and
every other decoder is imported inside the source that needs it, at
``open()`` time, so the pure-logic modules and their tests run on a machine
with no codecs installed.
"""

from __future__ import annotations

from station.core.config import SourceConfig

from station.ingest.base import (
    DEFAULT_FPS,
    Frame,
    FrameSource,
    IngestError,
    MissingDependencyError,
    PtsClock,
    PtsOrigin,
    ReconnectPolicy,
    SourceConfigError,
    SourceInfo,
    SourceStats,
    SourceUnavailableError,
    FpsDecimator,
    backend_failure,
)
from station.ingest.file_source import FileSource
from station.ingest.hdmi_source import DEFAULT_HDMI_URI, HdmiSource
from station.ingest.rtmp_source import DEFAULT_RTMP_URI, RtmpSource
from station.ingest.rtsp_source import DEFAULT_RTSP_URI, RtspSource
from station.ingest.synthetic_source import (
    SyntheticScene,
    SyntheticSource,
    synthetic_config,
)

__all__ = [
    "open_source",
    "SOURCE_TYPES",
    "Frame",
    "FrameSource",
    "SourceInfo",
    "SourceStats",
    "PtsClock",
    "PtsOrigin",
    "FpsDecimator",
    "ReconnectPolicy",
    "IngestError",
    "MissingDependencyError",
    "SourceUnavailableError",
    "SourceConfigError",
    "backend_failure",
    "FileSource",
    "RtspSource",
    "RtmpSource",
    "HdmiSource",
    "SyntheticSource",
    "SyntheticScene",
    "synthetic_config",
    "DEFAULT_FPS",
    "DEFAULT_RTSP_URI",
    "DEFAULT_RTMP_URI",
    "DEFAULT_HDMI_URI",
]

#: Every value ``SourceConfig.type`` may take, with the one-line description
#: that ends up in the error message when somebody mistypes one.
SOURCE_TYPES: dict[str, str] = {
    "file": "a recorded video file on disk (development, regression tests, after-action replay)",
    "rtsp": f"a live RTSP stream, e.g. the SIYI MK15 downlink at {DEFAULT_RTSP_URI}",
    "rtmp": f"a live RTMP stream from a relay, e.g. {DEFAULT_RTMP_URI}",
    "hdmi": "a USB/HDMI capture device, by index or path, e.g. hdmi://0 or /dev/video0",
    "synthetic": "generated frames with known ground truth; needs no codec and no video file",
}

_BUILDERS = {
    "file": FileSource,
    "rtsp": RtspSource,
    "rtmp": RtmpSource,
    "hdmi": HdmiSource,
    "synthetic": SyntheticSource,
}


def open_source(cfg: SourceConfig, *, autostart: bool = True) -> FrameSource:
    """Build the video source described by ``cfg``.

    Args:
        cfg: The ``source`` section of the station config.
        autostart: Open the source before returning it. Pass ``False`` to
            construct it and open it later -- useful when a supervisor wants to
            own the retry loop around a link that is not up yet.

    Returns:
        An open :class:`~station.ingest.base.FrameSource`, ready to iterate.
        Close it when finished, or use it as a context manager.

    Raises:
        SourceConfigError: ``cfg.type`` is not a known source type, or the
            configuration cannot describe a source that could be opened (a
            missing file, a URI with the wrong scheme).
        MissingDependencyError: the decoder this source needs is not installed.
            Raised at open time, with the install command in the message.
        SourceUnavailableError: the source exists but would not open.
    """
    kind = (cfg.type or "").strip().lower()
    builder = _BUILDERS.get(kind)
    if builder is None:
        valid = "\n".join(f"  {name:<10} {desc}" for name, desc in SOURCE_TYPES.items())
        raise SourceConfigError(
            f"unknown source.type {cfg.type!r}. Valid types are:\n{valid}"
        )
    source = builder(cfg)
    if autostart:
        source.open()
    return source
