"""WebRTC transport: one inference pass, many tablets.

Sub-500 ms glass-to-glass is a hard requirement here, which rules out HLS and
LL-HLS (2-30 s). A firefighter looking at a box drawn over a scene that has
already moved on is worse off than one looking at plain video, so the transport
has to be the low-latency one.

Two paths are provided and they are not alternatives to be chosen at random:

* :mod:`station.stream.webrtc` -- aiortc in-process. Publishes the video track
  *and* the per-peer ``detections`` data channel, and can recover the RTP
  timestamp of each frame, which is tier 1 of the overlay synchronisation
  algorithm in ``docs/CONTRACT.md``.
* :mod:`station.stream.mediamtx` -- drive the MediaMTX binary (RTSP in,
  WebRTC/WHEP out) for the video half only. Far less code to go wrong for
  video; no data channel, no ``rtp_ts``. Read that module's docstring before
  choosing it.

Nothing in this package imports aiortc, aiohttp or PyAV at module scope. The
station's pure-logic modules have to import and test on a laptop with none of
that installed, and ``import station.stream`` must not be what breaks that.

Safety note that constrains every message emitted from here: this layer
publishes detections when the model produced them and a liveness heartbeat
always. It never emits a message meaning "the scene is clear" -- there is no
such message in the wire contract, and there must never be one. An empty
``detections`` list is published faithfully and renders as nothing at all.
"""

from __future__ import annotations

from typing import Any

from station.stream.mediamtx import (
    MediaMtx,
    MediaMtxError,
    MediaMtxNotFoundError,
    MediaMtxStatus,
    default_config_path,
    find_binary,
    whep_url,
)
from station.stream.signaling import (
    SignalingError,
    add_routes,
    client_config,
    create_app,
    run_standalone,
)
from station.stream.webrtc import (
    FrameBroadcaster,
    FrameLike,
    FrameSource,
    FrameSubscription,
    PeerSession,
    PublishedFrame,
    RtpTimestampProbe,
    StreamDependencyError,
    StreamError,
    StreamerStats,
    VIDEO_CLOCK_RATE,
    WebRtcStreamer,
)

__all__ = [
    # webrtc
    "VIDEO_CLOCK_RATE",
    "StreamError",
    "StreamDependencyError",
    "FrameLike",
    "FrameSource",
    "PublishedFrame",
    "FrameSubscription",
    "FrameBroadcaster",
    "RtpTimestampProbe",
    "PeerSession",
    "StreamerStats",
    "WebRtcStreamer",
    "DetectionVideoTrack",
    # signaling
    "SignalingError",
    "add_routes",
    "create_app",
    "client_config",
    "run_standalone",
    # mediamtx
    "MediaMtx",
    "MediaMtxError",
    "MediaMtxNotFoundError",
    "MediaMtxStatus",
    "find_binary",
    "default_config_path",
    "whep_url",
]


def __getattr__(name: str) -> Any:
    """Lazily materialise names that need an optional dependency.

    ``DetectionVideoTrack`` subclasses ``aiortc.MediaStreamTrack``, so the class
    object itself cannot exist until aiortc is importable. Resolving it here
    keeps ``import station.stream`` working on a machine with no aiortc while
    still letting ``from station.stream import DetectionVideoTrack`` work on one
    that has it.
    """
    if name == "DetectionVideoTrack":
        from station.stream.webrtc import detection_video_track_class

        return detection_video_track_class()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
