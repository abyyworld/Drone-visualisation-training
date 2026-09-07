"""aiortc WebRTC transport: one inference pass fanned out to every tablet.

What this module is for
-----------------------
The ground station decodes the drone feed once and runs the model once. This
module takes the resulting frames and :class:`~station.core.types.FrameDetections`
and delivers them to an arbitrary number of tablets: video on an RTP track,
detections as JSON on a per-peer reliable data channel named ``detections``,
plus a liveness heartbeat on a fixed cadence. Adding the fifth firefighter to
an incident must cost one more encoder, not one more inference -- that is the
whole reason inference is centralised, and :class:`FrameBroadcaster` is where
that promise is kept.

Detections are never burned into the video pixels. Burned-in boxes cannot be
turned off, cannot be labelled stale, and cannot be re-examined afterwards
against the frame they were computed from. They travel as JSON and are drawn on
a canvas over the video by the tablet.

What this module never says
---------------------------
There is no message here that asserts an absence of fire. Detections are
published when the model produced them; an empty ``detections`` list is
published faithfully and renders as nothing at all. :class:`PipelineStatus`
reports whether the pipeline is still *looking*, which is a fact about the
station and not a claim about the scene. See ``station/core/safety.py``.

The heartbeat is load-bearing rather than cosmetic: without it, "the model
found nothing" and "the station fell over" are the same empty screen on the
tablet. With it, silence on the data channel is itself diagnosable, and the
state flips to ``stalled`` after ``StreamConfig.stall_after_s`` so the tablet
stops drawing boxes instead of leaving the last ones over live video.

rtp_ts, and why it is worth the trouble
---------------------------------------
Tier 1 of the overlay synchronisation algorithm (``docs/CONTRACT.md``) matches
each payload to the exact frame it was computed from, using the 90 kHz RTP
timestamp that the browser reports through
``requestVideoFrameCallback().rtpTimestamp``. It removes overlay drift instead
of estimating it, and at 15 m/s ground speed 300 ms of drift is about 4.5 m of
error in where a crew is pointed.

aiortc computes the on-the-wire timestamp as
``(timestamp_origin + convert_timebase(frame.pts, frame.time_base, 1/90000)) % 2**32``
-- but ``timestamp_origin`` is a random 32-bit value held in a **local variable**
inside ``RTCRtpSender._run_rtp`` (verified against aiortc 1.15.0). There is no
public API that returns it, and none that lets it be set. So
:class:`RtpTimestampProbe` recovers it by
observing the first outgoing RTP packets of the sender's own SSRC, once, and
then computing every subsequent ``rtp_ts`` arithmetically. If the probe cannot
attach (aiortc internals moved, no video sender, packets never observed), the
station publishes ``rtp_ts: null`` and logs exactly why at WARNING -- the tablet
then falls back to tier 2 and *says so on screen*, which is why an honest null
matters more than a plausible guess.

Because ``timestamp_origin`` is per-sender, it differs per tablet. ``rtp_ts`` is
therefore stamped **per peer** at send time, not once at publish time, and the
copy written to the incident log keeps ``rtp_ts=None`` because no single value
is true for every viewer.

Dependencies
------------
``aiortc``, ``av`` and ``numpy`` are imported inside the functions that need
them. Importing this module on a laptop with none of them installed must work,
and does.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Protocol, runtime_checkable

from station.core.config import StreamConfig
from station.core.types import (
    WIRE_VERSION,
    FrameDetections,
    ModelInfo,
    PipelineState,
    PipelineStatus,
    utc_now_iso,
)

__all__ = [
    "VIDEO_CLOCK_RATE",
    "DETECTIONS_CHANNEL",
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
    "detection_video_track_class",
]

log = logging.getLogger(__name__)

#: RTP video clock. Fixed by RFC 3551 for every video payload type WebRTC uses,
#: and the unit the browser reports in ``rtpTimestamp``.
VIDEO_CLOCK_RATE = 90000

#: Data channel label. Normative -- see ``docs/CONTRACT.md``.
DETECTIONS_CHANNEL = "detections"

#: RTP timestamps are 32-bit and wrap. At 90 kHz that is every ~13.25 hours,
#: which a long incident can genuinely reach, so every arithmetic result is
#: reduced modulo this.
_RTP_TS_MODULO = 1 << 32

#: Stop queueing detection payloads for a peer whose SCTP buffer is this deep.
#: Dropping payloads is safe -- the tablet's staleness rule turns the overlay
#: off. Letting the buffer grow is not: it delivers boxes late, and a late box
#: is a box over the wrong frame, which is this system's worst failure mode.
_MAX_CHANNEL_BUFFER_BYTES = 256 * 1024


class StreamError(RuntimeError):
    """Base class for failures raised by this module."""


class StreamDependencyError(StreamError):
    """A package needed to actually stream is not installed.

    Raised when a peer is created or a frame is converted -- never at import
    time, so the pure-logic modules keep importing on a machine with no aiortc.
    """


@runtime_checkable
class FrameLike(Protocol):
    """Structural view of :class:`station.ingest.base.Frame`.

    Declared structurally rather than imported so this module does not depend
    on the ingest package (and therefore on numpy) at import time. Any object
    with these attributes will stream.
    """

    image: Any
    frame_id: int
    pts: float
    width: int
    height: int


@runtime_checkable
class FrameSource(Protocol):
    """Structural view of :class:`station.ingest.base.FrameSource`.

    Only :meth:`read` is required: it returns the next frame, or ``None`` when
    the source is exhausted or has nothing to give right now.
    """

    def read(self) -> FrameLike | None: ...


@dataclass(slots=True)
class PublishedFrame:
    """One frame on its way to every subscribed peer.

    Holds the *decoded array*, not an ``av.VideoFrame``. PyAV's per-frame
    reformatter is lazily created and not thread-safe, and aiortc encodes each
    peer's copy in its own executor thread -- so a single shared
    ``av.VideoFrame`` handed to two encoders is a data race waiting for the
    second tablet to connect. Each peer wraps the array itself, which costs one
    plane memcpy and is negligible next to the encode.
    """

    frame_id: int
    #: Media-timeline seconds. Same number as ``FrameDetections.pts``.
    pts: float
    #: ``pts`` on the 90 kHz RTP clock, rounded once here so that every peer's
    #: RTP timestamp derives from an identical integer.
    pts_90k: int
    image: Any
    width: int
    height: int


class FrameSubscription:
    """One peer's view of the frame stream: latest frame wins, never a queue.

    A tablet on a weak corner of the WiFi, or a laptop whose encoder is
    momentarily behind, must not be able to add latency to anyone else or to
    itself. So there is exactly one slot: a frame that has not been collected
    when the next one arrives is dropped and counted. Queueing would trade the
    sub-500 ms budget for a backlog of frames nobody wants any more.
    """

    def __init__(self, name: str) -> None:
        """Create an empty subscription.

        Args:
            name: Peer identifier, used only in logs.
        """
        self.name = name
        self.dropped = 0
        self.delivered = 0
        self._slot: PublishedFrame | None = None
        self._event = asyncio.Event()
        self._closed = False

    @property
    def closed(self) -> bool:
        """Whether :meth:`close` has been called."""
        return self._closed

    def push(self, frame: PublishedFrame) -> None:
        """Offer a frame. Called on the event loop thread only."""
        if self._closed:
            return
        if self._slot is not None:
            self.dropped += 1
        self._slot = frame
        self._event.set()

    async def get(self) -> PublishedFrame:
        """Wait for and return the newest frame.

        Raises:
            StreamError: If the subscription is closed while waiting, which is
                how the video track learns to stop.
        """
        # No ``await`` between the emptiness test and ``clear()``, and both
        # ``push`` and ``get`` run on the event loop thread, so a frame cannot
        # slip in between them and be lost behind a cleared event.
        while self._slot is None:
            if self._closed:
                raise StreamError(f"frame subscription {self.name!r} closed")
            self._event.clear()
            await self._event.wait()
        frame, self._slot = self._slot, None
        self.delivered += 1
        return frame

    def close(self) -> None:
        """Close the subscription and wake anyone waiting on it."""
        self._closed = True
        self._slot = None
        self._event.set()


class FrameBroadcaster:
    """Fans one decoded frame out to every connected peer.

    This is the object that makes "N firefighters watching does not mean N
    inferences" true. The pipeline calls :meth:`publish` once per frame,
    whatever the number of viewers, including zero.
    """

    def __init__(self) -> None:
        self._subs: list[FrameSubscription] = []
        self._last_pts_90k: int | None = None
        self.published = 0
        self.repaired_timestamps = 0

    @property
    def subscriber_count(self) -> int:
        """How many peers are currently receiving video."""
        return len(self._subs)

    def subscribe(self, name: str) -> FrameSubscription:
        """Register a new peer and return its subscription."""
        sub = FrameSubscription(name)
        self._subs.append(sub)
        return sub

    def unsubscribe(self, sub: FrameSubscription) -> None:
        """Remove and close a subscription. Safe to call twice."""
        sub.close()
        try:
            self._subs.remove(sub)
        except ValueError:
            pass

    def publish(self, frame: FrameLike) -> PublishedFrame | None:
        """Fan one frame out to every subscriber.

        Args:
            frame: Any object satisfying :class:`FrameLike`.

        Returns:
            The :class:`PublishedFrame` handed to subscribers, or ``None`` when
            nobody is watching -- in which case no work at all is done, so a
            station with no tablets connected still runs the pipeline at full
            rate for the incident log.
        """
        if not self._subs:
            return None

        pts = float(frame.pts)
        pts_90k = round(pts * VIDEO_CLOCK_RATE)
        # A repeated or backwards RTP timestamp would make the tablet's tier-1
        # interpolation ambiguous, and an ambiguous match is a box on the wrong
        # frame. Ingest guarantees strictly increasing pts; this is the belt to
        # that braces, and it counts what it repaired so it is visible.
        if self._last_pts_90k is not None and pts_90k <= self._last_pts_90k:
            pts_90k = self._last_pts_90k + 1
            self.repaired_timestamps += 1
        self._last_pts_90k = pts_90k

        published = PublishedFrame(
            frame_id=int(frame.frame_id),
            pts=pts,
            pts_90k=pts_90k,
            image=frame.image,
            width=int(frame.width),
            height=int(frame.height),
        )
        for sub in self._subs:
            sub.push(published)
        self.published += 1
        return published

    def reset_timeline(self) -> None:
        """Forget the last RTP timestamp, after a source discontinuity."""
        self._last_pts_90k = None

    def close(self) -> None:
        """Close every subscription."""
        for sub in list(self._subs):
            sub.close()
        self._subs.clear()


# --------------------------------------------------------------------------
# RTP timestamp recovery (sync tier 1)
# --------------------------------------------------------------------------


class RtpTimestampProbe:
    """Recovers a sender's RTP timestamp origin so ``rtp_ts`` can be published.

    The problem is stated in the module docstring: aiortc keeps the random
    per-sender ``timestamp_origin`` in a local variable, so it can neither be
    read nor set through the public API. It *is* however present in every RTP
    packet the sender emits, and we know the media pts of the frame each packet
    was made from, so one observation determines it:

        ``origin = (wire_timestamp - frame_pts_90k) mod 2**32``

    The probe wraps the DTLS transport's packet-send path, filters to the video
    sender's own SSRC (RTX retransmissions use a different one), takes two
    observations from two *different* frames, and only trusts the origin when
    both agree. Then it unhooks itself: from that point every ``rtp_ts`` is pure
    arithmetic, with no interception in the hot path at all.

    Everything here touches aiortc internals, so every step is guarded and any
    failure degrades to ``rtp_ts = None`` with a WARNING that names the reason.
    Publishing a *guessed* rtp_ts would be far worse than publishing none: the
    tablet would sit in tier 1 telling the operator the overlay is exact while
    drawing boxes on the wrong frames.
    """

    def __init__(self, peer_id: str) -> None:
        """Create a detached probe.

        Args:
            peer_id: Peer identifier, used in log messages.
        """
        self.peer_id = peer_id
        self.origin: int | None = None
        self.reason: str | None = "probe not attached"
        self._transport: Any = None
        self._original_send: Any = None
        self._ssrc: int | None = None
        self._pts_of_last_encoded: Callable[[], int | None] | None = None
        self._candidate: tuple[int, int] | None = None  # (origin, pts_90k)

    @property
    def available(self) -> bool:
        """Whether a confirmed origin is known and ``rtp_ts`` can be stamped."""
        return self.origin is not None

    def attach(self, pc: Any, pts_getter: Callable[[], int | None]) -> bool:
        """Hook the peer connection's outgoing RTP path.

        Must be called after ``setLocalDescription``, because the sender's SSRC
        and transport do not exist before that.

        Args:
            pc: The ``RTCPeerConnection``.
            pts_getter: Returns the 90 kHz pts of the most recent frame handed
                to the encoder, or ``None``. aiortc's send loop awaits one
                ``track.recv()``, encodes it and emits its packets before
                calling ``recv()`` again, so at packet time this is exactly the
                frame the packet was made from.

        Returns:
            True if the hook was installed. False (with :attr:`reason` set)
            when the aiortc build does not expose what is needed.
        """
        try:
            sender = next(
                (s for s in pc.getSenders() if getattr(getattr(s, "track", None), "kind", None) == "video"),
                None,
            )
            if sender is None:
                self.reason = "no video sender on this peer connection"
                return False
            ssrc = getattr(sender, "_ssrc", None)
            transport = getattr(sender, "transport", None)
            send = getattr(transport, "_send_rtp", None)
            if ssrc is None or transport is None or send is None or not callable(send):
                self.reason = (
                    "this aiortc build does not expose RTCRtpSender._ssrc / .transport._send_rtp; "
                    "rtp_ts cannot be recovered"
                )
                return False
        except Exception as exc:  # pragma: no cover - depends on aiortc internals
            self.reason = f"could not inspect aiortc sender: {exc}"
            return False

        self._transport = transport
        self._original_send = send
        self._ssrc = int(ssrc)
        self._pts_of_last_encoded = pts_getter
        self.reason = "waiting for the first outgoing RTP packets"

        async def _send_rtp(data: bytes) -> None:
            try:
                self._observe(data)
            except Exception:  # pragma: no cover - never break the media path
                log.debug("rtp_ts probe observation failed", exc_info=True)
            await send(data)

        transport._send_rtp = _send_rtp  # noqa: SLF001 -- documented internals hook
        return True

    def detach(self) -> None:
        """Restore the transport's original send path. Safe to call twice."""
        transport, original = self._transport, self._original_send
        self._transport = None
        self._original_send = None
        self._pts_of_last_encoded = None
        if transport is None or original is None:
            return
        try:
            # Only restore if nobody else wrapped us afterwards, otherwise we
            # would silently unhook their wrapper too.
            if getattr(transport._send_rtp, "__name__", "") == "_send_rtp":  # noqa: SLF001
                transport._send_rtp = original  # noqa: SLF001
        except Exception:  # pragma: no cover
            log.debug("could not restore transport._send_rtp", exc_info=True)

    def rtp_ts_for(self, pts: float) -> int | None:
        """Convert a media pts to the RTP timestamp the browser will see.

        Args:
            pts: Media-timeline seconds, the same value as ``FrameDetections.pts``.

        Returns:
            The wrapped 32-bit RTP timestamp, or ``None`` when the origin is not
            known -- in which case the payload must carry ``rtp_ts: null`` so
            the tablet knows to use tier 2 rather than trusting a fiction.
        """
        if self.origin is None:
            return None
        return (self.origin + round(pts * VIDEO_CLOCK_RATE)) % _RTP_TS_MODULO

    # ---------------------------------------------------------- internals

    def _observe(self, data: bytes) -> None:
        """Extract (timestamp, ssrc) from one outgoing RTP packet."""
        if self.origin is not None or self._pts_of_last_encoded is None:
            return
        # RTP fixed header: V/P/X/CC | M/PT | seq(2) | timestamp(4) | ssrc(4).
        if len(data) < 12 or (data[0] >> 6) != 2:
            return  # not an RTP/RTCP packet at all
        # aiortc sends RTCP down this same call, and an RTCP sender report's
        # bytes 8:12 are an NTP fragment that could in principle collide with
        # our SSRC. RFC 5761's demultiplexing rule separates them: payload
        # types 72..76 are RTCP, everything else is RTP.
        if 72 <= (data[1] & 0x7F) <= 76:
            return
        if int.from_bytes(data[8:12], "big") != self._ssrc:
            return  # another sender, or this sender's RTX stream
        pts_90k = self._pts_of_last_encoded()
        if pts_90k is None:
            return
        wire_ts = int.from_bytes(data[4:8], "big")
        origin = (wire_ts - pts_90k) % _RTP_TS_MODULO

        if self._candidate is None:
            self._candidate = (origin, pts_90k)
            return
        prev_origin, prev_pts = self._candidate
        if pts_90k == prev_pts:
            return  # same frame, tells us nothing new
        if origin != prev_origin:
            # Two frames disagreeing means the packet-to-frame correspondence
            # this probe assumes does not hold on this build. Give up cleanly.
            self.origin = None
            self.reason = (
                f"RTP origin disagreed between frames ({prev_origin} vs {origin}); "
                "aiortc's send loop no longer emits one frame's packets before reading the next"
            )
            self._candidate = None
            self._pts_of_last_encoded = None
            log.warning("[%s] rtp_ts unavailable: %s", self.peer_id, self.reason)
            self.detach()
            return
        self.origin = origin
        self.reason = None
        log.info("[%s] rtp_ts available: recovered RTP timestamp origin %d", self.peer_id, origin)
        # Confirmed. Unhook so the media path is untouched from here on.
        self.detach()


# --------------------------------------------------------------------------
# Video track
# --------------------------------------------------------------------------

_track_class: Any = None


def detection_video_track_class() -> Any:
    """Build (once) and return the ``DetectionVideoTrack`` class.

    The class subclasses ``aiortc.MediaStreamTrack``, so it cannot exist until
    aiortc is importable. This factory is what
    ``from station.stream import DetectionVideoTrack`` resolves to.

    Returns:
        The ``DetectionVideoTrack`` class object.

    Raises:
        StreamDependencyError: If aiortc or PyAV is not installed.
    """
    global _track_class
    if _track_class is not None:
        return _track_class

    try:
        from aiortc import MediaStreamTrack  # noqa: PLC0415 -- lazy by design.
        from aiortc.mediastreams import MediaStreamError  # noqa: PLC0415
    except ImportError as exc:
        raise StreamDependencyError(
            "aiortc is not installed, so the station cannot serve WebRTC itself.\n"
            "  pip install aiortc\n"
            "Alternative: run MediaMTX for the video half (station.stream.mediamtx). "
            "Note that the MediaMTX path carries no detections data channel and no "
            "rtp_ts, so the tablet overlay drops to tier-2 synchronisation."
        ) from exc

    class DetectionVideoTrack(MediaStreamTrack):  # type: ignore[misc, valid-type]
        """Publishes the station's decoded frames to one peer.

        One instance per tablet, all fed from a single
        :class:`FrameBroadcaster`, so the decode and the inference happen once
        however many tablets are watching.

        The track does not pace itself: it emits frames at whatever cadence the
        pipeline publishes them, and drops rather than queues when a peer's
        encoder falls behind. Pacing a live feed would only add latency to the
        500 ms budget; a file source that needs pacing gets it in
        :meth:`WebRtcStreamer.attach_source`.

        No detection is ever drawn into these pixels. The boxes travel as JSON
        on the data channel; see the module docstring.
        """

        kind = "video"

        def __init__(self, subscription: FrameSubscription, *, peer_id: str) -> None:
            """Wrap one peer's frame subscription as an aiortc track.

            Args:
                subscription: This peer's slot on the broadcaster.
                peer_id: Peer identifier, for logs.
            """
            super().__init__()
            self._sub = subscription
            self.peer_id = peer_id
            #: 90 kHz pts of the most recent frame handed to the encoder. Read
            #: by :class:`RtpTimestampProbe` at packet time.
            self.last_pts_90k: int | None = None
            #: Media pts of the first frame this peer received. This is what
            #: seeds the tablet's tier-2 offset estimator, because the browser's
            #: ``video.currentTime`` starts near zero at this frame while our
            #: pts keeps counting from the start of the station's session.
            self.first_pts: float | None = None
            self.frames_sent = 0

        async def recv(self) -> Any:
            """Return the next frame as an ``av.VideoFrame``.

            Raises:
                MediaStreamError: When the subscription closes, which is how
                    aiortc is told the track has ended.
            """
            try:
                published = await self._sub.get()
            except StreamError as exc:
                raise MediaStreamError(str(exc)) from exc

            frame = _to_av_frame(published)
            if self.first_pts is None:
                self.first_pts = published.pts
            self.last_pts_90k = published.pts_90k
            self.frames_sent += 1
            return frame

        def stop(self) -> None:
            """Stop the track and release its subscription."""
            self._sub.close()
            super().stop()

    _track_class = DetectionVideoTrack
    return _track_class


def _to_av_frame(published: PublishedFrame) -> Any:
    """Wrap a decoded BGR array as an ``av.VideoFrame`` on the 90 kHz clock.

    Setting ``time_base`` to 1/90000 and ``pts`` to the media pts in those units
    means aiortc's ``convert_timebase`` is the identity, so the RTP timestamp on
    the wire is exactly ``origin + pts_90k``. That is what makes
    :meth:`RtpTimestampProbe.rtp_ts_for` exact rather than approximate.

    Args:
        published: The frame to convert.

    Returns:
        An ``av.VideoFrame`` with pts and time_base set.

    Raises:
        StreamDependencyError: If PyAV or numpy is missing.
    """
    try:
        import av  # noqa: PLC0415 -- lazy by design.
        import numpy as np  # noqa: PLC0415
    except ImportError as exc:
        raise StreamDependencyError(
            "PyAV and numpy are needed to encode video for WebRTC.\n"
            "  pip install av numpy\n"
            f"(missing: {exc.name})"
        ) from exc
    from fractions import Fraction  # noqa: PLC0415 -- stdlib, kept local for symmetry

    image = published.image
    # PyAV requires a C-contiguous buffer; a cropped or transposed array from a
    # decoder is not necessarily one, and the failure mode without this is a
    # torn image rather than an exception.
    if not getattr(image, "flags", None) or not image.flags["C_CONTIGUOUS"]:
        image = np.ascontiguousarray(image)
    frame = av.VideoFrame.from_ndarray(image, format="bgr24")
    frame.pts = published.pts_90k
    frame.time_base = Fraction(1, VIDEO_CLOCK_RATE)
    return frame


# --------------------------------------------------------------------------
# Per-peer session
# --------------------------------------------------------------------------


class PeerSession:
    """One tablet: a peer connection, a video track and a data channel.

    Owns everything that has to be torn down when that tablet goes away. In the
    field tablets disconnect constantly -- a truck moves, someone walks behind a
    engine, the screen locks -- and a peer connection leaked per reconnect will
    exhaust the station part-way through a long incident. Every exit path from
    this object goes through :meth:`close`, which is idempotent.
    """

    def __init__(
        self,
        peer_id: str,
        pc: Any,
        track: Any,
        subscription: FrameSubscription,
        *,
        max_channel_buffer_bytes: int = _MAX_CHANNEL_BUFFER_BYTES,
    ) -> None:
        """Create a session around an already-built peer connection.

        Args:
            peer_id: Opaque identifier handed back to the tablet.
            pc: The ``RTCPeerConnection``.
            track: The ``DetectionVideoTrack`` added to it.
            subscription: The peer's slot on the broadcaster.
            max_channel_buffer_bytes: Drop detection payloads once the data
                channel has this much unsent.
        """
        self.peer_id = peer_id
        self.pc = pc
        self.track = track
        self.subscription = subscription
        self.probe = RtpTimestampProbe(peer_id)
        self.created_monotonic = time.monotonic()
        self.channel: Any = None
        self.messages_sent = 0
        self.messages_dropped = 0
        self.closed = False
        self._sent_stream_start = False
        self._max_buffer = int(max_channel_buffer_bytes)
        self._close_callbacks: list[Callable[["PeerSession"], None]] = []

    # ------------------------------------------------------------- wiring

    def adopt_channel(self, channel: Any) -> None:
        """Take ownership of the data channel the tablet opened."""
        self.channel = channel
        if getattr(channel, "label", None) != DETECTIONS_CHANNEL:
            log.warning(
                "[%s] data channel is labelled %r, expected %r; the tablet build may be "
                "out of step with docs/CONTRACT.md",
                self.peer_id, getattr(channel, "label", None), DETECTIONS_CHANNEL,
            )

    @property
    def overlay_available(self) -> bool:
        """Whether this peer can receive detections at all.

        False means the tablet negotiated video but no data channel. That is a
        degraded but *safe* state -- plain video, which the operator is watching
        regardless -- and the answer payload says so, so the tablet can label
        the overlay as unavailable rather than silently drawing nothing.
        """
        return self.channel is not None

    def on_close(self, callback: Callable[["PeerSession"], None]) -> None:
        """Register a callback run once, when this session closes."""
        self._close_callbacks.append(callback)

    # ------------------------------------------------------------ sending

    def send_detections(self, detections: FrameDetections, shared_json: str) -> None:
        """Send one frame's detections to this tablet.

        Args:
            detections: The frame result, as published by the pipeline.
            shared_json: ``detections.to_json()``, serialised once by the
                streamer for every peer that does not need a per-peer
                ``rtp_ts``. Re-serialising per peer at 10 fps x N tablets is
                pure waste when the bytes are identical.
        """
        rtp_ts = self.probe.rtp_ts_for(detections.pts)
        if rtp_ts is None:
            payload = shared_json
        else:
            # Per-peer, because aiortc randomises the RTP timestamp origin per
            # sender: one tablet's rtp_ts is meaningless to another.
            payload = dataclasses.replace(detections, rtp_ts=rtp_ts).to_json()
        self._send(payload)

    def send_status(self, status: PipelineStatus) -> None:
        """Send one heartbeat to this tablet.

        ``stream_start_pts`` is added on the first heartbeat for which a value
        exists, and only then. It is the pts of the first frame *this* peer
        received: the tablet's ``video.currentTime`` starts near zero there, so
        that number is the seed for its pts-offset estimator (tier 2 of
        ``docs/CONTRACT.md``). A tablet that joined an hour into an incident
        needs its own seed, not the station's.
        """
        if not self._sent_stream_start:
            first_pts = getattr(self.track, "first_pts", None)
            if first_pts is not None:
                status = dataclasses.replace(status, stream_start_pts=float(first_pts))
                self._sent_stream_start = True
        self._send(status.to_json())

    def _send(self, payload: str) -> None:
        """Write one JSON message to the data channel, if it can take it."""
        channel = self.channel
        if channel is None or self.closed:
            return
        if getattr(channel, "readyState", None) != "open":
            return
        buffered = getattr(channel, "bufferedAmount", 0) or 0
        if buffered > self._max_buffer:
            # Congested. Drop this payload rather than deepening the queue: a
            # payload delivered late is drawn over a scene that has moved on,
            # and the tablet's staleness rule handles the resulting gap by
            # turning the overlay off, which is the safe state.
            self.messages_dropped += 1
            if self.messages_dropped % 100 == 1:
                log.warning(
                    "[%s] detections channel congested (%d bytes buffered); dropping payloads "
                    "(%d dropped so far)", self.peer_id, buffered, self.messages_dropped,
                )
            return
        try:
            channel.send(payload)
        except Exception as exc:
            # A send failure on a dying channel must not take the pipeline with
            # it, and must not be retried into a loop.
            self.messages_dropped += 1
            log.debug("[%s] data channel send failed: %s", self.peer_id, exc)
            return
        self.messages_sent += 1

    # ------------------------------------------------------------ teardown

    async def close(self) -> None:
        """Tear the session down. Idempotent, and never raises."""
        if self.closed:
            return
        self.closed = True
        self.probe.detach()
        try:
            if self.channel is not None and getattr(self.channel, "readyState", None) == "open":
                self.channel.close()
        except Exception:  # pragma: no cover
            log.debug("[%s] channel close failed", self.peer_id, exc_info=True)
        try:
            self.track.stop()
        except Exception:  # pragma: no cover
            log.debug("[%s] track stop failed", self.peer_id, exc_info=True)
        self.subscription.close()
        try:
            await self.pc.close()
        except Exception:  # pragma: no cover
            log.debug("[%s] peer connection close failed", self.peer_id, exc_info=True)
        for callback in self._close_callbacks:
            try:
                callback(self)
            except Exception:  # pragma: no cover
                log.debug("[%s] close callback failed", self.peer_id, exc_info=True)
        self._close_callbacks.clear()
        log.info(
            "[%s] peer closed after %.1fs: %d frames, %d messages, %d frames dropped, %d messages dropped",
            self.peer_id, time.monotonic() - self.created_monotonic,
            getattr(self.track, "frames_sent", 0), self.messages_sent,
            self.subscription.dropped, self.messages_dropped,
        )

    def describe(self) -> dict[str, Any]:
        """Diagnostic snapshot for the operator-facing peers endpoint."""
        return {
            "peer_id": self.peer_id,
            "connection_state": getattr(self.pc, "connectionState", "unknown"),
            "ice_state": getattr(self.pc, "iceConnectionState", "unknown"),
            "age_s": round(time.monotonic() - self.created_monotonic, 1),
            "frames_sent": getattr(self.track, "frames_sent", 0),
            "frames_dropped": self.subscription.dropped,
            "messages_sent": self.messages_sent,
            "messages_dropped": self.messages_dropped,
            "overlay_available": self.overlay_available,
            "sync_tier": 1 if self.probe.available else 2,
            "rtp_ts": self.probe.available,
            "rtp_ts_unavailable_reason": self.probe.reason,
        }


# --------------------------------------------------------------------------
# The streamer
# --------------------------------------------------------------------------


@dataclass(slots=True)
class StreamerStats:
    """Counters for the operator-facing diagnostics endpoint.

    Every field is about the *station*. None of them is about the scene.
    """

    peers: int = 0
    frames_published: int = 0
    detections_published: int = 0
    statuses_published: int = 0
    peers_accepted: int = 0
    peers_rejected: int = 0
    peers_closed: int = 0
    repaired_timestamps: int = 0


class WebRtcStreamer:
    """Serves video and detections to every connected tablet.

    Lifecycle::

        streamer = WebRtcStreamer(cfg.stream, model=runner.model_info)
        await streamer.start()
        ...
        streamer.publish_frame(frame)              # from the decode loop
        streamer.publish_detections(frame_result)  # from the inference loop
        ...
        await streamer.close()

    :meth:`publish_frame` and :meth:`publish_detections` are safe to call from
    any thread: the station's decode and inference loops are threads, not
    coroutines, and forcing them to be async just to talk to the transport would
    put encoding latency on the same loop that has to answer signalling.

    Example:
        >>> from station.core.config import StreamConfig
        >>> streamer = WebRtcStreamer(StreamConfig())
        >>> streamer.stats.peers
        0
    """

    def __init__(
        self,
        cfg: StreamConfig,
        *,
        model: ModelInfo | None = None,
        source: str | None = None,
        rtp_timestamp_probe: bool = True,
        connect_timeout_s: float = 20.0,
        max_peers: int = 16,
        max_channel_buffer_bytes: int = _MAX_CHANNEL_BUFFER_BYTES,
    ) -> None:
        """Create a streamer. Does not touch the network.

        Args:
            cfg: Stream configuration; supplies the ICE servers, the heartbeat
                cadence and the stall timeout.
            model: Model provenance, echoed in every heartbeat so the tablet can
                show which model is behind the boxes it is drawing.
            source: Source URI, echoed in the heartbeat.
            rtp_timestamp_probe: Recover the RTP timestamp origin so payloads
                can carry ``rtp_ts`` (sync tier 1). Turn it off to force the
                tablet onto tier 2 -- useful when bisecting an alignment bug,
                and harmless otherwise because tier 2 is the documented
                fallback.
            connect_timeout_s: A peer that has not reached ``connected`` within
                this many seconds is closed. Tablets vanish mid-handshake all
                the time; without this each one leaks a peer connection.
            max_peers: Refuse further offers beyond this many live peers. Each
                peer costs an encoder, and a station that has run out of CPU
                serves nobody rather than serving everybody badly.
            max_channel_buffer_bytes: Per-peer data channel congestion limit.
        """
        self.cfg = cfg
        self.model = model
        self.source = source
        self.rtp_timestamp_probe = rtp_timestamp_probe
        self.connect_timeout_s = float(connect_timeout_s)
        self.max_peers = int(max_peers)
        self.max_channel_buffer_bytes = int(max_channel_buffer_bytes)

        self.broadcaster = FrameBroadcaster()
        self.stats = StreamerStats()

        self._peers: dict[str, PeerSession] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._started_monotonic = time.monotonic()
        self._closed = False

        self._desired_state = PipelineState.STARTING
        self._note: str | None = None
        self._source_fps: float | None = None
        self._inference_fps: float | None = None
        self._dropped_frames = 0
        self._last_inference_pts: float | None = None
        self._last_inference_wall_time: str | None = None
        self._last_inference_monotonic: float | None = None

        self._pump_thread: threading.Thread | None = None
        self._pump_stop = threading.Event()

    # -------------------------------------------------------------- lifecycle

    async def start(self) -> None:
        """Bind to the running event loop and start the heartbeat.

        Raises:
            StreamError: If called twice, or after :meth:`close`.
        """
        if self._closed:
            raise StreamError("streamer is closed")
        if self._loop is not None:
            raise StreamError("streamer already started")
        self._loop = asyncio.get_running_loop()
        self._started_monotonic = time.monotonic()
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop(), name="wildfire-heartbeat")
        log.info(
            "webrtc streamer started: heartbeat %.2gs, stall after %.2gs, ice_servers=%s",
            self.cfg.status_interval_s, self.cfg.stall_after_s,
            self.cfg.ice_servers or "none (host candidates only)",
        )

    async def close(self) -> None:
        """Stop the heartbeat, drop every peer, and release the broadcaster."""
        if self._closed:
            return
        self._closed = True
        self._desired_state = PipelineState.STOPPED
        self.detach_source()

        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            except Exception:  # pragma: no cover - already logged by the loop
                log.debug("heartbeat task ended with an error", exc_info=True)
            self._heartbeat_task = None

        # A last heartbeat saying ``stopped`` is worth sending: it is the
        # difference between a tablet showing "station stopped" and a tablet
        # showing nothing while the operator wonders which it is.
        self._broadcast_status(PipelineState.STOPPED)
        await asyncio.gather(*(peer.close() for peer in list(self._peers.values())), return_exceptions=True)
        self._peers.clear()
        self.broadcaster.close()
        log.info("webrtc streamer closed")

    # ------------------------------------------------------ publishing (any thread)

    def publish_frame(self, frame: FrameLike) -> None:
        """Hand one decoded frame to every connected tablet.

        Callable from the decode thread. Does no work at all when no tablet is
        connected, so the pipeline's cost does not depend on who is watching.

        Args:
            frame: Any object satisfying :class:`FrameLike` -- in practice a
                :class:`station.ingest.base.Frame`.
        """
        self._on_loop(self._publish_frame_locked, frame)

    def publish_detections(self, detections: FrameDetections) -> None:
        """Publish one frame's detections to every connected tablet.

        This is also the liveness signal: a completed inference resets the stall
        timer. An empty ``detections`` tuple counts fully -- the model looked and
        proposed nothing, which is a real observation and keeps the pipeline
        ``running``. Not publishing empty frames would make a quiet scene
        indistinguishable from a dead model.

        Callable from the inference thread.

        Args:
            detections: The frame result, after the temporal filter.
        """
        self._on_loop(self._publish_detections_locked, detections)

    def note_inference(self, pts: float, wall_time: str | None = None) -> None:
        """Record a completed inference without publishing it.

        For the rare pipeline that logs a frame but suppresses its payload. The
        stall timer is reset either way, because the model *did* look.

        Args:
            pts: Media pts of the inferred frame.
            wall_time: RFC 3339 stamp; defaults to now.
        """
        self._on_loop(self._note_inference_locked, float(pts), wall_time or utc_now_iso())

    def set_state(self, state: str) -> None:
        """Set the pipeline state the heartbeat reports.

        The streamer may still override this with ``stalled`` when no inference
        has completed inside ``StreamConfig.stall_after_s``; it will never
        override it with anything more reassuring.

        Args:
            state: One of :class:`station.core.types.PipelineState`.

        Raises:
            ValueError: If the state is not a known one.
        """
        if state not in PipelineState.ALL:
            raise ValueError(f"unknown pipeline state {state!r}; expected one of {PipelineState.ALL}")
        self._desired_state = state

    def set_rates(self, *, source_fps: float | None = None, inference_fps: float | None = None) -> None:
        """Update the throughput numbers carried in the heartbeat.

        The gap between them is the sampling rate, and the operator is entitled
        to see it: it bounds how briefly a fire could appear and be missed
        entirely.
        """
        if source_fps is not None:
            self._source_fps = float(source_fps)
        if inference_fps is not None:
            self._inference_fps = float(inference_fps)

    def set_dropped_frames(self, count: int) -> None:
        """Set the cumulative dropped-frame count reported in the heartbeat."""
        self._dropped_frames = int(count)

    def set_note(self, note: str | None) -> None:
        """Set the operator-facing note in the heartbeat.

        For conditions the pipeline can self-detect, such as a source
        reconnecting. Never for anything about the contents of the scene.
        """
        self._note = note

    def set_model(self, model: ModelInfo | None) -> None:
        """Set the model provenance echoed in the heartbeat."""
        self.model = model

    def set_source(self, source: str | None) -> None:
        """Set the source URI echoed in the heartbeat."""
        self.source = source

    def reset_timeline(self) -> None:
        """Tell the transport the media timeline restarted (source reconnect)."""
        self._on_loop(self.broadcaster.reset_timeline)

    # --------------------------------------------------------- source pump

    def attach_source(self, source: FrameSource, *, paced: bool = True) -> None:
        """Read a :class:`FrameSource` in a thread and publish every frame.

        This is the video-only path: it is what you use to put a recorded file
        or a live feed on the wire without an inference stage, for a bench test
        or a demo. In production the pipeline calls :meth:`publish_frame`
        itself, so that the frame published and the frame inferred are the same
        object.

        Args:
            source: Anything with a ``read()`` returning frames or ``None``.
            paced: Sleep so that frames leave at their media-timeline rate. A
                live source arrives in real time already and pacing is a no-op;
                a file read as fast as the disk allows would otherwise arrive as
                a burst that no encoder or tablet can use.

        Raises:
            StreamError: If a source is already attached.
        """
        if self._pump_thread is not None and self._pump_thread.is_alive():
            raise StreamError("a source is already attached; call detach_source() first")
        self._pump_stop.clear()
        thread = threading.Thread(
            target=self._pump, args=(source, paced), name="wildfire-frame-pump", daemon=True
        )
        self._pump_thread = thread
        thread.start()

    def detach_source(self) -> None:
        """Stop the source pump started by :meth:`attach_source`."""
        self._pump_stop.set()
        thread = self._pump_thread
        self._pump_thread = None
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)

    def _pump(self, source: FrameSource, paced: bool) -> None:
        """Body of the source pump thread."""
        wall_origin: float | None = None
        pts_origin: float | None = None
        try:
            while not self._pump_stop.is_set():
                frame = source.read()
                if frame is None:
                    # Exhausted, or nothing available yet. A short sleep keeps a
                    # non-blocking source from spinning a core.
                    if self._pump_stop.wait(0.005):
                        break
                    continue
                if paced:
                    if wall_origin is None or pts_origin is None:
                        wall_origin, pts_origin = time.monotonic(), float(frame.pts)
                    due = wall_origin + (float(frame.pts) - pts_origin)
                    delay = due - time.monotonic()
                    # Only ever sleep to slow down. If we are behind -- a slow
                    # disk, a stalled loop -- we publish immediately rather than
                    # trying to catch up, because catching up means a burst.
                    if delay > 0 and self._pump_stop.wait(min(delay, 1.0)):
                        break
                self.publish_frame(frame)
        except Exception:
            log.exception("frame pump stopped on an unhandled error")

    # ------------------------------------------------------------- peers

    async def handle_offer(
        self,
        sdp: str,
        offer_type: str = "offer",
        *,
        peer_id: str | None = None,
    ) -> dict[str, Any]:
        """Answer one tablet's SDP offer and start serving it.

        Args:
            sdp: The tablet's offer SDP.
            offer_type: Must be ``"offer"``.
            peer_id: Optional caller-supplied identifier; one is generated if
                omitted.

        Returns:
            A mapping with ``sdp``, ``type``, ``peer_id``, ``wire_version`` and
            the overlay parameters the tablet needs. Serialise it as the HTTP
            response body.

        Raises:
            StreamDependencyError: If aiortc or PyAV is not installed.
            StreamError: If the streamer is closed, the offer is malformed, or
                the peer limit is reached.
        """
        if self._closed:
            raise StreamError("streamer is closed; not accepting peers")
        if self._loop is None:
            raise StreamError("streamer not started; call await start() first")
        if offer_type != "offer":
            raise StreamError(f"expected an SDP offer, got type {offer_type!r}")
        if not sdp or "m=" not in sdp:
            raise StreamError("offer SDP is empty or has no media sections")
        if len(self._peers) >= self.max_peers:
            self.stats.peers_rejected += 1
            raise StreamError(
                f"peer limit reached ({self.max_peers} tablets connected). Each viewer costs an "
                "encoder on the station; raise max_peers only if the laptop has the headroom."
            )

        RTCPeerConnection, RTCConfiguration, RTCIceServer, RTCSessionDescription = _aiortc()
        track_cls = detection_video_track_class()

        peer_id = peer_id or uuid.uuid4().hex[:12]

        # ICE: host candidates only unless the operator configured otherwise.
        # aiortc's default when ``iceServers`` is None is Google's public STUN
        # server -- on a fire ground with no internet that resolves to nothing
        # and adds a gathering timeout to every single connection, on a LAN
        # where host candidates already work. An *explicit empty list* is not
        # the same as leaving it unset; it is what turns the default off.
        ice_servers = [RTCIceServer(urls=url) for url in self.cfg.ice_servers]
        pc = RTCPeerConnection(RTCConfiguration(iceServers=ice_servers))

        subscription = self.broadcaster.subscribe(peer_id)
        track = track_cls(subscription, peer_id=peer_id)
        pc.addTrack(track)
        session = PeerSession(
            peer_id, pc, track, subscription,
            max_channel_buffer_bytes=self.max_channel_buffer_bytes,
        )

        @pc.on("datachannel")
        def _on_datachannel(channel: Any) -> None:
            session.adopt_channel(channel)
            # Answer the tablet immediately with a heartbeat rather than making
            # it wait up to status_interval_s to learn the pipeline is alive.
            session.send_status(self._build_status())

        @pc.on("connectionstatechange")
        def _on_state_change() -> None:
            state = pc.connectionState
            log.info("[%s] connection state -> %s", peer_id, state)
            if state in ("failed", "closed", "disconnected"):
                # ``disconnected`` can recover on its own, but in the field it
                # almost never does before the tablet simply re-offers, and a
                # peer kept alive per reconnect is the leak that kills a long
                # incident. Dropping it is cheap: the tablet re-offers.
                asyncio.ensure_future(self._drop_peer(peer_id))

        try:
            await pc.setRemoteDescription(RTCSessionDescription(sdp=sdp, type=offer_type))
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)
        except Exception as exc:
            await session.close()
            self.broadcaster.unsubscribe(subscription)
            raise StreamError(f"could not negotiate with peer {peer_id}: {exc}") from exc

        if self.rtp_timestamp_probe:
            if not session.probe.attach(pc, lambda: track.last_pts_90k):
                log.warning(
                    "[%s] rtp_ts will be published as null: %s. The tablet falls back to the "
                    "pts-offset estimator (sync tier 2) and labels the overlay accordingly.",
                    peer_id, session.probe.reason,
                )
        else:
            session.probe.reason = "disabled by configuration (rtp_timestamp_probe=False)"

        self._peers[peer_id] = session
        self.stats.peers_accepted += 1
        session.on_close(lambda s: self.broadcaster.unsubscribe(s.subscription))
        asyncio.ensure_future(self._connect_watchdog(peer_id))

        log.info(
            "[%s] peer accepted (%d connected); ice=%s",
            peer_id, len(self._peers), self.cfg.ice_servers or "host-only",
        )
        return {
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type,
            "peer_id": peer_id,
            **self.client_parameters(),
        }

    def client_parameters(self) -> dict[str, Any]:
        """Overlay parameters the tablet must be told rather than hard-code.

        ``max_overlay_age_s`` in particular is a safety parameter configured in
        one place (``StreamConfig``) and enforced on the tablet; shipping it in
        the answer means a station and a tablet cannot disagree about when boxes
        stop being drawn.
        """
        return {
            "wire_version": WIRE_VERSION,
            "detections_channel": DETECTIONS_CHANNEL,
            "max_overlay_age_s": self.cfg.max_overlay_age_s,
            "overlay_buffer_s": self.cfg.overlay_buffer_s,
            "status_interval_s": self.cfg.status_interval_s,
            "stall_after_s": self.cfg.stall_after_s,
            "ice_servers": list(self.cfg.ice_servers),
        }

    async def close_peer(self, peer_id: str) -> bool:
        """Close one peer by id.

        Returns:
            True if a peer with that id was open.
        """
        return await self._drop_peer(peer_id)

    async def _drop_peer(self, peer_id: str) -> bool:
        session = self._peers.pop(peer_id, None)
        if session is None:
            return False
        self.stats.peers_closed += 1
        await session.close()
        return True

    async def _connect_watchdog(self, peer_id: str) -> None:
        """Close a peer that never finishes connecting.

        A tablet that walks out of range between the offer and the ICE handshake
        leaves a peer connection and an encoder behind. Over a six-hour incident
        with people coming and going, that is the leak that ends the stream.
        """
        await asyncio.sleep(self.connect_timeout_s)
        session = self._peers.get(peer_id)
        if session is None or session.closed:
            return
        if getattr(session.pc, "connectionState", None) not in ("connected", "completed"):
            log.warning(
                "[%s] never reached connected within %.0fs (state=%s); dropping it",
                peer_id, self.connect_timeout_s, getattr(session.pc, "connectionState", "?"),
            )
            await self._drop_peer(peer_id)

    @property
    def peers(self) -> tuple[PeerSession, ...]:
        """The currently connected peers."""
        return tuple(self._peers.values())

    def describe_peers(self) -> list[dict[str, Any]]:
        """Diagnostic snapshot of every peer, for the operator."""
        return [peer.describe() for peer in self._peers.values()]

    # --------------------------------------------------------- heartbeat

    async def _heartbeat_loop(self) -> None:
        """Emit a :class:`PipelineStatus` on a fixed cadence, forever.

        Independent of detections on purpose. A tablet that receives nothing
        cannot tell "the model found nothing" from "the station fell over";
        a heartbeat on a known cadence makes silence itself diagnosable.
        """
        interval = max(0.05, float(self.cfg.status_interval_s))
        try:
            while not self._closed:
                self._broadcast_status()
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            raise
        except Exception:  # pragma: no cover - the heartbeat must not die quietly
            log.exception("heartbeat loop failed; tablets will see the pipeline as silent")
            raise

    def _effective_state(self) -> str:
        """The state to report, after applying the stall rule.

        The stall rule can only ever make the reported state *less* reassuring.
        """
        if self._closed or self._desired_state == PipelineState.STOPPED:
            return PipelineState.STOPPED
        # Before the first inference the reference is start-up, so a pipeline
        # that never manages to infer at all is reported as stalled rather than
        # sitting on ``starting`` indefinitely.
        reference = self._last_inference_monotonic
        if reference is None:
            reference = self._started_monotonic
        if time.monotonic() - reference > self.cfg.stall_after_s:
            return PipelineState.STALLED
        return self._desired_state

    def _build_status(self, state: str | None = None) -> PipelineStatus:
        """Assemble the heartbeat. Says nothing about the scene."""
        effective = state or self._effective_state()
        note = self._note
        if effective == PipelineState.STALLED:
            since = time.monotonic() - (self._last_inference_monotonic or self._started_monotonic)
            stall_note = f"no completed inference for {since:.1f}s"
            note = f"{note}; {stall_note}" if note else stall_note
        return PipelineStatus(
            state=effective,
            source=self.source,
            model=self.model,
            source_fps=self._source_fps,
            inference_fps=self._inference_fps,
            last_inference_pts=self._last_inference_pts,
            last_inference_wall_time=self._last_inference_wall_time,
            dropped_frames=self._dropped_frames,
            uptime_s=time.monotonic() - self._started_monotonic,
            note=note,
        )

    def _broadcast_status(self, state: str | None = None) -> None:
        """Send one heartbeat to every peer. Loop thread only."""
        if not self._peers:
            return
        status = self._build_status(state)
        for session in list(self._peers.values()):
            session.send_status(status)
        self.stats.statuses_published += 1

    # --------------------------------------------------------- internals

    def _on_loop(self, fn: Callable[..., Any], *args: Any) -> None:
        """Run ``fn`` on the event loop, from whichever thread called us.

        aiortc's data channels and tracks are only safe to touch on the loop
        thread, while the station's decode and inference stages are threads.
        This is the one seam between them.
        """
        loop = self._loop
        if loop is None or self._closed:
            return
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is loop:
            fn(*args)
            return
        try:
            loop.call_soon_threadsafe(fn, *args)
        except RuntimeError:
            # Loop already closed -- the station is shutting down and the
            # publishing thread has not noticed yet. Dropping is correct.
            log.debug("event loop closed; dropped a publish from thread %s", threading.current_thread().name)

    def _publish_frame_locked(self, frame: FrameLike) -> None:
        published = self.broadcaster.publish(frame)
        if published is not None:
            self.stats.frames_published += 1
        self.stats.peers = len(self._peers)
        self.stats.repaired_timestamps = self.broadcaster.repaired_timestamps

    def _publish_detections_locked(self, detections: FrameDetections) -> None:
        self._note_inference_locked(
            detections.pts, detections.wall_time or utc_now_iso()
        )
        if not self._peers:
            return
        # Serialised once. Only peers whose RTP origin is known pay for a
        # second serialisation, and only because their rtp_ts genuinely differs.
        shared_json = detections.to_json()
        for session in list(self._peers.values()):
            session.send_detections(detections, shared_json)
        self.stats.detections_published += 1

    def _note_inference_locked(self, pts: float, wall_time: str) -> None:
        self._last_inference_pts = float(pts)
        self._last_inference_wall_time = wall_time
        self._last_inference_monotonic = time.monotonic()


def _aiortc() -> tuple[Any, Any, Any, Any]:
    """Import aiortc, with an error a person standing next to a drone can act on.

    Returns:
        ``(RTCPeerConnection, RTCConfiguration, RTCIceServer, RTCSessionDescription)``.

    Raises:
        StreamDependencyError: If aiortc is not installed.
    """
    try:
        from aiortc import (  # noqa: PLC0415 -- lazy by design; see module docstring.
            RTCConfiguration,
            RTCIceServer,
            RTCPeerConnection,
            RTCSessionDescription,
        )
    except ImportError as exc:
        raise StreamDependencyError(
            "aiortc is not installed, so the station cannot serve WebRTC itself.\n"
            "  pip install aiortc\n"
            "Alternative: run MediaMTX for the video half (station.stream.mediamtx). "
            "That path has no detections data channel and no rtp_ts, so the tablet "
            "overlay drops to tier-2 synchronisation -- see docs/CONTRACT.md."
        ) from exc
    return RTCPeerConnection, RTCConfiguration, RTCIceServer, RTCSessionDescription
