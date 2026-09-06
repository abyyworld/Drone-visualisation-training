"""The runtime that joins ingest, inference, transport and the incident log.

One :class:`Pipeline` owns the whole path from a decoded frame to a JSON
payload on a tablet:

.. code-block:: text

    FrameSource --> [decode thread] --+--> WebRTC video track (every frame)
                                      +--> incident video recorder
                                      |
                                      +--> bounded queue (DROPS, never grows)
                                            |
                                            v
                                    [inference thread] ModelRunner
                                            |
                                            v
                                       TemporalFilter
                                            |
                    +-----------------------+-----------------------+
                    v                       v                       v
             incident log            WebRTC data channel      subscribers
             (every frame,           (JSON detections)        (anything else)
              empty ones too)

Two threads and one event loop, and the division is not arbitrary. Decoding and
inference are blocking, CPU/GPU-bound calls into C libraries; running them on
the event loop would stall the loop that has to answer WebRTC signalling and
send heartbeats, and a station that stops sending heartbeats looks stalled to
every tablet watching it. The loop keeps the parts that must stay responsive.

**Frame drop policy -- the important decision in this file.** When inference
cannot keep up, frames are *dropped*, never queued. The queue holds two frames
and, when full, the oldest is discarded to make room for the newest. Queueing
instead would trade a bounded overlay lag for an unbounded one: every frame the
model falls behind adds permanently to the delay between what the operator sees
in the video and where the boxes are drawn, and that delay is exactly the
failure ``docs/CONTRACT.md``'s staleness rule exists to catch. A dropped frame
costs one sample of a slowly-spreading fire. A growing queue costs the
correctness of every box drawn afterwards. See :meth:`Pipeline._offer_frame`.

Safety invariants this module is responsible for:

* Every inferred frame is logged, **including the ones with no detections**. A
  frame the model looked at and found nothing in is a real observation and the
  only thing that makes the incident log an audit rather than a highlight reel.
* A frame the model *skipped* (the ``max_fps`` gate, or a backpressure drop) is
  not logged as empty. ``Runner.infer`` returns ``None`` for those and ``None``
  is not an empty result -- conflating them would fabricate evidence that the
  scene was examined.
* Nothing here emits a claim about the scene. :meth:`Pipeline.status` reports
  liveness -- is the model still looking, and how well is it keeping up -- and
  the ``note`` field carries only conditions the pipeline can self-detect.
* A source disconnect degrades the pipeline; it does not end it. The reader
  reopens the source for as long as the pipeline is running.

Heavy dependencies are imported inside the functions that need them, so this
module imports on a laptop with no codecs, no GPU and no aiortc.
"""

from __future__ import annotations

import asyncio
import json
import logging
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from station.core.config import Config, SourceConfig
from station.core.types import (
    FrameDetections,
    ModelInfo,
    PipelineState,
    PipelineStatus,
    utc_now_iso,
)
from station.inference.runner import InferenceUnavailableError, Runner
from station.inference.temporal import TemporalFilter

__all__ = [
    "PipelineError",
    "PipelineMetrics",
    "Pipeline",
    "ReplayRunner",
    "load_incident_detections",
    "incident_video_segments",
    "replay_source_config",
    "build_runner",
]

log = logging.getLogger(__name__)

#: Frames held between the decode thread and the inference thread. Two, not
#: two hundred: one being worked on, one ready to go. Anything larger is
#: overlay latency wearing a queue costume.
DEFAULT_QUEUE_SIZE = 2

#: Consecutive :class:`InferenceUnavailableError` s tolerated before the
#: pipeline stops. A transient CUDA fault can recover; a model that has fallen
#: over cannot, and a station that silently emitted nothing for the rest of an
#: incident would be indistinguishable from a quiet scene.
DEFAULT_MAX_INFERENCE_ERRORS = 5

#: Seconds over which ``source_fps`` and ``inference_fps`` are measured. Long
#: enough to be readable on a 1 Hz heartbeat, short enough that a GPU that
#: starts thermal-throttling shows up while the operator is still looking.
_RATE_WINDOW_S = 5.0

#: Achieved inference rate below this fraction of the rate that *should* be
#: achievable marks the pipeline degraded.
_DEGRADED_FPS_RATIO = 0.6

#: Extra grace before an unstarted pipeline is called stalled. Opening an RTSP
#: link and decoding to the first keyframe routinely takes longer than
#: ``stall_after_s``, and a station that declares itself stalled every time it
#: starts trains operators to ignore the word.
_STARTUP_GRACE_S = 5.0

#: Sentinel: build the temporal filter from ``cfg.temporal``. Distinct from
#: ``None``, which means "do not filter at all" (replay of an already-filtered
#: incident log).
_FROM_CONFIG: Any = object()

DetectionSink = Callable[[FrameDetections], Any]
StatusSink = Callable[[PipelineStatus], Any]


class PipelineError(RuntimeError):
    """The pipeline could not be started, or died in a way it cannot recover."""


@dataclass(slots=True)
class PipelineMetrics:
    """Counters describing what the station is doing. Never the scene.

    Every field is a fact about this process: frames handled, frames discarded,
    inferences completed, sources reopened. None of them is evidence about
    whether anything is burning.
    """

    #: Frames handed out by the source (after its own ``target_fps``
    #: decimation, which is configured sampling rather than a failure).
    frames_read: int = 0
    #: Frames the inference thread accepted from the queue.
    frames_dequeued: int = 0
    #: Frames discarded at the queue because inference was busy. This is the
    #: number the drop policy is about.
    frames_dropped: int = 0
    #: Frames the model actually ran on -- ``infer()`` returned a result.
    frames_inferred: int = 0
    #: Frames ``infer()`` declined by the ``max_fps`` gate. Not a failure and
    #: never logged as an empty observation.
    frames_gated: int = 0
    #: Payloads handed to the incident log and the transport.
    frames_published: int = 0
    #: Frames written to the incident video recording.
    frames_recorded: int = 0
    #: Times the source was reopened after ending or failing.
    source_restarts: int = 0
    #: Times ``infer()`` raised. Consecutive failures stop the pipeline.
    inference_errors: int = 0
    #: Exceptions raised by user-supplied subscribers, which are isolated.
    sink_errors: int = 0
    started_monotonic: float | None = None
    stopped_monotonic: float | None = None
    last_frame_monotonic: float | None = None
    last_inference_monotonic: float | None = None
    last_inference_pts: float | None = None
    last_inference_wall_time: str | None = None
    stream_start_pts: float | None = None

    @property
    def uptime_s(self) -> float:
        """Wall seconds since :meth:`Pipeline.start`, frozen once stopped."""
        if self.started_monotonic is None:
            return 0.0
        end = self.stopped_monotonic if self.stopped_monotonic is not None else time.monotonic()
        return max(0.0, end - self.started_monotonic)

    def as_dict(self) -> dict[str, Any]:
        """JSON-ready snapshot, for ``/healthz`` and the incident log."""
        return {
            "frames_read": self.frames_read,
            "frames_dequeued": self.frames_dequeued,
            "frames_dropped": self.frames_dropped,
            "frames_inferred": self.frames_inferred,
            "frames_gated": self.frames_gated,
            "frames_published": self.frames_published,
            "frames_recorded": self.frames_recorded,
            "source_restarts": self.source_restarts,
            "inference_errors": self.inference_errors,
            "sink_errors": self.sink_errors,
            "uptime_s": round(self.uptime_s, 3),
            "last_inference_pts": self.last_inference_pts,
            "stream_start_pts": self.stream_start_pts,
        }


class _RateWindow:
    """Events per wall second over a sliding window.

    Wall clock, not media pts: the question this answers is "is the station
    keeping up right now", which is a question about the laptop, not about the
    recording. ``PtsRateLimiter`` is the one that has to be pts-based.
    """

    def __init__(self, window_s: float = _RATE_WINDOW_S) -> None:
        self.window_s = float(window_s)
        self._times: deque[float] = deque()
        self._lock = threading.Lock()

    def tick(self, now: float | None = None) -> None:
        """Record one event."""
        stamp = time.monotonic() if now is None else now
        with self._lock:
            self._times.append(stamp)
            self._trim(stamp)

    def rate(self) -> float | None:
        """Events per second, or ``None`` before there is enough to divide by.

        ``None`` rather than ``0.0``: a rate of zero is a measurement, and this
        is the absence of one. The distinction matters because the heartbeat
        omits ``None`` fields entirely rather than sending a misleading zero.
        """
        now = time.monotonic()
        with self._lock:
            self._trim(now)
            if len(self._times) < 2:
                return None
            span = self._times[-1] - self._times[0]
            if span <= 0.0:
                return None
            # n-1 intervals between n events.
            return (len(self._times) - 1) / span

    def reset(self) -> None:
        """Forget every recorded event (source discontinuity)."""
        with self._lock:
            self._times.clear()

    def _trim(self, now: float) -> None:
        cutoff = now - self.window_s
        while self._times and self._times[0] < cutoff:
            self._times.popleft()


class Pipeline:
    """Drives frames from a source through the model and out to the tablets.

    Example:
        >>> import asyncio
        >>> from station.core.config import Config
        >>> from station.inference.stub import StubModelRunner
        >>> cfg = Config()
        >>> cfg.source.type = "synthetic"
        >>> cfg.source.uri = "synthetic:?fps=10&frames=20"
        >>> cfg.incident_log.enabled = False
        >>> seen = []
        >>> async def main():
        ...     pipe = Pipeline(cfg, runner=StubModelRunner(cfg.inference))
        ...     pipe.subscribe(seen.append)
        ...     await pipe.run()
        >>> asyncio.run(main())            # doctest: +SKIP
        >>> bool(seen)                     # doctest: +SKIP
        True

    Threading contract: :meth:`start`, :meth:`stop` and :meth:`run` are
    coroutines and must be called from the event loop. :meth:`status`,
    :meth:`state` and :attr:`metrics` are safe from any thread. Subscribers
    registered with :meth:`subscribe` are always invoked on the event loop, so
    they may be ordinary functions or coroutine functions.
    """

    def __init__(
        self,
        cfg: Config,
        *,
        runner: Runner | None = None,
        source: Any | None = None,
        source_factory: Callable[[], Any] | None = None,
        temporal: TemporalFilter | None = _FROM_CONFIG,
        streamer: Any | None = None,
        incident_log: Any | None = None,
        recorder: Any | None = None,
        queue_size: int = DEFAULT_QUEUE_SIZE,
        max_inference_errors: int = DEFAULT_MAX_INFERENCE_ERRORS,
        incident_slug: str | None = None,
    ) -> None:
        """Wire a pipeline. Nothing is opened and no thread starts here.

        Args:
            cfg: The whole station configuration.
            runner: The model. Defaults to a real
                :class:`~station.inference.runner.ModelRunner` built from
                ``cfg.inference``; pass
                :class:`~station.inference.stub.StubModelRunner` to exercise
                the station with no model and no GPU.
            source: An already-constructed
                :class:`~station.ingest.base.FrameSource`. It is reopened, not
                rebuilt, across disconnects.
            source_factory: Called to build a fresh source, including after a
                disconnect. Defaults to ``open_source(cfg.source)``. Mutually
                exclusive with ``source``.
            temporal: The N-of-M filter. Defaults to one built from
                ``cfg.temporal``. Pass ``None`` to publish the runner's output
                unfiltered -- correct only when the runner already emits
                filtered detections, as :class:`ReplayRunner` does.
            streamer: A :class:`station.stream.webrtc.WebRtcStreamer`, or
                anything with the same publish/set methods. Optional: the
                pipeline runs and logs perfectly well with no tablets attached.
            incident_log: A
                :class:`~station.incidentlog.writer.IncidentLogWriter`.
                Built from ``cfg.incident_log`` when omitted.
            recorder: A :class:`~station.incidentlog.recorder.VideoRecorder`.
                Built from ``cfg.incident_log`` when omitted.
            queue_size: Frames buffered between decode and inference. Keep it
                small; see the module docstring.
            max_inference_errors: Consecutive inference failures tolerated
                before the pipeline stops rather than emitting nothing.
            incident_slug: Label for the generated incident directory.

        Raises:
            ValueError: If both ``source`` and ``source_factory`` are given, or
                ``queue_size`` is below 1.
        """
        if source is not None and source_factory is not None:
            raise ValueError("pass either source or source_factory, not both")
        if queue_size < 1:
            raise ValueError(f"queue_size must be >= 1, got {queue_size}")

        self.cfg = cfg
        self.streamer = streamer
        self.max_inference_errors = int(max_inference_errors)
        self._incident_slug = incident_slug

        self._runner: Runner | None = runner
        self._temporal: TemporalFilter | None = (
            TemporalFilter(cfg.temporal) if temporal is _FROM_CONFIG else temporal
        )
        self._incident_log = incident_log
        self._incident_dir: Path | None = getattr(incident_log, "dir", None)
        self._recorder = recorder
        self._owns_incident_log = incident_log is None
        self._owns_recorder = recorder is None

        self._source: Any = source
        self._source_provided = source is not None
        self._source_ever_opened = False
        self._source_factory = source_factory

        self._queue: queue.Queue[Any] = queue.Queue(maxsize=int(queue_size))
        self._stop = threading.Event()
        self._reader: threading.Thread | None = None
        self._worker: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._status_task: asyncio.Task[None] | None = None
        self._finished = asyncio.Event()

        self._detection_sinks: list[DetectionSink] = []
        self._status_sinks: list[StatusSink] = []

        self._metrics = PipelineMetrics()
        self._source_rate = _RateWindow()
        self._inference_rate = _RateWindow()

        self._lock = threading.Lock()
        self._state = PipelineState.STARTING
        self._note: str | None = None
        self._fatal: BaseException | None = None
        self._started = False
        self._stopping = False
        self._source_exhausted = False

    # ------------------------------------------------------------- inspection

    @property
    def state(self) -> str:
        """Current :class:`~station.core.types.PipelineState`. Thread-safe."""
        with self._lock:
            return self._state

    @property
    def metrics(self) -> PipelineMetrics:
        """Live counters. Mutated in place; snapshot with ``as_dict()``."""
        return self._metrics

    @property
    def runner(self) -> Runner | None:
        """The model runner, once :meth:`start` has built it."""
        return self._runner

    @property
    def incident_log(self) -> Any | None:
        """The incident log writer, once :meth:`start` has built it."""
        return self._incident_log

    @property
    def incident_dir(self) -> Path | None:
        """Directory this run recorded into, or ``None``.

        Remembered rather than read back off the writer, so it is still
        answerable after :meth:`stop` -- which is when a caller wants it, to
        tell the operator where the record went.
        """
        return self._incident_dir

    @property
    def error(self) -> BaseException | None:
        """The exception that stopped the pipeline, if one did."""
        return self._fatal

    @property
    def model_info(self) -> ModelInfo | None:
        """Provenance of whatever is producing detections."""
        runner = self._runner
        return runner.model_info if runner is not None else None

    def status(self) -> PipelineStatus:
        """Build the heartbeat for right now. Safe from any thread.

        Returns:
            A :class:`~station.core.types.PipelineStatus`. It describes the
            pipeline -- rates, drops, uptime, whether an inference has landed
            recently -- and contains no statement about the scene.

        The state is recomputed here rather than read from the last heartbeat.
        ``/healthz`` and the heartbeat must never disagree, and a cached state
        would let a pipeline that stalled half a second ago still answer
        "running" to whoever asked first -- reporting liveness the station no
        longer has is precisely the failure this field exists to catch.
        """
        state, note = self._evaluate_state()
        self._set_state(state, note)
        m = self._metrics
        source = self._source
        source_fps = None
        if source is not None:
            source_fps = getattr(getattr(source, "stats", None), "measured_source_fps", None)
        if source_fps is None:
            source_fps = self._source_rate.rate()
        return PipelineStatus(
            state=state,
            source=self.cfg.source.uri or self.cfg.source.type,
            model=self.model_info,
            source_fps=source_fps,
            inference_fps=self._inference_rate.rate(),
            last_inference_pts=m.last_inference_pts,
            last_inference_wall_time=m.last_inference_wall_time,
            dropped_frames=m.frames_dropped,
            uptime_s=m.uptime_s,
            stream_start_pts=m.stream_start_pts,
            note=note,
        )

    # ----------------------------------------------------------- subscription

    def subscribe(self, sink: DetectionSink) -> Callable[[], None]:
        """Receive every published :class:`FrameDetections`.

        The callback runs on the event loop and may be a coroutine function.
        It receives frames with empty ``detections`` too, because those are
        observations; a subscriber that filters them out is choosing to record
        a highlight reel.

        Args:
            sink: Called with each published frame result.

        Returns:
            A callable that removes the subscription.
        """
        self._detection_sinks.append(sink)
        return lambda: self._detection_sinks.remove(sink) if sink in self._detection_sinks else None

    def subscribe_status(self, sink: StatusSink) -> Callable[[], None]:
        """Receive each heartbeat. Same threading rules as :meth:`subscribe`.

        Args:
            sink: Called with each :class:`PipelineStatus`.

        Returns:
            A callable that removes the subscription.
        """
        self._status_sinks.append(sink)
        return lambda: self._status_sinks.remove(sink) if sink in self._status_sinks else None

    # -------------------------------------------------------------- lifecycle

    async def start(self) -> None:
        """Load the model, open the incident log, and start the threads.

        The model is loaded here rather than in the decode loop so that a
        missing weights file or a broken CUDA install fails at startup, in
        front of the operator, instead of during an incident.

        Raises:
            PipelineError: If the pipeline has already been started.
            InferenceUnavailableError: If the model cannot be loaded.
        """
        if self._started:
            raise PipelineError("pipeline already started")
        self._started = True
        self._loop = asyncio.get_running_loop()
        self._finished.clear()
        self._metrics.started_monotonic = time.monotonic()
        self._set_state(PipelineState.STARTING, None)

        if self._runner is None:
            self._runner = build_runner(self.cfg)
        self._runner.load()

        self._open_incident_log()

        if self.streamer is not None:
            self.streamer.set_model(self.model_info)
            self.streamer.set_source(self.cfg.source.uri or self.cfg.source.type)
            self.streamer.set_state(PipelineState.STARTING)

        self._stop.clear()
        self._reader = threading.Thread(target=self._reader_loop, name="wildfire-decode", daemon=True)
        self._worker = threading.Thread(target=self._inference_loop, name="wildfire-inference", daemon=True)
        self._reader.start()
        self._worker.start()
        self._status_task = asyncio.create_task(self._status_loop(), name="wildfire-pipeline-status")
        log.info(
            "pipeline started: source=%s model=%s queue=%d",
            self.cfg.source.uri or self.cfg.source.type,
            self.model_info.name if self.model_info else "none",
            self._queue.maxsize,
        )

    async def stop(self) -> None:
        """Stop the threads, close the source, model, log and recorder.

        Idempotent, and safe to call on a pipeline that never started.
        """
        if self._stopping:
            return
        self._stopping = True
        self._stop.set()
        # Replace whatever note was current: "stopped, source reconnecting" is
        # a contradiction, and the operator needs to know why it stopped rather
        # than what it was worrying about a second earlier.
        if self._fatal is not None:
            reason: str | None = f"stopped on an error: {self._fatal}"
        elif self._source_exhausted:
            reason = "source finished"
        else:
            reason = None
        self._set_state(PipelineState.STOPPED, reason)
        self._metrics.stopped_monotonic = time.monotonic()

        if self._status_task is not None:
            self._status_task.cancel()
            try:
                await self._status_task
            except asyncio.CancelledError:
                pass
            except Exception:  # pragma: no cover - already logged in the loop
                log.debug("status task ended with an error", exc_info=True)
            self._status_task = None

        # Closing the source is what unblocks a reader parked in a blocking
        # read on a link that has gone away; FrameSource.close() is documented
        # to be safe from another thread precisely for this.
        source = self._source
        if source is not None:
            try:
                source.close()
            except Exception:
                log.debug("closing the source raised during shutdown", exc_info=True)

        # Threads are joined off the loop: they can be in a multi-second
        # blocking read, and blocking the loop here would stop the final
        # heartbeat and the WebRTC teardown from ever being sent.
        for thread in (self._reader, self._worker):
            if thread is not None and thread.is_alive():
                await asyncio.to_thread(thread.join, 5.0)
                if thread.is_alive():
                    log.warning("%s did not stop within 5s; abandoning it", thread.name)
        self._reader = self._worker = None

        if self.streamer is not None:
            try:
                self.streamer.set_state(PipelineState.STOPPED)
            except Exception:
                log.debug("streamer refused the final state", exc_info=True)

        self._close_incident_log()

        runner = self._runner
        if runner is not None:
            try:
                runner.close()
            except Exception:
                log.debug("runner.close() raised", exc_info=True)

        self._finished.set()
        log.info("pipeline stopped: %s", json.dumps(self._metrics.as_dict(), separators=(",", ":")))

    async def wait(self) -> None:
        """Block until the pipeline finishes on its own.

        A live source runs until :meth:`stop`; a file source finishes at end of
        file (unless ``source.loop``).
        """
        await self._finished.wait()

    async def run(self) -> None:
        """:meth:`start`, wait for the source to finish, then :meth:`stop`.

        Raises:
            Exception: Whatever stopped the pipeline, once it has been shut
                down cleanly. A pipeline that dies on a model fault must still
                close its incident log.
        """
        try:
            await self.start()
            await self.wait()
        finally:
            # Also on a failed start: by then the incident log may already be
            # open, and an unclosed log leaves a directory with no final
            # metadata for a run that never happened.
            await self.stop()
        if self._fatal is not None:
            raise self._fatal

    async def __aenter__(self) -> "Pipeline":
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.stop()

    # ------------------------------------------------------------ decode side

    def _reader_loop(self) -> None:
        """Read frames for as long as the pipeline is running.

        Owns the source, including reopening it. A radio link that drops must
        not end the run: the operator is standing next to a drone, not next to
        the laptop, and the station coming back by itself is the difference
        between a gap in the record and the end of it.
        """
        backoff = max(0.1, float(self.cfg.source.reconnect_s))
        while not self._stop.is_set():
            source = self._source
            if source is None or not getattr(source, "is_open", False):
                if not self._open_source():
                    if self._stop.wait(backoff):
                        break
                    continue
                source = self._source

            try:
                frame = source.read()
            except Exception as exc:
                # Any decode-level failure is treated as a disconnect. The
                # ingest layer already retries what it can retry; anything
                # reaching here means the source object itself is finished.
                self._note_trouble(f"source error: {exc}")
                log.warning("source read failed (%s); reopening", exc, exc_info=True)
                self._close_source()
                if self._stop.wait(backoff):
                    break
                continue

            if frame is None:
                if self._stop.is_set():
                    break
                if self._handle_source_end():
                    break
                if self._stop.wait(backoff):
                    break
                continue

            self._handle_frame(frame)

        log.debug("decode thread finished")

    def _handle_frame(self, frame: Any) -> None:
        """Publish, record and enqueue one decoded frame."""
        m = self._metrics
        m.frames_read += 1
        m.last_frame_monotonic = time.monotonic()
        self._source_rate.tick(m.last_frame_monotonic)
        if m.stream_start_pts is None:
            m.stream_start_pts = float(frame.pts)

        # Video first, and unconditionally. The operator is watching the video;
        # it must keep flowing even when inference is behind, missing or
        # broken. Detections augment it -- they are not a precondition for it.
        if self.streamer is not None:
            try:
                self.streamer.publish_frame(frame)
            except Exception:
                log.debug("streamer refused a frame", exc_info=True)

        recorder = self._recorder
        if recorder is not None:
            try:
                if recorder.write(frame):
                    m.frames_recorded += 1
            except Exception:
                # The recorder is documented never to raise into the frame
                # loop; belt and braces, because losing the recording is bad
                # and losing the live feed during an incident is not survivable.
                log.debug("video recorder raised", exc_info=True)

        self._offer_frame(frame)

    def _offer_frame(self, frame: Any) -> None:
        """Hand a frame to inference, dropping the oldest rather than queueing.

        This is the drop policy. When the queue is full the *oldest* waiting
        frame is thrown away and the new one takes its place, so what inference
        sees next is always the freshest frame available.

        Dropping the newest instead would be the intuitive choice and it is the
        wrong one: it pins the model to a moment further and further in the
        past while live video runs on without it. Dropping the oldest keeps the
        overlay's lag bounded by one inference, which is what makes the boxes
        land on the frame they were computed from. Not dropping at all -- an
        unbounded queue -- makes that lag grow without limit until the tablet's
        staleness rule blanks the overlay entirely.

        Args:
            frame: The decoded frame to offer.
        """
        try:
            self._queue.put_nowait(frame)
            return
        except queue.Full:
            pass
        try:
            self._queue.get_nowait()
            self._metrics.frames_dropped += 1
        except queue.Empty:
            # The worker drained it between the two calls; nothing was lost.
            pass
        try:
            self._queue.put_nowait(frame)
        except queue.Full:  # pragma: no cover - only with several producers
            self._metrics.frames_dropped += 1

    def _open_source(self) -> bool:
        """Open (or reopen) the video source.

        Returns:
            ``True`` when a source is open and ready to read.
        """
        try:
            if self._source_provided and self._source is not None:
                self._source.open()
            else:
                self._source = self._build_source()
            self._source_ever_opened = True
            self._note_source_recovered()
            info = getattr(self._source, "info", None)
            log.info("source open: %s", info.describe() if info is not None else self._source)
            self._configure_recorder_fps(info)
            return True
        except Exception as exc:
            self._note_trouble(f"source unavailable: {exc}")
            log.warning("cannot open source %r: %s", self.cfg.source.uri or self.cfg.source.type, exc)
            return False

    def _build_source(self) -> Any:
        """Construct a fresh source from the factory or the config."""
        if self._source_factory is not None:
            return self._source_factory()
        # Imported here: station.ingest pulls in numpy, and the pure-logic
        # modules must import on a machine without it.
        from station.ingest import open_source  # noqa: PLC0415

        return open_source(self.cfg.source)

    def _close_source(self) -> None:
        """Close the current source, tolerating any failure."""
        source = self._source
        if source is None:
            return
        try:
            source.close()
        except Exception:
            log.debug("closing a failed source raised", exc_info=True)
        if not self._source_provided:
            self._source = None

    def _handle_source_end(self) -> bool:
        """React to ``read()`` returning ``None``.

        Returns:
            ``True`` when the run is genuinely over (a file played to its end),
            ``False`` when the reader should reopen and carry on.
        """
        source = self._source
        info = getattr(source, "info", None)
        is_live = bool(getattr(info, "is_live", False))
        if not is_live and not self.cfg.source.loop:
            log.info("source exhausted after %d frames; finishing", self._metrics.frames_read)
            self._source_exhausted = True
            self._request_finish()
            return True

        self._metrics.source_restarts += 1
        self._note_trouble("source ended; reopening")
        log.warning(
            "source ended (restart %d); reopening in %.1fs",
            self._metrics.source_restarts, self.cfg.source.reconnect_s,
        )
        self._close_source()
        # The media timeline restarts from the source's own zero, so the
        # transport's pts bookkeeping and the model's rate limiter both have to
        # be told: a pts that jumps backwards would otherwise look like a
        # decode fault to one and a stall to the other.
        self._reset_timeline()
        return False

    def _reset_timeline(self) -> None:
        """Tell every stateful component that the media clock restarted."""
        self._metrics.stream_start_pts = None
        self._source_rate.reset()
        self._inference_rate.reset()
        runner = self._runner
        if runner is not None:
            try:
                runner.reset()
            except Exception:
                log.debug("runner.reset() raised", exc_info=True)
        if self._temporal is not None:
            self._temporal.reset()
        if self.streamer is not None:
            try:
                self.streamer.reset_timeline()
            except Exception:
                log.debug("streamer.reset_timeline() raised", exc_info=True)
        # Drop whatever was queued from the old timeline: those frames carry
        # pts values from before the discontinuity and inferring them now would
        # publish boxes stamped for a moment that has already been replaced.
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    # --------------------------------------------------------- inference side

    def _inference_loop(self) -> None:
        """Run the model on queued frames until the pipeline stops."""
        consecutive_errors = 0
        while not self._stop.is_set():
            try:
                frame = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if self._stop.is_set():
                break

            self._metrics.frames_dequeued += 1
            try:
                result = self._infer(frame)
            except InferenceUnavailableError as exc:
                consecutive_errors += 1
                self._metrics.inference_errors += 1
                log.error(
                    "inference failed (%d/%d consecutive): %s",
                    consecutive_errors, self.max_inference_errors, exc,
                )
                self._note_trouble(f"inference failing: {exc}")
                if consecutive_errors >= self.max_inference_errors:
                    # Emitting nothing forever would be indistinguishable from
                    # a working model looking at a quiet scene. Stop loudly.
                    self._fail(exc)
                    break
                continue
            except Exception as exc:  # pragma: no cover - runner contract break
                self._metrics.inference_errors += 1
                log.exception("unexpected error in the inference loop")
                self._fail(exc)
                break

            consecutive_errors = 0
            if result is None:
                # The rate limiter declined this frame. Not an observation, so
                # nothing is logged and nothing is published: an empty payload
                # here would claim the model looked when it did not.
                self._metrics.frames_gated += 1
                continue
            self._publish(result)

        log.debug("inference thread finished")

    def _infer(self, frame: Any) -> FrameDetections | None:
        """Run the model and the temporal filter on one frame."""
        runner = self._runner
        if runner is None:  # pragma: no cover - start() guarantees one
            raise InferenceUnavailableError("pipeline has no model runner")
        raw = runner.infer(
            frame.image,
            pts=float(frame.pts),
            frame_id=int(frame.frame_id),
            rtp_ts=getattr(frame, "rtp_ts", None),
            source_id=self.cfg.source.uri or None,
        )
        if raw is None:
            return None
        if self._temporal is None:
            return raw
        return self._temporal.filter_frame(raw)

    def _publish(self, result: FrameDetections) -> None:
        """Record, transmit and fan out one frame's result.

        Called on the inference thread. The incident log and the transport are
        both written from here rather than being posted to the event loop: the
        log must record what the model saw even when the loop is busy, and the
        transport is documented as safe to call from any thread. Deferring
        either would add latency to the overlay for no benefit.
        """
        m = self._metrics
        now = time.monotonic()
        m.frames_inferred += 1
        m.frames_published += 1
        m.last_inference_monotonic = now
        m.last_inference_pts = result.pts
        m.last_inference_wall_time = result.wall_time or utc_now_iso()
        self._inference_rate.tick(now)

        # Transport first, disk second. Publishing is a cheap hand-off to the
        # event loop; the incident log fsyncs every frame by default, and a
        # slow disk must not sit between the model finishing and the tablet
        # getting the box. The log still records the frame a moment later, and
        # nothing downstream can tell the difference.
        if self.streamer is not None:
            try:
                self.streamer.publish_detections(result)
            except Exception:
                log.debug("streamer refused a detections payload", exc_info=True)

        writer = self._incident_log
        if writer is not None:
            try:
                # Every frame, including the empty ones. That is what makes
                # this an audit trail rather than a highlight reel, and it is
                # what the false-negative review in docs/VALIDATION.md needs.
                writer.write(result)
            except Exception:
                log.debug("incident log write raised", exc_info=True)

        self._dispatch(self._detection_sinks, result)

    # ----------------------------------------------------------- status loop

    async def _status_loop(self) -> None:
        """Recompute the pipeline state and emit the heartbeat on a cadence."""
        interval = max(0.1, float(self.cfg.stream.status_interval_s))
        last_state: str | None = None
        try:
            while not self._stop.is_set():
                status = self.status()  # recomputes and caches the state
                state, note = status.state, status.note

                if self.streamer is not None:
                    try:
                        self.streamer.set_state(state)
                        self.streamer.set_note(note)
                        self.streamer.set_rates(
                            source_fps=status.source_fps, inference_fps=status.inference_fps
                        )
                        self.streamer.set_dropped_frames(status.dropped_frames)
                    except Exception:
                        log.debug("streamer refused a status update", exc_info=True)

                if state != last_state:
                    # State changes go into the incident log as events, so an
                    # after-action review can see exactly when the station
                    # stopped keeping up rather than inferring it from gaps.
                    log.info("pipeline state -> %s%s", state, f" ({note})" if note else "")
                    writer = self._incident_log
                    if writer is not None:
                        try:
                            writer.log_status(status)
                        except Exception:
                            log.debug("incident log status write raised", exc_info=True)
                    last_state = state

                for sink in list(self._status_sinks):
                    self._call_sink(sink, status)

                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            raise
        except Exception:  # pragma: no cover - defensive
            log.exception("status loop stopped on an unhandled error")

    def _evaluate_state(self) -> tuple[str, str | None]:
        """Decide the current state and operator-facing note.

        Returns:
            ``(state, note)``. The note describes the *pipeline* -- a
            reconnecting source, a model behind its configured rate, a failing
            incident log. It never describes the scene.
        """
        if self._stop.is_set() or self._stopping:
            return PipelineState.STOPPED, self._note
        m = self._metrics
        now = time.monotonic()
        stall_after = max(0.1, float(self.cfg.stream.stall_after_s))

        if m.last_inference_monotonic is None:
            # Nothing has come through yet. Starting is the honest answer until
            # the startup grace runs out; after that the station is not looking
            # and must say so.
            if m.uptime_s < stall_after + _STARTUP_GRACE_S:
                return PipelineState.STARTING, self._source_note() or "waiting for the first frame"
            return PipelineState.STALLED, self._source_note() or "no inference has completed since start"

        age = now - m.last_inference_monotonic
        if age > stall_after:
            return (
                PipelineState.STALLED,
                self._source_note() or f"no inference completed in {age:.1f}s",
            )

        source_note = self._source_note()
        if source_note:
            return PipelineState.DEGRADED, source_note

        if not self._log_healthy():
            return PipelineState.DEGRADED, "incident log is not recording"

        achieved = self._inference_rate.rate()
        if achieved is not None:
            source_fps = self._source_rate.rate()
            # The model cannot beat its own rate cap, nor the arrival rate.
            expected = float(self.cfg.inference.max_fps)
            if source_fps is not None:
                expected = min(expected, source_fps)
            if expected > 0 and achieved < expected * _DEGRADED_FPS_RATIO:
                return (
                    PipelineState.DEGRADED,
                    f"inference at {achieved:.1f}/s against {expected:.1f}/s available",
                )
        return PipelineState.RUNNING, None

    def _source_note(self) -> str | None:
        """Operator-facing note about the source, if it is in trouble."""
        source = self._source
        stats = getattr(source, "stats", None) if source is not None else None
        if source is None or not getattr(source, "is_open", False):
            # "Not open yet" and "was open and went away" are different facts
            # to the operator: one is a station still coming up, the other is a
            # link that has dropped.
            return "source reconnecting" if self._source_ever_opened else "opening source"
        if stats is not None and getattr(stats, "reconnecting", False):
            return "source reconnecting"
        return None

    def _log_healthy(self) -> bool:
        """Whether the incident log is still recording, when there is one."""
        writer = self._incident_log
        if writer is None:
            return True
        return bool(getattr(writer, "healthy", True))

    # ---------------------------------------------------------------- sinks

    def _dispatch(self, sinks: Sequence[Callable[[Any], Any]], payload: Any) -> None:
        """Fan a payload out to subscribers on the event loop.

        Called from the inference thread, so the hop through the loop is what
        lets subscribers be ordinary async code without any locking of their
        own.
        """
        if not sinks:
            return
        loop = self._loop
        if loop is None or loop.is_closed():  # pragma: no cover - shutdown race
            return
        snapshot = list(sinks)
        try:
            loop.call_soon_threadsafe(self._dispatch_now, snapshot, payload)
        except RuntimeError:  # pragma: no cover - loop closed under us
            log.debug("event loop gone; dropping a subscriber dispatch")

    def _dispatch_now(self, sinks: Sequence[Callable[[Any], Any]], payload: Any) -> None:
        for sink in sinks:
            self._call_sink(sink, payload)

    def _call_sink(self, sink: Callable[[Any], Any], payload: Any) -> None:
        """Invoke one subscriber, isolating its failures from the pipeline.

        A subscriber that raises must not be able to stop the station: it is
        third-party code hanging off a system whose job is to keep showing
        video.
        """
        try:
            result = sink(payload)
        except Exception:
            self._metrics.sink_errors += 1
            log.exception("a pipeline subscriber raised; continuing")
            return
        if asyncio.iscoroutine(result):
            loop = self._loop
            if loop is None or loop.is_closed():  # pragma: no cover
                result.close()
                return
            task = loop.create_task(result)
            task.add_done_callback(self._sink_task_done)

    def _sink_task_done(self, task: "asyncio.Task[Any]") -> None:
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            self._metrics.sink_errors += 1
            log.error("an async pipeline subscriber failed: %s", exc, exc_info=exc)

    # ----------------------------------------------------------- bookkeeping

    def _set_state(self, state: str, note: str | None) -> None:
        with self._lock:
            self._state = state
            self._note = note

    def _note_trouble(self, note: str) -> None:
        with self._lock:
            self._note = note

    def _note_source_recovered(self) -> None:
        with self._lock:
            if self._note and ("source" in self._note):
                self._note = None

    def _fail(self, exc: BaseException) -> None:
        """Record a fatal error and ask the run to finish."""
        if self._fatal is None:
            self._fatal = exc
        self._request_finish()

    def _request_finish(self) -> None:
        """Ask :meth:`wait` to return, from a worker thread."""
        self._stop.set()
        loop = self._loop
        if loop is None or loop.is_closed():  # pragma: no cover
            return
        try:
            loop.call_soon_threadsafe(self._finished.set)
        except RuntimeError:  # pragma: no cover
            pass

    # -------------------------------------------------------- incident log

    def _open_incident_log(self) -> None:
        """Create and start the incident log and video recorder."""
        if self._owns_incident_log and self.cfg.incident_log.enabled:
            from station.incidentlog.writer import IncidentLogWriter  # noqa: PLC0415

            self._incident_log = IncidentLogWriter.from_config(
                self.cfg, model=self.model_info, slug=self._incident_slug
            )
        writer = self._incident_log
        if writer is not None:
            try:
                writer.start()
            except Exception:
                # The writer degrades rather than raising, but a filesystem
                # that is gone entirely should not stop the station streaming.
                log.exception("could not start the incident log; continuing without it")
                self._incident_log = None
                writer = None

        if writer is not None and self._owns_recorder and self.cfg.incident_log.record_video:
            from station.incidentlog.recorder import VideoRecorder  # noqa: PLC0415

            incident_dir = writer.dir
            if incident_dir is not None:
                self._recorder = VideoRecorder.from_config(self.cfg, incident_dir)
        if writer is not None:
            self._incident_dir = writer.dir
            if self._incident_dir is not None:
                log.info("incident log: %s", self._incident_dir)

    def _configure_recorder_fps(self, info: Any) -> None:
        """Tell the recorder the source's real frame rate, once it is known.

        The OpenCV encoder fallback has no timestamps at all and plays back at
        whatever rate it was told, so getting this wrong makes a recording that
        no longer lines up with the pts values in ``detections.jsonl``.
        """
        recorder = self._recorder
        if recorder is None or info is None:
            return
        fps = getattr(info, "output_fps", None)
        if fps and fps > 0 and not getattr(recorder, "recording", False):
            recorder.fps = float(fps)

    def _close_incident_log(self) -> None:
        """Close the recorder and the writer, in that order.

        Order matters: the recorder's manifest is what tells a reviewer which
        video file each ``pts`` in ``detections.jsonl`` refers to, and it has to
        be complete before it is stored in ``meta.json``.
        """
        recorder = self._recorder
        if recorder is not None:
            try:
                recorder.close()
            except Exception:
                log.debug("closing the video recorder raised", exc_info=True)
            if self._incident_log is not None:
                try:
                    self._incident_log.set_video_manifest(recorder.manifest())
                except Exception:
                    log.debug("storing the video manifest raised", exc_info=True)
            if self._owns_recorder:
                self._recorder = None

        writer = self._incident_log
        if writer is not None:
            try:
                writer.log_event("pipeline_stopped", **self._metrics.as_dict())
                writer.close()
            except Exception:
                log.debug("closing the incident log raised", exc_info=True)
            if self._owns_incident_log:
                self._incident_log = None


# --------------------------------------------------------------------- runner


def build_runner(cfg: Config) -> Runner:
    """Build the model runner described by ``cfg.inference``.

    Args:
        cfg: The station configuration.

    Returns:
        A :class:`~station.inference.runner.ModelRunner`, unloaded.

    Note:
        Deliberately never falls back to the stub when the weights are missing.
        A station that quietly ran a stub would emit empty payloads that are
        byte-identical to a working model finding nothing -- the exact
        confusion ``station/core/safety.py`` exists to prevent. Selecting the
        stub is an explicit choice made at the command line.
    """
    from station.inference.runner import ModelRunner  # noqa: PLC0415

    return ModelRunner(cfg.inference)


# --------------------------------------------------------------------- replay


class ReplayRunner:
    """Serves detections back out of a recorded incident log.

    Satisfies :class:`~station.inference.runner.Runner`, so ``station replay``
    drives the *same* pipeline, transport and PWA as a live deployment. That is
    the point: a training session or a demo exercises the code that will be
    running at the next incident, rather than a parallel implementation of it
    that can drift.

    The recorded payloads have already been through the temporal filter, so a
    replaying pipeline is constructed with ``temporal=None``. Filtering them
    again would apply N-of-M twice and quietly drop the first frames of every
    track.

    Example:
        >>> from station.core.types import FrameDetections
        >>> runner = ReplayRunner([FrameDetections(frame_id=0, pts=1.0)])
        >>> out = runner.infer(None, pts=1.001, frame_id=7)
        >>> out.pts, len(out)
        (1.0, 0)
        >>> runner.infer(None, pts=9.0) is None   # nothing recorded near 9.0
        True
    """

    def __init__(
        self,
        frames: Iterable[FrameDetections],
        *,
        tolerance_s: float = 0.05,
        model: ModelInfo | None = None,
    ) -> None:
        """Create a replay runner.

        Args:
            frames: Recorded results, in any order; sorted here by ``pts``.
            tolerance_s: How near a video frame's ``pts`` must be to a recorded
                one to be considered the same frame. Half a frame period at
                10 fps inference by default.
            model: Provenance to report. Defaults to the model recorded in the
                first frame that carries one, so a replay is attributed to the
                model that produced it rather than to whatever is installed on
                the laptop doing the replaying.

        Raises:
            ValueError: If ``tolerance_s`` is negative.
        """
        if tolerance_s < 0:
            raise ValueError(f"tolerance_s must be >= 0, got {tolerance_s}")
        self._frames: list[FrameDetections] = sorted(frames, key=lambda f: f.pts)
        self._pts = [f.pts for f in self._frames]
        self.tolerance_s = float(tolerance_s)
        self._model = model or next((f.model for f in self._frames if f.model is not None), None)
        self._stats = _ReplayStats()
        self._cursor = 0
        self._served: set[int] = set()

    @property
    def frames(self) -> tuple[FrameDetections, ...]:
        """The recorded results, ordered by ``pts``."""
        return tuple(self._frames)

    @property
    def model_info(self) -> ModelInfo:
        """The model that produced the recording, as far as it was recorded."""
        if self._model is not None:
            return self._model
        return ModelInfo(name="replay", version="unknown")

    @property
    def stats(self) -> Any:
        """Counters, shaped like :class:`~station.inference.runner.RunnerStats`."""
        return self._stats

    def load(self) -> None:
        """No model to load; logs loudly that this is a replay.

        A replay looks exactly like a live feed on a tablet, which is what
        makes it useful for training and dangerous to mistake for live. The
        station log says which it is.
        """
        log.warning(
            "REPLAY: serving %d recorded results from a previous incident. No model is running "
            "and this is not a live feed.",
            len(self._frames),
        )

    def reset(self) -> None:
        """Rewind to the start of the recording (the source looped)."""
        self._cursor = 0
        self._served.clear()

    def close(self) -> None:
        """Nothing to release."""

    def infer(
        self,
        frame: Any,
        pts: float,
        frame_id: int | None = None,
        *,
        rtp_ts: int | None = None,
        source_id: str | None = None,
    ) -> FrameDetections | None:
        """Return the recorded result for the frame at ``pts``.

        Args:
            frame: Ignored -- the pixels are not re-examined.
            pts: Media timestamp of the video frame being replayed.
            frame_id: Ignored; the recorded ``frame_id`` is preserved so the
                replayed payloads match the log they came from.
            rtp_ts: The live stream's RTP timestamp for this frame, which
                replaces the recorded one so the tablet's tier-1 alignment
                works against *this* transmission.
            source_id: Ignored; provenance stays with the recording.

        Returns:
            The recorded :class:`FrameDetections`, or ``None`` when no result
            was recorded within ``tolerance_s`` of this frame.

            ``None`` is the honest answer for a frame the original run never
            inferred -- most frames, at 10 fps inference over 30 fps video.
            Returning an empty result instead would invent an observation that
            was never made.
        """
        self._stats.frames_seen += 1
        index = self._nearest(float(pts))
        if index is None:
            self._stats.frames_skipped += 1
            return None
        # One recorded result is served once. Without this, a video whose frame
        # rate exceeds the recorded inference rate would republish the same
        # payload several times, and the tablet would show a box that appears
        # to persist longer than it really did.
        if index in self._served:
            self._stats.frames_skipped += 1
            return None
        self._served.add(index)
        self._cursor = index + 1
        result = self._frames[index]
        self._stats.frames_inferred += 1
        self._stats.last_inference_pts = result.pts
        self._stats.last_inference_wall_time = utc_now_iso()
        if rtp_ts is not None and result.rtp_ts != rtp_ts:
            import dataclasses  # noqa: PLC0415

            result = dataclasses.replace(result, rtp_ts=rtp_ts)
        return result

    def _nearest(self, pts: float) -> int | None:
        """Index of the recorded result nearest ``pts``, within tolerance."""
        if not self._pts:
            return None
        import bisect  # noqa: PLC0415

        i = bisect.bisect_left(self._pts, pts)
        best: int | None = None
        best_gap = self.tolerance_s
        for candidate in (i - 1, i):
            if 0 <= candidate < len(self._pts):
                gap = abs(self._pts[candidate] - pts)
                if gap <= best_gap:
                    best, best_gap = candidate, gap
        return best


@dataclass(slots=True)
class _ReplayStats:
    """Counters mirroring :class:`~station.inference.runner.RunnerStats`."""

    frames_seen: int = 0
    frames_inferred: int = 0
    frames_skipped: int = 0
    last_inference_pts: float | None = None
    last_inference_wall_time: str | None = None
    mean_inference_ms: float | None = None

    @property
    def inference_fps(self) -> float | None:
        """Always ``None``: nothing was inferred, so there is no rate."""
        return None


def load_incident_detections(incident_dir: str | Path) -> list[FrameDetections]:
    """Read ``detections.jsonl`` from a recorded incident.

    Args:
        incident_dir: The incident directory written by
            :class:`~station.incidentlog.writer.IncidentLogWriter`.

    Returns:
        Every recorded frame result, ordered by ``pts``. Includes the empty
        ones, which is the whole point of the record.

    Raises:
        FileNotFoundError: If the directory has no ``detections.jsonl``.

    Note:
        Malformed lines are logged and skipped rather than aborting the load. A
        log truncated by a power cut mid-write is a normal way for one to end,
        and the recoverable prefix is still worth reviewing.
    """
    from station.core.types import parse_message  # noqa: PLC0415

    path = Path(incident_dir).expanduser()
    if path.is_file():
        detections_file = path
    else:
        detections_file = path / "detections.jsonl"
    if not detections_file.is_file():
        raise FileNotFoundError(
            f"no detections.jsonl in {path}. Point 'station replay' at an incident directory "
            "written by station.incidentlog (it contains detections.jsonl, meta.json and video/)."
        )

    out: list[FrameDetections] = []
    bad = 0
    with detections_file.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                message = parse_message(line)
            except (ValueError, KeyError, TypeError) as exc:
                bad += 1
                if bad <= 5:
                    log.warning("%s:%d is not a readable payload (%s); skipping", detections_file, line_no, exc)
                continue
            if isinstance(message, FrameDetections):
                out.append(message)
    if bad:
        log.warning("%s: skipped %d unreadable line(s)", detections_file, bad)
    out.sort(key=lambda f: f.pts)
    log.info("loaded %d recorded frame results from %s", len(out), detections_file)
    return out


def incident_video_segments(incident_dir: str | Path) -> list[Path]:
    """List the recorded video segments of an incident, in timeline order.

    Args:
        incident_dir: The incident directory.

    Returns:
        Segment paths ordered by their recorded ``start_pts`` when
        ``video/segments.json`` is present, otherwise sorted by filename. An
        empty list when nothing was recorded.
    """
    root = Path(incident_dir).expanduser()
    video_dir = root / "video"
    if not video_dir.is_dir():
        return []

    manifest = video_dir / "segments.json"
    if manifest.is_file():
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
            segments = data.get("segments") or []
            ordered = sorted(segments, key=lambda s: (s.get("start_pts") or 0.0, s.get("index") or 0))
            paths = [video_dir / str(s["file"]) for s in ordered if s.get("file")]
            existing = [p for p in paths if p.is_file()]
            if existing:
                return existing
        except (OSError, ValueError, KeyError, TypeError) as exc:
            log.warning("%s is unreadable (%s); falling back to a directory listing", manifest, exc)

    return sorted(p for p in video_dir.iterdir() if p.is_file() and p.suffix.lower() in {".mp4", ".mkv", ".avi", ".webm"})


def replay_source_config(segment: Path, *, loop: bool = False, target_fps: float | None = None) -> SourceConfig:
    """Build the ``source`` config that replays one recorded segment.

    Args:
        segment: A video file from ``incident_video_segments``.
        loop: Replay the segment forever -- useful for an unattended demo.
        target_fps: Cap the replay rate; ``None`` uses the file's own rate.

    Returns:
        A :class:`~station.core.config.SourceConfig` for the file source.
    """
    return SourceConfig(type="file", uri=str(segment), loop=loop, target_fps=target_fps)
