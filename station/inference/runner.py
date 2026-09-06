"""YOLO inference on the ground station, and only on the ground station.

Inference runs here, once, on the laptop that holds the RTSP feed. Tablets
receive JSON boxes over the data channel and draw them on a canvas; they never
run a model and detections are never burned into video pixels. That is an
architectural choice, not an optimisation: burned-in boxes cannot be turned
off, cannot be labelled stale, and cannot be re-examined after the fact
against the frame they were computed from.

Every heavyweight dependency -- ``ultralytics``, ``torch``, ``numpy`` --
is imported inside the function that needs it. The station's pure-logic
modules (``temporal``, ``core``) must import and test on a laptop with no GPU,
no CUDA and no ultralytics, and ``import station.inference`` must not be what
breaks that.

Missing dependencies fail loudly at load time with the package name and the
weights path in the message. They must never fail quietly into "no detections
this frame", which is byte-identical to a working model looking at an empty
field -- see ``station/core/safety.py``.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence, runtime_checkable

from station.core.config import InferenceConfig
from station.core.types import CLASSES, BBox, Detection, FrameDetections, ModelInfo, utc_now_iso

__all__ = [
    "InferenceUnavailableError",
    "PtsRateLimiter",
    "RunnerStats",
    "Runner",
    "ModelRunner",
    "resolve_device",
    "weights_sha",
]

log = logging.getLogger(__name__)

#: Model class names we accept as aliases for the two wire classes. Trained
#: weights in the wild label these half a dozen ways; mapping them here is
#: better than shipping a model whose boxes arrive on the tablet under a label
#: it has no colour for.
_CLASS_ALIASES: dict[str, str] = {
    "fire": "fire",
    "flame": "fire",
    "flames": "fire",
    "wildfire": "fire",
    "smoke": "smoke",
    "smoky": "smoke",
    "smog": "smoke",
}


class InferenceUnavailableError(RuntimeError):
    """The model cannot run: missing package, missing weights, or bad device.

    Raised at load time, never swallowed per-frame. A pipeline that caught
    this and carried on would emit empty frames indistinguishable from a
    working model that found nothing.
    """


def resolve_device(requested: str) -> str:
    """Resolve ``inference.device`` to a concrete torch device string.

    Args:
        requested: ``"auto"`` (or empty) to probe, otherwise an explicit torch
            device such as ``"cuda:0"``, ``"cpu"`` or ``"mps"``, which is
            returned unchanged so an operator can always pin it.

    Returns:
        ``"cuda"`` when CUDA is usable, otherwise ``"cpu"``. Probing failures
        resolve to ``"cpu"``: a station that runs slowly is a station that
        runs, and the operator sees the reduced ``inference_fps`` in the
        heartbeat rather than a dead pipeline.
    """
    if requested and requested.lower() != "auto":
        return requested
    try:
        import torch  # noqa: PLC0415 -- lazy by design; see module docstring.
    except ImportError:
        return "cpu"
    try:
        if torch.cuda.is_available():
            return "cuda"
    except Exception:  # pragma: no cover - driver/runtime mismatches
        log.warning("CUDA probe failed; falling back to CPU inference", exc_info=True)
    return "cpu"


def weights_sha(path: str | Path, length: int = 7) -> str:
    """Short SHA-256 of a weights file.

    This is the definitive answer to "which model was actually running", and it
    is logged with every frame. ``model_version`` in the config is a human
    label that someone will eventually forget to bump; the hash cannot be
    forgotten.

    Args:
        path: Weights file to hash.
        length: Hex characters to keep. Seven, matching the wire example.

    Returns:
        The truncated lowercase hex digest.

    Raises:
        InferenceUnavailableError: If the file cannot be read.
    """
    resolved = Path(path).expanduser()
    digest = hashlib.sha256()
    try:
        with resolved.open("rb") as fh:
            # Weights are hundreds of MB; chunked so hashing never doubles
            # the station's resident memory at startup.
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise InferenceUnavailableError(f"cannot read weights file {resolved}: {exc}") from exc
    return digest.hexdigest()[:length]


class PtsRateLimiter:
    """Decides, from media timestamps alone, whether a frame gets inferred.

    Gating on ``pts`` rather than on wall clock is what makes a run
    reproducible: the same recording replayed at any speed -- faster than real
    time in a regression test, slower on a loaded laptop -- samples exactly the
    same frames, so an incident log can be re-derived and compared.
    """

    def __init__(self, max_fps: float) -> None:
        """Create a limiter.

        Args:
            max_fps: Maximum inferences per second of media time.

        Raises:
            ValueError: If ``max_fps`` is not positive.
        """
        if max_fps <= 0:
            raise ValueError(f"max_fps must be positive, got {max_fps}")
        self.max_fps = float(max_fps)
        self.min_interval = 1.0 / self.max_fps
        self._last_pts: float | None = None

    def should_run(self, pts: float) -> bool:
        """Return whether the frame at ``pts`` should be inferred.

        Accepting the frame also commits to it, so call this exactly once per
        frame. A backwards ``pts`` (file loop, reconnect, seek) resets the
        limiter and always runs, so the first frame after a discontinuity is
        never skipped -- that is precisely the frame the operator is waiting on.
        """
        if self._last_pts is None or pts < self._last_pts:
            self._last_pts = pts
            return True
        # 1e-6 absorbs the float error in accumulating frame durations; without
        # it a source at exactly max_fps drops every other frame.
        if pts - self._last_pts + 1e-6 < self.min_interval:
            return False
        self._last_pts = pts
        return True

    def reset(self) -> None:
        """Forget the last accepted frame."""
        self._last_pts = None


@dataclass(slots=True)
class RunnerStats:
    """Counters for the ``status`` heartbeat. Says nothing about the scene."""

    frames_seen: int = 0
    frames_inferred: int = 0
    frames_skipped: int = 0
    last_inference_pts: float | None = None
    last_inference_wall_time: str | None = None
    #: Exponential moving average of wall-clock inference cost, in ms.
    mean_inference_ms: float | None = None

    @property
    def inference_fps(self) -> float | None:
        """Model throughput implied by the mean inference cost.

        A ceiling, not a measurement of achieved rate: it ignores decode and
        the ``max_fps`` gate. The pipeline computes the real achieved rate from
        wall clock; this is here so a runner used standalone still has a number.
        """
        if not self.mean_inference_ms:
            return None
        return 1000.0 / self.mean_inference_ms


@runtime_checkable
class Runner(Protocol):
    """The interface the pipeline depends on.

    :class:`ModelRunner` and :class:`station.inference.stub.StubModelRunner`
    both satisfy it, which is what lets the whole station -- ingest, temporal
    filter, WebRTC, incident log, PWA -- be exercised end to end with no model
    and no GPU.
    """

    @property
    def model_info(self) -> ModelInfo: ...

    @property
    def stats(self) -> RunnerStats: ...

    def load(self) -> None: ...

    def reset(self) -> None: ...

    def infer(
        self,
        frame: Any,
        pts: float,
        frame_id: int | None = None,
        *,
        rtp_ts: int | None = None,
        source_id: str | None = None,
    ) -> FrameDetections | None: ...

    def close(self) -> None: ...


class ModelRunner:
    """Runs an ultralytics YOLO model and emits wire-format detections.

    Construction is cheap and does no I/O; :meth:`load` imports ultralytics,
    hashes the weights and warms the model up. :meth:`infer` is then
    synchronous and does not allocate a model. Call it from a worker thread --
    ultralytics releases the GIL inside the torch forward pass, so the decode
    loop keeps running.

    Example:
        >>> from station.core.config import InferenceConfig
        >>> runner = ModelRunner(InferenceConfig())  # doctest: +SKIP
        >>> runner.load()                            # doctest: +SKIP
        >>> frame = runner.infer(bgr_array, pts=12.5, frame_id=300)  # doctest: +SKIP
    """

    def __init__(self, cfg: InferenceConfig) -> None:
        """Create a runner. Does not load the model.

        Args:
            cfg: Inference configuration.

        Raises:
            ValueError: If ``max_fps`` is not positive.
        """
        self.cfg = cfg
        self._limiter = PtsRateLimiter(cfg.max_fps)
        self._stats = RunnerStats()
        self._model: Any = None
        self._device: str = ""
        self._half: bool = False
        self._weights_sha: str | None = None
        self._model_info: ModelInfo | None = None
        self._frame_counter: int = 0
        self._unknown_classes: set[str] = set()

    # ------------------------------------------------------------------ API

    @property
    def device(self) -> str:
        """Resolved torch device. Empty until :meth:`load` has been called."""
        return self._device

    @property
    def loaded(self) -> bool:
        """Whether the weights are in memory and ready."""
        return self._model is not None

    @property
    def stats(self) -> RunnerStats:
        """Live counters. Mutated in place; do not cache the fields."""
        return self._stats

    @property
    def model_info(self) -> ModelInfo:
        """Provenance for the incident log and the tablet's model badge.

        Available before :meth:`load` (with ``weights_sha`` filled in if the
        file is readable) so that a station which fails to start can still say
        what it was trying to start.
        """
        if self._model_info is not None:
            return self._model_info
        sha = self._weights_sha
        if sha is None:
            try:
                sha = weights_sha(self.cfg.weights)
            except InferenceUnavailableError:
                sha = None
        return ModelInfo(
            name=self.cfg.model_name,
            version=self.cfg.model_version,
            classes=CLASSES,
            weights_sha=sha,
            imgsz=self.cfg.imgsz,
            conf_threshold=self.cfg.conf_threshold,
        )

    def load(self) -> None:
        """Import ultralytics, load the weights, and warm the model up.

        Idempotent. Warm-up matters: the first forward pass through a CUDA
        model spends seconds in kernel autotuning, and paying that during
        startup rather than on the first frame of a live incident is the
        difference between a slow launch and a stalled overlay.

        Raises:
            InferenceUnavailableError: If ultralytics is not installed, the
                weights file is missing or unreadable, or the model refuses to
                load on the selected device. The message names the package or
                the path, because the person reading it is standing next to a
                drone.
        """
        if self._model is not None:
            return

        weights = Path(self.cfg.weights).expanduser()
        if not weights.is_file():
            raise InferenceUnavailableError(
                f"weights file not found: {weights.resolve()}\n"
                f"  (config key inference.weights = {self.cfg.weights!r})\n"
                "Put the trained .pt there, point inference.weights at it, or run the "
                "station with the stub runner (station.inference.stub.StubModelRunner) "
                "to exercise the pipeline without a model."
            )

        try:
            from ultralytics import YOLO  # noqa: PLC0415 -- lazy by design.
        except ImportError as exc:
            raise InferenceUnavailableError(
                "ultralytics is not installed, so no model can be run.\n"
                "  pip install ultralytics\n"
                "(this also pulls in torch; on the ground station install the CUDA build "
                "first, see docs/HARDWARE.md). To develop with no model at all, use "
                "station.inference.stub.StubModelRunner."
            ) from exc

        self._device = resolve_device(self.cfg.device)
        # Half precision is a CUDA tensor-core feature. On CPU it is either
        # rejected outright or emulated slowly, so it is silently downgraded --
        # and the downgrade is logged, because a station running 4x slower than
        # the operator expects should be explicable from the log.
        self._half = bool(self.cfg.half) and self._device.startswith("cuda")
        if self.cfg.half and not self._half:
            log.info("half precision requested but device is %s; running fp32", self._device)

        self._weights_sha = weights_sha(weights)
        try:
            model = YOLO(str(weights))
            model.to(self._device)
        except Exception as exc:
            raise InferenceUnavailableError(
                f"failed to load {weights.resolve()} on device {self._device!r}: {exc}\n"
                "Check the file is a YOLO checkpoint and that the CUDA build of torch "
                "matches the installed driver (set inference.device: cpu to rule the GPU out)."
            ) from exc

        self._model = model
        self._model_info = ModelInfo(
            name=self.cfg.model_name,
            version=self.cfg.model_version,
            classes=CLASSES,
            weights_sha=self._weights_sha,
            imgsz=self.cfg.imgsz,
            conf_threshold=self.cfg.conf_threshold,
        )
        self._warmup()
        log.info(
            "loaded %s v%s (sha %s) on %s, fp%d, imgsz=%d, conf>=%.2f",
            self.cfg.model_name, self.cfg.model_version, self._weights_sha,
            self._device, 16 if self._half else 32, self.cfg.imgsz, self.cfg.conf_threshold,
        )

    def infer(
        self,
        frame: Any,
        pts: float,
        frame_id: int | None = None,
        *,
        rtp_ts: int | None = None,
        source_id: str | None = None,
    ) -> FrameDetections | None:
        """Run the model on one frame, subject to the ``max_fps`` gate.

        Args:
            frame: An image ultralytics accepts -- normally an ``H x W x 3``
                uint8 BGR ``numpy`` array straight from the decoder.
            pts: Media-timeline presentation timestamp in seconds. This is the
                overlay's alignment key and it must be the media clock, not
                wall time (see ``docs/CONTRACT.md``).
            frame_id: Source frame number. Defaults to an internal counter of
                frames offered.
            rtp_ts: 90 kHz RTP timestamp, when the stream layer can supply it.
                Enables exact tier-1 overlay alignment on the tablet.
            source_id: Source URI, recorded for the incident log.

        Returns:
            A :class:`FrameDetections` with the raw (unfiltered) detections, or
            ``None`` when the frame was skipped by the rate limiter.

            ``None`` and an empty ``detections`` tuple are different facts and
            the caller must not conflate them: ``None`` means the model did not
            look at this frame, an empty tuple means it looked and proposed
            nothing. Only the latter belongs in the incident log; logging a
            skipped frame as empty would fabricate evidence that the scene was
            examined.

        Raises:
            InferenceUnavailableError: If called before a successful
                :meth:`load`, or if the forward pass fails.
        """
        self._stats.frames_seen += 1
        self._frame_counter += 1
        if not self._limiter.should_run(pts):
            self._stats.frames_skipped += 1
            return None

        if self._model is None:
            raise InferenceUnavailableError("ModelRunner.infer() called before load(); call load() first")

        started = time.perf_counter()
        try:
            results = self._model.predict(
                frame,
                imgsz=self.cfg.imgsz,
                conf=self.cfg.conf_threshold,
                iou=self.cfg.iou_nms,
                device=self._device,
                half=self._half,
                verbose=False,
            )
        except Exception as exc:
            raise InferenceUnavailableError(
                f"inference failed on frame {frame_id if frame_id is not None else self._frame_counter} "
                f"(pts={pts:.3f}) on device {self._device!r}: {exc}"
            ) from exc
        inference_ms = (time.perf_counter() - started) * 1000.0

        detections = self._to_detections(results, frame)

        self._stats.frames_inferred += 1
        self._stats.last_inference_pts = pts
        self._stats.last_inference_wall_time = utc_now_iso()
        prev = self._stats.mean_inference_ms
        # EMA, alpha 0.2: smooth enough to be readable on a 1 Hz heartbeat,
        # fast enough that thermal throttling shows up within a few seconds.
        self._stats.mean_inference_ms = inference_ms if prev is None else prev * 0.8 + inference_ms * 0.2

        return FrameDetections(
            frame_id=frame_id if frame_id is not None else self._frame_counter,
            pts=pts,
            detections=detections,
            model=self._model_info,
            inference_ms=inference_ms,
            rtp_ts=rtp_ts,
            source_id=source_id,
        )

    def reset(self) -> None:
        """Forget stream position after a source discontinuity.

        Only the rate limiter has any state to clear; the model does not.
        Kept for parity with :class:`~station.inference.stub.StubModelRunner`
        so a pipeline can call it without knowing which runner it holds.
        """
        self._limiter.reset()

    def close(self) -> None:
        """Release the model and free GPU memory. Safe to call twice."""
        self._model = None
        try:
            import torch  # noqa: PLC0415 -- lazy by design.
        except ImportError:
            return
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # pragma: no cover
            log.debug("torch.cuda.empty_cache() failed during close", exc_info=True)

    # ------------------------------------------------------------- internals

    def _warmup(self) -> None:
        """One throwaway forward pass on a black frame."""
        try:
            import numpy as np  # noqa: PLC0415 -- lazy by design.
        except ImportError:
            log.debug("numpy unavailable; skipping warm-up pass")
            return
        blank = np.zeros((self.cfg.imgsz, self.cfg.imgsz, 3), dtype=np.uint8)
        try:
            self._model.predict(
                blank, imgsz=self.cfg.imgsz, conf=self.cfg.conf_threshold,
                iou=self.cfg.iou_nms, device=self._device, half=self._half, verbose=False,
            )
        except Exception:
            # Not fatal: a warm-up failure on a model that then works would be
            # a bad reason to refuse to start an incident.
            log.warning("model warm-up pass failed; first live frame will be slow", exc_info=True)

    def _to_detections(self, results: Sequence[Any], frame: Any) -> tuple[Detection, ...]:
        """Convert ultralytics results to wire-format detections.

        Boxes come back in the *original* frame's pixel space (ultralytics
        undoes its own letterboxing), so they are normalised against
        ``orig_shape`` rather than against ``imgsz``. Getting that wrong
        offsets every box by the letterbox padding, which looks plausible and
        is wrong.
        """
        out: list[Detection] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None or len(boxes) == 0:
                continue
            height, width = self._frame_size(result, frame)
            names = getattr(result, "names", None) or {}
            # .tolist() once, rather than per-row tensor indexing: it moves the
            # results off the GPU in one transfer instead of 3N of them.
            xyxy = boxes.xyxy.tolist()
            confs = boxes.conf.tolist()
            classes = boxes.cls.tolist()
            for (x1, y1, x2, y2), conf, cls_idx in zip(xyxy, confs, classes):
                out.append(
                    Detection(
                        cls=self._class_name(int(cls_idx), names),
                        # Clamped, not asserted: a model that returns 1.0000001
                        # must not take the station down mid-incident.
                        conf=min(max(float(conf), 0.0), 1.0),
                        box=BBox.from_xyxy_pixels(x1, y1, x2, y2, width, height),
                    )
                )
        return tuple(out)

    @staticmethod
    def _frame_size(result: Any, frame: Any) -> tuple[int, int]:
        """Return ``(height, width)`` of the frame the boxes are expressed in."""
        shape = getattr(result, "orig_shape", None)
        if shape and len(shape) >= 2:
            return int(shape[0]), int(shape[1])
        shape = getattr(frame, "shape", None)
        if shape and len(shape) >= 2:
            return int(shape[0]), int(shape[1])
        raise InferenceUnavailableError(
            "cannot determine frame size: the result has no orig_shape and the frame has no "
            "shape attribute. Boxes cannot be normalised without it, and an unnormalised box "
            "would be drawn in the wrong place on the tablet."
        )

    def _class_name(self, index: int, names: Any) -> str:
        """Map a model class index to a wire class name.

        An unrecognised label is passed through rather than dropped. Dropping
        it would discard a region the model did propose, and a silently
        discarded fire is the one failure this system is built to avoid; an
        odd label on screen is recoverable, a missing box is not.
        """
        raw = None
        if isinstance(names, dict):
            raw = names.get(index)
        elif isinstance(names, (list, tuple)) and 0 <= index < len(names):
            raw = names[index]
        if raw is None and 0 <= index < len(CLASSES):
            raw = CLASSES[index]
        if raw is None:
            raw = f"class_{index}"
        key = str(raw).strip().lower()
        mapped = _CLASS_ALIASES.get(key)
        if mapped is not None:
            return mapped
        if key not in self._unknown_classes:
            self._unknown_classes.add(key)
            log.warning(
                "model emitted class %r which is not one of %s; passing it through unchanged. "
                "Check the weights match the class order in station.core.types.CLASSES.",
                raw, CLASSES,
            )
        return key
