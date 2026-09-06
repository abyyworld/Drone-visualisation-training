"""A runner that needs no model, no GPU and no weights.

:class:`StubModelRunner` satisfies the same :class:`~station.inference.runner.Runner`
interface as :class:`~station.inference.runner.ModelRunner`, so the whole
station -- ingest, temporal filter, WebRTC transport, incident log, PWA -- can
be brought up and exercised end to end on a laptop with nothing installed.
That matters more than it sounds: the parts of this system most likely to be
wrong are the overlay synchronisation and the staleness rule, and neither of
those needs a real model to be tested. It needs boxes that move.

Two sources of detections, usable together:

* **Scripted** -- exact detections per frame, for deterministic tests of the
  N-of-M filter, of ``persisted`` counts, and of the tablet's rendering.
* **Blob** -- threshold the frame's bright pixels and box the connected
  components. Point it at the synthetic source's moving bright blob and the
  overlay tracks something genuinely present in the video, so a demo shows
  real end-to-end alignment rather than a box drawn from a script that happens
  to agree with the picture.

The stub is not a model and never claims to be one: its :class:`ModelInfo`
reports ``name="stub"`` regardless of what ``inference.model_name`` says, so
no incident log can record stub output under the trained model's identity.

``numpy`` is imported lazily inside the blob path only, so scripted mode works
on an interpreter with nothing but the standard library.
"""

from __future__ import annotations

import logging
import random
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from station.core.config import InferenceConfig
from station.core.types import CLASS_FIRE, CLASSES, BBox, Detection, FrameDetections, ModelInfo, utc_now_iso
from station.inference.runner import PtsRateLimiter, RunnerStats

__all__ = ["StubModelRunner", "drifting_script"]

log = logging.getLogger(__name__)

#: A frame is downsampled so its long side is at most this many pixels before
#: blob labelling. Connected components in pure Python on a 1080p frame would
#: cost ~2 M iterations per frame; at 160 px it is ~14 k and runs in about a
#: millisecond. Fire-sized blobs survive the decimation easily -- anything
#: smaller than one cell at 160 px is well below what an operator could see.
_BLOB_MAX_DIM = 160

ScriptSource = (
    Sequence[Sequence[Detection]]
    | Mapping[int, Sequence[Detection]]
    | Callable[[Any, float, int], Sequence[Detection]]
)


def drifting_script(
    frames: int,
    *,
    cls: str = CLASS_FIRE,
    start: tuple[float, float] = (0.15, 0.5),
    end: tuple[float, float] = (0.85, 0.5),
    size: float = 0.08,
    growth: float = 1.0,
    conf: float = 0.7,
    visible: Callable[[int], bool] | None = None,
) -> list[tuple[Detection, ...]]:
    """Build a script of one box drifting across the frame.

    Useful as the moving target for overlay-alignment work: a box that does not
    move cannot reveal a synchronisation bug, because a stale overlay and a
    correct one look identical.

    Args:
        frames: Number of frames to generate.
        cls: Wire class for the detection.
        start: Normalised ``(cx, cy)`` centre on the first frame.
        end: Normalised ``(cx, cy)`` centre on the last frame.
        size: Half-extent of the box on the first frame, normalised.
        growth: Multiplier applied to ``size`` linearly across the run, so
            ``2.0`` ends at twice the starting size. Exercises the tracker's
            ability to hold identity through a growing plume.
        conf: Confidence reported on every frame.
        visible: Optional ``frame_index -> bool``. Frames where it returns
            false yield no detection, which is how you script the flicker and
            occlusion cases the temporal filter exists to absorb.

    Returns:
        One tuple of detections per frame, ready to pass as ``script``.
    """
    if frames <= 0:
        raise ValueError(f"frames must be positive, got {frames}")
    out: list[tuple[Detection, ...]] = []
    for i in range(frames):
        t = i / max(1, frames - 1)
        if visible is not None and not visible(i):
            out.append(())
            continue
        cx = start[0] + (end[0] - start[0]) * t
        cy = start[1] + (end[1] - start[1]) * t
        half = size * (1.0 + (growth - 1.0) * t)
        out.append((
            Detection(
                cls=cls,
                conf=conf,
                box=BBox(
                    max(0.0, cx - half), max(0.0, cy - half),
                    min(1.0, cx + half), min(1.0, cy + half),
                ),
            ),
        ))
    return out


class StubModelRunner:
    """Drop-in replacement for :class:`~station.inference.runner.ModelRunner`.

    Example:
        Deterministic scripted detections::

            >>> from station.core.config import InferenceConfig
            >>> runner = StubModelRunner(InferenceConfig(max_fps=10.0),
            ...                          script=drifting_script(30))
            >>> runner.load()
            >>> frame = runner.infer(None, pts=0.0, frame_id=1)
            >>> len(frame.detections), frame.model.name
            (1, 'stub')

        Or track the bright region of an actual frame::

            >>> runner = StubModelRunner(detect_blobs=True)  # doctest: +SKIP
            >>> runner.infer(bgr_array, pts=0.0)             # doctest: +SKIP
    """

    def __init__(
        self,
        cfg: InferenceConfig | None = None,
        *,
        script: ScriptSource | None = None,
        loop_script: bool = True,
        detect_blobs: bool = False,
        blob_threshold: float = 200.0,
        blob_min_area: float = 0.0004,
        blob_cls: str = CLASS_FIRE,
        max_blobs: int = 8,
        miss_rate: float = 0.0,
        jitter: float = 0.0,
        latency_ms: float = 0.0,
        seed: int | None = 0,
    ) -> None:
        """Create a stub runner.

        Args:
            cfg: Inference config. Only ``max_fps`` and ``conf_threshold`` are
                honoured; ``weights`` is never read. Defaults are used if
                omitted.
            script: Scripted detections, as one of:
                a sequence indexed by inferred-frame step; a mapping from step
                to detections (steps absent from the mapping yield nothing);
                or a callable ``(frame, pts, frame_id) -> detections``.
                "Step" counts frames that passed the ``max_fps`` gate, not
                frames offered, so a script stays aligned when the source runs
                faster than ``max_fps``.
            loop_script: Whether a sequence script repeats. When false, steps
                past its end yield nothing.
            detect_blobs: Enable brightness-threshold blob detection on the
                frame. Combines with ``script`` -- both contribute.
            blob_threshold: Mean channel value, 0-255, above which a pixel is
                considered part of a blob.
            blob_min_area: Minimum blob area as a fraction of the frame.
                Suppresses single-pixel highlights and sensor noise.
            blob_cls: Wire class reported for blobs.
            max_blobs: Cap on blobs reported per frame, largest first, so a
                pathological frame cannot stall the pipeline.
            miss_rate: Probability of dropping each scripted detection on each
                frame. Simulates the intermittent model output the temporal
                filter exists to smooth.
            jitter: Standard deviation of normalised noise added to scripted
                box corners, simulating per-frame box wobble.
            latency_ms: Wall-clock sleep per inference, to imitate a slow GPU
                and exercise the degraded/stall paths in the heartbeat.
            seed: Seed for ``miss_rate``/``jitter``. Fixed by default because
                a test that fails one run in twenty is worse than no test.

        Raises:
            ValueError: If ``max_fps`` is not positive or a probability is out
                of range.
        """
        self.cfg = cfg or InferenceConfig()
        if not 0.0 <= miss_rate <= 1.0:
            raise ValueError(f"miss_rate must be in 0..1, got {miss_rate}")
        if jitter < 0.0:
            raise ValueError(f"jitter must be >= 0, got {jitter}")
        self._script = script
        self._loop_script = loop_script
        self._detect_blobs = detect_blobs
        self._blob_threshold = float(blob_threshold)
        self._blob_min_area = float(blob_min_area)
        self._blob_cls = blob_cls
        self._max_blobs = max(1, int(max_blobs))
        self._miss_rate = float(miss_rate)
        self._jitter = float(jitter)
        self._latency_s = max(0.0, latency_ms) / 1000.0
        self._rng = random.Random(seed)
        self._limiter = PtsRateLimiter(self.cfg.max_fps)
        self._stats = RunnerStats()
        self._step = 0
        self._frame_counter = 0
        self._loaded = False
        self._warned_no_numpy = False

    # ------------------------------------------------------------------ API

    @property
    def device(self) -> str:
        """Always ``"stub"``. There is no compute device here."""
        return "stub"

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def stats(self) -> RunnerStats:
        return self._stats

    @property
    def model_info(self) -> ModelInfo:
        """Provenance that identifies this as a stub, and cannot be overridden.

        ``cfg.model_name`` is deliberately ignored. The incident log records
        this object with every frame and is the evidence base for the
        false-negative audit; a log that attributed stub boxes to
        ``yolo11s-fire v0.3.1`` would poison that audit permanently.
        """
        modes = []
        if self._script is not None:
            modes.append("script")
        if self._detect_blobs:
            modes.append("blob")
        return ModelInfo(
            name="stub",
            version="stub-" + ("+".join(modes) if modes else "idle"),
            classes=CLASSES,
            weights_sha=None,
            imgsz=self.cfg.imgsz,
            conf_threshold=self.cfg.conf_threshold,
        )

    def load(self) -> None:
        """No-op, present so the stub is interchangeable with the real runner.

        It does check that ``inference.weights`` is absent-or-present without
        caring, and logs loudly, so that a station accidentally left on the
        stub in the field is obvious in the first lines of the log.
        """
        self._loaded = True
        weights = Path(self.cfg.weights).expanduser()
        log.warning(
            "StubModelRunner active: NO MODEL IS RUNNING. Detections are %s. "
            "(configured weights %s were not loaded and were %s)",
            self.model_info.version, weights,
            "present" if weights.is_file() else "absent",
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
        """Produce scripted and/or blob detections for one frame.

        Args:
            frame: The decoded frame. Ignored in scripted mode -- ``None`` is
                fine -- and required (as an array with a ``shape``) for blobs.
            pts: Media-timeline timestamp in seconds.
            frame_id: Source frame number; defaults to an internal counter.
            rtp_ts: 90 kHz RTP timestamp, passed straight through.
            source_id: Source URI, passed straight through.

        Returns:
            A :class:`FrameDetections`, or ``None`` if the frame was skipped by
            the ``max_fps`` gate. As with the real runner, ``None`` ("did not
            look") and empty detections ("looked, proposed nothing") are
            different facts and must not be conflated by the caller.
        """
        self._stats.frames_seen += 1
        self._frame_counter += 1
        if not self._limiter.should_run(pts):
            self._stats.frames_skipped += 1
            return None

        resolved_id = frame_id if frame_id is not None else self._frame_counter
        started = time.perf_counter()
        if self._latency_s:
            time.sleep(self._latency_s)

        dets: list[Detection] = list(self._scripted(frame, pts, resolved_id))
        if self._detect_blobs:
            dets.extend(self._blobs(frame))

        # The real runner asks the model to threshold; the stub does it here so
        # that a low-confidence scripted detection behaves the same either way.
        dets = [d for d in dets if d.conf >= self.cfg.conf_threshold]

        inference_ms = (time.perf_counter() - started) * 1000.0
        self._step += 1
        self._stats.frames_inferred += 1
        self._stats.last_inference_pts = pts
        self._stats.last_inference_wall_time = utc_now_iso()
        prev = self._stats.mean_inference_ms
        self._stats.mean_inference_ms = inference_ms if prev is None else prev * 0.8 + inference_ms * 0.2

        return FrameDetections(
            frame_id=resolved_id,
            pts=pts,
            detections=tuple(dets),
            model=self.model_info,
            inference_ms=inference_ms,
            rtp_ts=rtp_ts,
            source_id=source_id,
        )

    def close(self) -> None:
        """Release nothing. Present for interface parity."""
        self._loaded = False

    def reset(self) -> None:
        """Rewind the script and the rate limiter, e.g. when a file loops."""
        self._step = 0
        self._limiter.reset()

    # ------------------------------------------------------------- internals

    def _scripted(self, frame: Any, pts: float, frame_id: int) -> tuple[Detection, ...]:
        """Detections from the script for the current step, with noise applied."""
        script = self._script
        if script is None:
            return ()
        if callable(script):
            raw: Sequence[Detection] = script(frame, pts, frame_id)
        elif isinstance(script, Mapping):
            raw = script.get(self._step, ())
        else:
            if not script:
                return ()
            if self._step >= len(script) and not self._loop_script:
                return ()
            raw = script[self._step % len(script)]

        out: list[Detection] = []
        for det in raw:
            if self._miss_rate and self._rng.random() < self._miss_rate:
                continue
            out.append(self._jittered(det) if self._jitter else det)
        return tuple(out)

    def _jittered(self, det: Detection) -> Detection:
        """Wobble a box's corners, keeping it a valid normalised BBox."""
        j = self._jitter
        x1, y1, x2, y2 = det.box.as_tuple()
        x1 = min(max(x1 + self._rng.gauss(0.0, j), 0.0), 1.0)
        y1 = min(max(y1 + self._rng.gauss(0.0, j), 0.0), 1.0)
        x2 = min(max(x2 + self._rng.gauss(0.0, j), 0.0), 1.0)
        y2 = min(max(y2 + self._rng.gauss(0.0, j), 0.0), 1.0)
        # Corners can cross under noise on a small box; BBox rejects inverted
        # corners, so re-order rather than raise mid-stream.
        lo_x, hi_x = sorted((x1, x2))
        lo_y, hi_y = sorted((y1, y2))
        return Detection(
            cls=det.cls,
            conf=min(max(det.conf + self._rng.gauss(0.0, j), 0.0), 1.0),
            box=BBox(lo_x, lo_y, hi_x, hi_y),
        )

    def _blobs(self, frame: Any) -> tuple[Detection, ...]:
        """Box the bright connected regions of ``frame``.

        Returns:
            Up to ``max_blobs`` detections, largest first. Empty when the frame
            is unusable (no numpy, wrong shape) -- and that emptiness is
            logged, because a silently blind stub would be exactly the failure
            mode this project refuses to tolerate in the real runner.
        """
        try:
            import numpy as np  # noqa: PLC0415 -- lazy by design.
        except ImportError:
            if not self._warned_no_numpy:
                self._warned_no_numpy = True
                log.error("blob mode needs numpy (pip install numpy); no blob detections will be produced")
            return ()

        if frame is None:
            return ()
        arr = np.asarray(frame)
        if arr.ndim not in (2, 3):
            log.error("blob mode expected an HxW or HxWx3 frame, got shape %r", getattr(arr, "shape", None))
            return ()
        height, width = int(arr.shape[0]), int(arr.shape[1])
        if height == 0 or width == 0:
            return ()

        # Decimate *before* the channel mean and the float conversion. Striding
        # first selects exactly the same pixels and costs ~1 ms on a 1080p frame
        # instead of ~35 ms -- which matters, because the stub has to keep up
        # with a live source for the end-to-end test to mean anything.
        step = max(1, -(-max(height, width) // _BLOB_MAX_DIM))  # ceil division
        sub = arr[::step, ::step]
        if sub.ndim == 3:
            # Mean across channels, so this works on BGR and RGB alike -- the
            # decoder's channel order is not worth a bug in a test fixture.
            small = sub.astype(np.float32).mean(axis=2)
        else:
            small = sub.astype(np.float32)
        # Float frames in 0..1 are common from some decoders; put everything on
        # the 0..255 scale the threshold is expressed in.
        if small.max() <= 1.0:
            small = small * 255.0

        mask = small >= self._blob_threshold
        if not mask.any():
            return ()

        sh, sw = mask.shape
        components = _label_components(mask.tolist(), sh, sw)

        frame_area = float(height * width)
        cell_area = float(step * step)
        found: list[tuple[float, Detection]] = []
        for pixels, (r0, c0, r1, c1) in components:
            area_frac = pixels * cell_area / frame_area
            if area_frac < self._blob_min_area:
                continue
            # +1 on the far edge: the component's last cell covers `step`
            # source pixels, and a box that stopped at its first pixel would
            # systematically undercut the blob by up to a cell on two sides.
            x1, y1 = c0 * step, r0 * step
            x2, y2 = min(width, (c1 + 1) * step), min(height, (r1 + 1) * step)
            region = small[r0:r1 + 1, c0:c1 + 1]
            bright = float(region[region >= self._blob_threshold].mean())
            headroom = max(1.0, 255.0 - self._blob_threshold)
            # Brighter blob -> higher confidence, capped at 0.95: a stub must
            # never emit a 1.0, which would read as certainty in the log.
            conf = min(0.95, 0.35 + 0.60 * (bright - self._blob_threshold) / headroom)
            found.append((
                area_frac,
                Detection(
                    cls=self._blob_cls,
                    conf=max(0.0, conf),
                    box=BBox.from_xyxy_pixels(x1, y1, x2, y2, width, height),
                ),
            ))

        found.sort(key=lambda item: -item[0])
        return tuple(det for _area, det in found[: self._max_blobs])


def _label_components(mask: list[list[bool]], height: int, width: int) -> list[tuple[int, tuple[int, int, int, int]]]:
    """8-connected component labelling on a small boolean mask.

    Iterative flood fill over a plain nested list: a recursive version blows
    the Python stack on a blob a few thousand pixels across, and pulling in
    scipy for this would defeat the point of a dependency-free stub.

    Args:
        mask: ``height x width`` booleans, true where the pixel is bright.
        height: Rows in ``mask``.
        width: Columns in ``mask``.

    Returns:
        One ``(pixel_count, (row_min, col_min, row_max, col_max))`` per
        component, in discovery order.
    """
    seen = [[False] * width for _ in range(height)]
    out: list[tuple[int, tuple[int, int, int, int]]] = []
    for row in range(height):
        mask_row = mask[row]
        for col in range(width):
            if not mask_row[col] or seen[row][col]:
                continue
            seen[row][col] = True
            stack = [(row, col)]
            count = 0
            r0 = r1 = row
            c0 = c1 = col
            while stack:
                r, c = stack.pop()
                count += 1
                if r < r0:
                    r0 = r
                elif r > r1:
                    r1 = r
                if c < c0:
                    c0 = c
                elif c > c1:
                    c1 = c
                for dr in (-1, 0, 1):
                    rr = r + dr
                    if rr < 0 or rr >= height:
                        continue
                    seen_rr = seen[rr]
                    mask_rr = mask[rr]
                    for dc in (-1, 0, 1):
                        cc = c + dc
                        if cc < 0 or cc >= width or seen_rr[cc] or not mask_rr[cc]:
                            continue
                        seen_rr[cc] = True
                        stack.append((rr, cc))
            out.append((count, (r0, c0, r1, c1)))
    return out
