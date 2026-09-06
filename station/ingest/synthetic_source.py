"""A deterministic, dependency-free video source for tests and demos.

This is the source that lets the rest of the system be tested at all. It needs
nothing but numpy -- no codec, no capture card, no video file, no drone -- and
it produces a *known* answer: for every frame it renders, it can state exactly
where the bright region is, in the same normalised coordinates the wire
protocol uses. That turns otherwise untestable questions into assertions:

* Does the N-of-M temporal filter confirm a track after exactly ``n`` frames,
  and drop it ``max_age`` frames after the region leaves?
* Does the overlay land on the frame it was computed from, or one frame late?
* Does rate decimation preserve pts monotonicity?

The scene is a function of the frame index alone, so frame 137 renders
identically whether it was reached by iterating or asked for directly, in this
process or the next one. Nothing here is random unless a seed is given, and
even then the seed is mixed with the frame index rather than carried in a
generator's hidden state.

The blob is a plausible target, not a fire simulation: a warm, bright,
roughly circular region on a sky/ground gradient. It exercises geometry and
timing, and asserting a real model's behaviour on it would prove nothing.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Iterator, Sequence
from urllib.parse import parse_qsl, urlsplit

import numpy as np

from station.core.config import SourceConfig
from station.core.types import CLASS_FIRE, BBox

from station.ingest.base import (
    Frame,
    FrameSource,
    PtsOrigin,
    SourceConfigError,
)

__all__ = [
    "SyntheticScene",
    "SyntheticSource",
    "synthetic_config",
    "TRUTH_CLASS",
    "DEFAULT_FRAME_COUNT",
]

#: The class a confirmed detection of the blob should carry, for tests that
#: compare pipeline output against :meth:`SyntheticScene.box`.
TRUTH_CLASS = CLASS_FIRE

#: A source that never ends is a hazard in a test suite. Ten seconds at 30 fps
#: is long enough for any temporal-filter scenario and short enough that a
#: forgotten ``for frame in src`` terminates. ``SourceConfig.loop`` makes it
#: endless when that is what a demo wants.
DEFAULT_FRAME_COUNT = 300

#: Warm orange in BGR -- the channel order every Frame in this system uses.
_BLOB_BGR = (30.0, 140.0, 250.0)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "on"):
        return True
    if text in ("0", "false", "no", "off"):
        return False
    raise SourceConfigError(f"expected a boolean, got {value!r}")


def _parse_ranges(spec: Any) -> tuple[tuple[int, int], ...]:
    """Parse ``"10-40,60-80"`` into half-open index ranges."""
    if spec is None or spec == "":
        return ()
    if isinstance(spec, (list, tuple)):
        return tuple((int(a), int(b)) for a, b in spec)
    out: list[tuple[int, int]] = []
    for chunk in str(spec).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        lo, _, hi = chunk.partition("-")
        if not hi:
            raise SourceConfigError(f"range {chunk!r} must look like 'start-end'")
        try:
            start, end = int(lo), int(hi)
        except ValueError as exc:
            raise SourceConfigError(f"range {chunk!r} must be two integers") from exc
        if end <= start:
            raise SourceConfigError(f"range {chunk!r} must have end > start")
        out.append((start, end))
    return tuple(out)


@dataclass(frozen=True, slots=True)
class SyntheticScene:
    """The generator behind :class:`SyntheticSource`, usable on its own.

    Every method is a pure function of the frame index, which is what makes the
    ground truth trustworthy: :meth:`box` does not read back what :meth:`render`
    drew, it computes the same geometry from the same parameters.

    Attributes:
        width: Frame width in pixels.
        height: Frame height in pixels.
        fps: Nominal frame rate; also fixes the media timeline (``pts =
            index / fps``).
        seed: Mixed with the frame index to drive background noise and optional
            jitter. The same seed always gives the same pixels.
        radius: Blob radius as a fraction of the frame height.
        amp_x: Horizontal travel, as a fraction of frame width, either side of
            centre.
        amp_y: Vertical travel, as a fraction of frame height.
        period_x: Seconds for one horizontal sweep.
        period_y: Seconds for one vertical sweep.
        pulse_period: Seconds for one cycle of the radius pulse.
        noise: Standard deviation of the background luma noise, in 0..255
            levels. Kept small so the blob stays the brightest thing present.
        jitter: Random displacement of the blob centre, as a fraction of frame
            height. Included in :meth:`box`, so ground truth stays exact.
        present: Half-open index ranges in which the blob exists. Empty means
            always present. Use it to test appearance and disappearance.
    """

    width: int = 640
    height: int = 360
    fps: float = 30.0
    seed: int = 0
    radius: float = 0.09
    amp_x: float = 0.30
    amp_y: float = 0.18
    period_x: float = 6.0
    period_y: float = 4.0
    pulse_period: float = 2.5
    noise: float = 3.0
    jitter: float = 0.0
    present: tuple[tuple[int, int], ...] = ()

    def __post_init__(self) -> None:
        if self.width < 16 or self.height < 16:
            raise SourceConfigError(f"synthetic frame must be at least 16x16, got {self.width}x{self.height}")
        if self.fps <= 0:
            raise SourceConfigError(f"synthetic fps must be positive, got {self.fps}")
        if not 0.0 < self.radius < 0.5:
            raise SourceConfigError(f"synthetic radius must be in (0, 0.5), got {self.radius}")

    # -- ground truth -------------------------------------------------------

    def is_present(self, index: int) -> bool:
        """Whether the blob exists on this frame."""
        if not self.present:
            return True
        return any(start <= index < end for start, end in self.present)

    def centre_px(self, index: int) -> tuple[float, float, float]:
        """``(cx, cy, radius)`` in pixels for the given frame index.

        The single definition of where the blob is. :meth:`render` and
        :meth:`box` both call it, so the drawn pixels and the published box can
        never drift apart.
        """
        t = index / self.fps
        cx = self.width * (0.5 + self.amp_x * math.sin(2.0 * math.pi * t / self.period_x))
        cy = self.height * (0.5 + self.amp_y * math.sin(2.0 * math.pi * t / self.period_y + math.pi / 3.0))
        pulse = 1.0 + 0.15 * math.sin(2.0 * math.pi * t / self.pulse_period)
        r = self.radius * self.height * pulse
        if self.jitter:
            rng = self._rng(index, stream=1)
            offset = rng.normal(0.0, self.jitter * self.height, size=2)
            cx += float(offset[0])
            cy += float(offset[1])
        # Keep the whole blob inside the frame: a target half off the edge has
        # a ground-truth box that is clipped, and a clipped box is a poor
        # reference for an IoU assertion.
        cx = min(max(cx, r), self.width - r)
        cy = min(max(cy, r), self.height - r)
        return cx, cy, r

    def box(self, index: int) -> BBox | None:
        """Ground-truth box for a frame, or ``None`` when the blob is absent.

        Returns:
            The exact bounding box of the rendered blob, normalised 0..1 with
            the origin at top-left -- the same convention as the wire protocol,
            so it can be compared with a :class:`~station.core.types.Detection`
            box by IoU with no conversion.
        """
        if not self.is_present(index):
            return None
        cx, cy, r = self.centre_px(index)
        # The blob has compact support: intensity is exactly zero outside
        # radius r, so this box is tight rather than a convention.
        return BBox.from_xyxy_pixels(cx - r, cy - r, cx + r, cy + r, self.width, self.height)

    def boxes(self, count: int, start: int = 0) -> list[BBox | None]:
        """Ground-truth sequence for ``count`` frames from ``start``."""
        return [self.box(i) for i in range(start, start + count)]

    def pts(self, index: int) -> float:
        """Media timestamp of a frame index, in seconds."""
        return index / self.fps

    # -- rendering ----------------------------------------------------------

    def render(self, index: int) -> np.ndarray:
        """Render one frame as an ``HxWx3`` uint8 BGR array."""
        image = self._background(index)
        if self.is_present(index):
            cx, cy, r = self.centre_px(index)
            yy, xx = self._grid()
            # Squared normalised distance; clipped so the disc has hard support
            # and the ground-truth box is exactly its extent.
            falloff = np.clip(1.0 - (((xx - cx) ** 2 + (yy - cy) ** 2) / (r * r)), 0.0, 1.0) ** 0.6
            alpha = falloff[:, :, None]
            colour = np.array(_BLOB_BGR, dtype=np.float32)
            image = image * (1.0 - alpha) + colour * alpha
        return np.clip(image, 0.0, 255.0).astype(np.uint8)

    def _background(self, index: int) -> np.ndarray:
        """Sky-to-ground gradient plus a little per-frame noise."""
        h, w = self.height, self.width
        ramp = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
        sky = np.array([200.0, 170.0, 140.0], dtype=np.float32)
        ground = np.array([40.0, 70.0, 45.0], dtype=np.float32)
        image = sky[None, None, :] * (1.0 - ramp[:, :, None]) + ground[None, None, :] * ramp[:, :, None]
        image = np.repeat(image, w, axis=1)
        if self.noise > 0.0:
            rng = self._rng(index, stream=0)
            # Luma-only noise: one channel of noise added to all three keeps
            # the background grey-ish rather than colourful confetti, which is
            # closer to sensor noise and cheaper to generate.
            image = image + rng.normal(0.0, self.noise, size=(h, w, 1)).astype(np.float32)
        return image

    def _grid(self) -> tuple[np.ndarray, np.ndarray]:
        """Pixel coordinate grids, cached per (width, height)."""
        key = (self.width, self.height)
        cached = _GRID_CACHE.get(key)
        if cached is None:
            yy, xx = np.mgrid[0 : self.height, 0 : self.width]
            cached = (yy.astype(np.float32), xx.astype(np.float32))
            _GRID_CACHE[key] = cached
        return cached

    def _rng(self, index: int, stream: int) -> np.random.Generator:
        """A generator seeded by (seed, index, stream), not by iteration order.

        Frame 137 must look the same however it was reached; a single
        long-lived generator would make the pixels depend on how many frames
        had been drawn before, and a test that skipped frames would see a
        different image.
        """
        return np.random.default_rng([int(self.seed), int(index), int(stream)])


#: Coordinate grids are the only mutable state in this module; they depend on
#: frame size alone, so sharing them across scenes is safe and saves rebuilding
#: two float arrays per frame.
_GRID_CACHE: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}


#: Every knob that can be set from the URI query string.
_SCENE_KEYS: dict[str, Any] = {
    "width": int,
    "height": int,
    "fps": float,
    "seed": int,
    "radius": float,
    "amp_x": float,
    "amp_y": float,
    "period_x": float,
    "period_y": float,
    "pulse_period": float,
    "noise": float,
    "jitter": float,
    "present": _parse_ranges,
}
_SOURCE_KEYS: dict[str, Any] = {
    "frames": lambda v: None if str(v).lower() in ("", "none", "0") else int(v),
    "realtime": _as_bool,
}


class SyntheticSource(FrameSource):
    """Generates frames from a :class:`SyntheticScene`. Needs only numpy.

    Configure it either through ``SourceConfig.uri`` as a query string --
    ``synthetic:?width=1280&height=720&fps=30&seed=7&present=30-90`` -- or by
    passing keyword overrides to the constructor, which win over the URI.

    Example:
        >>> cfg = synthetic_config(fps=10.0, frames=20, seed=1)
        >>> with SyntheticSource(cfg) as src:
        ...     frames = list(src)
        >>> len(frames), round(frames[3].pts, 3)
        (20, 0.3)

    Args:
        cfg: The source configuration. ``target_fps`` decimates the generated
            stream exactly as it would a real one; ``loop`` makes the source
            endless.
        **overrides: Scene or source parameters, overriding the URI.
    """

    def __init__(self, cfg: SourceConfig, **overrides: Any) -> None:
        params = _parse_uri(cfg.uri)
        params.update(overrides)
        scene_kwargs = {k: _SCENE_KEYS[k](v) for k, v in params.items() if k in _SCENE_KEYS}
        self.scene = SyntheticScene(**scene_kwargs)
        frames = params.get("frames", DEFAULT_FRAME_COUNT)
        if not isinstance(frames, (int, type(None))):
            frames = _SOURCE_KEYS["frames"](frames)
        # loop wins over any frame count: the scene is a continuous function of
        # time, so looping simply means never stopping -- there is no seam to
        # jump back to and therefore no pts discontinuity, unlike a file loop.
        self._frame_limit: int | None = None if cfg.loop else frames
        self._realtime = _as_bool(params.get("realtime", False))
        super().__init__(cfg, source_type="synthetic", uri=cfg.uri or _default_uri(self.scene))
        self._index = 0
        self._start_monotonic = 0.0

    @property
    def frame_limit(self) -> int | None:
        """How many frames will be generated, or ``None`` when endless."""
        return self._frame_limit

    def _open_impl(self) -> None:
        self._index = 0
        self._start_monotonic = time.monotonic()
        self._set_info(
            width=self.scene.width,
            height=self.scene.height,
            fps=self.scene.fps,
            backend="numpy",
            is_live=self._realtime,
        )
        # The scene defines its own timeline exactly, so these timestamps are
        # real rather than synthesised -- pts is index/fps by construction.
        self._init_clock(PtsOrigin.GENERATED, self.scene.fps)

    def _read_raw(self) -> tuple[np.ndarray, float | None] | None:
        if self._frame_limit is not None and self._index >= self._frame_limit:
            return None
        index = self._index
        if self._realtime:
            # Demo mode only. Tests leave this off so a hundred frames take
            # milliseconds instead of seconds.
            due = self._start_monotonic + self.scene.pts(index)
            delay = due - time.monotonic()
            if delay > 0:
                time.sleep(delay)
        image = self.scene.render(index)
        self._index += 1
        return image, self.scene.pts(index)

    def _close_impl(self) -> None:
        return None

    # -- ground truth, in the same terms the pipeline reports ---------------

    def ground_truth(self, index: int) -> BBox | None:
        """Ground-truth box for a *generated* frame index.

        Note the distinction from ``Frame.frame_id``: with ``target_fps`` set,
        generated indices and emitted frame ids differ. Use
        :meth:`ground_truth_for` when you have a frame in hand.
        """
        return self.scene.box(index)

    def ground_truth_for(self, frame: Frame) -> BBox | None:
        """Ground-truth box for a frame this source produced.

        Recovers the generated index from the frame's pts, so it stays correct
        under rate decimation.
        """
        return self.scene.box(self.index_of(frame))

    def index_of(self, frame: Frame) -> int:
        """Generated index a frame came from, recovered from its pts."""
        return int(round(frame.pts * self.scene.fps))

    def ground_truth_sequence(self, count: int, start: int = 0) -> list[BBox | None]:
        """The first ``count`` ground-truth boxes, ``None`` where absent."""
        return self.scene.boxes(count, start)


def _default_uri(scene: SyntheticScene) -> str:
    """A URI that reproduces this scene, so ``source_id`` identifies it."""
    return (
        f"synthetic:?width={scene.width}&height={scene.height}"
        f"&fps={scene.fps:g}&seed={scene.seed}"
    )


def _parse_uri(uri: str) -> dict[str, Any]:
    """Parse ``synthetic:?k=v&...`` into a parameter mapping.

    Unknown keys are rejected rather than ignored, for the same reason
    ``station.core.config`` rejects unknown config keys: a typo that silently
    left a default in place would be a test asserting against a scene it did
    not configure.
    """
    if not uri:
        return {}
    query = urlsplit(uri).query or (uri.split("?", 1)[1] if "?" in uri else "")
    if not query:
        return {}
    params: dict[str, Any] = {}
    valid = set(_SCENE_KEYS) | set(_SOURCE_KEYS)
    for key, value in parse_qsl(query, keep_blank_values=True):
        if key not in valid:
            raise SourceConfigError(
                f"unknown synthetic source parameter {key!r} in {uri!r}; "
                f"valid parameters are {sorted(valid)}"
            )
        converter = _SCENE_KEYS.get(key) or _SOURCE_KEYS[key]
        try:
            params[key] = converter(value)
        except SourceConfigError:
            raise
        except Exception as exc:
            raise SourceConfigError(f"bad value for synthetic parameter {key}={value!r}: {exc}") from exc
    return params


def synthetic_config(
    *,
    target_fps: float | None = None,
    loop: bool = False,
    **params: Any,
) -> SourceConfig:
    """Build a ``SourceConfig`` for a synthetic source. Convenience for tests.

    Args:
        target_fps: Rate cap applied by the ingest layer, as for any source.
        loop: Generate frames forever.
        **params: Scene parameters (``width``, ``height``, ``fps``, ``seed``,
            ``present``, ...) and source parameters (``frames``, ``realtime``).

    Returns:
        A ``SourceConfig`` whose URI encodes the parameters, so the resulting
        source is reproducible from config alone.
    """
    unknown = set(params) - set(_SCENE_KEYS) - set(_SOURCE_KEYS)
    if unknown:
        raise SourceConfigError(
            f"unknown synthetic parameter(s) {sorted(unknown)}; "
            f"valid parameters are {sorted(set(_SCENE_KEYS) | set(_SOURCE_KEYS))}"
        )
    parts = []
    for key, value in params.items():
        if isinstance(value, bool):
            value = "true" if value else "false"
        elif isinstance(value, (list, tuple)):
            value = ",".join(f"{a}-{b}" for a, b in value)
        parts.append(f"{key}={value}")
    uri = "synthetic:?" + "&".join(parts) if parts else "synthetic:"
    return SourceConfig(type="synthetic", uri=uri, target_fps=target_fps, loop=loop)
