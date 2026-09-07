"""A crude colour-threshold detector, for demonstrating the pipeline only.

This is NOT the model. It is classical computer vision -- a few colour
thresholds and a blob merge -- standing in for a trained network so the full
pipeline can be shown working before any training has happened.

It is deliberately kept, rather than replaced with a replay of the ground
truth, because its failure modes are real and they are the point: it finds a
bright flame front easily and loses thin smoke against light ground, which is
exactly the false-negative behaviour that the safety invariants are designed
around. A demo driven from ground truth would hide that and would show a
system that never misses anything, which no detector does.

Do not deploy this. Its confidences are heuristic and carry no calibration.
"""

from __future__ import annotations

import numpy as np

from station.core.types import BBox, Detection

BLOCK = 16  # mask is analysed on a coarse grid; blob shapes here are approximate


def _components(mask: np.ndarray, min_blocks: int) -> list[tuple[int, int, int, int, float]]:
    """Connected components over a coarse boolean grid, 8-connected.

    Iterative flood fill: a recursive one blows the stack on a large plume,
    which is precisely the case that matters here.
    """
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[tuple[int, int, int, int, float]] = []
    for sy in range(h):
        for sx in range(w):
            if not mask[sy, sx] or seen[sy, sx]:
                continue
            stack = [(sy, sx)]
            seen[sy, sx] = True
            cells = []
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            stack.append((ny, nx))
            if len(cells) < min_blocks:
                continue
            ys = [c[0] for c in cells]
            xs = [c[1] for c in cells]
            fill = len(cells) / max(1, (max(ys) - min(ys) + 1) * (max(xs) - min(xs) + 1))
            out.append((min(xs), min(ys), max(xs), max(ys), fill))
    return out


def detect(bgr: np.ndarray) -> list[Detection]:
    """Return fire and smoke detections for one frame.

    Args:
        bgr: ``HxWx3`` uint8 array in **BGR** order, as ``station.ingest``
            produces and ``ModelRunner`` consumes.

    Channel order is worth stating loudly. Feeding this RGB does not fail --
    it silently stops detecting fire while smoke keeps working, because smoke
    is grey and survives a channel swap. The result is a system that looks
    healthy and has lost the class it exists to find. This detector was
    written against RGB by mistake and did exactly that.
    """
    img = bgr.astype(np.float32)
    b, g, r = img[..., 0], img[..., 1], img[..., 2]
    mx = img.max(axis=2)
    mn = img.min(axis=2)

    # Flame: bright, strongly warm-shifted. Tight thresholds keep sunlit ground
    # out; a real model has to learn this distinction rather than assert it.
    fire_px = (r > 170) & ((r - g) > 38) & ((g - b) > 12)

    # Smoke: bright but desaturated. This threshold is where the false
    # negatives live -- thin smoke sits below it, and raising it to catch the
    # thin plume starts swallowing pale ground.
    smoke_px = (mx > 128) & ((mx - mn) < 34) & ~fire_px

    h, w = fire_px.shape
    gh, gw = h // BLOCK, w // BLOCK

    def grid(px: np.ndarray, frac: float) -> np.ndarray:
        cut = px[: gh * BLOCK, : gw * BLOCK].reshape(gh, BLOCK, gw, BLOCK)
        return cut.mean(axis=(1, 3)) > frac

    dets: list[Detection] = []
    for cls, mask, min_blocks in (("fire", grid(fire_px, 0.30), 1), ("smoke", grid(smoke_px, 0.42), 4)):
        for x0, y0, x1, y1, fill in _components(mask, min_blocks):
            box = BBox(
                x1=(x0 * BLOCK) / w,
                y1=(y0 * BLOCK) / h,
                x2=min(1.0, ((x1 + 1) * BLOCK) / w),
                y2=min(1.0, ((y1 + 1) * BLOCK) / h),
            )
            # Heuristic only: denser, more solid blobs score higher. Calibrated
            # against nothing, which is why it must never be reported as skill.
            conf = round(min(0.94, 0.34 + 0.55 * float(fill)), 3)
            dets.append(Detection(cls=cls, conf=conf, box=box))
    return dets
