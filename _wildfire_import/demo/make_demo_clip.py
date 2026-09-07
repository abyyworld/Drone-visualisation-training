"""Generate a synthetic aerial wildfire clip, with ground truth.

Why this exists: the file-source spike (build step 1) needs real video, and
this environment cannot reach any footage archive. So the clip is generated --
procedurally, but with the properties that make aerial fire footage hard:
a drifting camera, a flame front that flickers frame to frame, and a smoke
plume with no crisp boundary that grows and thins as it travels downwind.

It ships with a per-frame ground-truth sidecar. That is what separates this
from decoration: the demo can state exactly what the model saw versus what was
there, and the same clip doubles as a regression fixture for the pipeline.

The smoke is deliberately awkward. Smoke is a poor fit for bounding boxes --
it is amorphous, and its "extent" depends entirely on the alpha threshold you
pick. The ground truth here uses a fixed threshold and says so, because a box
around thin smoke is an approximation and pretending otherwise would make the
evaluation numbers look better than they are.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

WORLD_W, WORLD_H = 1760, 1000


def value_noise(shape: tuple[int, int], cells: int, rng: np.random.Generator) -> np.ndarray:
    """Smooth value noise by bilinear upsampling of a coarse random grid."""
    h, w = shape
    gh, gw = max(2, cells), max(2, int(cells * w / h))
    grid = rng.random((gh, gw))
    ys = np.linspace(0, gh - 1, h)
    xs = np.linspace(0, gw - 1, w)
    y0 = np.floor(ys).astype(int).clip(0, gh - 2)
    x0 = np.floor(xs).astype(int).clip(0, gw - 2)
    fy = (ys - y0)[:, None]
    fx = (xs - x0)[None, :]
    # smoothstep keeps the interpolation from looking like a grid of diamonds
    fy = fy * fy * (3 - 2 * fy)
    fx = fx * fx * (3 - 2 * fx)
    g00 = grid[np.ix_(y0, x0)]
    g01 = grid[np.ix_(y0, x0 + 1)]
    g10 = grid[np.ix_(y0 + 1, x0)]
    g11 = grid[np.ix_(y0 + 1, x0 + 1)]
    return (g00 * (1 - fx) * (1 - fy) + g01 * fx * (1 - fy) + g10 * (1 - fx) * fy + g11 * fx * fy)


def fbm(shape: tuple[int, int], rng: np.random.Generator, octaves: int = 5,
        persistence: float = 0.5) -> np.ndarray:
    """Fractal noise.

    ``persistence`` is the knob that decides whether ground looks airbrushed or
    photographed. At 0.5 the ninth octave contributes 0.4% and the fine detail
    is invisible; forest canopy from altitude needs the high frequencies to
    carry real weight, so terrain uses ~0.68.
    """
    out = np.zeros(shape)
    amp, cells, norm = 1.0, 3, 0.0
    for _ in range(octaves):
        out += amp * value_noise(shape, cells, rng)
        norm += amp
        amp *= persistence
        cells *= 2
    return out / norm


def build_terrain(rng: np.random.Generator) -> np.ndarray:
    """Forested terrain: greens and browns broken up by clearings.

    The octave count matters more than it looks. Too few and the ground reads as
    smooth blobs -- the giveaway that says 'render' rather than 'aerial photo'.
    Canopy at altitude is high-frequency, so the fine octaves carry the realism.
    """
    h = fbm((WORLD_H, WORLD_W), rng, octaves=6)
    detail = fbm((WORLD_H, WORLD_W), rng, octaves=9, persistence=0.68)
    speckle = fbm((WORLD_H, WORLD_W), rng, octaves=10, persistence=0.72)
    canopy = 0.65 * h + 0.35 * detail

    img = np.zeros((WORLD_H, WORLD_W, 3))
    # dark conifer -> lighter scrub -> dry grass, by elevation-ish value
    grain = (speckle - 0.5) * 74.0          # individual tree crowns
    img[..., 0] = 30 + 78 * np.clip((canopy - 0.45) * 2.2, 0, 1) + 26 * detail + grain
    img[..., 1] = 46 + 92 * np.clip((canopy - 0.35) * 1.9, 0, 1) + 22 * detail + grain
    img[..., 2] = 24 + 38 * np.clip((canopy - 0.55) * 1.6, 0, 1) + 14 * detail + grain * 0.6
    # a few clearings so the scene is not uniform texture
    clear = (fbm((WORLD_H, WORLD_W), rng, octaves=3) > 0.72)
    img[clear] = img[clear] * 0.75 + np.array([120, 115, 78]) * 0.25
    return img


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="demo/assets/wildfire_demo.mp4")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    import av

    rng = np.random.default_rng(args.seed)
    terrain = build_terrain(rng)
    n_frames = int(args.seconds * args.fps)
    W, H = args.width, args.height

    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)

    # One noise field, generated once and slid per frame. Regenerating noise
    # every frame cost 4 minutes of wall clock for 20 seconds of video.
    TEX_W, TEX_H = W + 420, H + 420
    texfield = fbm((TEX_H, TEX_W), np.random.default_rng(args.seed + 999), octaves=8, persistence=0.62)

    # Fire ignites at t0 and the front grows; smoke lags and drifts downwind.
    # Starting a few seconds in gives the demo an honest "before" -- the model
    # correctly showing nothing over unburnt forest is part of what it does.
    ignite_at = 3.0
    wind = np.array([1.0, -0.42])
    wind /= np.linalg.norm(wind)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(str(out_path), mode="w")
    stream = container.add_stream("libx264", rate=args.fps)
    stream.width, stream.height = W, H
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "20", "preset": "medium"}

    truth = []
    fire_world = np.array([760.0, 560.0])

    for i in range(n_frames):
        t = i / args.fps
        # Drone drift: slow diagonal pan plus a little low-frequency wobble, so
        # the scene moves the way handheld aerial video actually moves. This is
        # what makes overlay drift visible if the sync is wrong.
        cam_x = 210 + 9.0 * t + 14 * math.sin(t * 0.55)
        cam_y = 120 + 3.4 * t + 10 * math.cos(t * 0.41)
        cam_x = float(np.clip(cam_x, 0, WORLD_W - W - 1))
        cam_y = float(np.clip(cam_y, 0, WORLD_H - H - 1))
        x0, y0 = int(cam_x), int(cam_y)

        frame = terrain[y0:y0 + H, x0:x0 + W].copy()
        fx, fy = fire_world[0] - x0, fire_world[1] - y0

        burn = max(0.0, t - ignite_at)
        fire_box = smoke_box = None

        if burn > 0:
            # --- smoke plume -----------------------------------------------
            # Each puff is drawn into its own small window rather than across
            # the whole frame: 120 full-frame gaussians per frame is what made
            # the first version unusably slow.
            #
            # The shape rule that matters: downwind drift must outrun radial
            # growth, or the plume collapses into a disc instead of stretching
            # into a plume. Drift ~70px per age-unit against radius +9.
            smoke_alpha = np.zeros((H, W))
            max_age = min(burn, 9.0)
            n_puff = 130
            for k in range(n_puff):
                age = max_age * (k + 0.5) / n_puff
                drift = wind * age * 70.0
                turb_x = math.sin(k * 0.83 + t * 0.6) * (10 + age * 7)
                turb_y = math.cos(k * 1.27 + t * 0.5) * (7 + age * 6)
                cx, cy = fx + drift[0] + turb_x, fy + drift[1] + turb_y
                radius = 15.0 + age * 9.0
                a = 0.235 * math.exp(-age / 6.2) * (0.52 + 0.48 * math.sin(k * 2.1 + t))
                if a <= 0.004:
                    continue
                r3 = int(radius * 2.6)
                lx0, lx1 = max(0, int(cx) - r3), min(W, int(cx) + r3)
                ly0, ly1 = max(0, int(cy) - r3), min(H, int(cy) + r3)
                if lx1 <= lx0 or ly1 <= ly0:
                    continue
                sub_x = xx[ly0:ly1, lx0:lx1] - cx
                sub_y = yy[ly0:ly1, lx0:lx1] - cy
                smoke_alpha[ly0:ly1, lx0:lx1] += a * np.exp(
                    -(sub_x * sub_x + sub_y * sub_y) / (2 * radius * radius))

            # Break up the smooth gaussian sum so the plume has internal
            # structure. Reusing one precomputed noise field and sliding it with
            # the camera is far cheaper than regenerating noise every frame.
            ty0 = (y0 + int(t * 9)) % (TEX_H - H)
            tx0 = (x0 + int(t * 13)) % (TEX_W - W)
            smoke_alpha *= 0.30 + 1.30 * texfield[ty0:ty0 + H, tx0:tx0 + W]
            smoke_alpha = np.clip(smoke_alpha, 0, 0.88)

            grey = np.array([182, 179, 175])
            frame = frame * (1 - smoke_alpha[..., None]) + grey * smoke_alpha[..., None]

            m = smoke_alpha > 0.14
            if m.any():
                ys_, xs_ = np.where(m)
                smoke_box = (xs_.min() / W, ys_.min() / H, xs_.max() / W, ys_.max() / H)

            # --- flame front ------------------------------------------------
            # An irregular perimeter, not an ellipse: angular noise modulates
            # the radius so the front has the ragged edge a real fire has.
            fr_x = 30 + burn * 5.4
            fr_y = 13 + burn * 1.9
            r3 = int(max(fr_x, fr_y) * 2.2)
            lx0, lx1 = max(0, int(fx) - r3), min(W, int(fx) + r3)
            ly0, ly1 = max(0, int(fy) - r3), min(H, int(fy) + r3)
            if lx1 > lx0 and ly1 > ly0:
                sx_ = xx[ly0:ly1, lx0:lx1] - fx
                sy_ = yy[ly0:ly1, lx0:lx1] - fy
                ang = np.arctan2(sy_, sx_)
                ragged = (1.0
                          + 0.085 * np.sin(ang * 6 + t * 3.1)
                          + 0.055 * np.sin(ang * 11 - t * 4.7)
                          + 0.035 * np.sin(ang * 19 + t * 8.3)
                          + 0.025 * np.sin(ang * 31 - t * 11.2))
                e = ((sx_ / (fr_x * ragged)) ** 2 + (sy_ / (fr_y * ragged)) ** 2)
                flick = 0.86 + 0.14 * math.sin(t * 13.0) + 0.07 * math.sin(t * 27.7)
                core = np.clip(1.0 - e, 0, 1) ** 0.75 * flick
                hot = core > 0.05
                if hot.any():
                    sub = frame[ly0:ly1, lx0:lx1]
                    # white-hot centre grading out to deep orange at the edge
                    c2 = core * core
                    sub[..., 0] = np.where(hot, np.clip(sub[..., 0] * (1 - core) + (255 * core), 0, 255), sub[..., 0])
                    sub[..., 1] = np.where(hot, np.clip(sub[..., 1] * (1 - core) + (110 + 145 * c2) * core, 0, 255), sub[..., 1])
                    sub[..., 2] = np.where(hot, np.clip(sub[..., 2] * (1 - core) + (20 + 180 * c2 * c2) * core, 0, 255), sub[..., 2])
                    frame[ly0:ly1, lx0:lx1] = sub
                    ys_, xs_ = np.where(core > 0.12)
                    if len(xs_):
                        fire_box = ((xs_.min() + lx0) / W, (ys_.min() + ly0) / H,
                                    (xs_.max() + lx0) / W, (ys_.max() + ly0) / H)
                    # glow cast on the smoke just downwind of the front
                    glow = np.clip(1.0 - e * 0.34, 0, 1) ** 2 * 0.30 * flick
                    sub = frame[ly0:ly1, lx0:lx1]
                    sub[..., 0] = np.clip(sub[..., 0] + 96 * glow, 0, 255)
                    sub[..., 1] = np.clip(sub[..., 1] + 48 * glow, 0, 255)
                    frame[ly0:ly1, lx0:lx1] = sub

        # atmospheric haze so it does not look like clip art
        frame = frame * 0.94 + 12.0
        arr = np.clip(frame, 0, 255).astype(np.uint8)

        vf = av.VideoFrame.from_ndarray(arr, format="rgb24")
        for packet in stream.encode(vf):
            container.mux(packet)

        boxes = []
        if fire_box:
            boxes.append({"cls": "fire", "box": [round(v, 5) for v in fire_box]})
        if smoke_box:
            boxes.append({"cls": "smoke", "box": [round(v, 5) for v in smoke_box]})
        truth.append({"frame_id": i, "pts": round(i / args.fps, 5), "boxes": boxes})

    for packet in stream.encode():
        container.mux(packet)
    container.close()

    sidecar = out_path.with_suffix(".truth.json")
    sidecar.write_text(json.dumps({
        "video": out_path.name,
        "fps": args.fps,
        "width": W,
        "height": H,
        "classes": ["fire", "smoke"],
        "smoke_alpha_threshold": 0.14,
        "note": ("Synthetic clip. Ground-truth smoke extent is threshold-dependent because "
                 "smoke has no crisp boundary; treat smoke boxes as approximate."),
        "frames": truth,
    }, indent=1))

    n_fire = sum(1 for f in truth if any(b["cls"] == "fire" for b in f["boxes"]))
    n_smoke = sum(1 for f in truth if any(b["cls"] == "smoke" for b in f["boxes"]))
    size_kb = out_path.stat().st_size / 1024
    print(f"wrote {out_path} ({size_kb:.0f} KB), {n_frames} frames @ {args.fps}fps")
    print(f"  frames with fire: {n_fire}, with smoke: {n_smoke}, clean (pre-ignition): {n_frames - n_smoke}")
    print(f"  ground truth: {sidecar}")


if __name__ == "__main__":
    main()
