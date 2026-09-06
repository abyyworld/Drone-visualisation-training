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


def fbm(shape: tuple[int, int], rng: np.random.Generator, octaves: int = 5) -> np.ndarray:
    out = np.zeros(shape)
    amp, cells, norm = 1.0, 3, 0.0
    for _ in range(octaves):
        out += amp * value_noise(shape, cells, rng)
        norm += amp
        amp *= 0.5
        cells *= 2
    return out / norm


def build_terrain(rng: np.random.Generator) -> np.ndarray:
    """Forested terrain: greens and browns broken up by clearings."""
    h = fbm((WORLD_H, WORLD_W), rng, octaves=6)
    detail = fbm((WORLD_H, WORLD_W), rng, octaves=7)
    canopy = 0.65 * h + 0.35 * detail

    img = np.zeros((WORLD_H, WORLD_W, 3))
    # dark conifer -> lighter scrub -> dry grass, by elevation-ish value
    img[..., 0] = 38 + 90 * np.clip((canopy - 0.45) * 2.2, 0, 1) + 18 * detail
    img[..., 1] = 55 + 85 * np.clip((canopy - 0.35) * 1.9, 0, 1) + 14 * detail
    img[..., 2] = 30 + 40 * np.clip((canopy - 0.55) * 1.6, 0, 1) + 10 * detail
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
            # --- smoke: blobs emitted along the burn history, advected downwind,
            # expanding and thinning. Thin, wide and soft-edged on purpose.
            smoke_alpha = np.zeros((H, W))
            n_puff = int(min(burn, 17.0) * 7)
            for k in range(n_puff):
                age = burn - k * (min(burn, 17.0) / max(n_puff, 1))
                if age <= 0:
                    continue
                drift = wind * age * 27.0
                jitter = np.array([math.sin(k * 1.7 + t * 0.9) * 16, math.cos(k * 2.3 + t * 0.7) * 11])
                cx, cy = fx + drift[0] + jitter[0], fy + drift[1] + jitter[1]
                radius = 26 + age * 15.0
                # thins as it disperses -- the far plume is the hard case
                a = 0.60 * math.exp(-age / 9.0)
                d2 = (xx - cx) ** 2 + (yy - cy) ** 2
                smoke_alpha += a * np.exp(-d2 / (2 * radius * radius))
            smoke_alpha = np.clip(smoke_alpha, 0, 0.93)
            texture = 0.75 + 0.5 * fbm((H, W), np.random.default_rng(args.seed + i), octaves=4)
            smoke_alpha *= np.clip(texture, 0, 1.35)
            smoke_alpha = np.clip(smoke_alpha, 0, 0.93)

            grey = np.array([176, 172, 168])
            frame = frame * (1 - smoke_alpha[..., None]) + grey * smoke_alpha[..., None]

            # ground truth at a stated threshold; see module docstring
            m = smoke_alpha > 0.14
            if m.any():
                ys_, xs_ = np.where(m)
                smoke_box = (xs_.min() / W, ys_.min() / H, xs_.max() / W, ys_.max() / H)

            # --- flame front: an ellipse that grows, with per-frame flicker
            fr_x = 30 + burn * 5.2
            fr_y = 19 + burn * 3.1
            flick = 0.80 + 0.20 * math.sin(t * 13.0) + 0.10 * math.sin(t * 27.7)
            e = ((xx - fx) ** 2) / (fr_x ** 2) + ((yy - fy) ** 2) / (fr_y ** 2)
            core = np.clip(1.0 - e, 0, 1) ** 0.55 * flick
            hot = core > 0.06
            if hot.any():
                frame[..., 0] = np.where(hot, np.clip(frame[..., 0] * (1 - core) + 255 * core, 0, 255), frame[..., 0])
                frame[..., 1] = np.where(hot, np.clip(frame[..., 1] * (1 - core) + 165 * core, 0, 255), frame[..., 1])
                frame[..., 2] = np.where(hot, np.clip(frame[..., 2] * (1 - core) + 45 * core, 0, 255), frame[..., 2])
                ys_, xs_ = np.where(core > 0.12)
                if len(xs_):
                    fire_box = (xs_.min() / W, ys_.min() / H, xs_.max() / W, ys_.max() / H)

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
