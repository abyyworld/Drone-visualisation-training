#!/usr/bin/env python3
"""Fly a REAL video sequence the way the tablet looks at it.

    python3 tools/fly_video.py --sequence <dir> --annotations <file.txt> --out flight.json
    node tools/score_flight.mjs flight.json

WHY THIS AND NOT tools/fly.py
    fly.py pans across one still frame. That models the camera moving and nothing else. The
    people are frozen relative to each other, nobody walks behind anybody, and there is no
    motion blur, no rolling shutter, and no compression that gets worse exactly when the
    scene does. Every tracking number this project had came out of that.

    VisDrone's MOT sequences are consecutive frames of real flights, and every row carries a
    TARGET ID. So the `who` this writes out is the real person, not an inference about which
    person a box mostly sat on, and a second number issued to somebody is a measured identity
    switch rather than an estimate of one.

WHAT IT MODELS ABOUT THE DEVICE
    The video runs at its own frame rate and the detector does not. A cycle takes about
    250 ms, so roughly every seventh frame of 30 fps footage gets looked at, and the tracker
    is handed the gap in milliseconds rather than in frames. Tiles rotate exactly as
    Tiles.region says. What the tracker sees here is what it sees on the tablet.

ANNOTATION FORMAT (VisDrone MOT)
    frame, target_id, x, y, w, h, score, category, truncation, occlusion
    Categories 1 and 2 are pedestrian and people, which is what the app keeps.
"""
import argparse, collections, json, pathlib, sys
import numpy as np
import onnxruntime as ort
from PIL import Image

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
MODELS = ROOT / "web" / "models"
OVERLAP = 0.18                      # Tiles.OVERLAP

SPEC = SIZE = KEEP = SESSION = None


def load(onnx):
    global SPEC, SIZE, KEEP, SESSION
    onnx = pathlib.Path(onnx)
    SPEC = json.loads(onnx.with_suffix(".json").read_text())
    SIZE, KEEP = SPEC["imgsz"], sorted(SPEC["keepClasses"])
    SESSION = ort.InferenceSession(str(onnx), providers=["CPUExecutionProvider"])


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)


def nms(boxes, threshold=0.55):
    kept = []
    for box in sorted(boxes, key=lambda d: -d[4]):
        if all(iou(box, other) <= threshold for other in kept):
            kept.append(box)
    return kept


def region(index, columns, rows, width, height):
    """Tiles.region, in Python."""
    n = columns * rows
    wrapped = index % n
    column, row = wrapped % columns, wrapped // columns
    tw, th = width / columns, height / rows
    px, py = tw * OVERLAP, th * OVERLAP
    x = max(0.0, column * tw - px)
    y = max(0.0, row * th - py)
    return [x, y, min(width - x, tw + px * 2), min(height - y, th + py * 2)]


def detect(view, conf, offset=(0.0, 0.0)):
    scale = min(SIZE / view.width, SIZE / view.height)
    dw, dh = round(view.width * scale), round(view.height * scale)
    px, py = (SIZE - dw) // 2, (SIZE - dh) // 2
    square = Image.new("RGB", (SIZE, SIZE), (114, 114, 114))
    square.paste(view.resize((dw, dh), Image.BILINEAR), (px, py))
    head = SESSION.run(None, {SESSION.get_inputs()[0].name:
                              np.asarray(square, np.float32).transpose(2, 0, 1)[None] / 255})[0][0]
    best = head[4:][KEEP].max(axis=0)
    found = []
    for i in np.nonzero(best >= conf)[0]:
        cx, cy, w, h = head[0, i], head[1, i], head[2, i], head[3, i]
        found.append([
            offset[0] + max(0.0, (cx - w / 2 - px) / scale),
            offset[1] + max(0.0, (cy - h / 2 - py) / scale),
            offset[0] + min(view.width, (cx + w / 2 - px) / scale),
            offset[1] + min(view.height, (cy + h / 2 - py) / scale),
            float(best[i]),
        ])
    return [b for b in found if b[2] - b[0] > 1 and b[3] - b[1] > 1]


def truth_by_frame(path):
    """frame index -> [(target_id, box), ...] for people only."""
    people = collections.defaultdict(list)
    for line in pathlib.Path(path).read_text().splitlines():
        parts = [p for p in line.replace(",", " ").split() if p]
        if len(parts) < 8:
            continue
        frame, target, x, y, w, h, score, category = (int(float(v)) for v in parts[:8])
        if score == 0 or category not in (1, 2):
            continue
        people[frame].append((target, [x, y, x + w, y + h]))
    return people


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sequence", required=True, help="directory of consecutive frames")
    ap.add_argument("--annotations", required=True, help="the MOT .txt for that sequence")
    ap.add_argument("--fps", type=float, default=30.0, help="the footage's own frame rate")
    ap.add_argument("--period", type=int, default=250, help="ms per detect cycle")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--tiles-per-cycle", type=int, default=2)
    ap.add_argument("--grid", default="2x1")
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--model", default=str(MODELS / "person-640.onnx"))
    ap.add_argument("--out", default="flight.json")
    args = ap.parse_args()

    load(args.model)
    columns, rows = (int(v) for v in args.grid.lower().split("x"))
    frames = sorted(pathlib.Path(args.sequence).glob("*.jpg"))
    if not frames:
        raise SystemExit(f"no frames in {args.sequence}")
    truth = truth_by_frame(args.annotations)

    # The detector does not see every frame. At 30 fps and a 250 ms cycle it sees every
    # seventh or eighth, which is the actual relationship on the tablet.
    step = max(1, round(args.fps * args.period / 1000))
    # Bounded by how many cycles the footage HOLDS, not by how many frames it has. Every
    # cycle consumes `step` frames, so a 500-frame sequence at every eighth frame is 62
    # cycles and asking for more walks off the end.
    wanted = min(len(frames) // step, int(args.seconds * args.fps / step))
    if wanted < 2:
        raise SystemExit(f"{len(frames)} frames at every {step} is too short to fly")
    first = Image.open(frames[0])
    width, height = first.size

    out_frames, present, turn = [], set(), 0
    for n in range(wanted):
        path = frames[n * step]
        # VisDrone MOT frames are numbered from one, in the file name.
        index = int(path.stem)
        view = Image.open(path).convert("RGB")
        here = truth.get(index, [])
        present.update(t for t, _ in here)

        found = []
        for _ in range(args.tiles_per_cycle):
            x, y, w, h = region(turn, columns, rows, width, height)
            turn += 1
            crop = view.crop((int(x), int(y), int(x + w), int(y + h)))
            found += detect(crop, args.conf, offset=(x, y))
        found = nms(found, 0.55)

        out_frames.append({
            "detections": [{"label": "person", "confidence": round(float(b[4]), 4),
                            "box": [round(float(v), 2) for v in b[:4]]} for b in found],
            "truth": [[round(float(v), 2) for v in box] for _, box in here],
            # The real person, from the dataset. Not an inference about which person a box
            # mostly sat on - which is what makes an identity switch measured here.
            "who": [target for target, _ in here],
        })
        if n % 10 == 0:
            print(f"  {n}/{wanted} cycles", file=sys.stderr, flush=True)

    # A truth lookup that matches nothing reads exactly like a flight over an empty field:
    # every cycle reports nobody present, the scorer divides by zero people and prints
    # tidy-looking zeroes, and the run goes green. That is how a fire model once shipped
    # that answered the same score for every picture. If the frame numbering in the
    # annotations does not line up with the file names, say so here.
    matched = sum(len(f["who"]) for f in out_frames)
    if not matched:
        raise SystemExit(
            f"no annotation row matched any frame of {args.sequence}. The .txt has "
            f"{len(truth)} annotated frame indices ({sorted(truth)[:5]}...) and the files "
            f"are named {[p.stem for p in frames[:5]]}; the two do not line up, so this "
            f"would have reported an empty field rather than a failure.")

    name = pathlib.Path(args.sequence).name
    pathlib.Path(args.out).write_text(json.dumps({
        "period": args.period,
        "config": {**vars(args), "grid": args.grid},
        "runs": [{"name": name, "peoplePresent": len(present), "frames": out_frames}],
    }))
    print(f"wrote {args.out}: {name}, {len(out_frames)} cycles of {args.period} ms, "
          f"{len(present)} distinct people")


if __name__ == "__main__":
    main()
