"""Fly the camera the way the TABLET actually looks at it, not the way it is convenient to.

    python3 tools/fly.py --out flight.json && node tools/score_flight.mjs flight.json

Needs onnxruntime, pillow and numpy, and a VisDrone-DET validation directory (--data, or
VISDRONE_VAL). Every number in PERSON-640.md and in the tracker's constants came out of
these two files, so a change to either can be argued with rather than believed.

WHY THIS EXISTS AND A WHOLE-FRAME BENCHMARK DOES NOT REPLACE IT
    Handing the tracker whole-frame detections every cycle gives every person a look every
    cycle, which is a thing the device may or may not do depending on how the grid and the
    tile budget are set. Get that wrong and every number is measured under a cadence the
    device has never had - which is how this project came to quote figures for a shape where
    five people in six were coasting on a guess at any moment.

    So the cadence is the parameter, and the flight is flown the way the tablet flies it.

WHAT IT VARIES
    --tiles-per-cycle   how many of the six tiles are looked at each cycle (device: 1)
    --wide-every        how often the whole frame is looked at (device: 3, 0 for never)
    --period            milliseconds per cycle (device target: 200, real: 250-370)
    --grid              tile columns x rows, e.g. 3x2 or 4x3

    --model             which weights to fly with, so a candidate can be judged on the
                        tablet's cadence rather than on a still frame

WHY THE PERIOD MOVES THE CAMERA
    A slower model does not get the same flight in slow motion. The drone keeps flying while
    it thinks, so a cycle that takes twice as long sees the ground twice as far along and
    gets half as many looks at it. Cycle count and pan per cycle are both derived from the
    period here, and the flight covers the same ground in the same seconds whatever model is
    running. Comparing a slow model against a fast one on a fixed cycle count would hand the
    slow one twice the flight.

Writes device.json for device.mjs to run the real tracker over.
"""
import argparse, json, math, os, pathlib, sys
import numpy as np
import onnxruntime as ort
from PIL import Image

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
# Not in the repository: tens of gigabytes, licensed separately. Fetch it with
# training/prepare_datasets.py, or point --data at a copy you already have.
DATA = pathlib.Path(os.environ.get("VISDRONE_VAL")
                    or ROOT / "data" / "VisDrone2019-DET-val")
MODELS = ROOT / "web" / "models"
SPEC, SIZE, KEEP, SESSION = None, None, None, None


def load(onnx):
    global SPEC, SIZE, KEEP, SESSION
    onnx = pathlib.Path(onnx)
    SPEC = json.loads(onnx.with_suffix(".json").read_text())
    SIZE, KEEP = SPEC["imgsz"], sorted(SPEC["keepClasses"])
    SESSION = ort.InferenceSession(str(onnx), providers=["CPUExecutionProvider"])
OVERLAP = 0.18                      # Tiles.OVERLAP
VIEW = (1920, 1080)                 # what the drone sends
FRAME = "0000295_02400_d_0000033"


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)


def nms(boxes, threshold=0.5):
    kept = []
    for box in sorted(boxes, key=lambda d: -d[4]):
        if all(iou(box, other) <= threshold for other in kept):
            kept.append(box)
    return kept


def truth(path):
    people = []
    for line in path.read_text().splitlines():
        parts = [p for p in line.replace(",", " ").split() if p]
        if len(parts) < 6:
            continue
        x, y, w, h, score, category = (int(float(v)) for v in parts[:6])
        if score != 0 and category in (1, 2):
            people.append([x, y, x + w, y + h])
    return people


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
    """The real model on one crop, boxes in FULL-FRAME pixels."""
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


def render(source, step, pan):
    """The whole 1920x1080 view on this cycle, and where a source box lands in it."""
    cx = 700 + pan * step
    cy = 500
    left, top = cx - VIEW[0] / 2, cy - VIEW[1] / 2
    view = source.transform(VIEW, Image.AFFINE, (1, 0, left, 0, 1, top), Image.BILINEAR)
    return view, (lambda b: [b[0] - left, b[1] - top, b[2] - left, b[3] - top])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tiles-per-cycle", type=int, default=2)
    ap.add_argument("--wide-every", type=int, default=0, help="0 for never")
    ap.add_argument("--period", type=int, default=250, help="ms per cycle")
    ap.add_argument("--grid", default="2x1")
    ap.add_argument("--cycles", type=int, default=0,
                    help="0 to derive from --seconds and --period")
    ap.add_argument("--seconds", type=float, default=10.0, help="wall clock of each flight")
    ap.add_argument("--model", default=str(MODELS / "person-640.onnx"))
    ap.add_argument("--frame", default=FRAME, help="which labelled frame to fly over")
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--data", default=str(DATA), help="a VisDrone-DET val directory")
    ap.add_argument("--out", default="flight.json")
    args = ap.parse_args()

    load(args.model)
    columns, rows = (int(v) for v in args.grid.lower().split("x"))
    tiles = columns * rows
    if not args.cycles:
        args.cycles = max(2, round(args.seconds * 1000 / args.period))
    # The ground speeds below are per 250 ms, the cadence they were chosen at. A model with a
    # longer cycle covers more ground between looks; it does not get a slower drone.
    stretch = args.period / 250.0
    data = pathlib.Path(args.data)
    source = Image.open(data / "images" / f"{args.frame}.jpg").convert("RGB")
    people = truth(data / "annotations" / f"{args.frame}.txt")

    runs = []
    for name, speed in [("still", 0.0), ("slow forward", 6.0), ("fast forward", 18.0)]:
        pan = speed * stretch
        frames, present, turn = [], set(), 0
        for step in range(args.cycles):
            view, into = render(source, step, pan)
            here = [i for i, p in enumerate(people)
                    if (lambda b: b[0] >= 0 and b[1] >= 0 and b[2] <= VIEW[0]
                        and b[3] <= VIEW[1] and b[2]-b[0] > 2 and b[3]-b[1] > 2)(into(p))]
            present.update(here)

            found = []
            if args.wide_every and step % args.wide_every == 0:
                found += detect(view.resize((640, 360), Image.BILINEAR), args.conf)
                found = [[b[0]*3, b[1]*3, b[2]*3, b[3]*3, b[4]] for b in found]
            for _ in range(args.tiles_per_cycle):
                x, y, w, h = region(turn, columns, rows, VIEW[0], VIEW[1])
                turn += 1
                crop = view.crop((int(x), int(y), int(x + w), int(y + h)))
                found += detect(crop, args.conf, offset=(x, y))
            found = nms(found, 0.55)

            frames.append({
                "detections": [{"label": "person", "confidence": round(float(b[4]), 4),
                                "box": [round(float(v), 2) for v in b[:4]]} for b in found],
                "truth": [[round(float(v), 2) for v in into(people[i])] for i in here],
                # Which person each truth box IS. Without this a scorer can count boxes but
                # cannot tell one person numbered twice from two people numbered once, which
                # is the only question the product actually asks.
                "who": here,
            })
        runs.append({"name": name, "peoplePresent": len(present), "frames": frames})
        print(f"  flew {name}", file=sys.stderr)

    pathlib.Path(args.out).write_text(json.dumps(
        {"period": args.period, "config": vars(args), "runs": runs}))
    print(f"wrote {args.out} ({args.cycles} cycles of {args.period} ms, "
          f"{pathlib.Path(args.model).name})")


if __name__ == "__main__":
    main()
