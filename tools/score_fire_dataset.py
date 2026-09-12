#!/usr/bin/env python3
"""Score a fire model against labelled pictures, kept apart by where the camera was.

    python3 tools/score_fire_dataset.py --model web/models/wildfire-640.onnx

WHY THE SOURCE COLUMN IS NOT COLLAPSED
    Because it is the whole question. Measured on the model that ships, at the threshold it
    ships at, over the same 311 pictures:

        dfire, close fire from the ground        41/48   85%
        aiformankind, long range smoke            4/11   36%
        pyro, wildfire smoke at distance        15/120   12%

    One averaged number would have read as 34% and hidden the thing that matters: a plume
    seen from far away has no flame edge, no sharp colour boundary and no shape, and a model
    trained on fires photographed across a room has never been shown one. A drone sees the
    third row. So an average is not a summary here, it is a way of not noticing.

    Neither column is coverage. A picture with nothing marked has not been cleared of
    anything; it failed to meet a threshold.
"""
import argparse, collections, sys
import numpy as np
import onnxruntime as ort
from PIL import Image

THRESHOLDS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.35, 0.50)


def load_rows(name, split, per_source):
    from datasets import load_dataset
    # Downloaded rather than streamed. Streaming the first 600 rows once reached only the
    # two corpora at the top of the file and produced a confident table headed AERIAL with
    # no aerial picture in it. A sample that stops early is the beginning of the file.
    data = load_dataset(name, split=split)
    taken = collections.Counter()
    rows = []
    for row in data:
        source = row.get("source")
        if taken[source] >= per_source:
            continue
        taken[source] += 1
        rows.append(row)
    return rows


def scorer(path, size):
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name

    def top(img):
        img = img.convert("RGB")
        scale = min(size / img.width, size / img.height)
        dw, dh = round(img.width * scale), round(img.height * scale)
        canvas = Image.new("RGB", (size, size), (114, 114, 114))
        canvas.paste(img.resize((dw, dh), Image.BILINEAR), ((size - dw) // 2, (size - dh) // 2))
        x = np.asarray(canvas, np.float32).transpose(2, 0, 1)[None] / 255
        head = session.run(None, {name: x})[0][0]
        # Every class this model has is a kind of burning, so the strongest of them is the
        # answer to "is something on fire here".
        return float(head[4:].max())

    return top


def table(rows, title):
    fire = [s for _, burning, s in rows if burning]
    clear = [s for _, burning, s in rows if not burning]
    out = [f"  {title}  ({len(fire)} burning, {len(clear)} not)"]
    if not fire and not clear:
        return out + ["    nothing here", ""]
    out.append(f"    {'conf':>6}{'found, of burning':>22}{'marked, of clear':>20}")
    for conf in THRESHOLDS:
        found = sum(1 for s in fire if s >= conf)
        marked = sum(1 for s in clear if s >= conf)
        out.append(f"    {conf:>6.2f}"
                   + f"{f'{found}/{len(fire)}' if fire else '-':>22}"
                   + f"{f'{marked}/{len(clear)}' if clear else '-':>20}")
    return out + [""]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--label", default="", help="what to call it in the report")
    ap.add_argument("--dataset",
                    default="baizhanquan/FireDetectionDataset-flame-forest-flameye-wildfire")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--per-source", type=int, default=150)
    ap.add_argument("--report", default="")
    args = ap.parse_args()

    rows = load_rows(args.dataset, args.split, args.per_source)
    if not rows:
        print("::warning::no rows came back", file=sys.stderr)
        return 0
    missing = [k for k in ("image", "has_fire", "has_smoke", "source") if k not in rows[0]]
    if missing:
        print(f"::warning::the schema has changed, missing {missing}", file=sys.stderr)
        return 0

    top = scorer(args.model, args.imgsz)
    scored = []
    for row in rows:
        try:
            burning = bool(row["has_fire"]) or bool(row["has_smoke"])
            scored.append((row["source"], burning, top(row["image"])))
        except Exception:
            continue
    if not scored:
        print("::warning::nothing could be scored", file=sys.stderr)
        return 0

    label = args.label or args.model
    lines = ["", "=" * 78, f"{label}, at {args.imgsz}",
             f"{len(scored)} pictures from {args.dataset}", ""]
    lines += table(scored, "every source together")
    for source in sorted({s for s, _, _ in scored}):
        lines += table([r for r in scored if r[0] == source], source)
    lines += ["  Neither column is coverage. A picture with nothing marked has not been",
              "  cleared of anything; it failed to meet a threshold.", "=" * 78, ""]
    text = "\n".join(lines)
    print(text)
    if args.report:
        with open(args.report, "a") as out:
            out.write(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
