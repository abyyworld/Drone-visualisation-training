#!/usr/bin/env python3
"""Does the fire model see fire, and does it cry wolf?

WHY THIS AND NOT tools/evaluate_wildfire.py
    That one scores boxes against boxes and is the better instrument when there are
    labelled boxes to score against. This one answers the question that comes first and
    that nothing had ever asked of the shipped model: on a picture with fire in it, does
    anything get marked at all, and on a picture with no fire in it, does anything get
    marked anyway.

    Frame level, deliberately. It needs no boxes, so it runs against any pile of fire
    images and any pile of fire-free ones, which is what is actually obtainable. A
    detector that fails this does not deserve to be scored on box overlap, and one that
    passes it can then be handed to evaluate_wildfire.py - this writes predictions in
    that tool's own format for exactly that.

WHAT THE TWO NUMBERS MEAN, AND WHICH WAY EACH ONE FAILS
    marked, of fire frames        recall. Low means the model is silent on fire, which
                                  is the failure this whole subject fails toward: thin
                                  smoke on a bright sky, smouldering with no flame, fire
                                  under canopy, fire at night. Silence is not safety.

    marked, of fire-free frames   false alarms. High means the overlay marks things on
                                  ordinary footage, and an operator who has learned to
                                  ignore the boxes has lost the boxes.

    Neither number is coverage. A frame with nothing marked has not been cleared of
    anything; it has failed to meet a threshold. See station/core/safety.py.

    A control set is not optional. The false-alarm rate is what makes recall mean
    anything: a model that marks every frame scores perfect recall and is worthless.

USAGE
    python3 tools/evaluate_fire.py --fire datasets/fire/images --control datasets/visdrone
    python3 tools/evaluate_fire.py --control datasets/visdrone --predictions preds.jsonl
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
THRESHOLDS = (0.15, 0.25, 0.35, 0.50)
ROOT = pathlib.Path(__file__).resolve().parent.parent


def pictures(directory, limit=0):
    """Every image under a directory, in a stable order."""
    if directory is None:
        return []
    found = sorted(p for p in pathlib.Path(directory).rglob("*")
                   if p.suffix.lower() in SUFFIXES and p.is_file())
    return found[:limit] if limit else found


def spec_for(manifest, subject):
    """The manifest entry, which is what the app decodes by."""
    entry = json.loads(pathlib.Path(manifest).read_text())[subject]
    if not entry.get("labels"):
        raise SystemExit(f"the manifest gives {subject} no labels, so it marks nothing")
    return entry


class Detector:
    """The shipped model, decoded exactly the way web/js/detect.js decodes it.

    The decode is repeated here rather than approximated, because approximating it is how
    this model came to be shipped raising on every frame: its head is a segmentation one,
    four box numbers and one class score and then 32 mask coefficients, and reading those
    coefficients as scores marks every frame at over 1.0 confidence. classChannels below
    is that rule, and getting it wrong is the whole failure being measured against.
    """

    def __init__(self, model, spec):
        import numpy as np
        import onnxruntime as ort

        self.np = np
        self.session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
        self.name = self.session.get_inputs()[0].name
        self.size = spec["imgsz"]
        self.labels = spec["labels"]
        self.keep = set(spec.get("keepClasses", range(len(self.labels))))
        self.iou_threshold = spec.get("iouThreshold", 0.45)

    def __call__(self, path, floor):
        """Every box above `floor`, normalised 0..1, with its score and class."""
        from PIL import Image

        np = self.np
        image = Image.open(path).convert("RGB")
        size = self.size
        scale = min(size / image.width, size / image.height)
        dw, dh = round(image.width * scale), round(image.height * scale)
        px, py = (size - dw) // 2, (size - dh) // 2
        canvas = Image.new("RGB", (size, size), (114, 114, 114))
        canvas.paste(image.resize((dw, dh), Image.BILINEAR), (px, py))

        head = self.session.run(None, {self.name: np.asarray(canvas, np.float32)
                                       .transpose(2, 0, 1)[None] / 255.0})[0][0]
        if head.shape[0] > head.shape[1]:
            head = head.T                      # per-anchor rows rather than channel-major

        # Four box numbers, then one score per label, then whatever else the export
        # carries. Anything past the labels is not a score and is not read.
        scores = head[4:4 + len(self.labels)]
        wanted = sorted(c for c in self.keep if c < len(self.labels))
        best = scores[wanted].max(axis=0)
        which = [wanted[i] for i in scores[wanted].argmax(axis=0)]

        found = []
        for i in np.nonzero(best >= floor)[0]:
            cx, cy, w, h = (float(head[j, i]) for j in range(4))
            box = [max(0.0, (cx - w / 2 - px) / scale), max(0.0, (cy - h / 2 - py) / scale),
                   min(image.width, (cx + w / 2 - px) / scale),
                   min(image.height, (cy + h / 2 - py) / scale)]
            if box[2] - box[0] <= 1 or box[3] - box[1] <= 1:
                continue
            found.append({"cls": self.labels[which[i]], "conf": float(best[i]),
                          "box": [box[0] / image.width, box[1] / image.height,
                                  box[2] / image.width, box[3] / image.height]})
        return suppress(found, self.iou_threshold)


def overlap(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / union if union > 0 else 0.0


def suppress(found, threshold):
    kept = []
    for box in sorted(found, key=lambda d: -d["conf"]):
        if all(box["cls"] != other["cls"] or overlap(box["box"], other["box"]) <= threshold
               for other in kept):
            kept.append(box)
    return kept


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=str(ROOT / "web/models/wildfire-320.onnx"))
    parser.add_argument("--manifest", default=str(ROOT / "web/models/manifest.json"))
    parser.add_argument("--subject", default="wildfire")
    parser.add_argument("--fire", help="a directory of pictures that DO contain fire or smoke")
    parser.add_argument("--control", help="a directory of pictures that do NOT")
    parser.add_argument("--limit", type=int, default=0, help="at most this many from each")
    parser.add_argument("--predictions",
                        help="write raw detections here, in evaluate_wildfire.py's format")
    parser.add_argument("--report", help="write the report here as well as to stdout")
    parser.add_argument("--misses", type=int, default=15,
                        help="how many fire pictures it said nothing about to name")
    args = parser.parse_args(argv)

    fire = pictures(args.fire, args.limit)
    control = pictures(args.control, args.limit)
    if not fire and not control:
        raise SystemExit("nothing to evaluate: give --fire, --control, or both")

    spec = spec_for(args.manifest, args.subject)
    detector = Detector(args.model, spec)
    floor = min(THRESHOLDS)

    rows = []
    written = None
    if args.predictions:
        written = open(args.predictions, "w", encoding="utf-8")
    try:
        for group, paths in (("fire", fire), ("control", control)):
            for path in paths:
                found = detector(path, floor)
                rows.append((group, path, found))
                if written:
                    written.write(json.dumps({"image": str(path),
                                              "detections": found}) + "\n")
    finally:
        if written:
            written.close()

    lines = []
    say = lines.append
    say(f"model      {pathlib.Path(args.model).name}")
    say(f"labels     {spec['labels']}")
    say(f"pictures   {len(fire)} with fire or smoke, {len(control)} without")
    say("")
    say(f"{'conf':>6} {'marked, of fire':>18} {'marked, of fire-free':>22}")
    say(f"{'':>6} {'(higher is better)':>18} {'(lower is better)':>22}")
    for threshold in THRESHOLDS:
        hit = sum(1 for g, _, d in rows if g == "fire"
                  and any(x["conf"] >= threshold for x in d))
        cry = sum(1 for g, _, d in rows if g == "control"
                  and any(x["conf"] >= threshold for x in d))
        say(f"{threshold:>6.2f} "
            f"{f'{hit}/{len(fire)}' if fire else 'no fire set':>18} "
            f"{f'{cry}/{len(control)}' if control else 'no control set':>22}")

    shipped = spec.get("confThreshold", 0.25)
    say("")
    say(f"the app runs at {shipped:.2f}")

    silent = [p for g, p, d in rows
              if g == "fire" and not any(x["conf"] >= shipped for x in d)]
    if silent:
        say("")
        say(f"said nothing about {len(silent)} of {len(fire)} pictures with fire in them:")
        for path in silent[:args.misses]:
            say(f"  {path}")
        if len(silent) > args.misses:
            say(f"  ... and {len(silent) - args.misses} more")

    say("")
    say("Neither column is coverage. A picture with nothing marked has not been cleared")
    say("of anything; it failed to meet a threshold. Thin smoke on a bright sky, fire")
    say("under canopy and fire at night all return an empty list, and that list is")
    say("identical to the one an empty field returns.")

    report = "\n".join(lines)
    print(report)
    if args.report:
        pathlib.Path(args.report).write_text(report + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
