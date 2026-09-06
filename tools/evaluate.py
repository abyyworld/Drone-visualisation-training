#!/usr/bin/env python3
"""Evaluate a detector and report the numbers that actually mean something.

The v1 run only ever recorded aggregate mAP. With `healthy` at 62% of instances and
trivially separable, that aggregate said almost nothing about defect detection — which is
the entire job. This prints per-class AP, precision and recall, and a confusion matrix.

Two things worth knowing before comparing v1 against v2:

  * They have different class sets. v1 has four classes including `healthy`; v2 has three
    and treats "no detections" as healthy. Aggregate mAP is therefore NOT comparable —
    v2 loses an easy majority class and its aggregate will look worse while the model is
    better. Compare the three shared defect classes.
  * They have different test splits, because v2 re-split on capture-ID blocks to remove
    near-duplicate leakage. v1's split flatters v1.

So the honest read is per-class defect AP, each model on its own split, plus the visual
check below.

Usage:
    # per-class metrics on a labelled split
    python3 tools/evaluate.py best.pt --data data.yaml --split test

    # same for the exported ONNX, to measure what quantisation cost
    python3 tools/evaluate.py web/models/turbine.onnx --data datasets/turbine_v2/data.yaml

    # reality check: run on unlabelled images and write annotated results to look at
    python3 tools/evaluate.py best.pt --predict path/to/real_photos --out runs/reality_check

Requires `pip install ultralytics`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(weights: Path):
    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("ultralytics is not installed — run `pip install ultralytics`")
    return YOLO(str(weights))


def evaluate(weights: Path, data: Path, split: str, imgsz: int, out: Path) -> dict:
    model = load(weights)
    print(f"Evaluating {weights} on {data} [{split}] at imgsz={imgsz}\n")

    results = model.val(
        data=str(data),
        split=split,
        imgsz=imgsz,
        plots=True,          # writes the confusion matrix and PR curves
        project=str(out.parent),
        name=out.name,
        exist_ok=True,
        verbose=False,
    )

    box = results.box
    names = results.names
    per_class = {}

    header = f"  {'class':<20}{'images':>8}{'inst':>8}{'P':>9}{'R':>9}{'mAP50':>9}{'mAP50-95':>11}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for index, class_id in enumerate(box.ap_class_index):
        name = names[class_id]
        precision, recall, ap50, ap = (
            float(box.p[index]), float(box.r[index]),
            float(box.ap50[index]), float(box.ap[index]),
        )
        instances = int(results.nt_per_class[class_id]) if hasattr(results, "nt_per_class") else 0
        per_class[name] = {
            "precision": precision, "recall": recall,
            "mAP50": ap50, "mAP50_95": ap, "instances": instances,
        }
        # Below ~30 instances AP is dominated by sampling noise, so flag it rather than
        # letting a confident-looking number get quoted.
        flag = "  (too few instances to trust)" if 0 < instances < 30 else ""
        print(f"  {name:<20}{'':>8}{instances:>8}{precision:>9.3f}{recall:>9.3f}"
              f"{ap50:>9.3f}{ap:>11.3f}{flag}")

    print("  " + "-" * (len(header) - 2))
    print(f"  {'all':<20}{'':>8}{'':>8}{box.mp:>9.3f}{box.mr:>9.3f}{box.map50:>9.3f}{box.map:>11.3f}")

    summary = {
        "weights": str(weights),
        "data": str(data),
        "split": split,
        "imgsz": imgsz,
        "aggregate": {
            "precision": float(box.mp), "recall": float(box.mr),
            "mAP50": float(box.map50), "mAP50_95": float(box.map),
        },
        "per_class": per_class,
    }

    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(f"\n  metrics.json      -> {out / 'metrics.json'}")
    print(f"  confusion matrix  -> {out / 'confusion_matrix.png'}")

    weakest = min(per_class.items(), key=lambda kv: kv[1]["mAP50"], default=None)
    if weakest:
        name, stats = weakest
        print(f"\n  Weakest class: {name} at mAP50 {stats['mAP50']:.3f}. Aggregate mAP hides this;")
        print("  it is the number to improve, and the one to quote honestly.")
    return summary


def predict(weights: Path, images: Path, out: Path, imgsz: int, conf: float) -> None:
    """Run on unlabelled images and save annotated results.

    This is the check that matters most and the one the v1 run never did: metrics on a split
    drawn from the same pool as training cannot tell you whether the model works on a photo
    it has never seen the like of.
    """
    model = load(weights)
    files = sorted(
        p for p in images.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    if not files:
        raise SystemExit(f"no images found under {images}")

    print(f"Running {weights} over {len(files)} images from {images}\n")
    results = model.predict(
        source=[str(p) for p in files],
        imgsz=imgsz,
        conf=conf,
        save=True,
        project=str(out.parent),
        name=out.name,
        exist_ok=True,
        verbose=False,
    )

    empty = 0
    counts: dict[str, int] = {}
    for result in results:
        if not len(result.boxes):
            empty += 1
        for class_id in result.boxes.cls.tolist():
            name = result.names[int(class_id)]
            counts[name] = counts.get(name, 0) + 1

    print(f"  images with no detections: {empty} of {len(files)}")
    for name, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {name:<20}{count:>6}")
    print(f"\n  annotated images -> {out}")
    print("\n  Look at them. A model that boxes sky, ground or blade edges is failing in a way")
    print("  no metric on an in-distribution split will show you.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("weights", type=Path)
    parser.add_argument("--data", type=Path, help="data.yaml for metric evaluation")
    parser.add_argument("--split", default="test", choices=["train", "val", "valid", "test"])
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--predict", type=Path, help="directory of unlabelled images")
    parser.add_argument("--out", type=Path, default=Path("runs/evaluate"))
    args = parser.parse_args()

    if not args.weights.exists():
        raise SystemExit(f"not found: {args.weights}")
    if not args.data and not args.predict:
        raise SystemExit("give --data (metrics) or --predict (visual check), or both")

    if args.data:
        evaluate(args.weights, args.data, args.split, args.imgsz, args.out)
    if args.predict:
        if args.data:
            print()
        predict(args.weights, args.predict, args.out.with_name(f"{args.out.name}_predict"),
                args.imgsz, args.conf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
