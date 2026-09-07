#!/usr/bin/env python3
"""Count how often the model boxes a healthy blade. The check v1 never had.

WHY THIS IS THE NUMBER THAT MATTERS
    Every metric so far is measured on images that contain defects. mAP asks "did it find
    the defect?" - never "did it invent one?". The original complaint about v1 was not that
    it missed defects; it was that it drew boxes on things that were not defects at all.
    No score on a defect split can detect that.

    The dataset contains thousands of healthy blade photographs that carry no labels after
    the rebuild, because they became background negatives. Any box the model draws on those
    is, by construction, a false positive. That makes them a ready-made test set for
    precisely the failure that made v1 useless - no new photographs required.

WHAT A GOOD RESULT LOOKS LIKE
    A low share of healthy images with any detection at the deployed confidence threshold.
    The absolute number matters less than the shape: if the model fires on most healthy
    frames, it has learned "blade texture" rather than "defect", and the deployed
    confidence threshold needs raising or the negatives need to be in-domain.

    Reported per confidence threshold, since that is the knob the web app actually exposes.

Usage:
    python3 tools/false_positive_check.py [--limit 300] [--model web/models/turbine.onnx]

Requires `pip install onnxruntime numpy Pillow`.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# The letterbox/decode pair is imported from the export-contract test rather than copied.
# That test's entire purpose is proving this decoder matches Ultralytics and the browser's;
# a second copy here would be a second thing to keep in sync, and the first to drift.
_spec = importlib.util.spec_from_file_location("contract", ROOT / "tests/test_export_contract.py")
_contract = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_contract)
letterbox, decode = _contract.letterbox, _contract.decode

# The families holding healthy blade photographs. They carry no labels after the rebuild.
HEALTHY_PREFIXES = ("Healthy_Train", "Areial_Healthy")
THRESHOLDS = (0.25, 0.40, 0.50, 0.70)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, default=ROOT / "web/models/turbine.onnx")
    parser.add_argument("--limit", type=int, default=300, help="how many healthy images to test")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=ROOT / "docs/false-positives.json")
    parser.add_argument("--save-worst", type=Path, help="copy the worst offenders here to look at")
    args = parser.parse_args()

    import numpy as np
    import onnxruntime as ort
    from PIL import Image

    if not args.model.exists():
        raise SystemExit(f"no model at {args.model}")

    manifest = json.loads((args.model.parent / "manifest.json").read_text())["turbine"]
    labels, imgsz = manifest["labels"], manifest["imgsz"]

    images = sorted(
        p for split in ("train", "valid", "test")
        for p in (ROOT / split / "images").glob("*.jpg")
        if p.name.startswith(HEALTHY_PREFIXES)
    )
    if not images:
        raise SystemExit("no healthy images found - expected train/valid/test image folders")

    rng = random.Random(args.seed)
    rng.shuffle(images)
    sample = images[: args.limit]

    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    print(f"Running {args.model.name} over {len(sample)} healthy blade images "
          f"(of {len(images)} available) at imgsz={imgsz}\n")
    print("Every detection below is a false positive: these images contain no defect.\n")

    fired = {t: 0 for t in THRESHOLDS}
    per_class = {t: {name: 0 for name in labels} for t in THRESHOLDS}
    worst = []

    for index, path in enumerate(sample, 1):
        with Image.open(path) as handle:
            image = handle.convert("RGB")
        width, height = image.size
        tensor, scale, pad_x, pad_y = letterbox(image, imgsz)
        output = session.run(None, {input_name: tensor})[0]

        detections = decode(output, scale, pad_x, pad_y, width, height, min(THRESHOLDS))
        for threshold in THRESHOLDS:
            hits = [d for d in detections if d["conf"] >= threshold]
            if hits:
                fired[threshold] += 1
                for hit in hits:
                    per_class[threshold][labels[hit["cls"]]] += 1
        if detections:
            worst.append((max(d["conf"] for d in detections), path))
        if index % 25 == 0:
            print(f"  {index}/{len(sample)}", end="\r", flush=True)
    print(" " * 30, end="\r")

    header = f"  {'conf':>6}{'images firing':>16}{'rate':>9}   boxes by class"
    print(header)
    print("  " + "-" * (len(header) + 24))
    for threshold in THRESHOLDS:
        rate = fired[threshold] / len(sample)
        detail = ", ".join(f"{k} {v}" for k, v in per_class[threshold].items() if v)
        print(f"  {threshold:>6.2f}{fired[threshold]:>16}{rate:>8.1%}   {detail or '-'}")

    deployed = manifest.get("confThreshold", 0.25)
    rate = fired.get(deployed, fired[min(THRESHOLDS)]) / len(sample)
    print(f"\n  At the deployed threshold ({deployed}), {rate:.1%} of healthy blade images")
    print("  get at least one box drawn on them.")

    worst.sort(reverse=True, key=lambda pair: pair[0])
    payload = {
        "model": str(args.model.relative_to(ROOT)), "sampled": len(sample),
        "available": len(images), "imgsz": imgsz,
        "by_threshold": {str(t): {"images_firing": fired[t],
                                  "rate": round(fired[t] / len(sample), 4),
                                  "boxes_by_class": per_class[t]} for t in THRESHOLDS},
        "worst": [{"confidence": round(c, 4), "image": p.name} for c, p in worst[:40]],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\n  -> {args.out.relative_to(ROOT)}")

    if args.save_worst and worst:
        import shutil
        args.save_worst.mkdir(parents=True, exist_ok=True)
        for rank, (confidence, path) in enumerate(worst[:40], 1):
            shutil.copy2(path, args.save_worst / f"{rank:02d}_{confidence:.2f}_{path.name}")
        print(f"  -> {min(40, len(worst))} images copied to {args.save_worst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
