#!/usr/bin/env python3
"""Prove that a real Ultralytics ONNX export matches what the browser app expects.

The browser suite (tests/test_web.mjs) validates the JavaScript decoder against ONNX
fixtures whose outputs I chose by hand. That proves the decoder is self-consistent. It does
NOT prove the decoder agrees with what Ultralytics actually exports — output layout, class
ordering, normalisation and opset are all assumptions until something checks them against a
genuine model.

This is where that mismatch would otherwise be found: after burning GPU hours on a good
model, at deployment, with detections that are subtly wrong rather than obviously broken.

What it checks:
  1. The exported graph's input is [1, 3, imgsz, imgsz] float32.
  2. The output is a layout web/js/detect.js handles — [1, 4+nc, anchors] or [1, N, 6].
  3. The class count matches the label list in web/models/manifest.json.
  4. **A Python port of the JS decoder, run on the ONNX output, reproduces Ultralytics' own
     predictions on the same image.** This is the real test: if these agree, the browser is
     doing the same arithmetic as `model.predict()`.

Usage:
    python3 tests/test_export_contract.py --weights runs/turbine_v2/weights/best.pt \\
        --image path/to/photo.jpg --imgsz 960 --name turbine

    # No trained model yet? This trains a throwaway one first (CPU is fine, it is tiny):
    python3 tests/test_export_contract.py --smoke-train

Requires `pip install ultralytics onnxruntime`.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CONF = 0.25
IOU = 0.45


# --------------------------------------------------------------------------------------
# Python port of web/js/detect.js + preprocess.js. Kept deliberately literal — this only
# has value if it mirrors the JavaScript, so prefer an awkward transcription over an
# idiomatic rewrite.
# --------------------------------------------------------------------------------------

def letterbox(image, size):
    """Mirror of preprocess.js letterbox(): aspect-preserving resize onto a 114-grey pad."""
    import numpy as np
    from PIL import Image

    source_w, source_h = image.size
    scale = min(size / source_w, size / source_h)
    draw_w, draw_h = round(source_w * scale), round(source_h * scale)
    pad_x, pad_y = (size - draw_w) // 2, (size - draw_h) // 2

    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    canvas.paste(image.resize((draw_w, draw_h), Image.BILINEAR), (pad_x, pad_y))

    array = np.asarray(canvas, dtype=np.float32) / 255.0      # HWC, 0..1
    tensor = array.transpose(2, 0, 1)[None]                    # -> NCHW
    return np.ascontiguousarray(tensor), scale, pad_x, pad_y


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    w, h = x1 - x0, y1 - y0
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def nms(detections, threshold):
    """Mirror of detect.js nms(): greedy, per class."""
    kept = []
    for class_id in {d["cls"] for d in detections}:
        candidates = sorted(
            (d for d in detections if d["cls"] == class_id),
            key=lambda d: -d["conf"],
        )
        while candidates:
            best = candidates.pop(0)
            kept.append(best)
            candidates = [c for c in candidates if iou(best["box"], c["box"]) <= threshold]
    return kept


def decode(output, scale, pad_x, pad_y, width, height):
    """Mirror of detect.js decode() + unletterbox()."""
    dims = output.shape

    if len(dims) == 3 and dims[2] == 6:          # end-to-end / NMS-free export
        raw = [
            {"cls": int(round(row[5])), "conf": float(row[4]),
             "box": [float(row[0]), float(row[1]), float(row[2]), float(row[3])]}
            for row in output[0] if row[4] >= CONF
        ]
    else:                                         # classic [1, 4+nc, anchors]
        _, channels, anchors = dims
        num_classes = channels - 4
        values = output[0]
        raw = []
        for i in range(anchors):
            scores = values[4:, i]
            best = int(scores.argmax())
            score = float(scores[best])
            if score < CONF:
                continue
            cx, cy, w, h = (float(values[j, i]) for j in range(4))
            raw.append({
                "cls": best, "conf": score,
                "box": [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2],
            })
        raw = nms(raw, IOU)

    out = []
    for d in raw:
        x0, y0, x1, y1 = d["box"]
        box = [
            max(0.0, (x0 - pad_x) / scale),
            max(0.0, (y0 - pad_y) / scale),
            min(float(width), (x1 - pad_x) / scale),
            min(float(height), (y1 - pad_y) / scale),
        ]
        if box[2] - box[0] > 1 and box[3] - box[1] > 1:
            out.append({**d, "box": box})
    return sorted(out, key=lambda d: -d["conf"])


# --------------------------------------------------------------------------------------

FAILURES = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"\n          {detail}" if not ok and detail else ""))
    if not ok:
        FAILURES.append(name)


def smoke_train(dataset: Path, out: Path) -> Path:
    """Train a deliberately worthless model, fast, just to get a real .pt to export.

    Accuracy is irrelevant here — this exists so the export path can be exercised on CPU
    without waiting on a real training run.
    """
    from ultralytics import YOLO
    import random, yaml

    subset = out / "subset"
    if subset.exists():
        shutil.rmtree(subset)

    random.seed(0)
    for split, cap in (("train", 200), ("valid", 60)):
        images = sorted((dataset / split / "images").iterdir())
        random.shuffle(images)
        (subset / split / "images").mkdir(parents=True)
        (subset / split / "labels").mkdir(parents=True)
        for image in images[:cap]:
            shutil.copy2(image, subset / split / "images" / image.name)
            label = dataset / split / "labels" / f"{image.stem}.txt"
            if label.exists():
                shutil.copy2(label, subset / split / "labels" / f"{image.stem}.txt")

    names = yaml.safe_load((dataset / "data.yaml").read_text())["names"]
    (subset / "data.yaml").write_text(
        f"path: {subset.resolve()}\ntrain: train/images\nval: valid/images\n"
        f"nc: {len(names)}\nnames: {names}\n"
    )

    print(f"Smoke-training on {subset} (yolo11n @ 320, 2 epochs — the model will be bad on purpose)\n")
    model = YOLO("yolo11n.pt")
    model.train(
        data=str(subset / "data.yaml"), epochs=2, imgsz=320, batch=8,
        device="cpu", workers=2, cache=False, plots=False, val=False,
        project=str(out), name="smoke", exist_ok=True, verbose=False, seed=0,
    )
    return out / "smoke" / "weights" / "best.pt"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--image", type=Path, help="image to compare predictions on")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--name", default="turbine", help="manifest key to check labels against")
    parser.add_argument("--dataset", type=Path, default=ROOT / "datasets" / "turbine_v2")
    parser.add_argument("--out", type=Path, default=ROOT / "runs" / "contract")
    parser.add_argument("--smoke-train", action="store_true")
    args = parser.parse_args()

    try:
        import numpy as np
        import onnxruntime as ort
        from PIL import Image
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(f"missing dependency ({exc}) — pip install ultralytics onnxruntime")

    args.out.mkdir(parents=True, exist_ok=True)

    weights = args.weights
    if args.smoke_train or weights is None:
        if not (args.dataset / "data.yaml").exists():
            raise SystemExit(f"{args.dataset} not found — run tools/rebuild_turbine.py first")
        weights = smoke_train(args.dataset, args.out)

    print(f"\nWeights: {weights}")
    model = YOLO(str(weights))
    names = [model.names[i] for i in sorted(model.names)]
    print(f"Classes: {names}\n")

    exported = Path(model.export(format="onnx", imgsz=args.imgsz, opset=12,
                                 dynamic=False, simplify=True, nms=False))
    session = ort.InferenceSession(str(exported))
    inp, out = session.get_inputs()[0], session.get_outputs()[0]

    print("Export contract")
    check("input is NCHW float32", list(inp.shape)[:2] == [1, 3] and "float" in inp.type,
          f"got shape {inp.shape} type {inp.type}")
    check(f"input spatial dims are {args.imgsz}", list(inp.shape)[2:] == [args.imgsz, args.imgsz],
          f"got {inp.shape}")

    shape = list(out.shape)
    classic = len(shape) == 3 and shape[1] == 4 + len(names)
    endtoend = len(shape) == 3 and shape[2] == 6
    check("output layout is one detect.js handles", classic or endtoend,
          f"got {shape}; expected [1,{4 + len(names)},anchors] or [1,N,6]")
    if classic:
        check("class-channel count matches the model's classes", shape[1] - 4 == len(names),
              f"{shape[1] - 4} channels vs {len(names)} names")

    manifest_path = ROOT / "web" / "models" / "manifest.json"
    if manifest_path.exists():
        entry = json.loads(manifest_path.read_text()).get(args.name, {})
        check(f"manifest['{args.name}'].labels matches the model", entry.get("labels") == names,
              f"manifest {entry.get('labels')} vs model {names}")
        check(f"manifest['{args.name}'].imgsz matches the export", entry.get("imgsz") == args.imgsz,
              f"manifest {entry.get('imgsz')} vs export {args.imgsz}")

    # ---- the real test -----------------------------------------------------------------
    image_path = args.image
    if image_path is None:
        candidates = sorted((args.dataset / "valid" / "images").iterdir())
        image_path = candidates[0] if candidates else None
    if image_path is None or not Path(image_path).exists():
        print("\nNo image available — skipping the decoder-agreement check.")
        return 1 if FAILURES else 0

    image = Image.open(image_path).convert("RGB")
    tensor, scale, pad_x, pad_y = letterbox(image, args.imgsz)
    onnx_out = session.run(None, {inp.name: tensor})[0]
    ours = decode(np.asarray(onnx_out), scale, pad_x, pad_y, image.width, image.height)

    truth = model.predict(source=str(image_path), imgsz=args.imgsz, conf=CONF, iou=IOU, verbose=False)[0]
    theirs = sorted(
        [{"cls": int(c), "conf": float(f), "box": [float(v) for v in b]}
         for b, c, f in zip(truth.boxes.xyxy.tolist(), truth.boxes.cls.tolist(), truth.boxes.conf.tolist())],
        key=lambda d: -d["conf"],
    )

    print(f"\nDecoder agreement on {Path(image_path).name}")
    print(f"  browser decoder: {len(ours)} detections | ultralytics: {len(theirs)} detections")
    check("same number of detections", len(ours) == len(theirs))

    if len(ours) == len(theirs) and ours:
        worst_box = max(
            max(abs(a - b) for a, b in zip(o["box"], t["box"]))
            for o, t in zip(ours, theirs)
        )
        worst_conf = max(abs(o["conf"] - t["conf"]) for o, t in zip(ours, theirs))
        same_classes = all(o["cls"] == t["cls"] for o, t in zip(ours, theirs))
        # A couple of pixels is resampling and fp32 noise; more means a real logic mismatch.
        check("classes agree", same_classes)
        check("boxes agree within 2 px", worst_box < 2.0, f"largest disagreement {worst_box:.2f} px")
        check("confidences agree within 0.02", worst_conf < 0.02, f"largest disagreement {worst_conf:.4f}")
    elif not ours and not theirs:
        print("  (both found nothing — expected from a smoke-trained model; "
              "re-run with --image on something the model fires on to test box maths)")

    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) failed: {', '.join(FAILURES)}")
        return 1
    print("Export contract holds — the browser decoder matches Ultralytics.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
