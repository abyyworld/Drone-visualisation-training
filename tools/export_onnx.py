#!/usr/bin/env python3
"""Export a trained model to ONNX for the browser app, and register it in the manifest.

The web app downloads these files to every visitor, so size is a hard constraint, not a
preference. A yolo11s export is roughly 38 MB at fp32 and roughly 10 MB after dynamic int8
quantisation.

Quantisation is on by default because a 38 MB download is a bad first visit, but it is
**not free**: dynamic int8 typically costs a couple of points of mAP on a detection head,
occasionally more on small objects. Measure it rather than assuming:

    python3 tools/evaluate.py runs/turbine/weights/best.pt   --data datasets/turbine_v2/data.yaml
    python3 tools/evaluate.py web/models/turbine.onnx        --data datasets/turbine_v2/data.yaml

If the drop is unacceptable, re-export with --no-quantize and accept the larger download.

Usage:
    python3 tools/export_onnx.py runs/turbine/weights/best.pt --name turbine --imgsz 960
    python3 tools/export_onnx.py gate.onnx --name gate --quantize-only

Requires `pip install ultralytics onnx onnxruntime` (the first pulls in torch).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

WEB_MODELS = Path("web/models")

# opset 12 is what onnxruntime-web's WebGPU backend is happiest with. Newer opsets export
# fine and then silently fall back to WASM in the browser, which is 5-10x slower.
OPSET = 12


def export_ultralytics(weights: Path, imgsz: int) -> tuple[Path, list[str]]:
    """Export an Ultralytics checkpoint, returning the .onnx path and its class names."""
    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("ultralytics is not installed — run `pip install ultralytics`")

    model = YOLO(str(weights))
    names = [model.names[i] for i in sorted(model.names)]

    print(f"Exporting {weights} at imgsz={imgsz}, opset={OPSET}")
    exported = model.export(
        format="onnx",
        imgsz=imgsz,
        opset=OPSET,
        # Static shapes and no built-in NMS: the browser does its own letterboxing and
        # suppression (web/js/detect.js), and a dynamic graph is markedly slower under WASM.
        dynamic=False,
        simplify=True,
        nms=False,
    )
    return Path(exported), names


def quantize(source: Path, target: Path) -> None:
    """Dynamic int8 quantisation — weights only, no calibration dataset needed."""
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except ImportError:
        raise SystemExit("onnxruntime is not installed — run `pip install onnxruntime`")

    quantize_dynamic(
        model_input=str(source),
        model_output=str(target),
        weight_type=QuantType.QUInt8,
    )


def update_manifest(name: str, filename: str, imgsz: int, labels: list[str]) -> None:
    """Keep models/manifest.json in step with what was actually exported.

    Class order and image size must match the export exactly — the web app trusts the
    manifest, so a stale entry mislabels every detection rather than failing loudly.
    """
    path = WEB_MODELS / "manifest.json"
    manifest = json.loads(path.read_text()) if path.exists() else {}

    entry = manifest.get(name, {})
    previous = entry.get("labels")
    entry.update({"file": filename, "imgsz": imgsz, "labels": labels})
    entry.setdefault("task", "classify" if name == "gate" else "detect")
    if entry["task"] == "detect":
        entry.setdefault("confThreshold", 0.25)
        entry.setdefault("iouThreshold", 0.45)
        # Unweighted classes score 1.0 each; tune these by hand afterwards.
        weights = entry.setdefault("severityWeights", {})
        for label in labels:
            weights.setdefault(label, 1.0)
        for stale in set(weights) - set(labels):
            del weights[stale]

    manifest[name] = entry
    path.write_text(json.dumps(manifest, indent=2) + "\n")

    if previous and previous != labels:
        print(f"  ! class list changed: {previous} -> {labels}")
    print(f"  manifest updated: {name} -> {filename}, imgsz {imgsz}, {len(labels)} classes")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("weights", type=Path, help="Ultralytics .pt, or an .onnx with --quantize-only")
    parser.add_argument("--name", required=True, choices=["turbine", "solar", "gate"])
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--no-quantize", action="store_true", help="keep fp32 (bigger, more accurate)")
    parser.add_argument("--quantize-only", action="store_true", help="input is already ONNX")
    parser.add_argument(
        "--labels", help="comma-separated class names; required when the source is a bare .onnx"
    )
    args = parser.parse_args()

    if not args.weights.exists():
        raise SystemExit(f"not found: {args.weights}")
    WEB_MODELS.mkdir(parents=True, exist_ok=True)

    if args.quantize_only:
        source = args.weights
        if not args.labels:
            raise SystemExit("--labels is required with --quantize-only")
        labels = [name.strip() for name in args.labels.split(",") if name.strip()]
    else:
        source, labels = export_ultralytics(args.weights, args.imgsz)

    target = WEB_MODELS / f"{args.name}.onnx"

    if args.no_quantize:
        shutil.copy2(source, target)
    else:
        print("Quantising to int8 (compare mAP against the fp32 model before shipping)")
        quantize(source, target)

    size_mb = target.stat().st_size / 1e6
    print(f"\n  {target}  {size_mb:.1f} MB")
    if size_mb > 15:
        print("  ! over 15 MB — that is a slow first visit. Consider a smaller model "
              "(yolo11n), a lower imgsz, or quantisation if you disabled it.")

    update_manifest(args.name, target.name, args.imgsz, labels)
    print("\nDeployed. Commit web/models/ and the site will serve it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
