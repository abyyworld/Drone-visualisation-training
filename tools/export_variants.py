#!/usr/bin/env python3
"""Export several ONNX variants of a detector, measure each, and keep the best.

WHAT THIS WAS BUILT TO TEST, AND WHY THAT HYPOTHESIS WAS WRONG
    The shipped export was int8-quantised and measured well below the .pt it came from:

        best.pt        mAP50 0.759   mAP50-95 0.478
        int8 (all)     mAP50 0.667   mAP50-95 0.239

    mAP50 lost 12% while mAP50-95 lost 50%. Since mAP50 only asks whether a box overlaps
    the truth at all and mAP50-95 averages over tight IoU thresholds, that gap reads as
    "detections found, boxes badly placed" - and quantising the detection head's coordinate
    regression to 8 bits is a textbook cause. 25 of the 88 quantised convolutions sit in
    that head, which made the story fit.

    Running it settled the question the other way:

        fp32 (no quantisation)      38.1 MB   mAP50 0.643   mAP50-95 0.223
        int8 everywhere             10.1 MB   mAP50 0.664   mAP50-95 0.238
        int8 backbone, fp32 head    12.5 MB   mAP50 0.656   mAP50-95 0.228

    Full precision is no better than int8 - marginally worse, within noise. Quantisation
    costs essentially nothing. The loss is in the PyTorch-to-ONNX step itself and applies
    to every variant equally.

    A difference that lands identically on all three points at what they share: the export
    and the ONNX validation path, not the weights. The leading suspect is preprocessing
    rather than model damage - `val()` on a .pt uses rectangular inference by default,
    while a static ONNX export forces every image into a square 960x960 letterbox, which
    shrinks small defects relative to the frame. That is a hypothesis, not a finding; it
    would be tested by evaluating the .pt with rectangular inference disabled and seeing
    whether it drops to meet the ONNX numbers.

    Kept as a tool because the measurement is the point. The head-exclusion variant it was
    written to prove is now just one of three options it prices honestly.

Usage:
    python3 tools/export_variants.py runs/turbine_v2/weights/best.pt \
        --data datasets/turbine_v2/data.yaml --name turbine --imgsz 960

Requires `pip install ultralytics onnx onnxruntime`.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

OPSET = 12

# Ultralytics names ONNX nodes after the module that produced them, so the Detect head is
# the highest /model.N/ block. Derived rather than hardcoded: the index moves with the
# architecture (23 for yolo11s, different for other sizes).
HEAD_PATTERN = re.compile(r"/model\.(\d+)/")


def head_nodes(model_path: Path) -> list[str]:
    import onnx

    graph = onnx.load(str(model_path)).graph
    blocks = {int(m.group(1)) for n in graph.node if (m := HEAD_PATTERN.search(n.name))}
    if not blocks:
        return []
    last = max(blocks)
    return [n.name for n in graph.node if f"/model.{last}/" in n.name]


def quantize(source: Path, target: Path, exclude: list[str]) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(
        model_input=str(source), model_output=str(target),
        weight_type=QuantType.QUInt8, nodes_to_exclude=exclude,
    )


def evaluate(model: Path, data: Path, imgsz: int) -> dict:
    from ultralytics import YOLO

    # device=cpu: Ultralytics pulls in onnxruntime-gpu, which cannot load Kaggle's CUDA and
    # fails while binding the input rather than with a clear message.
    result = YOLO(str(model)).val(data=str(data), split="test", imgsz=imgsz,
                                  device="cpu", plots=False, verbose=False)
    box = result.box
    return {
        "mAP50": float(box.map50),
        "mAP50_95": float(box.map),
        "per_class": {
            result.names[c]: round(float(box.ap50[i]), 4)
            for i, c in enumerate(box.ap_class_index)
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("weights", type=Path, help="trained .pt")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--name", default="turbine")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--out", type=Path, default=Path("web/models"))
    parser.add_argument("--work", type=Path, default=Path("/kaggle/working/onnx_variants"))
    parser.add_argument("--tolerance", type=float, default=0.05,
                        help="keep the smallest variant within this relative mAP50-95 of "
                             "the full-precision export (default 5%%)")
    args = parser.parse_args()

    if not args.weights.exists():
        raise SystemExit(f"not found: {args.weights}")

    from ultralytics import YOLO

    args.work.mkdir(parents=True, exist_ok=True)

    # Ultralytics writes the .onnx beside the input .pt, and Kaggle mounts /kaggle/input
    # read-only - so exporting straight from an attached notebook output dies with
    # "[Errno 30] Read-only file system" after doing all the work. Copy somewhere writable
    # first; it costs one file copy and removes the whole class of problem.
    weights = args.work / args.weights.name
    if args.weights.resolve() != weights.resolve():
        shutil.copy(args.weights, weights)

    print(f"Exporting {args.weights} at imgsz={args.imgsz}, opset={OPSET}\n")
    exported = Path(YOLO(str(weights)).export(
        format="onnx", imgsz=args.imgsz, opset=OPSET, simplify=True, dynamic=False))

    fp32 = args.work / f"{args.name}_fp32.onnx"
    shutil.move(str(exported), fp32)

    excluded = head_nodes(fp32)
    print(f"detection head: {len(excluded)} nodes held at full precision in the mixed variant\n")

    int8_all = args.work / f"{args.name}_int8_all.onnx"
    quantize(fp32, int8_all, exclude=[])

    variants = [("fp32 (no quantisation)", fp32), ("int8 everywhere", int8_all)]
    if excluded:
        mixed = args.work / f"{args.name}_int8_body.onnx"
        quantize(fp32, mixed, exclude=excluded)
        variants.append(("int8 backbone, fp32 head", mixed))

    results = []
    for label, path in variants:
        size = path.stat().st_size / 1e6
        print(f"Evaluating {label} ({size:.1f} MB) ...")
        scores = evaluate(path, args.data, args.imgsz)
        results.append({"label": label, "path": path, "size_mb": size, **scores})

    header = f"  {'variant':<28}{'MB':>7}{'mAP50':>9}{'mAP50-95':>11}   per class"
    print("\n" + header)
    print("  " + "-" * (len(header) + 22))
    for r in results:
        detail = ", ".join(f"{k} {v:.3f}" for k, v in r["per_class"].items())
        print(f"  {r['label']:<28}{r['size_mb']:>7.1f}{r['mAP50']:>9.3f}"
              f"{r['mAP50_95']:>11.3f}   {detail}")

    reference = results[0]["mAP50_95"]
    acceptable = [r for r in results
                  if reference == 0 or (reference - r["mAP50_95"]) / reference <= args.tolerance]
    chosen = min(acceptable, key=lambda r: r["size_mb"]) if acceptable else results[0]

    loss = 0.0 if reference == 0 else (reference - chosen["mAP50_95"]) / reference * 100
    print(f"\n  Chosen: {chosen['label']} - smallest variant within {args.tolerance:.0%} of "
          f"full precision\n  ({loss:.1f}% of mAP50-95 given up for "
          f"{results[0]['size_mb'] / chosen['size_mb']:.1f}x smaller download)")

    args.out.mkdir(parents=True, exist_ok=True)
    target = args.out / f"{args.name}.onnx"
    shutil.copy(chosen["path"], target)

    manifest_path = args.out / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest[args.name]["imgsz"] = args.imgsz
    manifest[args.name]["labels"] = list(YOLO(str(weights)).names.values())
    manifest_path.write_text(json.dumps(manifest, indent=2))

    report = {"chosen": chosen["label"], "imgsz": args.imgsz,
              "variants": [{k: v for k, v in r.items() if k != "path"} for r in results]}
    Path("docs").mkdir(exist_ok=True)
    Path(f"docs/export-variants-{args.name}.json").write_text(json.dumps(report, indent=2))

    print(f"\n  {target}  ({chosen['size_mb']:.1f} MB)")
    print(f"  docs/export-variants-{args.name}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
