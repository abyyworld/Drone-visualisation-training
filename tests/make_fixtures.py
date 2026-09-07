#!/usr/bin/env python3
"""Build tiny ONNX models and images that exercise the web app's real inference path.

These are NOT trained models. They are fixtures with hand-computed outputs, so the browser
test can assert exact boxes, classes and severity scores. That catches the failures that a
"does the page load" smoke test never would - a transposed YOLO output, letterbox padding
applied in the wrong direction, NMS that suppresses nothing, an off-by-one in the class map.

The gate fixture is a real (if trivial) computation over the input: global-average-pool the
three channels and route red -> turbine, blue -> solar, green -> invalid. That makes the
ImageNet normalisation in preprocess.js part of what gets tested, rather than bypassed.

    python3 tests/make_fixtures.py [--out tests/fixtures]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from PIL import Image

OPSET = 12  # what tools/export_onnx.py targets, and what WebGPU needs.


def build_gate(path: Path) -> None:
    """[1,3,224,224] -> [1,3] logits. Routes by dominant colour channel."""
    # Column j of W selects which input channel drives logit j.
    #   logit0 (turbine) <- red, logit1 (solar) <- blue, logit2 (invalid) <- green
    weights = np.array(
        [
            [1.0, 0.0, 0.0],  # red   -> turbine
            [0.0, 0.0, 1.0],  # green -> invalid
            [0.0, 1.0, 0.0],  # blue  -> solar
        ],
        dtype=np.float32,
    ) * 5.0  # scale so softmax is decisively confident

    graph = helper.make_graph(
        nodes=[
            helper.make_node("GlobalAveragePool", ["images"], ["pooled"]),
            helper.make_node("Flatten", ["pooled"], ["flat"], axis=1),
            helper.make_node("MatMul", ["flat", "W"], ["logits"]),
        ],
        name="gate_fixture",
        inputs=[helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 224, 224])],
        outputs=[helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 3])],
        initializer=[numpy_helper.from_array(weights, name="W")],
    )
    save(graph, path)


def build_detector(path: Path, num_classes: int, detections: list, size: int, anchors: int) -> None:
    """[1,3,size,size] -> [1,4+nc,anchors] in the classic YOLOv8/YOLO11 channel-major layout.

    The input is intentionally unused: the point is to assert that the *decoder* handles a
    known tensor correctly, independent of any weights.
    """
    output = np.zeros((1, 4 + num_classes, anchors), dtype=np.float32)
    for index, (cx, cy, w, h, class_id, score) in enumerate(detections):
        output[0, 0, index] = cx
        output[0, 1, index] = cy
        output[0, 2, index] = w
        output[0, 3, index] = h
        output[0, 4 + class_id, index] = score

    graph = helper.make_graph(
        nodes=[
            helper.make_node(
                "Constant",
                inputs=[],
                outputs=["output0"],
                value=numpy_helper.from_array(output, name="const_out"),
            )
        ],
        name="detector_fixture",
        inputs=[helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, size, size])],
        outputs=[
            helper.make_tensor_value_info(
                "output0", TensorProto.FLOAT, [1, 4 + num_classes, anchors]
            )
        ],
    )
    save(graph, path)


def save(graph, path: Path) -> None:
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", OPSET)], producer_name="fixtures"
    )
    model.ir_version = 9  # onnxruntime-web 1.23 does not accept IR 10+
    onnx.checker.check_model(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, path)
    print(f"  {path.name:16s} {path.stat().st_size:>7,} bytes")


def solid(path: Path, rgb: tuple[int, int, int], size: int = 960) -> None:
    Image.new("RGB", (size, size), rgb).save(path)
    print(f"  {path.name:16s} {size}x{size} rgb{rgb}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("tests/fixtures"))
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    print("models:")
    build_gate(out / "gate.onnx")

    # Turbine: corrosion, crack, and a third box that overlaps the first and must be
    # suppressed by NMS. Values chosen so the expected result is calculable by hand.
    build_detector(
        out / "turbine.onnx",
        num_classes=3,
        size=960,
        anchors=8,
        detections=[
            (480, 480, 200, 100, 0, 0.90),  # corrosion -> (380,430,580,530)
            (200, 300, 60, 40, 1, 0.80),    # crack     -> (170,280,230,320)
            (485, 482, 200, 100, 0, 0.70),  # near-duplicate of the first, NMS should drop it
        ],
    )

    # Solar: soiling plus a missing module, to check the severity weighting differs by class.
    build_detector(
        out / "solar.onnx",
        num_classes=6,
        size=960,
        anchors=4,
        detections=[
            (300, 300, 400, 400, 0, 0.85),  # soiling
            (700, 700, 100, 100, 5, 0.75),  # missing_module
        ],
    )

    print("\nimages:")
    solid(out / "turbine_red.png", (255, 0, 0))
    solid(out / "solar_blue.png", (0, 0, 255))
    solid(out / "invalid_green.png", (0, 255, 0))
    (out / "not_an_image.txt").write_text("this is not an image\n")
    print(f"  {'not_an_image.txt':16s} plain text")

    print(f"\nFixtures written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
