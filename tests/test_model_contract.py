"""
The shipped detector against what the app believes about it.

WHY THIS EXISTS
    NativeDetector.java reads the model's own tensors at startup rather than being told
    their shape, and everything it then does follows from that reading. When the reading is
    wrong nothing raises: the model is handed a picture it has never seen and reports an
    empty scene, which looks exactly like a frame with nobody in it. That has happened once
    already, because the app took the input size from the second dimension - correct for
    almost every TFLite vision model, and wrong for the converter that produced this one,
    which emits [1, 3, 320, 320]. The size came out as three.

    So the rules the Java uses are written out here and checked against the file that ships.

    The conversion workflow makes the same checks with a runtime present. This one runs
    wherever a runtime happens to be installed, so a model swapped in by hand is caught too.
"""
import json
import pathlib

import pytest

MODELS = pathlib.Path(__file__).resolve().parent.parent / "web" / "models"
SPEC = MODELS / "person-320.json"
TFLITE = MODELS / "person-320-int8.tflite"
ONNX = MODELS / "person-320.onnx"


@pytest.fixture(scope="module")
def spec():
    if not SPEC.exists():
        pytest.skip("person-320.json is absent; the conversion workflow has not run here")
    return json.loads(SPEC.read_text())


def input_size(shape):
    """The rule in NativeDetector.java, written once so the two cannot drift apart."""
    planar = shape[1] == 3 and shape[3] != 3
    return shape[2] if planar else shape[1]


def test_the_class_list_is_self_consistent(spec):
    labels = spec["labels"]
    assert labels, "a detector with no classes cannot report anything"
    assert spec["keepClasses"], "nothing is kept, so nothing would ever be marked"
    for kept in spec["keepClasses"]:
        assert 0 <= kept < len(labels), f"keepClasses {kept} is outside {len(labels)} labels"


def test_the_kept_classes_are_the_person_ones(spec):
    # Crowd mode is about people. If a conversion ever pointed these at vehicles the app
    # would carry on quite happily, marking cars and tallying them.
    kept = {spec["labels"][i].lower() for i in spec["keepClasses"]}
    assert kept <= {"person", "pedestrian", "people"}, f"kept classes are not people: {kept}"


@pytest.mark.parametrize("model", [TFLITE, ONNX], ids=["tflite", "onnx"])
def test_the_app_would_read_this_models_input_correctly(spec, model):
    if not model.exists():
        pytest.skip(f"{model.name} is not present")

    if model.suffix == ".onnx":
        ort = pytest.importorskip("onnxruntime")
        session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
        shape = [int(v) for v in session.get_inputs()[0].shape]
    else:
        litert = pytest.importorskip("ai_edge_litert.interpreter")
        interpreter = litert.Interpreter(model_path=str(model))
        interpreter.allocate_tensors()
        shape = [int(v) for v in interpreter.get_input_details()[0]["shape"]]

    assert len(shape) == 4, f"the app can only read a four dimensional input: {shape}"
    assert 3 in (shape[1], shape[3]), f"neither dimension is the colour channels: {shape}"
    assert input_size(shape) == spec["imgsz"], (
        f"the app would read an input size of {input_size(shape)} from {shape}, "
        f"but person-320.json says {spec['imgsz']}"
    )


def test_the_head_carries_one_channel_per_label(spec):
    ort = pytest.importorskip("onnxruntime")
    if not ONNX.exists():
        pytest.skip("person-320.onnx is not present")

    import numpy as np

    session = ort.InferenceSession(str(ONNX), providers=["CPUExecutionProvider"])
    size = spec["imgsz"]
    blank = np.zeros((1, 3, size, size), dtype=np.float32)
    out = session.run(None, {session.get_inputs()[0].name: blank})[0]

    assert out.ndim == 3, f"the app cannot decode an output of shape {out.shape}"
    # Four box numbers and then one score per class. Channels is the small dimension;
    # anchors runs to thousands.
    channels = min(out.shape[1], out.shape[2])
    assert channels == 4 + len(spec["labels"]), (
        f"the head carries {channels} channels, so four plus {channels - 4} classes, "
        f"but person-320.json lists {len(spec['labels'])} labels"
    )
