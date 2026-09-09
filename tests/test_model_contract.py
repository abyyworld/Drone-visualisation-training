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
import sys

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


def box_scale(head, anchors, channels, size, per_anchor=False):
    """The rule in NativeDetector.calibrate(), written once so the two cannot drift."""
    largest = max(
        head[anchor * channels + c] if per_anchor else head[c * anchors + anchor]
        for anchor in range(anchors) for c in (0, 1)
    )
    return size if largest <= size / 8 else 1


def _probe(session_or_interpreter, size, tflite):
    """One flat grey frame, which is what the app calibrates on."""
    import numpy as np

    grey = np.full((1, 3, size, size), 114 / 255, dtype=np.float32)
    if tflite:
        entry = session_or_interpreter.get_input_details()[0]
        session_or_interpreter.set_tensor(entry["index"], grey.astype(entry["dtype"]))
        session_or_interpreter.invoke()
        exit_ = session_or_interpreter.get_output_details()[0]
        return session_or_interpreter.get_tensor(exit_["index"])[0]
    name = session_or_interpreter.get_inputs()[0].name
    return session_or_interpreter.run(None, {name: grey})[0][0]


def test_the_two_exports_disagree_about_box_units_and_both_are_handled():
    """
    The same weights, exported twice, do not use the same units for a box.

    The ONNX gives centres and sizes in pixels of the model's own square; the TFLite
    converter divides them by that square and gives fractions. Nothing in either file says
    so and the shapes are identical, so reading one as the other is not an error: every box
    comes out a fraction of a pixel across in the corner, the screen shows nothing, and the
    detector cheerfully reports a hundred people. That shipped once.

    The app measures it rather than assuming it. This checks the measurement lands on the
    right answer for both files, which is the part that would break silently.
    """
    ort = pytest.importorskip("onnxruntime")
    litert = pytest.importorskip("ai_edge_litert.interpreter")
    if not (ONNX.exists() and TFLITE.exists()):
        pytest.skip("both exports are needed to compare them")

    spec = json.loads(SPEC.read_text())
    size = spec["imgsz"]
    channels = 4 + len(spec["labels"])

    session = ort.InferenceSession(str(ONNX), providers=["CPUExecutionProvider"])
    onnx_head = _probe(session, size, tflite=False)
    anchors = onnx_head.shape[1]
    assert box_scale(onnx_head.reshape(-1), anchors, channels, size) == 1, (
        "the ONNX export used to give boxes in pixels; it no longer does, and the web app "
        "reads it as pixels"
    )

    interpreter = litert.Interpreter(model_path=str(TFLITE))
    interpreter.allocate_tensors()
    tflite_head = _probe(interpreter, size, tflite=True)
    assert box_scale(tflite_head.reshape(-1), anchors, channels, size) == size, (
        "the TFLite export used to give boxes as fractions of the model square; it no "
        "longer does, and the tablet would multiply them by 320 a second time"
    )


# ---------------------------------------------------------------------------------------
# Every detector the web app loads, not just the person one
# ---------------------------------------------------------------------------------------
#
# The test above pins one model. The wildfire model is what showed why that is not enough:
# it was fetched, converted, committed and described in the manifest without anything ever
# comparing the two, and the file that arrived is a segmentation export. Its head carries
# 4 box numbers, 1 class score and 32 mask coefficients, so 37 channels where the manifest
# says 1 label. web/js/detect.js checks the same thing and throws, which means wildfire
# analysis raised on every frame rather than marking anything.
#
# Read as class scores those 32 coefficients are not scores at all: they run from about
# -2.2 to +1.6, so on 60 real drone frames with no fire in any of them every single frame
# produced a box, at up to 1.81 "confidence". The only thing standing between that and the
# screen is keepClasses happening to select the one real channel.
#
# So this checks every detector in the manifest, not the one somebody remembered.

MANIFEST = MODELS / "manifest.json"


def shipped_detectors():
    """Each detect model in the manifest whose ONNX file is actually present."""
    if not MANIFEST.exists():
        return []
    manifest = json.loads(MANIFEST.read_text())
    found = []
    for name, entry in manifest.items():
        if not isinstance(entry, dict) or entry.get("task") != "detect":
            continue
        if not entry.get("labels"):
            # A subject with no model, described as having none. Nothing to check.
            continue
        path = MODELS / entry.get("file", "")
        # ONNX only. The tablet's own model is TFLite and is pinned by the tests above,
        # which read it with the runtime that actually loads it.
        if path.suffix == ".onnx" and path.exists():
            found.append((name, entry, path))
    return found


@pytest.mark.parametrize("name,entry,path", shipped_detectors(),
                         ids=lambda v: v if isinstance(v, str) else "")
def test_each_shipped_detector_has_one_channel_per_label(name, entry, path):
    ort = pytest.importorskip("onnxruntime")
    import numpy as np

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    size = entry["imgsz"]
    out = session.run(None, {session.get_inputs()[0].name:
                             np.zeros((1, 3, size, size), dtype=np.float32)})[0]

    assert out.ndim == 3, f"{name}: the app cannot decode an output of shape {out.shape}"
    channels = min(out.shape[1], out.shape[2])
    labels = entry["labels"]
    # A segmentation export carries mask coefficients after the class scores. The manifest
    # has to declare how many, and web/js/detect.js drops exactly that many. An undeclared
    # extra channel is still a failure: it is as likely to be the wrong file as anything.
    coefficients = entry.get("maskCoefficients", 0)
    assert channels - coefficients == 4 + len(labels), (
        f"{name}: {path.name} predicts {channels - coefficients - 4} classes but the "
        f"manifest lists {len(labels)} labels ({labels}). detect.js throws on exactly "
        f"this, so the subject raises on every frame instead of marking anything."
    )

    if coefficients:
        # The declared coefficients have to actually be coefficients rather than a label
        # list somebody trimmed. A segmentation export emits its mask prototypes as a
        # second output, and its width is the number of coefficients per box.
        others = [o.shape for o in session.get_outputs()[1:]]
        assert any(len(shape) == 4 and shape[1] == coefficients for shape in others), (
            f"{name}: the manifest declares {coefficients} mask coefficients, but no "
            f"prototype output of that width is present. Outputs: {others}"
        )


@pytest.mark.parametrize("name,entry,path", shipped_detectors(),
                         ids=lambda v: v if isinstance(v, str) else "")
def test_each_shipped_detector_reads_its_input(name, entry, path, tmp_path_factory):
    """A model that answers the same for every picture is not quiet, it is dead.

    This is the check that was missing when a fire model was fetched, converted,
    committed and described, and returns a score between 0.0123 and 0.0155 for black,
    white, noise, a flame-coloured block, sixty real drone photographs and fifty frames of
    the fire clip alike. Everything else passed: the graph ran, the output was a shape the
    app decodes, the head matched the label list.

    The judgement is tools/model_liveness.py, which is the same code the conversion
    workflow gates on, so a model cannot pass on the way in and fail here or the reverse.
    Read that file for why it asks about responsiveness rather than about accuracy, and
    for the measurements behind the floor.
    """
    ort = pytest.importorskip("onnxruntime")
    pytest.importorskip("PIL")
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "tools"))
    import model_liveness

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    cache = tmp_path_factory.mktemp("liveness")
    spread, named, enough = model_liveness.judge(
        session, len(entry["labels"]), entry["imgsz"], cache)
    if not enough:
        pytest.skip("no photograph could be fetched, and generated pictures alone cannot "
                    "tell a quiet model from a dead one")

    assert spread >= model_liveness.FLOOR, (
        f"{name}: {path.name} scores "
        f"{', '.join(f'{n} {v:.4f}' for n, v in named)} - a spread of {spread:.5f}. "
        f"It is not reading its input, so it would ship as a model that marks nothing "
        f"while the app reports it as working."
    )
