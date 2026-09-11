"""Reading a fire dataset's labels, which is where this measurement can go backwards.

WHY THIS IS TESTED AND THE REST OF THE FETCHING IS NOT
    Everything else in tools/fetch_fire_pictures.py is network and cannot run here. This
    part is pure logic, and it is the part that decides which pile each picture lands in.
    Get it wrong and recall is not merely inaccurate, it is inverted: the fire set fills
    with ordinary photographs and the model appears to miss everything, or the reverse and
    it appears to be perfect. Either reads exactly like a measurement.

    Two traps, both real:

    Negation. "no_fire", "nonfire" and "no_smoke" mean the opposite of the word inside
    them, and a plain substring test for "no" also matches "normal" - which would file
    every ordinary picture in the dataset as a fire.

    Bare numbers. A label of 0 means fire in some published sets and no fire in others,
    and the number carries nothing that says which. It is refused unless the dataset
    publishes names for its classes, because a coin flip here swaps the two sets.
"""
import importlib.util
import pathlib

import pytest

TOOL = pathlib.Path(__file__).resolve().parent.parent / "tools" / "fetch_fire_pictures.py"


@pytest.fixture(scope="module")
def fetcher():
    spec = importlib.util.spec_from_file_location("fetch_fire_pictures", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("label", [
    "fire", "smoke", "flame", "wildfire", "Fire", "forest fire", "fire_smoke",
    "fireorsmoke", "fire and smoke",
])
def test_a_label_naming_fire_means_fire(fetcher, label):
    assert fetcher.classify(label, None) == "fire"


@pytest.mark.parametrize("label", [
    "no_fire", "nofire", "non-fire", "nonfire", "no fire", "no_smoke", "without_fire",
])
def test_a_negated_label_means_the_opposite_of_the_word_inside_it(fetcher, label):
    assert fetcher.classify(label, None) == "clear"


@pytest.mark.parametrize("label", ["normal", "neutral", "background", "negative", "none"])
def test_an_ordinary_picture_is_not_read_as_a_negation(fetcher, label):
    # "normal" begins with "no" and must not be read as negating anything, which is what a
    # substring test for "no" would do.
    assert fetcher.classify(label, None) == "clear"


@pytest.mark.parametrize("label", [0, 1, 7, -1])
def test_a_bare_number_is_refused(fetcher, label):
    # 0 is fire in some published sets and no fire in others. Guessing swaps the two sets
    # and produces a recall figure that is backwards while looking like a measurement.
    assert fetcher.classify(label, None) is None


def test_a_number_is_read_only_through_the_names_the_dataset_publishes(fetcher):
    assert fetcher.classify(0, ["fire", "no_fire"]) == "fire"
    assert fetcher.classify(1, ["fire", "no_fire"]) == "clear"
    # The same number, the other way round, and the answer has to follow the names.
    assert fetcher.classify(0, ["no_fire", "fire"]) == "clear"
    assert fetcher.classify(1, ["no_fire", "fire"]) == "fire"
    # Out of range is not an index into anything.
    assert fetcher.classify(5, ["fire", "no_fire"]) is None


def test_true_and_false_are_unambiguous_where_nought_and_one_are_not(fetcher):
    assert fetcher.classify(True, None) == "fire"
    assert fetcher.classify(False, None) == "clear"


@pytest.mark.parametrize("label", ["cat", "street", "", "   ", None, 3.5, ["fire"]])
def test_anything_it_cannot_read_is_left_out_rather_than_guessed(fetcher, label):
    assert fetcher.classify(label, None) is None


# ---------------------------------------------------------------------------------------
# The other two shapes a published fire dataset comes in
# ---------------------------------------------------------------------------------------
#
# The first version of the fetcher read only a label column, and the run that followed
# rejected two perfectly usable datasets on its way past: one carried its labels as
# detection boxes, the other as a boolean saying "this is a counterexample". The column
# lists below are the real ones those datasets reported.

NAMES = ["fire", "smoke"]


@pytest.mark.parametrize("objects,names,expected", [
    # hiennguyen9874/fire-smoke-detection: image_id, image, width, height, objects
    ({"bbox": [[0, 0, 1, 1], [2, 2, 3, 3]], "category": [0, 1]}, NAMES, "fire"),
    ([{"bbox": [0, 0, 1, 1], "category": 0}], NAMES, "fire"),
    # No boxes is not missing data. It is what a detection set's empty frames are for.
    ({"bbox": [], "category": []}, NAMES, "clear"),
    ([], NAMES, "clear"),
    ({"category": [0]}, ["no_fire"], "clear"),
])
def test_boxes_say_whether_a_picture_has_fire_in_it(fetcher, objects, names, expected):
    assert fetcher.verdict_from_objects(objects, names) == expected


@pytest.mark.parametrize("objects,names", [
    # Categories with nothing to turn them into words: the same refusal as a bare label.
    ({"category": [0, 1]}, None),
    ({"category": [0]}, ["car"]),
    ("nonsense", NAMES),
    (None, NAMES),
    ({"bbox": [[0, 0, 1, 1]]}, NAMES),      # boxes but no categories at all
])
def test_boxes_it_cannot_read_are_refused(fetcher, objects, names):
    assert fetcher.verdict_from_objects(objects, names) is None


def test_a_negative_flag_means_a_counterexample(fetcher):
    # fireviewer/fire-smoke-detection-corpus-v1 carries a boolean "negative" column.
    keys = (None, None, "negative")
    assert fetcher.verdict_for_row({"negative": True}, keys, None) == "clear"
    assert fetcher.verdict_for_row({"negative": False}, keys, None) == "fire"
    assert fetcher.verdict_for_row({"negative": "maybe"}, keys, None) is None
    assert fetcher.verdict_for_row({}, keys, None) is None


def test_an_unreadable_label_falls_through_to_the_next_shape(fetcher):
    # A bare 0 says nothing, but the boxes beside it do, so the row is still usable.
    row = {"label": 0, "objects": {"category": [0]}}
    assert fetcher.verdict_for_row(row, ("label", "objects", None), ["fire"]) == "fire"
    # And when nothing is readable, the row is left out rather than guessed at.
    assert fetcher.verdict_for_row({"label": 0}, ("label", None, None), None) is None
