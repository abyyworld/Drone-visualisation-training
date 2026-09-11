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
