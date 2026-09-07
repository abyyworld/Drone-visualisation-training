"""The `person` class, and the rules that attach only to it.

Person detection from a drone has a different error profile from fire and
smoke, and the system is supposed to respect that rather than treat it as a
third label. At altitude a person is a handful of pixels, is routinely occluded
by canopy, smoke or terrain, and is easily confused with a rock or a stump.
Misses are the normal case.

So these tests check the consequences of that, not the label:

* the temporal filter confirms person sooner, because a track that never
  confirms is a person never drawn;
* nothing in the interface may render an empty result as an absence of people;
* the overlay never lets a life-safety class fall through to the anonymous
  unknown-class style.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from station.core.config import TemporalConfig
from station.core.safety import find_forbidden_phrases
from station.core.types import (
    CLASS_FIRE,
    CLASS_PERSON,
    CLASSES,
    LIFE_SAFETY_CLASSES,
    WIRE_VERSION,
    BBox,
    Detection,
    FrameDetections,
    parse_message,
)
from station.inference.temporal import TemporalFilter

REPO = Path(__file__).resolve().parent.parent


class TestTheContract:
    def test_person_is_appended_so_earlier_indices_survive(self):
        # Appending, not inserting: weights and incident logs written before
        # this class existed still decode to the same classes.
        assert CLASSES.index(CLASS_FIRE) == 0
        assert CLASSES.index(CLASS_PERSON) == len(CLASSES) - 1

    def test_person_is_marked_as_life_safety(self):
        assert CLASS_PERSON in LIFE_SAFETY_CLASSES
        assert CLASS_FIRE not in LIFE_SAFETY_CLASSES

    def test_a_person_detection_round_trips(self):
        det = Detection(cls=CLASS_PERSON, conf=0.41, box=BBox(0.5, 0.5, 0.52, 0.55), persisted=2)
        frame = FrameDetections(frame_id=7, pts=1.4, detections=(det,))
        back = parse_message(frame.to_json())
        assert back.detections[0].cls == CLASS_PERSON
        assert back.detections[0].conf == pytest.approx(0.41)

    def test_the_version_was_bumped_for_this_class(self):
        # A v1 tablet would have parsed a person payload and drawn it in the
        # unknown-class fallback style. Being silently mislabelled is worse for
        # this class than the tablet refusing to connect.
        assert WIRE_VERSION >= 2


class TestTheTemporalWindowIsLooserForPeople:
    def test_person_confirms_sooner_than_fire(self):
        cfg = TemporalConfig()
        assert cfg.per_class.get("person", {}).get("n", cfg.n) < cfg.n

    def test_a_flickering_person_is_still_drawn(self):
        # Seen, missed, seen. Under the global 3-of-5 this track would not yet
        # be confirmed; under person's 2-of-6 it is, which is the whole point.
        filt = TemporalFilter(TemporalConfig())
        box = BBox(0.40, 0.40, 0.43, 0.47)
        emitted = []
        for i, seen in enumerate((True, False, True)):
            dets = [Detection(cls=CLASS_PERSON, conf=0.5, box=box)] if seen else []
            emitted.append(len(filt.update(dets, pts=i * 0.1)))
        assert emitted == [0, 0, 1]

    def test_the_same_sequence_does_not_confirm_a_fire(self):
        filt = TemporalFilter(TemporalConfig())
        box = BBox(0.40, 0.40, 0.43, 0.47)
        emitted = []
        for i, seen in enumerate((True, False, True)):
            dets = [Detection(cls=CLASS_FIRE, conf=0.5, box=box)] if seen else []
            emitted.append(len(filt.update(dets, pts=i * 0.1)))
        assert emitted == [0, 0, 0]

    def test_an_override_that_cannot_be_satisfied_fails_at_startup(self):
        # On the bench, not mid-incident.
        cfg = TemporalConfig(per_class={"person": {"n": 9, "m": 4}})
        with pytest.raises(ValueError, match="per_class"):
            TemporalFilter(cfg)


class TestNothingClaimsAnAbsenceOfPeople:
    @pytest.mark.parametrize(
        "text",
        [
            "No one detected",
            "zero casualties found",
            "0 people found",
            "Nobody present",
            "The building is empty",
            "Sector evacuated",
            "area is clear of people",
            "All personnel accounted for",
            "Search complete",
            "counts the occupants in frame",
        ],
    )
    def test_absence_of_people_phrasings_are_refused(self, text):
        assert find_forbidden_phrases(text), f"{text!r} should be refused"

    def test_reporting_what_was_found_is_still_allowed(self):
        # The rule bans claims about absence, not the reporting of presence.
        for ok in ("PERSON 0.62", "3 person detections this frame", "person track 7 confirmed"):
            assert not find_forbidden_phrases(ok), f"{ok!r} should be allowed"


class TestTheOverlayNeverAnonymisesAPerson:
    def test_every_class_has_its_own_drawing_rule(self):
        overlay = (REPO / "app" / "js" / "overlay.js").read_text(encoding="utf-8")
        styled = set(re.findall(r"\[CLASS_(\w+)\]:", overlay))
        assert {c.upper() for c in CLASSES} <= styled, (
            "a class with no rule falls through to the unknown style, which draws "
            "an unlabelled white box -- unacceptable for a human being"
        )

    def test_person_is_given_a_minimum_drawn_size(self):
        overlay = (REPO / "app" / "js" / "overlay.js").read_text(encoding="utf-8")
        line = next(ln for ln in overlay.splitlines() if "[CLASS_PERSON]" in ln)
        minpx = int(re.search(r"minPx:\s*(\d+)", line).group(1))
        assert minpx > 0, (
            "a person box is a few pixels at altitude; drawn at natural size on a "
            "sunlit tablet it is invisible, which defeats the purpose of drawing it"
        )

    def test_the_browser_parser_agrees_on_the_class_list(self):
        wire = (REPO / "app" / "js" / "wire.js").read_text(encoding="utf-8")
        listed = re.search(r"CLASSES = Object\.freeze\(\[([^\]]+)\]\)", wire).group(1)
        names = tuple(x.strip().replace("CLASS_", "").lower() for x in listed.split(","))
        assert names == CLASSES
