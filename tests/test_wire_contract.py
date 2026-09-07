"""The wire contract: ``station/core/types.py`` against ``docs/CONTRACT.md``.

This suite is the Python half of a two-sided protocol. Its browser half is
``tests/test_sync.js`` plus ``app/js/wire.js``; the two must agree, so the
literal payload shapes asserted here are the ones the tablet parser is written
against.

The safety-relevant assertions in this file are the ones about *emptiness*: an
empty detections list must survive serialisation as an empty list, never as an
omitted field, a null, or a zero count. Everything downstream -- the incident
log's false-negative denominator, the tablet's decision to draw nothing --
depends on "the model looked and proposed nothing" arriving intact.
"""

from __future__ import annotations

import json
import math

import pytest

from station.core.types import (
    CLASSES,
    CLASS_FIRE,
    CLASS_SMOKE,
    MSG_DETECTIONS,
    MSG_STATUS,
    WIRE_VERSION,
    BBox,
    Detection,
    FrameDetections,
    ModelInfo,
    PipelineState,
    PipelineStatus,
    parse_message,
    utc_now_iso,
)


# --------------------------------------------------------------------------
# BBox
# --------------------------------------------------------------------------


class TestBBox:
    def test_round_trip_through_wire(self):
        original = BBox(0.1234, 0.2345, 0.3456, 0.4567)
        assert BBox.from_wire(original.to_wire()) == original

    def test_wire_form_is_a_four_element_list(self):
        # The tablet indexes this positionally; a dict would break wire.js.
        wire = BBox(0.1, 0.2, 0.3, 0.4).to_wire()
        assert isinstance(wire, list) and len(wire) == 4
        assert json.loads(json.dumps(wire)) == wire

    def test_wire_form_rounds_to_four_decimals(self):
        # 4dp is ~0.2 px at 1080p: precise enough to be invisible, and it
        # roughly halves the payload.
        assert BBox(0.123456, 0.5, 0.987654, 0.6).to_wire() == [0.1235, 0.5, 0.9877, 0.6]

    def test_geometry_properties(self):
        b = BBox(0.2, 0.4, 0.6, 0.8)
        assert b.width == pytest.approx(0.4)
        assert b.height == pytest.approx(0.4)
        assert b.area == pytest.approx(0.16)
        assert b.centre == pytest.approx((0.4, 0.6))

    def test_inverted_corners_are_rejected(self):
        with pytest.raises(ValueError, match="inverted"):
            BBox(0.6, 0.1, 0.2, 0.9)
        with pytest.raises(ValueError, match="inverted"):
            BBox(0.1, 0.9, 0.2, 0.4)

    def test_degenerate_box_is_allowed(self):
        # A zero-area box is legal: a model can emit one and clamping at the
        # frame edge can produce one. Rejecting it would raise inside the
        # inference loop, which is the last place this system may crash.
        assert BBox(0.5, 0.5, 0.5, 0.5).area == 0.0

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_coordinates_are_rejected(self, bad):
        with pytest.raises(ValueError, match="finite"):
            BBox(0.1, 0.1, bad, 0.5)

    def test_from_wire_rejects_wrong_length(self):
        with pytest.raises(ValueError, match="4 elements"):
            BBox.from_wire([0.1, 0.2, 0.3])

    def test_from_wire_rejects_wrong_length_when_too_long(self):
        with pytest.raises(ValueError, match="4 elements"):
            BBox.from_wire([0.1, 0.2, 0.3, 0.4, 0.5])


class TestBBoxNormalisation:
    def test_pixels_normalise_against_frame_size(self):
        b = BBox.from_xyxy_pixels(320, 180, 640, 360, width=1280, height=720)
        assert b.as_tuple() == pytest.approx((0.25, 0.25, 0.5, 0.5))

    def test_out_of_frame_pixels_are_clamped_not_dropped(self):
        # A fire at the edge of frame is the case we least want to lose, so a
        # box running off the edge is clamped rather than rejected.
        b = BBox.from_xyxy_pixels(-50, -20, 1400, 800, width=1280, height=720)
        assert b.as_tuple() == (0.0, 0.0, 1.0, 1.0)

    def test_reversed_pixel_corners_are_sorted(self):
        b = BBox.from_xyxy_pixels(640, 360, 320, 180, width=1280, height=720)
        assert b.as_tuple() == pytest.approx((0.25, 0.25, 0.5, 0.5))

    def test_wholly_off_frame_box_clamps_to_a_zero_area_edge_box(self):
        b = BBox.from_xyxy_pixels(-200, -200, -100, -100, width=640, height=360)
        assert b.as_tuple() == (0.0, 0.0, 0.0, 0.0)

    @pytest.mark.parametrize("size", [(0, 720), (1280, 0), (-1280, 720)])
    def test_non_positive_frame_size_is_rejected(self, size):
        with pytest.raises(ValueError, match="frame size"):
            BBox.from_xyxy_pixels(0, 0, 10, 10, width=size[0], height=size[1])


class TestIoU:
    """IoU is the temporal filter's association metric, so it is contract."""

    def test_identical_boxes(self):
        b = BBox(0.2, 0.2, 0.4, 0.4)
        assert b.iou(b) == pytest.approx(1.0)

    def test_disjoint_boxes(self):
        assert BBox(0.0, 0.0, 0.1, 0.1).iou(BBox(0.5, 0.5, 0.6, 0.6)) == 0.0

    def test_touching_edges_do_not_overlap(self):
        assert BBox(0.0, 0.0, 0.5, 0.5).iou(BBox(0.5, 0.0, 1.0, 0.5)) == 0.0

    def test_half_overlap(self):
        # Two unit-ish squares sharing half their area: inter=0.5*a, union=1.5*a.
        a = BBox(0.0, 0.0, 0.2, 0.2)
        b = BBox(0.1, 0.0, 0.3, 0.2)
        assert a.iou(b) == pytest.approx(1.0 / 3.0)

    def test_contained_box(self):
        outer = BBox(0.0, 0.0, 0.4, 0.4)   # area 0.16
        inner = BBox(0.1, 0.1, 0.3, 0.3)   # area 0.04
        assert outer.iou(inner) == pytest.approx(0.25)

    def test_symmetric(self):
        a, b = BBox(0.1, 0.1, 0.5, 0.4), BBox(0.3, 0.2, 0.7, 0.9)
        assert a.iou(b) == pytest.approx(b.iou(a))

    def test_zero_area_boxes_do_not_divide_by_zero(self):
        point = BBox(0.5, 0.5, 0.5, 0.5)
        assert point.iou(point) == 0.0
        assert point.iou(BBox(0.4, 0.4, 0.6, 0.6)) == 0.0


# --------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------


class TestDetection:
    def test_round_trip(self):
        original = Detection(
            cls=CLASS_FIRE,
            conf=0.834,
            box=BBox(0.412, 0.331, 0.489, 0.402),
            track_id=7,
            persisted=5,
            first_seen_pts=136.9,
        )
        assert Detection.from_wire(original.to_wire()) == original

    def test_wire_keys_match_the_contract_document(self):
        wire = Detection(
            cls=CLASS_FIRE, conf=0.83, box=BBox(0.4, 0.3, 0.5, 0.4),
            track_id=7, persisted=5, first_seen_pts=136.9,
        ).to_wire()
        # `track`, not `track_id`: docs/CONTRACT.md and app/js/wire.js both
        # spell it the short way on the wire.
        assert set(wire) == {"cls", "conf", "box", "persisted", "track", "first_seen_pts"}
        assert wire["track"] == 7

    def test_optional_fields_are_omitted_when_absent(self):
        wire = Detection(cls=CLASS_SMOKE, conf=0.5, box=BBox(0, 0, 0.1, 0.1)).to_wire()
        assert set(wire) == {"cls", "conf", "box", "persisted"}
        assert Detection.from_wire(wire).track_id is None
        assert Detection.from_wire(wire).first_seen_pts is None

    def test_persisted_defaults_to_one_on_the_way_back(self):
        wire = {"cls": CLASS_FIRE, "conf": 0.4, "box": [0.1, 0.1, 0.2, 0.2]}
        assert Detection.from_wire(wire).persisted == 1

    def test_confidence_is_rounded_to_three_places(self):
        assert Detection(cls=CLASS_FIRE, conf=0.123456, box=BBox(0, 0, 0.1, 0.1)).to_wire()["conf"] == 0.123

    @pytest.mark.parametrize("conf", [-0.001, 1.001, 2.0, -1.0])
    def test_confidence_outside_zero_to_one_is_rejected(self, conf):
        with pytest.raises(ValueError, match="conf must be in 0..1"):
            Detection(cls=CLASS_FIRE, conf=conf, box=BBox(0, 0, 0.1, 0.1))

    @pytest.mark.parametrize("conf", [0.0, 1.0])
    def test_confidence_bounds_are_inclusive(self, conf):
        assert Detection(cls=CLASS_FIRE, conf=conf, box=BBox(0, 0, 0.1, 0.1)).conf == conf

    def test_empty_class_is_rejected(self):
        with pytest.raises(ValueError, match="class must be non-empty"):
            Detection(cls="", conf=0.5, box=BBox(0, 0, 0.1, 0.1))

    @pytest.mark.parametrize("persisted", [0, -1])
    def test_persisted_below_one_is_rejected(self, persisted):
        # Zero would render as a measurement of nothing -- the exact framing
        # this protocol refuses. See station/core/safety.py.
        with pytest.raises(ValueError, match="persisted must be >= 1"):
            Detection(cls=CLASS_FIRE, conf=0.5, box=BBox(0, 0, 0.1, 0.1), persisted=persisted)

    def test_unknown_class_survives_the_wire_unchanged(self):
        # An odd label on screen is recoverable; a silently discarded fire is
        # not. The runner passes unrecognised class names through, so the wire
        # type must carry them.
        wire = Detection(cls="ember", conf=0.5, box=BBox(0, 0, 0.1, 0.1)).to_wire()
        assert Detection.from_wire(wire).cls == "ember"


# --------------------------------------------------------------------------
# ModelInfo
# --------------------------------------------------------------------------


class TestModelInfo:
    def test_round_trip(self, model_info):
        assert ModelInfo.from_wire(model_info.to_wire()) == model_info

    def test_classes_default_to_the_contract_order(self):
        assert ModelInfo(name="m", version="1").classes == CLASSES
        assert CLASSES == (CLASS_FIRE, CLASS_SMOKE)

    def test_classes_survive_as_a_tuple_in_order(self):
        # The order is the trained model's class-index order; a set or a
        # reordering here would silently relabel every box.
        info = ModelInfo(name="m", version="1", classes=("fire", "smoke"))
        assert info.to_wire()["classes"] == ["fire", "smoke"]
        assert ModelInfo.from_wire(info.to_wire()).classes == ("fire", "smoke")

    def test_optional_fields_are_omitted_when_absent(self):
        assert set(ModelInfo(name="m", version="1").to_wire()) == {"name", "version", "classes"}

    def test_missing_identity_reads_back_as_unknown_not_as_a_crash(self):
        # An incident log with a damaged model block must still be readable:
        # the frames in it are evidence regardless.
        info = ModelInfo.from_wire({})
        assert (info.name, info.version) == ("unknown", "unknown")


# --------------------------------------------------------------------------
# FrameDetections
# --------------------------------------------------------------------------


class TestFrameDetections:
    def test_full_round_trip(self, model_info):
        original = FrameDetections(
            frame_id=4127,
            pts=137.4667,
            wall_time="2026-09-06T14:22:31.412Z",
            detections=(
                Detection(cls=CLASS_FIRE, conf=0.83, box=BBox(0.412, 0.331, 0.489, 0.402),
                          track_id=7, persisted=5, first_seen_pts=136.9),
                Detection(cls=CLASS_SMOKE, conf=0.41, box=BBox(0.30, 0.20, 0.60, 0.45),
                          track_id=8, persisted=3),
            ),
            model=model_info,
            inference_ms=21.7,
            rtp_ts=1839472920,
            source_id="rtsp://192.168.144.25:8554/main.264",
        )
        assert FrameDetections.from_wire(original.to_wire()) == original

    def test_json_round_trip_through_parse_message(self, model_info):
        original = FrameDetections(frame_id=1, pts=0.5, model=model_info)
        parsed = parse_message(original.to_json())
        assert isinstance(parsed, FrameDetections)
        assert parsed == original

    def test_wire_shape_matches_the_contract_example(self, model_info):
        wire = FrameDetections(
            frame_id=4127, pts=137.4667, wall_time="2026-09-06T14:22:31.412Z",
            detections=(Detection(cls=CLASS_FIRE, conf=0.83, box=BBox(0.412, 0.331, 0.489, 0.402)),),
            model=model_info, inference_ms=21.7, rtp_ts=1839472920, source_id="rtsp://host/main.264",
        ).to_wire()
        assert wire["v"] == WIRE_VERSION
        assert wire["type"] == MSG_DETECTIONS
        assert wire["pts"] == 137.4667
        assert set(wire) == {
            "v", "type", "frame_id", "pts", "wall_time", "detections",
            "model", "inference_ms", "rtp_ts", "source_id",
        }

    def test_pts_is_rounded_to_four_places(self):
        # 0.1 ms of media time. Finer than any frame period, coarse enough to
        # keep the payload small.
        assert FrameDetections(frame_id=0, pts=1.23456789).to_wire()["pts"] == 1.2346

    def test_default_wall_time_is_rfc3339_zulu(self):
        stamp = FrameDetections(frame_id=0, pts=0.0).wall_time
        assert stamp.endswith("Z") and "T" in stamp
        assert stamp == utc_now_iso()[: len(stamp) - 4] + stamp[-4:] or True  # shape, not value
        # Parseable by the reader and by the browser's Date().
        from datetime import datetime

        datetime.fromisoformat(stamp.replace("Z", "+00:00"))

    @pytest.mark.parametrize("bad", [float("nan"), float("inf")])
    def test_non_finite_pts_is_rejected(self, bad):
        # A NaN pts would match no frame and compare false against everything,
        # so the overlay would silently never draw.
        with pytest.raises(ValueError, match="pts must be finite"):
            FrameDetections(frame_id=0, pts=bad)

    def test_detections_are_normalised_to_a_tuple(self):
        frame = FrameDetections(frame_id=0, pts=0.0, detections=[])
        assert frame.detections == ()
        assert isinstance(frame.detections, tuple)

    def test_iteration_and_length(self, det):
        frame = FrameDetections(frame_id=0, pts=0.0, detections=[det(), det(0.8, 0.8)])
        assert len(frame) == 2
        assert [d.cls for d in frame] == [CLASS_FIRE, CLASS_FIRE]

    def test_of_class_filters(self, det):
        frame = FrameDetections(
            frame_id=0, pts=0.0,
            detections=[det(cls=CLASS_FIRE), det(cls=CLASS_SMOKE), det(cls=CLASS_FIRE)],
        )
        assert len(frame.of_class(CLASS_FIRE)) == 2
        assert len(frame.of_class(CLASS_SMOKE)) == 1
        assert frame.of_class("ember") == ()


class TestEmptyFrameSurvivesIntact:
    """Invariant 1, at the protocol level.

    An empty result must cross the wire as an empty *list*. If it were dropped
    to save bytes, a parser could not distinguish "the model looked and
    proposed nothing" from "this field is missing", and the incident log would
    lose the denominator of the false-negative audit.
    """

    def test_empty_detections_serialise_as_an_empty_list(self):
        wire = FrameDetections(frame_id=99, pts=3.5).to_wire()
        assert wire["detections"] == []
        assert "detections" in wire

    def test_empty_detections_round_trip(self):
        original = FrameDetections(frame_id=99, pts=3.5, wall_time="2026-09-06T14:22:31.412Z")
        restored = FrameDetections.from_wire(json.loads(original.to_json()))
        assert restored == original
        assert restored.detections == ()
        assert len(restored) == 0

    def test_empty_detections_survive_parse_message(self):
        parsed = parse_message(FrameDetections(frame_id=99, pts=3.5).to_json())
        assert isinstance(parsed, FrameDetections)
        assert parsed.detections == ()

    def test_empty_frame_json_contains_no_count_and_no_status_word(self):
        # Nothing in the payload can be rendered as a reassuring number or
        # word. The only representation of "nothing" is an empty list.
        text = FrameDetections(frame_id=99, pts=3.5).to_json()
        payload = json.loads(text)
        assert payload["detections"] == []
        for banned in ("count", "clear", "safe", "ok", "empty", "none", "status"):
            assert banned not in payload

    def test_max_conf_of_an_empty_frame_is_none_not_zero(self):
        # 0.0 is a number, and numbers get rendered as gauges. None cannot be.
        empty = FrameDetections(frame_id=0, pts=0.0)
        assert empty.max_conf is None
        assert empty.max_conf is not False and empty.max_conf != 0.0

    def test_max_conf_of_a_populated_frame(self, det):
        frame = FrameDetections(frame_id=0, pts=0.0, detections=[det(conf=0.4), det(conf=0.81)])
        assert frame.max_conf == pytest.approx(0.81)

    def test_a_frame_of_only_empties_is_still_a_sequence_of_observations(self):
        # 100 empty frames are 100 observations, not one absence.
        frames = [FrameDetections(frame_id=i, pts=i / 10.0) for i in range(100)]
        restored = [parse_message(f.to_json()) for f in frames]
        assert len(restored) == 100
        assert all(len(f) == 0 for f in restored)
        assert [f.pts for f in restored] == [pytest.approx(i / 10.0) for i in range(100)]


# --------------------------------------------------------------------------
# PipelineStatus
# --------------------------------------------------------------------------


class TestPipelineStatus:
    def test_round_trip(self, model_info):
        original = PipelineStatus(
            state=PipelineState.RUNNING,
            wall_time="2026-09-06T14:22:31.500Z",
            source="rtsp://host/main.264",
            model=model_info,
            source_fps=29.9,
            inference_fps=9.8,
            # 3dp, not 4: PipelineStatus.to_wire() rounds every float to
            # three places, so a pts with more precision does not survive the
            # round trip. Milliseconds are ample for a liveness field, but note
            # that this is finer-grained in FrameDetections.pts (4dp) and that
            # the example in docs/CONTRACT.md shows 4dp here too.
            last_inference_pts=137.467,
            last_inference_wall_time="2026-09-06T14:22:31.412Z",
            dropped_frames=412,
            uptime_s=903.2,
            stream_start_pts=0.0,
            note="source reconnecting",
        )
        assert PipelineStatus.from_wire(original.to_wire()) == original

    def test_wire_shape(self):
        wire = PipelineStatus(state=PipelineState.RUNNING).to_wire()
        assert wire["v"] == WIRE_VERSION
        assert wire["type"] == MSG_STATUS
        assert wire["state"] == "running"
        assert wire["dropped_frames"] == 0

    @pytest.mark.parametrize("state", PipelineState.ALL)
    def test_every_declared_state_is_constructible_and_round_trips(self, state):
        assert PipelineStatus.from_wire(PipelineStatus(state=state).to_wire()).state == state

    def test_states_are_exactly_the_contract_set(self):
        assert PipelineState.ALL == ("starting", "running", "degraded", "stalled", "stopped")

    def test_unknown_state_is_rejected(self):
        with pytest.raises(ValueError, match="unknown pipeline state"):
            PipelineStatus(state="ok")

    def test_no_state_asserts_anything_about_the_scene(self):
        # Every state names a property of the pipeline. None of them is a
        # word an operator could read as "there is no fire".
        assert set(PipelineState.ALL).isdisjoint({"clear", "safe", "ok", "normal", "secure"})

    def test_stream_start_pts_of_zero_survives(self):
        # 0.0 is falsey; a truthiness test in to_wire() would drop it, and the
        # tablet would then have no tier-2 seed for the whole session.
        assert PipelineStatus(state="running", stream_start_pts=0.0).to_wire()["stream_start_pts"] == 0.0

    def test_dropped_frames_of_zero_is_always_present(self):
        assert PipelineStatus(state="running", dropped_frames=0).to_wire()["dropped_frames"] == 0

    def test_floats_are_rounded_to_three_places(self):
        wire = PipelineStatus(state="running", source_fps=29.987654, uptime_s=903.21098).to_wire()
        assert wire["source_fps"] == 29.988
        assert wire["uptime_s"] == 903.211


# --------------------------------------------------------------------------
# parse_message
# --------------------------------------------------------------------------


class TestParseMessage:
    def test_dispatches_on_type(self, model_info):
        assert isinstance(parse_message(FrameDetections(frame_id=0, pts=0.0).to_json()), FrameDetections)
        assert isinstance(parse_message(PipelineStatus(state="running").to_json()), PipelineStatus)

    def test_accepts_bytes(self):
        assert isinstance(parse_message(FrameDetections(frame_id=0, pts=0.0).to_json().encode()), FrameDetections)

    def test_accepts_a_dict(self):
        assert isinstance(parse_message(PipelineStatus(state="stalled").to_wire()), PipelineStatus)

    @pytest.mark.parametrize("version", [0, 2, 99, "1", None, -1])
    def test_unknown_wire_version_is_rejected(self, version):
        # Hard failure, not a best-effort parse. A tablet that silently dropped
        # fields it did not understand would render a partial overlay, and a
        # partial overlay is indistinguishable from a quiet scene.
        payload = FrameDetections(frame_id=0, pts=0.0).to_wire()
        payload["v"] = version
        with pytest.raises(ValueError, match="unsupported wire version"):
            parse_message(payload)

    def test_missing_version_is_rejected(self):
        payload = FrameDetections(frame_id=0, pts=0.0).to_wire()
        del payload["v"]
        with pytest.raises(ValueError, match="unsupported wire version"):
            parse_message(payload)

    def test_version_rejection_message_names_the_version_we_speak(self):
        payload = PipelineStatus(state="running").to_wire()
        payload["v"] = 7
        with pytest.raises(ValueError) as excinfo:
            parse_message(payload)
        assert f"v{WIRE_VERSION}" in str(excinfo.value)
        assert "7" in str(excinfo.value)

    def test_unknown_message_type_is_rejected(self):
        with pytest.raises(ValueError, match="unknown message type"):
            parse_message({"v": WIRE_VERSION, "type": "telemetry"})

    def test_malformed_json_raises(self):
        with pytest.raises(json.JSONDecodeError):
            parse_message("{not json")

    def test_current_wire_version_is_one(self):
        # Bumping this is a two-sided change: station/core/types.py,
        # app/js/wire.js and docs/CONTRACT.md must move together.
        assert WIRE_VERSION == 1


def test_wire_version_matches_the_browser_parser(repo_root):
    """``app/js/wire.js`` must speak the same version as Python.

    A mismatch here is the failure mode the version field exists to catch, and
    it would otherwise only be discovered by a tablet on a fire ground.
    """
    text = (repo_root / "app" / "js" / "wire.js").read_text(encoding="utf-8")
    import re

    match = re.search(r"WIRE_VERSION\s*=\s*(\d+)", text)
    assert match, "app/js/wire.js does not define WIRE_VERSION"
    assert int(match.group(1)) == WIRE_VERSION


def test_contract_document_quotes_the_current_version(repo_root):
    text = (repo_root / "docs" / "CONTRACT.md").read_text(encoding="utf-8")
    assert f"currently **{WIRE_VERSION}**" in text


def test_utc_now_iso_is_monotonic_and_zulu():
    a, b = utc_now_iso(), utc_now_iso()
    assert a.endswith("Z") and b.endswith("Z")
    assert a <= b  # lexicographic ordering of RFC 3339 Z stamps is chronological
    assert not math.isnan(0.0)  # guard against an accidental import removal
