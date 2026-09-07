"""The N-of-M temporal filter -- the critical suite.

This is the only place in the system where a detection's *history* is turned
into a decision to draw. Everything it can get wrong is operationally
dangerous in one of two directions:

* too eager -- a box appears from one frame of noise, and the operator learns
  to distrust the overlay;
* too timid -- a confirmed box blinks out because evidence dipped for one
  frame, which is the system silently retracting a warning.

So the assertions here are exact frame counts, not "eventually". Determinism
is part of the contract too: incident logs get replayed and compared, and a
filter whose output depended on dict ordering would make that impossible.
"""

from __future__ import annotations

import pytest

from station.core.config import TemporalConfig
from station.core.types import CLASS_FIRE, CLASS_SMOKE, BBox, Detection, FrameDetections, ModelInfo
from station.inference.temporal import TemporalFilter


def ids(emitted) -> list[int | None]:
    return [d.track_id for d in emitted]


# --------------------------------------------------------------------------
# construction and validation
# --------------------------------------------------------------------------


class TestConstruction:
    def test_rejects_n_greater_than_m(self):
        with pytest.raises(ValueError, match="1 <= n <= m"):
            TemporalFilter(TemporalConfig(n=6, m=5))

    def test_rejects_n_below_one(self):
        with pytest.raises(ValueError, match="1 <= n <= m"):
            TemporalFilter(TemporalConfig(n=0, m=5))

    @pytest.mark.parametrize("iou", [0.0, 1.0, -0.5, 1.5])
    def test_rejects_iou_outside_the_open_unit_interval(self, iou):
        # iou_match=0 would associate every pair of boxes anywhere in frame;
        # iou_match=1 would associate only pixel-identical ones.
        with pytest.raises(ValueError, match="iou_match"):
            TemporalFilter(TemporalConfig(iou_match=iou))

    def test_rejects_negative_max_age(self):
        with pytest.raises(ValueError, match="max_age"):
            TemporalFilter(TemporalConfig(max_age=-1))

    def test_starts_empty(self):
        filt = TemporalFilter(TemporalConfig())
        assert len(filt) == 0
        assert filt.tracks == ()
        assert filt.frames_seen == 0


# --------------------------------------------------------------------------
# the n-of-m boundary
# --------------------------------------------------------------------------


class TestNofMBoundary:
    """Nothing is drawn until the evidence bar is cleared -- exactly then."""

    def test_confirms_on_the_nth_frame_and_not_before(self, det, temporal_cfg):
        filt = TemporalFilter(temporal_cfg(n=3, m=5))
        d = det(0.5, 0.5)
        assert filt.update([d], pts=0.0) == ()          # 1 of 3
        assert filt.update([d], pts=0.1) == ()          # 2 of 3
        emitted = filt.update([d], pts=0.2)             # 3 of 3 -> confirmed
        assert len(emitted) == 1
        assert emitted[0].track_id == 1
        assert emitted[0].persisted == 3

    @pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
    def test_first_emission_is_always_on_frame_n(self, det, temporal_cfg, n):
        filt = TemporalFilter(temporal_cfg(n=n, m=5))
        d = det(0.5, 0.5)
        first_emit = None
        for i in range(10):
            if filt.update([d], pts=i * 0.1) and first_emit is None:
                first_emit = i
        assert first_emit == n - 1  # zero-based frame index

    def test_n_equals_one_confirms_immediately(self, det, temporal_cfg):
        # A legal, deliberately trigger-happy configuration. Worth pinning:
        # `_new_track` has to confirm at creation time for n=1, and getting
        # that wrong delays every box by one frame at every other n too.
        filt = TemporalFilter(temporal_cfg(n=1, m=3))
        emitted = filt.update([det(0.5, 0.5)], pts=0.0)
        assert ids(emitted) == [1]

    def test_hits_must_fall_within_the_window(self, det, temporal_cfg):
        # 3-of-5 with hits spread over seven frames must NOT confirm: the
        # window is the last m, not "three hits ever".
        filt = TemporalFilter(temporal_cfg(n=3, m=5, max_age=10))
        d = det(0.5, 0.5)
        pattern = [True, False, False, True, False, False, True]
        emitted = []
        for i, hit in enumerate(pattern):
            emitted = filt.update([d] if hit else [], pts=i * 0.1)
        # The last three frames of the window are hit/miss/miss/hit/... -- only
        # two hits are inside the trailing five frames.
        assert filt.tracks[0].hits_in_window == 2
        assert emitted == ()

    def test_scattered_noise_in_different_places_never_confirms(self, det, temporal_cfg):
        # This is the whole reason the module exists: uncorrelated false
        # positives do not persist in the same place, so they never clear the
        # bar however many of them there are.
        filt = TemporalFilter(temporal_cfg(n=3, m=5))
        spots = [(0.1, 0.1), (0.9, 0.2), (0.4, 0.8), (0.7, 0.5), (0.2, 0.6), (0.85, 0.85)]
        for i, (cx, cy) in enumerate(spots * 3):
            assert filt.update([det(cx, cy, 0.05)], pts=i * 0.1) == ()

    def test_persisted_reports_hits_within_the_window(self, det, temporal_cfg):
        filt = TemporalFilter(temporal_cfg(n=2, m=4, max_age=6))
        d = det(0.5, 0.5)
        seen = []
        for i in range(6):
            emitted = filt.update([d], pts=i * 0.1)
            seen.append(emitted[0].persisted if emitted else None)
        # Confirms on frame 2 (index 1), then the window fills to m and caps.
        assert seen == [None, 2, 3, 4, 4, 4]


# --------------------------------------------------------------------------
# anti-flicker: the reason the module exists
# --------------------------------------------------------------------------


class TestConfirmedTracksDoNotFlicker:
    def test_confirmation_latches_across_a_missed_frame(self, det, temporal_cfg):
        filt = TemporalFilter(temporal_cfg(n=3, m=5, max_age=5))
        d = det(0.5, 0.5)
        for i in range(3):
            filt.update([d], pts=i * 0.1)
        # The model drops it for one frame. The box must NOT blink out:
        # suppressing it would be the system quietly retracting a warning.
        emitted = filt.update([], pts=0.3)
        assert ids(emitted) == [1]

    def test_alternating_detection_holds_a_steady_box(self, det, temporal_cfg):
        """A marginal region at the confidence threshold drops in and out.

        This is the exact case the filter exists for. Once confirmed, the box
        must be emitted on every frame -- a box blinking at 5 Hz draws the
        operator's eye to the blinking rather than to the scene, and reads as
        instrument fault rather than as fire.
        """
        filt = TemporalFilter(temporal_cfg(n=3, m=5, max_age=5))
        d = det(0.5, 0.5)
        for i in range(3):
            filt.update([d], pts=i * 0.1)
        emissions = []
        for i in range(3, 23):
            present = (i % 2) == 1  # seen every other frame
            emissions.append(ids(filt.update([d] if present else [], pts=i * 0.1)))
        assert all(e == [1] for e in emissions), f"box flickered: {emissions}"

    def test_persisted_decays_while_a_track_coasts(self, det, temporal_cfg):
        # The box stays, but the number on screen weakens as the evidence
        # does. That is the operator's cue that it is coasting.
        filt = TemporalFilter(temporal_cfg(n=3, m=5, max_age=5))
        d = det(0.5, 0.5)
        for i in range(5):
            filt.update([d], pts=i * 0.1)
        decay = []
        for i in range(5, 10):
            emitted = filt.update([], pts=i * 0.1)
            decay.append(emitted[0].persisted)
        assert decay == [4, 3, 2, 1, 1]

    def test_persisted_never_reaches_zero(self, det, temporal_cfg):
        # A `persisted: 0` on the wire would be a zero count -- a measurement
        # of nothing -- and Detection forbids it outright.
        filt = TemporalFilter(temporal_cfg(n=1, m=2, max_age=10))
        filt.update([det(0.5, 0.5)], pts=0.0)
        for i in range(1, 11):
            emitted = filt.update([], pts=i * 0.1)
            assert emitted[0].persisted >= 1

    def test_a_coasting_box_keeps_its_last_observed_position(self, det, temporal_cfg):
        filt = TemporalFilter(temporal_cfg(n=2, m=4, max_age=4))
        filt.update([det(0.30, 0.50)], pts=0.0)
        filt.update([det(0.34, 0.50)], pts=0.1)
        coasted = filt.update([], pts=0.2)
        # No extrapolation, no smoothing: an invented position is a misplaced
        # box, which docs/CONTRACT.md calls the most dangerous failure mode.
        assert coasted[0].box == det(0.34, 0.50).box


# --------------------------------------------------------------------------
# track identity across gaps
# --------------------------------------------------------------------------


class TestTrackIdentityAcrossGaps:
    @pytest.mark.parametrize("gap", [1, 2, 3, 4, 5])
    def test_gap_shorter_than_max_age_keeps_the_track_id(self, det, temporal_cfg, gap):
        filt = TemporalFilter(temporal_cfg(n=2, m=5, max_age=5))
        d = det(0.5, 0.5)
        filt.update([d], pts=0.0)
        filt.update([d], pts=0.1)
        pts = 0.2
        for _ in range(gap):
            filt.update([], pts=pts)
            pts += 0.1
        emitted = filt.update([d], pts=pts)
        assert ids(emitted) == [1], "a gap within max_age must not mint a new track"
        assert emitted[0].first_seen_pts == pytest.approx(0.0)

    def test_gap_longer_than_max_age_starts_a_new_track(self, det, temporal_cfg):
        filt = TemporalFilter(temporal_cfg(n=2, m=5, max_age=3))
        d = det(0.5, 0.5)
        filt.update([d], pts=0.0)
        filt.update([d], pts=0.1)
        for i in range(2, 8):  # six misses, well past max_age=3
            filt.update([], pts=i * 0.1)
        assert len(filt) == 0
        filt.update([d], pts=0.8)
        emitted = filt.update([d], pts=0.9)
        assert ids(emitted) == [2], "a re-detection after ageing out is a new hypothesis"

    def test_track_ids_are_never_recycled(self, det, temporal_cfg):
        # Re-using an id would let the tablet draw a line between two
        # unrelated regions and call it one continuous fire.
        filt = TemporalFilter(temporal_cfg(n=1, m=3, max_age=0))
        seen_ids = []
        for i in range(6):
            emitted = filt.update([det(0.5, 0.5)], pts=i * 0.1)
            seen_ids.extend(ids(emitted))
        # max_age=0 kills a track on its first miss, but here it is matched
        # every frame, so this is one continuous identity.
        assert seen_ids == [1] * 6

    def test_ids_keep_counting_up_across_a_reset(self, det, temporal_cfg):
        filt = TemporalFilter(temporal_cfg(n=1, m=3))
        assert ids(filt.update([det(0.5, 0.5)], pts=0.0)) == [1]
        filt.reset()
        assert ids(filt.update([det(0.5, 0.5)], pts=0.0)) == [2]

    def test_first_seen_pts_is_the_first_detection_not_the_first_display(self, det, temporal_cfg):
        # After-action review uses this to find the earliest visible frame, so
        # it must predate confirmation.
        filt = TemporalFilter(temporal_cfg(n=3, m=5))
        d = det(0.5, 0.5)
        filt.update([d], pts=10.0)
        filt.update([d], pts=10.1)
        emitted = filt.update([d], pts=10.2)
        assert emitted[0].first_seen_pts == pytest.approx(10.0)


# --------------------------------------------------------------------------
# ageing out
# --------------------------------------------------------------------------


class TestAgeOut:
    def test_track_survives_exactly_max_age_misses(self, det, temporal_cfg):
        filt = TemporalFilter(temporal_cfg(n=2, m=5, max_age=5))
        d = det(0.5, 0.5)
        filt.update([d], pts=0.0)
        filt.update([d], pts=0.1)
        emissions = [ids(filt.update([], pts=(2 + i) * 0.1)) for i in range(7)]
        # Five coasted frames, then gone on the sixth miss.
        assert emissions == [[1], [1], [1], [1], [1], [], []]
        assert len(filt) == 0

    @pytest.mark.parametrize("max_age", [0, 1, 2, 5, 9])
    def test_max_age_is_the_number_of_coasted_frames(self, det, temporal_cfg, max_age):
        filt = TemporalFilter(temporal_cfg(n=1, m=3, max_age=max_age))
        filt.update([det(0.5, 0.5)], pts=0.0)
        coasted = 0
        for i in range(1, max_age + 5):
            if filt.update([], pts=i * 0.1):
                coasted += 1
            else:
                break
        assert coasted == max_age

    def test_unconfirmed_tracks_also_age_out(self, det, temporal_cfg):
        # An association hypothesis that was never confirmed must not linger:
        # it would keep absorbing detections that should start fresh tracks.
        filt = TemporalFilter(temporal_cfg(n=5, m=5, max_age=2))
        filt.update([det(0.5, 0.5)], pts=0.0)
        assert len(filt) == 1
        for i in range(1, 5):
            filt.update([], pts=i * 0.1)
        assert len(filt) == 0


# --------------------------------------------------------------------------
# two fires must not merge
# --------------------------------------------------------------------------


class TestSeparateTracks:
    def test_two_overlapping_same_class_fires_stay_separate(self, temporal_cfg):
        """Two adjacent fires, boxes overlapping well above iou_match.

        Greedy one-to-one matching is what stops them collapsing: the
        higher-IoU pair claims its partner and the loser keeps its own track.
        Merging them would put one box over two crews' worth of ground.
        """
        filt = TemporalFilter(temporal_cfg(n=2, m=4, iou_match=0.30, max_age=4))

        def pair(offset: float) -> list[Detection]:
            a = BBox(0.30 + offset, 0.40, 0.50 + offset, 0.60)
            b = BBox(0.3667 + offset, 0.40, 0.5667 + offset, 0.60)  # IoU(a, b) ~= 0.5
            return [
                Detection(cls=CLASS_FIRE, conf=0.8, box=a),
                Detection(cls=CLASS_FIRE, conf=0.7, box=b),
            ]

        assert pair(0.0)[0].box.iou(pair(0.0)[1].box) > 0.30, "fixture must actually overlap"

        for i in range(6):
            emitted = filt.update(pair(i * 0.005), pts=i * 0.1)
        assert sorted(ids(emitted)) == [1, 2]
        assert len(filt) == 2
        # And they kept their own geometry rather than averaging into one.
        by_id = {d.track_id: d for d in emitted}
        assert by_id[1].box.x1 < by_id[2].box.x1

    def test_one_detection_cannot_feed_two_tracks(self, temporal_cfg):
        filt = TemporalFilter(temporal_cfg(n=1, m=3, iou_match=0.10, max_age=3))
        a = Detection(cls=CLASS_FIRE, conf=0.8, box=BBox(0.30, 0.40, 0.50, 0.60))
        b = Detection(cls=CLASS_FIRE, conf=0.7, box=BBox(0.40, 0.40, 0.60, 0.60))
        filt.update([a, b], pts=0.0)
        assert len(filt) == 2
        # Now only one detection, sitting between the two tracks.
        middle = Detection(cls=CLASS_FIRE, conf=0.9, box=BBox(0.35, 0.40, 0.55, 0.60))
        filt.update([middle], pts=0.1)
        matched = [t for t in filt.tracks if t.age == 0]
        assert len(matched) == 1, "one detection matched two tracks"

    def test_distant_regions_never_associate(self, det, temporal_cfg):
        filt = TemporalFilter(temporal_cfg(n=1, m=3, max_age=3))
        filt.update([det(0.10, 0.10, 0.05)], pts=0.0)
        filt.update([det(0.90, 0.90, 0.05)], pts=0.1)
        assert len(filt) == 2
        assert {t.track_id for t in filt.tracks} == {1, 2}

    def test_matching_is_deterministic_regardless_of_input_order(self, temporal_cfg):
        # Incident logs get replayed and compared; a filter whose output
        # depended on detection ordering would make that meaningless.
        dets = [
            Detection(cls=CLASS_FIRE, conf=0.8, box=BBox(0.10, 0.10, 0.20, 0.20)),
            Detection(cls=CLASS_FIRE, conf=0.7, box=BBox(0.50, 0.50, 0.60, 0.60)),
            Detection(cls=CLASS_SMOKE, conf=0.6, box=BBox(0.30, 0.30, 0.45, 0.45)),
        ]
        outputs = []
        for order in ([0, 1, 2], [2, 1, 0], [1, 0, 2]):
            filt = TemporalFilter(temporal_cfg(n=2, m=3, max_age=3))
            ordered = [dets[i] for i in order]
            filt.update(ordered, pts=0.0)
            emitted = filt.update(ordered, pts=0.1)
            outputs.append(sorted((d.cls, d.box.as_tuple(), d.persisted) for d in emitted))
        assert outputs[0] == outputs[1] == outputs[2]


# --------------------------------------------------------------------------
# a growing plume keeps one identity
# --------------------------------------------------------------------------


class TestGrowingRegion:
    def test_a_growing_smoke_box_keeps_its_track_id(self, temporal_cfg):
        """Matching is against the LAST observed box, not the first.

        A plume that grows 12% per frame has near-zero IoU with its own first
        box after a couple of seconds, but high IoU with its box one frame
        ago. Chaining frame to frame is what keeps one identity, and a filter
        that matched against the creation box would split one fire into a
        dozen tracks as it grew.
        """
        filt = TemporalFilter(temporal_cfg(n=2, m=5, iou_match=0.30, max_age=5))
        cx = cy = 0.5
        half = 0.02
        first_box = None
        emitted = ()
        for i in range(32):
            half = min(half * 1.12, 0.49)
            b = BBox(max(0.0, cx - half), max(0.0, cy - half), min(1.0, cx + half), min(1.0, cy + half))
            if first_box is None:
                first_box = b
            emitted = filt.update([Detection(cls=CLASS_SMOKE, conf=0.6, box=b)], pts=i * 0.1)
        assert ids(emitted) == [1], "a growing plume was split into several tracks"
        assert len(filt) == 1
        final = emitted[0].box
        assert final.area > 0.9
        assert final.iou(first_box) < 0.01, "fixture did not actually grow away from its first box"

    def test_a_shrinking_region_keeps_its_track_id(self, temporal_cfg):
        filt = TemporalFilter(temporal_cfg(n=2, m=5, iou_match=0.30, max_age=5))
        half = 0.30
        emitted = ()
        for i in range(15):
            half = max(half * 0.90, 0.01)
            b = BBox(0.5 - half, 0.5 - half, 0.5 + half, 0.5 + half)
            emitted = filt.update([Detection(cls=CLASS_FIRE, conf=0.6, box=b)], pts=i * 0.1)
        assert ids(emitted) == [1]

    def test_a_drifting_region_keeps_its_track_id(self, det, temporal_cfg):
        filt = TemporalFilter(temporal_cfg(n=2, m=5, iou_match=0.30, max_age=5))
        emitted = ()
        for i in range(30):
            emitted = filt.update([det(0.15 + i * 0.02, 0.5, 0.12)], pts=i * 0.1)
        assert ids(emitted) == [1]

    def test_a_teleporting_region_does_not_keep_its_track_id(self, det, temporal_cfg):
        # The complement of the above: identity must not be preserved across a
        # jump the geometry cannot justify, or two fires a field apart become
        # one moving fire.
        filt = TemporalFilter(temporal_cfg(n=1, m=3, iou_match=0.30, max_age=3))
        assert ids(filt.update([det(0.15, 0.15, 0.08)], pts=0.0)) == [1]
        assert ids(filt.update([det(0.85, 0.85, 0.08)], pts=0.1)) == [1, 2]


# --------------------------------------------------------------------------
# class isolation
# --------------------------------------------------------------------------


class TestClassIsolation:
    def test_fire_and_smoke_at_the_same_place_are_two_tracks(self, temporal_cfg):
        """Flame sits at the base of its own plume, so co-located fire and
        smoke boxes are the normal geometry of one event -- not one object.
        Associating them would let smoke hits confirm a fire track."""
        filt = TemporalFilter(temporal_cfg(n=2, m=4, iou_match=0.30, max_age=4))
        b = BBox(0.40, 0.40, 0.60, 0.60)
        pair = [
            Detection(cls=CLASS_FIRE, conf=0.8, box=b),
            Detection(cls=CLASS_SMOKE, conf=0.6, box=b),
        ]
        filt.update(pair, pts=0.0)
        emitted = filt.update(pair, pts=0.1)
        assert len(emitted) == 2
        assert {d.cls for d in emitted} == {CLASS_FIRE, CLASS_SMOKE}
        assert len({d.track_id for d in emitted}) == 2

    def test_smoke_hits_do_not_confirm_a_fire_track(self, temporal_cfg):
        filt = TemporalFilter(temporal_cfg(n=3, m=5, iou_match=0.30, max_age=5))
        b = BBox(0.40, 0.40, 0.60, 0.60)
        fire = Detection(cls=CLASS_FIRE, conf=0.8, box=b)
        smoke = Detection(cls=CLASS_SMOKE, conf=0.6, box=b)
        filt.update([fire], pts=0.0)
        filt.update([smoke], pts=0.1)
        emitted = filt.update([smoke], pts=0.2)
        # Smoke reached 3-of-5 on its own? No -- it has only two hits. And the
        # fire track certainly did not.
        assert emitted == ()
        by_cls = {t.cls: t for t in filt.tracks}
        assert by_cls[CLASS_FIRE].hits_in_window == 1
        assert by_cls[CLASS_SMOKE].hits_in_window == 2

    def test_a_track_label_never_flips(self, temporal_cfg):
        # A box alternating between "fire" and "smoke" is unreadable, and a
        # fire reported under the smoke label understates what is on screen.
        filt = TemporalFilter(temporal_cfg(n=1, m=3, iou_match=0.30, max_age=6))
        b = BBox(0.40, 0.40, 0.60, 0.60)
        labels_by_id: dict[int, set[str]] = {}
        for i in range(10):
            cls = CLASS_FIRE if i % 2 == 0 else CLASS_SMOKE
            for d in filt.update([Detection(cls=cls, conf=0.7, box=b)], pts=i * 0.1):
                labels_by_id.setdefault(d.track_id, set()).add(d.cls)
        assert all(len(v) == 1 for v in labels_by_id.values()), labels_by_id

    def test_an_unknown_class_is_tracked_like_any_other(self, temporal_cfg):
        # The runner passes unrecognised model labels through rather than
        # dropping them; the filter must not be the thing that drops them.
        filt = TemporalFilter(temporal_cfg(n=2, m=3, max_age=3))
        d = Detection(cls="ember", conf=0.7, box=BBox(0.4, 0.4, 0.5, 0.5))
        filt.update([d], pts=0.0)
        emitted = filt.update([d], pts=0.1)
        assert [x.cls for x in emitted] == ["ember"]


# --------------------------------------------------------------------------
# empty frames, discontinuities, and the frame wrapper
# --------------------------------------------------------------------------


class TestEmptyFramesAreEvidence:
    def test_an_empty_frame_ages_the_window(self, det, temporal_cfg):
        # Skipping empty frames would freeze every window instead of ageing
        # it, and confirmed boxes would then coast forever.
        filt = TemporalFilter(temporal_cfg(n=3, m=5, max_age=5))
        d = det(0.5, 0.5)
        for i in range(3):
            filt.update([d], pts=i * 0.1)
        before = filt.tracks[0].hits_in_window
        filt.update([], pts=0.3)
        assert filt.tracks[0].hits_in_window == before
        assert filt.tracks[0].age == 1
        assert filt.frames_seen == 4

    def test_an_empty_frame_returns_an_empty_tuple_not_none(self, temporal_cfg):
        filt = TemporalFilter(temporal_cfg())
        result = filt.update([], pts=0.0)
        assert result == ()
        assert isinstance(result, tuple)

    def test_filter_frame_preserves_every_other_field(self, det, temporal_cfg, model_info):
        filt = TemporalFilter(temporal_cfg(n=1, m=3))
        frame = FrameDetections(
            frame_id=42,
            pts=7.5,
            wall_time="2026-09-06T14:22:31.412Z",
            detections=(det(0.5, 0.5),),
            model=model_info,
            inference_ms=21.7,
            rtp_ts=1839472920,
            source_id="rtsp://host/main.264",
        )
        out = filt.filter_frame(frame)
        assert (out.frame_id, out.pts, out.wall_time) == (42, 7.5, "2026-09-06T14:22:31.412Z")
        assert (out.model, out.inference_ms, out.rtp_ts, out.source_id) == (
            model_info, 21.7, 1839472920, "rtsp://host/main.264",
        )
        assert out.detections and out.detections[0].track_id == 1

    def test_filter_frame_on_an_empty_frame_stays_empty(self, temporal_cfg):
        filt = TemporalFilter(temporal_cfg())
        out = filt.filter_frame(FrameDetections(frame_id=1, pts=0.5))
        assert out.detections == ()
        assert out.pts == 0.5

    def test_output_detections_are_wire_legal(self, det, temporal_cfg):
        filt = TemporalFilter(temporal_cfg(n=2, m=3))
        d = det(0.5, 0.5)
        filt.update([d], pts=0.0)
        emitted = filt.update([d], pts=0.1)
        for detection in emitted:
            restored = Detection.from_wire(detection.to_wire())
            assert restored.track_id == detection.track_id
            assert restored.persisted == detection.persisted


class TestDiscontinuity:
    def test_backwards_pts_resets_the_tracker(self, det, temporal_cfg):
        # A file loop or a reconnect restarts the media clock. Associating
        # across it would join the end of one pass to the start of the next.
        filt = TemporalFilter(temporal_cfg(n=2, m=4, max_age=4))
        d = det(0.5, 0.5)
        filt.update([d], pts=10.0)
        assert ids(filt.update([d], pts=10.1)) == [1]
        assert filt.update([d], pts=0.0) == ()      # reset: this is track 2, 1-of-2
        emitted = filt.update([d], pts=0.1)
        assert ids(emitted) == [2]
        assert emitted[0].first_seen_pts == pytest.approx(0.0)

    def test_equal_pts_does_not_reset(self, det, temporal_cfg):
        # Two frames sharing a pts is a repaired timeline, not a rewind.
        filt = TemporalFilter(temporal_cfg(n=2, m=4))
        d = det(0.5, 0.5)
        filt.update([d], pts=5.0)
        assert ids(filt.update([d], pts=5.0)) == [1]

    def test_reset_clears_tracks_and_the_frame_counter(self, det, temporal_cfg):
        filt = TemporalFilter(temporal_cfg(n=1, m=3))
        filt.update([det(0.5, 0.5)], pts=0.0)
        assert len(filt) == 1 and filt.frames_seen == 1
        filt.reset()
        assert len(filt) == 0 and filt.frames_seen == 0


class TestEmitUnconfirmed:
    def test_off_by_default(self):
        assert TemporalConfig().emit_unconfirmed is False

    def test_unconfirmed_tracks_are_emitted_when_enabled(self, det, temporal_cfg):
        filt = TemporalFilter(temporal_cfg(n=3, m=5, emit_unconfirmed=True))
        emitted = filt.update([det(0.5, 0.5)], pts=0.0)
        assert ids(emitted) == [1]
        assert emitted[0].persisted == 1  # weak evidence, shown as weak evidence

    def test_unconfirmed_tracks_are_not_coasted(self, det, temporal_cfg):
        # Coasting an unconfirmed track would draw a box with no evidence
        # behind it at all.
        filt = TemporalFilter(temporal_cfg(n=3, m=5, max_age=5, emit_unconfirmed=True))
        filt.update([det(0.5, 0.5)], pts=0.0)
        assert filt.update([], pts=0.1) == ()
        assert len(filt) == 1, "the track should still be alive, just not drawn"

    def test_confirmed_tracks_still_coast_when_enabled(self, det, temporal_cfg):
        filt = TemporalFilter(temporal_cfg(n=2, m=5, max_age=5, emit_unconfirmed=True))
        d = det(0.5, 0.5)
        filt.update([d], pts=0.0)
        filt.update([d], pts=0.1)
        assert ids(filt.update([], pts=0.2)) == [1]


# --------------------------------------------------------------------------
# the filter cannot invent detections
# --------------------------------------------------------------------------


class TestFilterCannotInvent:
    def test_a_filter_fed_nothing_emits_nothing_forever(self, temporal_cfg):
        # The safety-relevant direction: this module narrows what the overlay
        # says "look here" about. It can never widen it.
        filt = TemporalFilter(temporal_cfg(n=1, m=1, max_age=10))
        for i in range(200):
            assert filt.update([], pts=i * 0.1) == ()
        assert len(filt) == 0

    def test_every_emitted_box_was_proposed_by_the_model(self, det, temporal_cfg):
        filt = TemporalFilter(temporal_cfg(n=2, m=4, max_age=4))
        proposed: set[tuple[float, ...]] = set()
        emitted_boxes: set[tuple[float, ...]] = set()
        for i in range(40):
            dets = [det(0.2 + (i % 5) * 0.01, 0.5, 0.12)] if i % 3 else []
            for d in dets:
                proposed.add(d.box.as_tuple())
            for out in filt.update(dets, pts=i * 0.1):
                emitted_boxes.add(out.box.as_tuple())
        assert emitted_boxes <= proposed

    def test_the_filter_never_raises_the_class_count(self, temporal_cfg):
        filt = TemporalFilter(temporal_cfg(n=1, m=2, max_age=3))
        emitted = filt.update(
            [Detection(cls=CLASS_FIRE, conf=0.6, box=BBox(0.4, 0.4, 0.5, 0.5))], pts=0.0
        )
        assert {d.cls for d in emitted} == {CLASS_FIRE}


def test_module_imports_without_numpy_or_a_model():
    """The filter is pure stdlib, so it tests on any machine.

    Enforced here as well as in CI because it is easy to break with a
    convenience import at the top of the module.
    """
    import station.inference.temporal as mod

    source = open(mod.__file__, encoding="utf-8").read()
    head = source.split("class Track", 1)[0]
    for heavy in ("import numpy", "import torch", "import cv2", "from ultralytics"):
        assert heavy not in head, f"{heavy!r} at module scope in temporal.py"


def test_doc_example_in_the_module_docstring_still_holds():
    """The docstring's worked example is executable documentation."""
    filt = TemporalFilter(TemporalConfig(n=2, m=3))
    box = BBox(0.4, 0.4, 0.5, 0.5)
    assert filt.update([Detection(cls=CLASS_FIRE, conf=0.6, box=box)], pts=0.0) == ()
    emitted = filt.update([Detection(cls=CLASS_FIRE, conf=0.7, box=box)], pts=0.1)
    assert (emitted[0].track_id, emitted[0].persisted) == (1, 2)


def test_model_info_is_not_touched_by_the_filter(det, temporal_cfg):
    info = ModelInfo(name="yolo11s-fire", version="0.3.1")
    filt = TemporalFilter(temporal_cfg(n=1, m=2))
    out = filt.filter_frame(FrameDetections(frame_id=0, pts=0.0, detections=(det(),), model=info))
    assert out.model is info
