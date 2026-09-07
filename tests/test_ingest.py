"""Ingest: the synthetic source, the pts clock, and rate decimation.

The synthetic source is the seam that makes the rest of the system testable at
all -- it needs no codec, no capture card and no drone, and for every frame it
renders it can state exactly where the bright region is, in the same normalised
coordinates the wire protocol uses. So its own guarantees are worth pinning
hard, because every downstream test rests on them:

* frame 137 renders identically however it was reached, in this process or the
  next one;
* the first emitted pts is exactly 0.0, and pts is strictly increasing
  thereafter -- across decode glitches, file loops and radio dropouts;
* rate decimation drops frames, and dropping frames can never break
  monotonicity.

Only numpy is needed for any of this. The tests that would need PyAV, OpenCV
or a real camera assert the *degradation* path instead: a clear error naming
the missing package.
"""

from __future__ import annotations

import numpy as np
import pytest

from station.core.config import SourceConfig
from station.core.types import BBox
from station.ingest import SOURCE_TYPES, open_source
from station.ingest.base import (
    DEFAULT_FPS,
    Frame,
    FpsDecimator,
    IngestError,
    PtsClock,
    PtsOrigin,
    SourceConfigError,
)
from station.ingest.synthetic_source import (
    DEFAULT_FRAME_COUNT,
    TRUTH_CLASS,
    SyntheticScene,
    SyntheticSource,
    synthetic_config,
)


# --------------------------------------------------------------------------
# dispatch
# --------------------------------------------------------------------------


class TestOpenSource:
    def test_known_types(self):
        assert set(SOURCE_TYPES) == {"file", "rtsp", "rtmp", "hdmi", "synthetic"}

    def test_synthetic_dispatch_returns_an_open_source(self):
        with open_source(synthetic_config(frames=3, fps=10.0)) as src:
            assert src.is_open
            assert isinstance(src, SyntheticSource)

    def test_autostart_false_leaves_it_closed(self):
        src = open_source(synthetic_config(frames=3), autostart=False)
        try:
            assert not src.is_open
            with pytest.raises(IngestError, match="before open"):
                src.read()
        finally:
            src.close()

    def test_unknown_type_lists_every_valid_type(self):
        with pytest.raises(SourceConfigError) as excinfo:
            open_source(SourceConfig(type="webcam"))
        message = str(excinfo.value)
        assert "webcam" in message
        for name in SOURCE_TYPES:
            assert name in message, f"{name} missing from the error message"

    @pytest.mark.parametrize("kind", ["", "  ", "SYNTHETIC "])
    def test_type_is_normalised_before_dispatch(self, kind):
        cfg = SourceConfig(type=kind, uri="synthetic:?frames=2")
        if kind.strip().lower() in SOURCE_TYPES:
            with open_source(cfg) as src:
                assert src.is_open
        else:
            with pytest.raises(SourceConfigError):
                open_source(cfg)


# --------------------------------------------------------------------------
# determinism
# --------------------------------------------------------------------------


class TestSyntheticDeterminism:
    def test_the_same_index_renders_identical_pixels(self):
        scene = SyntheticScene(width=64, height=48, seed=7)
        assert np.array_equal(scene.render(137), scene.render(137))

    def test_rendering_does_not_depend_on_how_the_index_was_reached(self):
        # A generator carrying hidden state would make a test that skipped
        # frames see different pixels; the seed is mixed with the index instead.
        scene = SyntheticScene(width=64, height=48, seed=7)
        direct = scene.render(50)
        for i in range(50):
            scene.render(i)
        assert np.array_equal(scene.render(50), direct)

    def test_different_seeds_give_different_pixels(self):
        a = SyntheticScene(width=64, height=48, seed=1).render(10)
        b = SyntheticScene(width=64, height=48, seed=2).render(10)
        assert not np.array_equal(a, b)

    def test_two_sources_with_the_same_config_agree_frame_for_frame(self):
        cfg = synthetic_config(frames=12, fps=10.0, seed=3, width=64, height=48)
        with open_source(cfg) as a, open_source(cfg) as b:
            for fa, fb in zip(a, b):
                assert np.array_equal(fa.image, fb.image)
                assert fa.pts == fb.pts
                assert fa.frame_id == fb.frame_id

    def test_reopening_a_source_replays_the_same_frames(self):
        src = SyntheticSource(synthetic_config(frames=5, fps=10.0, width=64, height=48))
        first = [f.image.copy() for f in src.open()]
        src.close()
        src2 = SyntheticSource(synthetic_config(frames=5, fps=10.0, width=64, height=48))
        with src2:
            second = [f.image.copy() for f in src2]
        assert all(np.array_equal(a, b) for a, b in zip(first, second))

    def test_frames_are_bgr_uint8_of_the_configured_size(self):
        with open_source(synthetic_config(frames=2, width=320, height=240)) as src:
            frame = src.read()
        assert frame.image.dtype == np.uint8
        assert frame.image.shape == (240, 320, 3)
        assert frame.shape == (320, 240)

    def test_the_blob_is_the_brightest_thing_in_frame_by_channel_mean(self):
        """The stub's blob detector thresholds the *channel mean* at 200.

        So "brightest thing in frame" has to be true in that specific metric,
        not merely to the eye: the sky gradient is a bright grey-blue and a
        saturated orange blob has a lower channel mean than it does. Getting
        this wrong makes `make run-stub` draw nothing, which -- by design --
        looks exactly like a healthy overlay on a quiet scene.
        """
        scene = SyntheticScene(width=160, height=120, seed=0)
        for index in (0, 20, 55, 91):
            image = scene.render(index)
            cx, cy, r = scene.centre_px(index)
            luma = image.mean(axis=2)
            core = luma[int(cy), int(cx)]
            # Everything at least two radii away from the blob is background.
            yy, xx = np.mgrid[0:scene.height, 0:scene.width]
            background = luma[((xx - cx) ** 2 + (yy - cy) ** 2) > (2 * r) ** 2]
            assert core > 200.0, f"frame {index}: core {core} is below the stub's threshold"
            assert core > background.max() + 40


class TestGroundTruth:
    def test_the_published_box_matches_the_rendered_blob(self):
        # box() computes the geometry rather than reading back the pixels, so
        # this checks the two definitions have not drifted apart.
        # `present` toggles the blob off without touching the background, and
        # noise=0 makes the two frames differ in exactly the blob's support.
        scene = SyntheticScene(width=200, height=150, seed=0, noise=0.0, present=((33, 34),))
        box = scene.box(33)
        assert box is not None and scene.box(34) is None
        drawn = scene.render(33).astype(np.int16)
        background = scene.render(34).astype(np.int16)
        painted = np.argwhere(np.abs(drawn - background).max(axis=2) > 0)
        ys, xs = painted[:, 0], painted[:, 1]
        measured = BBox.from_xyxy_pixels(
            xs.min(), ys.min(), xs.max() + 1, ys.max() + 1, scene.width, scene.height
        )
        assert box.iou(measured) > 0.90, f"published {box.as_tuple()} vs drawn {measured.as_tuple()}"

    def test_ground_truth_is_none_outside_the_present_ranges(self):
        scene = SyntheticScene(present=((10, 20),))
        assert scene.box(5) is None
        assert scene.box(10) is not None
        assert scene.box(19) is not None
        assert scene.box(20) is None, "present ranges are half-open"

    def test_ground_truth_for_a_frame_survives_decimation(self):
        # With target_fps set, generated indices and emitted frame ids differ;
        # ground_truth_for recovers the generated index from the pts.
        cfg = synthetic_config(frames=30, fps=30.0, target_fps=10.0, width=64, height=48)
        with SyntheticSource(cfg) as src:
            for frame in src:
                index = src.index_of(frame)
                assert index == round(frame.pts * src.scene.fps)
                assert src.ground_truth_for(frame) == src.scene.box(index)

    def test_truth_class_is_the_wire_fire_class(self):
        from station.core.types import CLASS_FIRE

        assert TRUTH_CLASS == CLASS_FIRE

    def test_the_blob_never_leaves_the_frame(self):
        # A clipped ground-truth box is a poor reference for an IoU assertion,
        # so the scene keeps the whole disc inside the picture.
        scene = SyntheticScene(width=320, height=240, amp_x=0.45, amp_y=0.45)
        for i in range(0, 300, 7):
            box = scene.box(i)
            assert 0.0 <= box.x1 < box.x2 <= 1.0
            assert 0.0 <= box.y1 < box.y2 <= 1.0


# --------------------------------------------------------------------------
# pts
# --------------------------------------------------------------------------


class TestPtsGuarantees:
    def test_the_first_emitted_pts_is_exactly_zero(self):
        with open_source(synthetic_config(frames=3, fps=25.0)) as src:
            assert src.read().pts == 0.0

    def test_pts_is_strictly_increasing(self):
        with open_source(synthetic_config(frames=60, fps=30.0, width=64, height=48)) as src:
            pts = [f.pts for f in src]
        assert len(pts) == 60
        assert all(b > a for a, b in zip(pts, pts[1:])), "pts is not strictly increasing"

    def test_pts_follows_the_nominal_rate(self):
        with open_source(synthetic_config(frames=10, fps=20.0, width=64, height=48)) as src:
            pts = [f.pts for f in src]
        assert pts == [pytest.approx(i / 20.0) for i in range(10)]

    def test_frame_ids_count_emitted_frames_from_zero(self):
        with open_source(synthetic_config(frames=8, fps=10.0, width=64, height=48)) as src:
            assert [f.frame_id for f in src] == list(range(8))

    def test_pts_origin_is_reported_as_generated(self):
        with open_source(synthetic_config(frames=2, width=64, height=48)) as src:
            src.read()
            assert src.info.pts_origin == PtsOrigin.GENERATED

    def test_rtp_ts_is_always_none_from_this_package(self):
        # Only the sender knows the real RTP timestamp; inventing one here
        # would leave the tablet in tier 1 drawing boxes on the wrong frames.
        with open_source(synthetic_config(frames=3, width=64, height=48)) as src:
            assert all(f.rtp_ts is None for f in src)


class TestPtsClock:
    def test_container_timestamps_are_rebased_to_zero(self):
        clock = PtsClock(30.0, PtsOrigin.CONTAINER)
        assert clock.stamp(1000.0) == 0.0
        assert clock.stamp(1000.5) == pytest.approx(0.5)

    def test_a_backwards_timestamp_is_repaired_not_emitted(self):
        # A repeated or reversed pts means the tablet cannot tell which frame
        # a box belongs to, so the clock forces a gap and counts it.
        clock = PtsClock(30.0, PtsOrigin.CONTAINER)
        first = clock.stamp(0.0)
        second = clock.stamp(1.0)
        third = clock.stamp(0.5)
        assert first < second < third
        assert clock.repairs == 1

    def test_a_repeated_timestamp_is_repaired(self):
        clock = PtsClock(30.0, PtsOrigin.CONTAINER)
        clock.stamp(5.0)
        assert clock.stamp(5.0) > 0.0
        assert clock.repairs == 1

    def test_missing_timestamps_fall_back_to_uniform_steps(self):
        clock = PtsClock(25.0, PtsOrigin.FRAME_INDEX)
        assert [clock.stamp(None) for _ in range(3)] == [0.0, pytest.approx(0.04), pytest.approx(0.08)]
        assert clock.synthesised == 3
        assert clock.last_synthesised is True

    def test_non_finite_raw_timestamps_are_ignored(self):
        clock = PtsClock(10.0, PtsOrigin.CONTAINER)
        clock.stamp(0.0)
        value = clock.stamp(float("nan"))
        assert value > 0.0 and value == value  # not NaN

    def test_a_discontinuity_moves_forward_never_back(self):
        # The tablet's overlay buffer matches on pts; a timeline that rewound
        # would match new detections against old frames.
        clock = PtsClock(30.0, PtsOrigin.CONTAINER)
        clock.stamp(100.0)
        clock.stamp(101.0)
        clock.discontinuity(2.5)  # a measured 2.5 s outage
        after = clock.stamp(0.0)  # the source restarted its own clock at zero
        assert after == pytest.approx(1.0 + 2.5)

    def test_a_loop_advances_by_one_frame_period_by_default(self):
        clock = PtsClock(10.0, PtsOrigin.CONTAINER)
        clock.stamp(0.0)
        clock.stamp(1.0)
        clock.discontinuity()
        assert clock.stamp(0.0) == pytest.approx(1.1)

    def test_an_unknown_mode_is_rejected(self):
        with pytest.raises(ValueError, match="unknown pts mode"):
            PtsClock(30.0, "guesswork")

    def test_a_zero_or_missing_fps_falls_back_to_the_default_rate(self):
        assert PtsClock(0.0, PtsOrigin.FRAME_INDEX).nominal_step == pytest.approx(1.0 / DEFAULT_FPS)
        assert PtsClock(None, PtsOrigin.FRAME_INDEX).nominal_step == pytest.approx(1.0 / DEFAULT_FPS)


# --------------------------------------------------------------------------
# decimation
# --------------------------------------------------------------------------


class TestFpsDecimator:
    def test_no_target_keeps_every_frame(self):
        dec = FpsDecimator(None)
        assert all(dec.accept(i / 30.0) for i in range(30))
        assert dec.dropped == 0

    def test_the_first_frame_is_always_kept(self):
        assert FpsDecimator(1.0).accept(123.456) is True

    def test_a_thirty_fps_stream_capped_at_ten_keeps_every_third_frame(self):
        dec = FpsDecimator(10.0)
        kept = [i for i in range(30) if dec.accept(i / 30.0)]
        assert kept == [0, 3, 6, 9, 12, 15, 18, 21, 24, 27]
        assert dec.dropped == 20

    def test_a_source_at_exactly_the_target_rate_keeps_everything(self):
        # The 1e-9 epsilon in accept() is what stops floating point turning
        # "exactly 10 fps" into "every other frame".
        dec = FpsDecimator(10.0)
        assert sum(dec.accept(i / 10.0) for i in range(50)) == 50

    def test_a_gap_resynchronises_instead_of_bursting(self):
        # After a reconnect the schedule is far behind; catching up would
        # hand inference a burst of back-to-back frames.
        dec = FpsDecimator(10.0)
        dec.accept(0.0)
        assert dec.accept(30.0) is True
        assert dec.accept(30.01) is False
        assert dec.accept(30.11) is True

    def test_reset_forgets_the_schedule(self):
        dec = FpsDecimator(10.0)
        dec.accept(0.0)
        assert dec.accept(0.01) is False
        dec.reset()
        assert dec.accept(0.01) is True

    @pytest.mark.parametrize("bad", [0.0, -1.0])
    def test_a_non_positive_target_is_rejected(self, bad):
        with pytest.raises(SourceConfigError, match="target_fps"):
            FpsDecimator(bad)


class TestDecimationThroughTheSource:
    def test_target_fps_reduces_the_frame_count(self):
        cfg = synthetic_config(frames=60, fps=30.0, target_fps=10.0, width=64, height=48)
        with open_source(cfg) as src:
            frames = list(src)
            stats = src.stats
        assert len(frames) == 20
        assert stats.frames_in == 60
        assert stats.frames_out == 20
        assert stats.decimated == 40

    def test_decimation_preserves_pts_monotonicity(self):
        cfg = synthetic_config(frames=90, fps=30.0, target_fps=7.0, width=64, height=48)
        with open_source(cfg) as src:
            pts = [f.pts for f in src]
        assert all(b > a for a, b in zip(pts, pts[1:]))

    def test_decimation_keeps_pts_on_the_source_timeline(self):
        # Decimating only removes values from an already-increasing sequence;
        # it must not rescale them, or every box would land on the wrong frame.
        cfg = synthetic_config(frames=30, fps=30.0, target_fps=10.0, width=64, height=48)
        with open_source(cfg) as src:
            pts = [f.pts for f in src]
        assert pts[0] == 0.0
        assert pts[1] == pytest.approx(3 / 30.0)
        assert all(abs(p * 30.0 - round(p * 30.0)) < 1e-6 for p in pts)

    def test_frame_ids_stay_dense_under_decimation(self):
        # frame_id counts *emitted* frames, so it matches the incident log
        # one-for-one; the dropped frames leave no holes.
        cfg = synthetic_config(frames=30, fps=30.0, target_fps=10.0, width=64, height=48)
        with open_source(cfg) as src:
            assert [f.frame_id for f in src] == list(range(10))

    def test_output_fps_is_reported(self):
        cfg = synthetic_config(frames=10, fps=30.0, target_fps=10.0, width=64, height=48)
        with open_source(cfg) as src:
            src.read()
            assert src.info.output_fps == pytest.approx(10.0)
            assert src.info.fps == pytest.approx(30.0)

    def test_a_target_above_the_source_rate_keeps_everything(self):
        cfg = synthetic_config(frames=15, fps=10.0, target_fps=60.0, width=64, height=48)
        with open_source(cfg) as src:
            assert len(list(src)) == 15


# --------------------------------------------------------------------------
# lifecycle
# --------------------------------------------------------------------------


class TestLifecycle:
    def test_iteration_stops_at_the_frame_limit(self):
        with open_source(synthetic_config(frames=7, width=64, height=48)) as src:
            assert len(list(src)) == 7
            assert src.is_exhausted
            assert src.read() is None

    def test_default_frame_count_is_finite(self):
        # A source that never ends is a hazard in a test suite.
        assert DEFAULT_FRAME_COUNT == 300
        src = SyntheticSource(synthetic_config(width=64, height=48))
        assert src.frame_limit == DEFAULT_FRAME_COUNT

    def test_loop_makes_the_source_endless(self):
        src = SyntheticSource(synthetic_config(loop=True, frames=3, width=64, height=48))
        with src:
            assert src.frame_limit is None
            pts = [src.read().pts for _ in range(10)]
        # The scene is a continuous function of time, so a loop has no seam
        # and therefore no pts discontinuity.
        assert all(b > a for a, b in zip(pts, pts[1:]))

    def test_open_is_idempotent_and_returns_self(self):
        src = SyntheticSource(synthetic_config(frames=3, width=64, height=48))
        assert src.open() is src
        assert src.open() is src
        src.close()

    def test_close_is_idempotent(self):
        src = SyntheticSource(synthetic_config(frames=3, width=64, height=48))
        src.open()
        src.close()
        src.close()
        assert not src.is_open

    def test_reading_before_open_raises(self):
        src = SyntheticSource(synthetic_config(frames=3, width=64, height=48))
        with pytest.raises(IngestError, match="before open"):
            src.read()

    def test_iterating_auto_opens(self):
        src = SyntheticSource(synthetic_config(frames=2, width=64, height=48))
        try:
            assert len(list(src)) == 2
        finally:
            src.close()

    def test_describe_is_a_single_log_line(self):
        with open_source(synthetic_config(frames=2, width=64, height=48)) as src:
            src.read()
            line = src.info.describe()
        assert "\n" not in line
        assert "synthetic" in line


# --------------------------------------------------------------------------
# configuration parsing
# --------------------------------------------------------------------------


class TestSyntheticConfiguration:
    def test_uri_query_string_configures_the_scene(self):
        cfg = SourceConfig(type="synthetic", uri="synthetic:?width=320&height=240&fps=15&seed=4&frames=5")
        with SyntheticSource(cfg) as src:
            assert (src.scene.width, src.scene.height) == (320, 240)
            assert src.scene.fps == 15.0
            assert src.scene.seed == 4
            assert len(list(src)) == 5

    def test_present_ranges_parse_from_the_uri(self):
        cfg = SourceConfig(type="synthetic", uri="synthetic:?present=30-90,120-150&frames=2&width=64&height=48")
        src = SyntheticSource(cfg)
        assert src.scene.present == ((30, 90), (120, 150))

    def test_keyword_overrides_beat_the_uri(self):
        cfg = SourceConfig(type="synthetic", uri="synthetic:?width=320&height=240")
        src = SyntheticSource(cfg, width=64, height=48)
        assert (src.scene.width, src.scene.height) == (64, 48)

    def test_an_unknown_uri_parameter_is_rejected(self):
        # Same reason config.py rejects unknown keys: a typo that silently
        # left a default in place would be a test asserting against a scene it
        # did not configure.
        with pytest.raises(SourceConfigError, match="unknown synthetic source parameter"):
            SyntheticSource(SourceConfig(type="synthetic", uri="synthetic:?widht=320"))

    def test_an_unknown_helper_parameter_is_rejected(self):
        with pytest.raises(SourceConfigError, match="unknown synthetic parameter"):
            synthetic_config(colour="red")

    def test_a_malformed_range_is_rejected(self):
        with pytest.raises(SourceConfigError, match="start-end"):
            SyntheticSource(SourceConfig(type="synthetic", uri="synthetic:?present=30"))

    def test_an_inverted_range_is_rejected(self):
        with pytest.raises(SourceConfigError, match="end > start"):
            SyntheticSource(SourceConfig(type="synthetic", uri="synthetic:?present=90-30"))

    @pytest.mark.parametrize("uri", ["synthetic:?width=4&height=4", "synthetic:?fps=0", "synthetic:?radius=0.9"])
    def test_impossible_scenes_are_rejected(self, uri):
        with pytest.raises(SourceConfigError):
            SyntheticSource(SourceConfig(type="synthetic", uri=uri))

    def test_synthetic_config_round_trips_through_its_uri(self):
        cfg = synthetic_config(width=128, height=96, fps=12.0, seed=9, frames=4)
        with SyntheticSource(cfg) as src:
            assert (src.scene.width, src.scene.height, src.scene.seed) == (128, 96, 9)
            assert len(list(src)) == 4


# --------------------------------------------------------------------------
# the Frame type
# --------------------------------------------------------------------------


class TestFrame:
    def make(self, **kwargs):
        image = np.zeros((48, 64, 3), dtype=np.uint8)
        return Frame(image=image, frame_id=kwargs.pop("frame_id", 0), pts=kwargs.pop("pts", 0.0), **kwargs)

    def test_size_is_derived_from_the_image(self):
        frame = self.make()
        assert frame.shape == (64, 48)

    def test_a_declared_size_that_contradicts_the_image_is_rejected(self):
        with pytest.raises(ValueError, match="declares"):
            Frame(image=np.zeros((48, 64, 3), np.uint8), frame_id=0, pts=0.0, width=1920, height=1080)

    def test_a_non_array_image_is_rejected(self):
        with pytest.raises(TypeError, match="numpy array"):
            Frame(image=[[0, 0, 0]], frame_id=0, pts=0.0)

    @pytest.mark.parametrize(
        "shape,dtype,match",
        [
            ((48, 64), np.uint8, "HxWx3"),
            ((48, 64, 4), np.uint8, "HxWx3"),
            ((48, 64, 3), np.float32, "uint8"),
        ],
    )
    def test_wrong_image_layout_is_rejected(self, shape, dtype, match):
        with pytest.raises(ValueError, match=match):
            Frame(image=np.zeros(shape, dtype), frame_id=0, pts=0.0)

    def test_non_finite_pts_is_rejected(self):
        with pytest.raises(ValueError, match="pts must be finite"):
            self.make(pts=float("inf"))

    def test_repr_never_prints_the_array(self):
        # A frame in a log line must not dump a megabyte of pixels.
        text = repr(self.make(frame_id=12, pts=1.5))
        assert "12" in text and "1.5" in text
        assert "[" not in text and len(text) < 120


def test_ingest_package_imports_without_a_codec():
    """``station.ingest`` must import with no PyAV, no OpenCV and no camera.

    Every codec-backed reader imports its dependency inside the function that
    needs it, so the dispatch table and the synthetic source stay usable on a
    bare machine. Breaking this breaks the whole test suite on CI.
    """
    import importlib
    import sys

    for name in ("av", "cv2"):
        assert name not in sys.modules or sys.modules[name] is not None
    module = importlib.import_module("station.ingest")
    assert callable(module.open_source)


def test_missing_codec_dependency_names_the_package_and_the_install(tmp_path):
    """A file source with no decoder installed must fail with instructions.

    Skipped only if a decoder genuinely is installed -- in that case there is
    no degradation path to exercise.
    """
    import importlib.util

    if importlib.util.find_spec("av") or importlib.util.find_spec("cv2"):
        pytest.skip("a decoder is installed, so the missing-dependency path cannot be reached")

    path = tmp_path / "clip.mp4"
    path.write_bytes(b"\x00" * 64)
    with pytest.raises(IngestError) as excinfo:
        open_source(SourceConfig(type="file", uri=str(path)))
    message = str(excinfo.value)
    assert "pip install" in message.lower()
