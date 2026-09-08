"""End to end: synthetic source -> stub runner -> temporal filter -> incident log.

No GPU, no codec, no drone, no model. Every component in this path is the real
one; only the *weights* are absent, and the stub runner exists precisely so
that the parts most likely to be wrong -- pts handling, the drop policy, what
gets logged and what does not -- can be exercised without them.

The two assertions this file exists for:

1. **The incident log contains the empty frames.** A log holding only frames
   with detections is a highlight reel of what the model happened to catch,
   and the false-negative audit in ``docs/VALIDATION.md`` has no denominator
   without the empties. Every frame the model *looked at* is a record.
2. **A rate-gated frame is not logged as an empty one.** ``infer()`` returning
   ``None`` means the model did not look. Writing that down as "looked,
   proposed nothing" fabricates evidence that the scene was examined, and it
   is the one lie this system's evidence base cannot survive.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import pytest

from station.core.types import CLASS_FIRE, BBox, Detection, FrameDetections, PipelineState
from station.incidentlog.reader import IncidentLogReader
from station.inference.stub import StubModelRunner
from station.inference.temporal import TemporalFilter
from station.ingest import open_source
from station.ingest.synthetic_source import synthetic_config
from station.ingest.synthetic_source import SyntheticSource
from station.pipeline import Pipeline, PipelineError, ReplayRunner, load_incident_detections

SOURCE_FPS = 10.0
SOURCE_FRAMES = 60
#: Generated-frame indices on which the script proposes a fire. Long enough to
#: clear a 3-of-5 gate with room to spare, and bounded so there are empty
#: frames on both sides of it -- and so the empties outnumber the detections,
#: which is what a real log looks like.
FIRE_FRAMES = range(20, 36)
FIRE_BOX = BBox(0.40, 0.35, 0.55, 0.50)


def fire_script(_frame, _pts, frame_id):
    """Propose a fire on a fixed window of frame ids.

    Keyed on ``frame_id`` rather than on the stub's internal step counter, so
    the script stays aligned no matter how many frames the drop policy
    discards -- the test asserts on pts, and a script that drifted with drops
    would make those assertions meaningless.
    """
    if frame_id in FIRE_FRAMES:
        return (Detection(cls=CLASS_FIRE, conf=0.90, box=FIRE_BOX),)
    return ()


class PacedSyntheticSource(SyntheticSource):
    """A synthetic source that decodes at a believable pace.

    Without this the decode thread generates 64x48 frames faster than the
    event loop can be scheduled, the two-frame queue overflows, and the drop
    policy discards a different, load-dependent set of frames on every run --
    which would make every assertion below either flaky or vacuous.

    3 ms per frame is still ~330 fps, far faster than any real source and far
    slower than the stub, so the queue stays empty and nothing is dropped.
    Note what this does *not* do: it does not raise the queue size. A large
    queue would end the run with frames still in it that the model never
    looked at, which is a different and worse kind of untruth.

    3 ms is right for tests about the drop policy, which need the source to be
    able to outrun inference. It is too tight for tests that assert *which*
    frame carries the first detection: on a contended CI runner the event loop
    is not always scheduled within 3 ms, the queue overflows, and the drop
    policy discards exactly the frames being asserted about. Those tests use
    :func:`unhurried_factory` instead.
    """

    PERIOD_S = 0.003

    def _read_raw(self):
        raw = super()._read_raw()
        if raw is not None:
            time.sleep(self.PERIOD_S)
        return raw


def paced_factory(cfg):
    """A ``source_factory`` producing :class:`PacedSyntheticSource`."""
    return lambda: PacedSyntheticSource(cfg.source).open()


class UnhurriedSyntheticSource(PacedSyntheticSource):
    """Paced slowly enough that a loaded machine still drops nothing.

    For tests that assert on the index of a particular frame. They are not
    about the drop policy, and a dropped frame makes them assert about a
    different frame than the one they mean - which is how a green suite became
    a red build that reproduced on no developer machine.
    """

    PERIOD_S = 0.010


def unhurried_factory(cfg):
    """A ``source_factory`` producing :class:`UnhurriedSyntheticSource`."""
    return lambda: UnhurriedSyntheticSource(cfg.source).open()


class LockstepSyntheticSource(SyntheticSource):
    """A source that will not produce a frame until the model has room for it.

    Pacing by a sleep is a guess about how fast the machine is, and on a
    shared CI runner the guess is wrong: the loop is not scheduled, the queue
    overflows, and the drop policy discards the exact frames a test is
    asserting about. Three milliseconds was wrong, ten was wrong less often,
    and no number is right, because the thing being guessed at is load.

    So this does not guess. It waits until the queue is empty before reading
    the next frame, which makes the whole path lossless whatever the machine
    is doing: nothing is ever dropped, so a test can say "the detection is on
    frame 20" and mean it. It is a slower source than any real one, which is
    the point -- a real source outruns inference and the drop policy is what
    handles that, and the tests that are about the drop policy use the paced
    source and the shipped queue instead.

    Note what this does *not* do: it does not raise the queue size. A deep
    queue ends the run with frames still in it that the model never looked at,
    because shutdown abandons whatever is queued. That was tried, and it
    turned a run that dropped frames 20 to 35 into a run that abandoned them,
    which is the same red build wearing a different hat.
    """

    #: A ceiling on the wait, so a stalled or dead inference thread ends the
    #: test with an ordinary assertion rather than hanging the suite.
    MAX_WAIT_S = 5.0

    def __init__(self, cfg, room):
        super().__init__(cfg)
        self._room = room

    def _read_raw(self):
        deadline = time.monotonic() + self.MAX_WAIT_S
        while not self._room() and time.monotonic() < deadline:
            time.sleep(0.001)
        return super()._read_raw()


def lockstep_factory(cfg, holder):
    """A ``source_factory`` producing :class:`LockstepSyntheticSource`.

    ``holder`` is filled in with the pipeline after it is constructed, because
    the source has to see the queue it is feeding and the factory runs inside
    the pipeline's own start-up.
    """
    def room():
        pipe = holder.get("pipe")
        if pipe is None:
            return True
        # Reaching for a private attribute, deliberately. The alternative is
        # guessing at timing again, and the queue is the thing being waited on.
        return pipe._queue.empty() or pipe._stop.is_set()

    return lambda: LockstepSyntheticSource(cfg.source, room).open()


def build(cfg, **kwargs) -> Pipeline:
    """Build a pipeline at the shipped queue size, fed in lockstep.

    The queue stays at its default of 2, because that is the drop policy the
    station actually ships and a test that raised it would be testing a
    configuration nobody runs -- and worse, would end the run with frames
    still queued that the model never saw.

    What changes instead is the source: it waits for the model rather than
    racing it, so nothing is dropped however loaded the machine is, and a test
    that says "frame 20" is asserting about frame 20. The tests that are
    genuinely about the drop policy pass their own fast source and want it to
    fire.
    """
    runner = kwargs.pop("runner", StubModelRunner(cfg.inference, script=fire_script))
    holder: dict[str, Pipeline] = {}
    kwargs.setdefault("source_factory", lockstep_factory(cfg, holder))
    pipe = Pipeline(cfg, runner=runner, **kwargs)
    holder["pipe"] = pipe
    return pipe


def first_with_detections(frames, what: str) -> int:
    """Index of the first frame carrying a detection.

    A helper rather than a bare ``next``, because ``next`` over an empty
    generator raises ``StopIteration`` and tells whoever reads the failure
    nothing at all -- not how many frames there were, not whether any survived
    the drop policy, not whether the run produced anything. That is precisely
    how this arrived from CI.
    """
    for index, frame in enumerate(frames):
        if len(frame) > 0:
            return index
    raise AssertionError(
        f"no {what} frame carried a detection. "
        f"{len(frames)} frames were logged inside the window; "
        "if that is 0 the drop policy discarded them all, which means the "
        "paced source is running faster than this machine can consume it."
    )


async def run_collecting(pipe: Pipeline) -> list[FrameDetections]:
    """Run a pipeline to completion, collecting everything it publishes."""
    seen: list[FrameDetections] = []
    pipe.subscribe(seen.append)
    await pipe.run()
    # Subscribers are dispatched onto the loop from the inference thread, so a
    # handful can still be in flight when run() returns. One turn plus a short
    # sleep drains them; the assertions below tolerate a short tail regardless.
    await asyncio.sleep(0.1)
    return seen


@pytest.fixture
def pipeline_cfg(station_config):
    """A config that runs a short, fully-deterministic synthetic clip."""
    return station_config(
        **{
            "source.uri": f"synthetic:?fps={SOURCE_FPS:g}&frames={SOURCE_FRAMES}&width=64&height=48",
            "inference.max_fps": 1000.0,   # gate everything through; gating has its own test
            "inference.conf_threshold": 0.25,
            "stream.status_interval_s": 0.05,
        }
    )


# --------------------------------------------------------------------------
# the end-to-end run
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def run_result(tmp_path_factory):
    """One full run, shared by every assertion in :class:`TestEndToEnd`.

    Module-scoped because the run takes a moment and every assertion is a
    different question about the same evidence -- which is exactly how an
    incident log is read after the fact.

    Returns ``(pipeline, published, logged)``. ``published`` is what reached
    the subscriber; ``logged`` is what reached ``detections.jsonl``. They are
    not interchangeable: the log is written synchronously inside ``_publish``
    while subscribers are dispatched onto the event loop, so the log is the
    complete record and the subscriber list may be a frame or two short of it.
    """
    from station.core.config import Config, InferenceConfig, SourceConfig

    tmp_path = tmp_path_factory.mktemp("e2e")
    cfg = Config()
    cfg.station_name = "test station"
    cfg.source = SourceConfig(
        type="synthetic",
        uri=f"synthetic:?fps={SOURCE_FPS:g}&frames={SOURCE_FRAMES}&width=64&height=48",
    )
    cfg.inference = InferenceConfig(max_fps=1000.0)
    cfg.incident_log.dir = str(tmp_path / "incidents")
    cfg.incident_log.record_video = False
    cfg.stream.status_interval_s = 0.05

    pipe = build(cfg)
    published = asyncio.run(run_collecting(pipe))
    return pipe, published, load_incident_detections(pipe.incident_dir)


class TestEndToEnd:

    def test_the_run_completed_and_stopped(self, run_result):
        pipe, _published, _logged = run_result
        assert pipe.state == PipelineState.STOPPED
        assert pipe.error is None

    def test_frames_reached_the_subscriber(self, run_result):
        _pipe, published, logged = run_result
        assert published, "no detections payloads reached the subscriber"
        assert all(isinstance(f, FrameDetections) for f in published)

    def test_published_pts_are_the_source_timeline_untouched(self, run_result):
        """pts survives ingest -> runner -> filter -> publish exactly.

        With no ``target_fps`` the synthetic source emits ``pts = frame_id /
        fps``, so this is an equality rather than a tolerance. A box drawn one
        frame late is a box in the wrong place, which docs/CONTRACT.md calls
        this system's most dangerous failure mode -- and pts is the only field
        that ties a payload to the frame it was computed from.
        """
        _pipe, published, logged = run_result
        for frame in published:
            assert frame.pts == pytest.approx(frame.frame_id / SOURCE_FPS, abs=1e-9)
        pts = [f.pts for f in published]
        assert all(b > a for a, b in zip(pts, pts[1:])), "published pts is not increasing"
        assert pts[0] == 0.0

    def test_the_subscriber_saw_empty_frames_as_well_as_populated_ones(self, run_result):
        _pipe, published, _logged = run_result
        empties = [f for f in published if len(f) == 0]
        populated = [f for f in published if len(f) > 0]
        assert empties, "the subscriber never saw an empty observation"
        assert populated, "the subscriber never saw a detection"

    def test_detections_are_the_scripted_box_after_the_filter_confirmed_it(self, run_result):
        """The box is the scripted one, stamped by the filter, and it appears
        on the third frame the model *looked at* inside the scripted window --
        3-of-5, counted in inferred frames rather than in source frames."""
        _pipe, _published, logged = run_result
        populated = [f for f in logged if len(f) > 0]
        assert populated
        for frame in populated:
            for d in frame.detections:
                assert d.cls == CLASS_FIRE
                assert d.box == FIRE_BOX
                assert d.track_id is not None, "the temporal filter did not stamp a track id"
                assert d.persisted >= 1

        inside = [f for f in logged if f.frame_id in FIRE_FRAMES]
        assert len(inside) >= 5, "too few scripted frames survived the drop policy to judge"
        first_with_box = next(i for i, f in enumerate(inside) if len(f) > 0)
        assert first_with_box == 2, "the 3-of-5 gate did not open on the third inferred frame"

    def test_boxes_do_not_outlive_the_evidence_by_more_than_max_age(self, run_result):
        """Coasting is bounded, and bounded in *inferred* frames.

        The filter ages a track once per frame the model looked at, not once
        per source frame -- so with frames being dropped at the queue, a box
        coasts for ``max_age`` inferences however many source frames that
        spans.
        """
        pipe, _published, logged = run_result
        max_age = pipe.cfg.temporal.max_age
        last_evidence = max(i for i, f in enumerate(logged) if f.frame_id in FIRE_FRAMES)
        trailing = logged[last_evidence + 1:]
        coasted = [f for f in trailing if len(f) > 0]
        # min(): if the clip ended sooner than max_age inferences later, the
        # box is still coasting when the log stops, and that is correct.
        assert len(coasted) == min(max_age, len(trailing)), (
            f"coasted for {len(coasted)} of {len(trailing)} trailing frames, max_age is {max_age}"
        )
        # ...and nothing was drawn before the model had seen anything.
        first_evidence = min(i for i, f in enumerate(logged) if f.frame_id in FIRE_FRAMES)
        assert all(len(f) == 0 for f in logged[:first_evidence])

    def test_every_published_frame_carries_the_stub_identity(self, run_result):
        # The incident log is the evidence base for the false-negative audit;
        # stub boxes recorded under a trained model's name would poison it.
        _pipe, published, logged = run_result
        for frame in published:
            assert frame.model is not None
            assert frame.model.name == "stub"
            assert frame.model.weights_sha is None

    def test_payloads_are_wire_legal(self, run_result):
        from station.core.types import parse_message

        _pipe, published, logged = run_result
        for frame in published:
            # Compared on the wire form: to_wire() rounds, so a re-parsed
            # payload is equal *as a message* rather than as an object.
            assert parse_message(frame.to_json()).to_wire() == frame.to_wire()


# --------------------------------------------------------------------------
# the incident log
# --------------------------------------------------------------------------


class TestIncidentLogContainsEmptyFrames:
    @pytest.fixture
    def logged(self, pipeline_cfg):
        pipe = build(pipeline_cfg)
        published = asyncio.run(run_collecting(pipe))
        assert pipe.incident_dir is not None
        return pipe, published, Path(pipe.incident_dir)

    def test_the_log_directory_was_created(self, logged):
        _pipe, _published, incident_dir = logged
        assert incident_dir.is_dir()
        assert (incident_dir / "detections.jsonl").is_file()
        assert (incident_dir / "meta.json").is_file()

    def test_the_log_contains_empty_frames(self, logged):
        """The assertion this whole file exists for.

        A log of only the frames that had detections is a highlight reel. The
        empties are what make it a record of what the model *saw*, and they are
        the denominator of every false-negative number in docs/VALIDATION.md.
        """
        _pipe, _published, incident_dir = logged
        records = [
            json.loads(line)
            for line in (incident_dir / "detections.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert records, "nothing was logged at all"
        empty = [r for r in records if r["detections"] == []]
        populated = [r for r in records if r["detections"]]
        assert empty, "the incident log holds no empty frames -- it is a highlight reel"
        assert populated, "the incident log holds no detections either"
        # 16 scripted frames out of 60, so the empties dominate -- as they do
        # in any real log.
        assert len(empty) > len(populated)

    def test_every_record_has_the_detections_key_even_when_empty(self, logged):
        # Omitting the key to save bytes would make "the model looked and
        # proposed nothing" indistinguishable from a truncated record.
        _pipe, _published, incident_dir = logged
        for line in (incident_dir / "detections.jsonl").read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            assert "detections" in record
            assert isinstance(record["detections"], list)

    def test_the_log_reads_back_through_the_reader_with_the_empties_intact(self, logged):
        _pipe, _published, incident_dir = logged
        reader = IncidentLogReader(incident_dir)
        frames = list(reader.frames(keep_empty=True))
        assert any(len(f) == 0 for f in frames)
        kept = list(reader.frames(keep_empty=False))
        assert len(kept) < len(frames), "keep_empty made no difference; the empties are missing"

    def test_the_summary_counts_the_empty_frames_separately(self, logged):
        _pipe, _published, incident_dir = logged
        summary = IncidentLogReader(incident_dir).summarise()
        assert summary.frames_empty > 0
        assert summary.frames_logged == summary.frames_empty + summary.frames_with_detections

    def test_logged_pts_match_the_published_pts_in_order(self, logged):
        _pipe, published, incident_dir = logged
        logged_pts = [round(f.pts, 4) for f in load_incident_detections(incident_dir)]
        published_pts = [round(f.pts, 4) for f in published]
        # The log is written before the subscriber is dispatched, so the log
        # may hold a couple of frames the subscriber has not been handed yet.
        assert logged_pts[: len(published_pts)] == published_pts
        assert 0 <= len(logged_pts) - len(published_pts) <= 3

    def test_logged_pts_are_strictly_increasing(self, logged):
        _pipe, _published, incident_dir = logged
        pts = [f.pts for f in load_incident_detections(incident_dir)]
        assert all(b > a for a, b in zip(pts, pts[1:]))

    def test_the_log_never_invents_a_frame(self, logged):
        _pipe, _published, incident_dir = logged
        legal = {round(i / SOURCE_FPS, 4) for i in range(SOURCE_FRAMES)}
        for frame in load_incident_detections(incident_dir):
            assert round(frame.pts, 4) in legal

    def test_meta_records_the_stub_identity_and_the_config(self, logged):
        _pipe, _published, incident_dir = logged
        meta = json.loads((incident_dir / "meta.json").read_text(encoding="utf-8"))
        assert meta["model"]["name"] == "stub"
        assert meta["closed_cleanly"] is True
        assert meta["config"]["inference"]["conf_threshold"] == 0.25

    def test_meta_carries_its_own_reading_instructions_for_empty_results(self, logged):
        # A log read years later has to explain itself: an empty list is a
        # null result, not a finding.
        _pipe, _published, incident_dir = logged
        meta = json.loads((incident_dir / "meta.json").read_text(encoding="utf-8"))
        blob = json.dumps(meta).lower()
        assert "empty" in blob
        from station.core.safety import find_forbidden_phrases

        assert find_forbidden_phrases(json.dumps(meta, indent=2)) == []

    def test_a_disabled_log_writes_nothing(self, pipeline_cfg):
        cfg = pipeline_cfg
        cfg.incident_log.enabled = False
        pipe = build(cfg)
        published = asyncio.run(run_collecting(pipe))
        assert published, "publishing must not depend on logging"
        assert pipe.incident_dir is None
        assert not Path(cfg.incident_log.dir).exists()


# --------------------------------------------------------------------------
# a gated frame is not an observation
# --------------------------------------------------------------------------


class TestRateGatingIsNotAnObservation:
    def test_gated_frames_are_neither_published_nor_logged(self, pipeline_cfg):
        """``infer()`` returning ``None`` means the model did not look.

        Recording that as an empty frame would fabricate evidence that the
        scene was examined -- and the false-negative audit reads exactly those
        records as "the model looked here and found nothing".
        """
        cfg = pipeline_cfg
        cfg.inference.max_fps = 2.0            # source runs at 10 fps
        pipe = build(cfg)
        published = asyncio.run(run_collecting(pipe))

        m = pipe.metrics
        assert m.frames_gated > 0, "the rate gate never fired; the test proves nothing"
        assert m.frames_published == m.frames_inferred
        assert m.frames_dequeued == m.frames_inferred + m.frames_gated

        logged = load_incident_detections(pipe.incident_dir)
        assert len(logged) <= m.frames_inferred
        assert len(published) <= m.frames_inferred

    def test_gating_keeps_the_sampling_visible_in_the_pts_gaps(self, pipeline_cfg):
        # The operator is entitled to see the sampling rate: it bounds how
        # briefly a fire could appear and still be missed entirely.
        cfg = pipeline_cfg
        cfg.inference.max_fps = 2.0
        pipe = build(cfg)
        asyncio.run(run_collecting(pipe))
        pts = [f.pts for f in load_incident_detections(pipe.incident_dir)]
        gaps = [round(b - a, 3) for a, b in zip(pts, pts[1:])]
        assert gaps and min(gaps) >= 0.4, gaps

    def test_dropped_frames_are_counted_and_never_fabricated(self, pipeline_cfg):
        # The drop policy discards the OLDEST queued frame. Whatever it drops,
        # it must not appear in the log as an observation.
        cfg = pipeline_cfg
        pipe = build(
            cfg,
            runner=StubModelRunner(cfg.inference, script=fire_script, latency_ms=3.0),
            queue_size=2,   # the shipped default: this is the policy under test
        )
        asyncio.run(run_collecting(pipe))
        m = pipe.metrics
        assert m.frames_read == m.frames_dequeued + m.frames_dropped + pipe._queue.qsize()
        logged = load_incident_detections(pipe.incident_dir)
        assert len(logged) == m.frames_published


# --------------------------------------------------------------------------
# the temporal filter in the pipeline
# --------------------------------------------------------------------------


class TestTemporalFilterIntegration:
    def test_the_configured_n_of_m_is_the_one_applied(self, pipeline_cfg):
        cfg = pipeline_cfg
        cfg.temporal.n, cfg.temporal.m = 5, 5
        pipe = build(cfg, source_factory=unhurried_factory(cfg))
        asyncio.run(run_collecting(pipe))
        logged = load_incident_detections(pipe.incident_dir)
        inside = [f for f in logged if f.frame_id in FIRE_FRAMES]
        assert first_with_detections(inside, "n-of-m filtered") == 4

    def test_temporal_none_publishes_the_runner_output_unfiltered(self, pipeline_cfg):
        # Replay uses this: the recorded payloads are already filtered, and
        # re-filtering would swallow the opening frames of every track.
        pipe = Pipeline(
            pipeline_cfg,
            runner=StubModelRunner(pipeline_cfg.inference, script=fire_script),
            temporal=None,
            source_factory=unhurried_factory(pipeline_cfg),
        )
        published = asyncio.run(run_collecting(pipe))
        logged = load_incident_detections(pipe.incident_dir)
        inside = [f for f in logged if f.frame_id in FIRE_FRAMES]
        assert first_with_detections(inside, "unfiltered") == 0
        assert all(d.track_id is None for f in published for d in f.detections)

    def test_an_explicit_filter_instance_is_used(self, pipeline_cfg):
        from station.core.config import TemporalConfig

        filt = TemporalFilter(TemporalConfig(n=1, m=2))
        pipe = Pipeline(
            pipeline_cfg,
            runner=StubModelRunner(pipeline_cfg.inference, script=fire_script),
            temporal=filt,
            source_factory=unhurried_factory(pipeline_cfg),
        )
        asyncio.run(run_collecting(pipe))
        assert filt.frames_seen > 0
        logged = load_incident_detections(pipe.incident_dir)
        inside = [f for f in logged if f.frame_id in FIRE_FRAMES]
        assert first_with_detections(inside, "unfiltered") == 0

    def test_the_filter_sees_every_inferred_frame_including_the_empty_ones(self, pipeline_cfg):
        from station.core.config import TemporalConfig

        filt = TemporalFilter(TemporalConfig(n=3, m=5))
        pipe = Pipeline(
            pipeline_cfg,
            runner=StubModelRunner(pipeline_cfg.inference, script=fire_script),
            temporal=filt,
            source_factory=paced_factory(pipeline_cfg),
        )
        asyncio.run(run_collecting(pipe))
        # Skipping empty frames would freeze every window instead of ageing
        # it, and confirmed boxes would then coast forever.
        assert filt.frames_seen == pipe.metrics.frames_inferred


# --------------------------------------------------------------------------
# status, lifecycle and failure handling
# --------------------------------------------------------------------------


class TestStatus:
    def test_status_is_answerable_before_start(self, pipeline_cfg):
        pipe = build(pipeline_cfg)
        status = pipe.status()
        assert status.state in PipelineState.ALL
        assert status.note  # says what it is waiting for

    def test_status_never_describes_the_scene(self, pipeline_cfg):
        from station.core.safety import find_forbidden_phrases

        pipe = build(pipeline_cfg)
        statuses = []
        pipe.subscribe_status(statuses.append)
        asyncio.run(run_collecting(pipe))
        assert statuses, "no heartbeat was emitted"
        for status in statuses:
            assert find_forbidden_phrases(status.to_json()) == []
            assert status.state in PipelineState.ALL

    def test_the_final_state_is_stopped_with_the_reason(self, pipeline_cfg):
        pipe = build(pipeline_cfg)
        asyncio.run(run_collecting(pipe))
        status = pipe.status()
        assert status.state == PipelineState.STOPPED
        assert status.note == "source finished"

    def test_status_recomputes_rather_than_replaying_the_last_heartbeat(self, pipeline_cfg):
        # /healthz and the data-channel heartbeat must never disagree.
        pipe = build(pipeline_cfg)
        asyncio.run(run_collecting(pipe))
        assert pipe.status().state == pipe.status().state == PipelineState.STOPPED

    def test_metrics_add_up(self, pipeline_cfg):
        pipe = build(pipeline_cfg)
        published = asyncio.run(run_collecting(pipe))
        m = pipe.metrics
        assert m.frames_read == SOURCE_FRAMES
        assert m.frames_published == m.frames_inferred
        assert m.frames_inferred > 0
        assert m.inference_errors == 0
        assert m.sink_errors == 0
        assert m.stream_start_pts == 0.0
        assert len(published) <= m.frames_published
        assert m.as_dict()["frames_published"] == m.frames_published


class TestLifecycle:
    def test_starting_twice_is_refused(self, pipeline_cfg):
        async def main():
            pipe = build(pipeline_cfg)
            await pipe.start()
            try:
                with pytest.raises(PipelineError, match="already started"):
                    await pipe.start()
            finally:
                await pipe.stop()

        asyncio.run(main())

    def test_stop_is_idempotent_and_safe_before_start(self, pipeline_cfg):
        async def main():
            pipe = build(pipeline_cfg)
            await pipe.stop()
            await pipe.stop()
            assert pipe.state == PipelineState.STOPPED

        asyncio.run(main())

    def test_async_context_manager_stops_the_pipeline(self, pipeline_cfg):
        async def main():
            async with build(pipeline_cfg) as pipe:
                assert pipe.state in PipelineState.ALL
            assert pipe.state == PipelineState.STOPPED
            return pipe

        pipe = asyncio.run(main())
        assert pipe.incident_dir is not None

    def test_unsubscribe_stops_delivery(self, pipeline_cfg):
        seen: list[FrameDetections] = []

        async def main():
            pipe = build(pipeline_cfg)
            unsubscribe = pipe.subscribe(seen.append)
            unsubscribe()
            unsubscribe()  # idempotent
            await pipe.run()
            await asyncio.sleep(0.05)

        asyncio.run(main())
        assert seen == []

    def test_a_raising_subscriber_cannot_stop_the_station(self, pipeline_cfg):
        # Subscribers are third-party code hanging off a system whose job is
        # to keep showing video.
        good: list[FrameDetections] = []

        def bad(_frame):
            raise RuntimeError("subscriber blew up")

        async def main():
            pipe = build(pipeline_cfg)
            pipe.subscribe(bad)
            pipe.subscribe(good.append)
            await pipe.run()
            await asyncio.sleep(0.1)
            return pipe

        pipe = asyncio.run(main())
        assert good, "a raising subscriber starved the others"
        assert pipe.metrics.sink_errors > 0
        assert pipe.error is None

    def test_an_async_subscriber_is_awaited(self, pipeline_cfg):
        seen: list[float] = []

        async def sink(frame):
            seen.append(frame.pts)

        async def main():
            pipe = build(pipeline_cfg)
            pipe.subscribe(sink)
            await pipe.run()
            await asyncio.sleep(0.1)

        asyncio.run(main())
        assert seen

    def test_both_source_and_source_factory_is_refused(self, pipeline_cfg):
        src = open_source(synthetic_config(frames=2, width=64, height=48), autostart=False)
        with pytest.raises(ValueError, match="not both"):
            Pipeline(pipeline_cfg, source=src, source_factory=lambda: src)

    @pytest.mark.parametrize("size", [0, -1])
    def test_a_useless_queue_size_is_refused(self, pipeline_cfg, size):
        with pytest.raises(ValueError, match="queue_size"):
            Pipeline(pipeline_cfg, queue_size=size)

    def test_an_explicit_source_object_is_used(self, pipeline_cfg):
        src = open_source(
            synthetic_config(frames=12, fps=SOURCE_FPS, width=64, height=48), autostart=False
        )
        pipe = Pipeline(
            pipeline_cfg,
            runner=StubModelRunner(pipeline_cfg.inference, script=fire_script),
            source=src,
        )
        asyncio.run(run_collecting(pipe))
        # Twelve is the number this source was built with and pipeline_cfg's is
        # sixty, so reading twelve is the assertion: the object handed in is the
        # one that was read.
        assert pipe.metrics.frames_read == 12
        # Deliberately not "assert published". This source is unpaced and the
        # queue is the shipped depth of two, so on a loaded machine the drop
        # policy can legitimately discard every frame before inference is
        # scheduled - and a test of which source object was used has no business
        # failing because the machine was busy. What must hold is that every
        # frame read was accounted for: inferred, or dropped and counted.
        m = pipe.metrics
        assert m.frames_dequeued + m.frames_dropped <= m.frames_read
        assert m.frames_dequeued + m.frames_dropped > 0


class TestInferenceFailure:
    def test_repeated_inference_failures_stop_the_pipeline_loudly(self, pipeline_cfg):
        """Emitting nothing forever is byte-identical to a working model on a
        quiet scene. A broken model must stop the run, not go quiet."""
        from station.inference.runner import InferenceUnavailableError

        class BrokenRunner(StubModelRunner):
            def infer(self, *args, **kwargs):
                raise InferenceUnavailableError("CUDA fell over")

        pipe = Pipeline(
            pipeline_cfg,
            runner=BrokenRunner(pipeline_cfg.inference),
            max_inference_errors=3,
            source_factory=paced_factory(pipeline_cfg),
        )

        async def main():
            with pytest.raises(InferenceUnavailableError, match="CUDA fell over"):
                await pipe.run()

        asyncio.run(main())
        assert pipe.error is not None
        assert pipe.metrics.inference_errors >= 3
        assert pipe.metrics.frames_published == 0

    def test_a_failed_run_still_closes_its_incident_log(self, pipeline_cfg):
        # An unclosed log leaves a directory with no final metadata for a run
        # that did happen, and closed_cleanly is how a reviewer tells the two
        # apart.
        from station.inference.runner import InferenceUnavailableError

        class BrokenRunner(StubModelRunner):
            def infer(self, *args, **kwargs):
                raise InferenceUnavailableError("no model")

        pipe = Pipeline(
            pipeline_cfg,
            runner=BrokenRunner(pipeline_cfg.inference),
            max_inference_errors=2,
            source_factory=paced_factory(pipeline_cfg),
        )

        async def main():
            with pytest.raises(InferenceUnavailableError):
                await pipe.run()

        asyncio.run(main())
        meta = json.loads((Path(pipe.incident_dir) / "meta.json").read_text(encoding="utf-8"))
        assert meta["closed_cleanly"] is True


# --------------------------------------------------------------------------
# replay
# --------------------------------------------------------------------------


class TestReplay:
    def test_a_recorded_incident_replays_frame_for_frame(self, pipeline_cfg):
        """Replay must reproduce the log exactly, empties included.

        The recorded payloads are already filtered, so replay runs with
        ``temporal=None``; re-filtering would re-apply N-of-M and swallow the
        opening frames of every track.
        """
        first = build(pipeline_cfg)
        asyncio.run(run_collecting(first))
        recorded = load_incident_detections(first.incident_dir)
        assert recorded

        replay_cfg = pipeline_cfg
        replay_cfg.incident_log.enabled = False
        pipe = Pipeline(
            replay_cfg,
            runner=ReplayRunner(recorded),
            temporal=None,
            source_factory=paced_factory(replay_cfg),
        )
        replayed = asyncio.run(run_collecting(pipe))

        assert replayed, "replay published nothing"
        by_pts = {round(f.pts, 4): f for f in recorded}
        for frame in replayed:
            original = by_pts[round(frame.pts, 4)]
            assert [d.box.as_tuple() for d in frame.detections] == [
                d.box.as_tuple() for d in original.detections
            ]

    def test_replay_preserves_the_empty_frames(self, pipeline_cfg):
        first = build(pipeline_cfg)
        asyncio.run(run_collecting(first))
        recorded = load_incident_detections(first.incident_dir)

        replay_cfg = pipeline_cfg
        replay_cfg.incident_log.enabled = False
        pipe = Pipeline(
            replay_cfg,
            runner=ReplayRunner(recorded),
            temporal=None,
            source_factory=paced_factory(replay_cfg),
        )
        replayed = asyncio.run(run_collecting(pipe))
        assert any(len(f) == 0 for f in replayed)


# --------------------------------------------------------------------------
# video recording, which is genuinely optional
# --------------------------------------------------------------------------


def test_video_recording_never_takes_the_detections_log_with_it(station_config):
    """With or without an encoder, the log must be complete.

    Losing the recording is bad; losing the record of what the model saw is
    not survivable, so the recorder is documented never to raise into the
    frame loop. This runs whether or not PyAV is installed -- when it is
    absent, the degradation path is what gets exercised.
    """
    cfg = station_config(
        **{
            "source.uri": f"synthetic:?fps={SOURCE_FPS:g}&frames=20&width=64&height=48",
            "inference.max_fps": 1000.0,
            "incident_log.record_video": True,
            "stream.status_interval_s": 0.05,
        }
    )
    pipe = build(cfg)
    published = asyncio.run(run_collecting(pipe))
    assert published
    logged = load_incident_detections(pipe.incident_dir)
    assert logged
    assert any(len(f) == 0 for f in logged)
