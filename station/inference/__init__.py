"""Inference for the ground station: the model runner and the temporal filter.

Nothing here imports ultralytics, torch, cv2 or numpy at module load, so this
package can be imported -- and the temporal filter fully tested -- on a machine
with none of them installed.

The two halves are deliberately separate. :class:`ModelRunner` turns pixels
into raw per-frame proposals; :class:`TemporalFilter` decides which of those
proposals have persisted long enough to be worth an operator's attention. The
filter is where most of this system's usable accuracy comes from, and it works
identically behind the real runner and behind :class:`StubModelRunner`.

Typical wiring::

    runner = ModelRunner(cfg.inference)      # or StubModelRunner(cfg.inference)
    runner.load()
    filt = TemporalFilter(cfg.temporal)

    frame = runner.infer(image, pts=pts, frame_id=n, rtp_ts=rtp)
    if frame is not None:                    # None means "not inferred", not "nothing there"
        published = filt.filter_frame(frame)
        incident_log.write(published)
        data_channel.send(published.to_json())
"""

from __future__ import annotations

from station.inference.runner import (
    InferenceUnavailableError,
    ModelRunner,
    PtsRateLimiter,
    Runner,
    RunnerStats,
    resolve_device,
    weights_sha,
)
from station.inference.stub import StubModelRunner, drifting_script
from station.inference.temporal import TemporalFilter, Track

__all__ = [
    "InferenceUnavailableError",
    "ModelRunner",
    "PtsRateLimiter",
    "Runner",
    "RunnerStats",
    "StubModelRunner",
    "TemporalFilter",
    "Track",
    "drifting_script",
    "resolve_device",
    "weights_sha",
]
