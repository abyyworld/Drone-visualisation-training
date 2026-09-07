"""Shared fixtures for the wildfire-watch test suite.

Two rules shape everything here:

1. **The core logic is tested without optional dependencies.** The wire
   contract, the config schema, the safety invariants, the N-of-M temporal
   filter and the synthetic ingest source must all be exercised on an
   interpreter with nothing but numpy, PyYAML and pytest. A test that only
   passes because it skipped is not a test, so ``pytest.importorskip`` appears
   only where a genuinely optional dependency (PyAV, aiortc, ultralytics) is
   the thing under test.
2. **Fixtures build wire-legal objects, not mocks.** Every helper below
   returns real :mod:`station.core.types` instances. A mock that accepts a box
   the real type would reject is a test that agrees with itself.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, Iterable, Sequence

import pytest

# The suite runs from a source checkout, which may not be pip-installed.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from station.core.config import (  # noqa: E402  (path fixed up above)
    Config,
    InferenceConfig,
    SourceConfig,
    TemporalConfig,
)
from station.core.types import BBox, Detection, FrameDetections, ModelInfo  # noqa: E402


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Absolute path to the repository root."""
    return REPO_ROOT


# --------------------------------------------------------------------------
# geometry / detection builders
# --------------------------------------------------------------------------


@pytest.fixture
def box() -> Callable[..., BBox]:
    """Build a square-ish normalised box from a centre and a size.

    Centre/size rather than corners because every temporal-filter scenario is
    naturally expressed as "the same region, moved a bit" or "the same region,
    grown a bit", and corner arithmetic obscures which of those a test means.
    """

    def _box(cx: float, cy: float, size: float = 0.10, aspect: float = 1.0) -> BBox:
        half_w = size * aspect / 2.0
        half_h = size / 2.0
        return BBox(
            max(0.0, cx - half_w),
            max(0.0, cy - half_h),
            min(1.0, cx + half_w),
            min(1.0, cy + half_h),
        )

    return _box


@pytest.fixture
def det(box: Callable[..., BBox]) -> Callable[..., Detection]:
    """Build a raw (pre-filter) detection.

    Raw detections carry no ``track_id`` and ``persisted == 1``: those fields
    are the temporal filter's output, and a fixture that pre-populated them
    would let a broken filter look correct.
    """

    def _det(
        cx: float = 0.5,
        cy: float = 0.5,
        size: float = 0.10,
        *,
        cls: str = "fire",
        conf: float = 0.80,
    ) -> Detection:
        return Detection(cls=cls, conf=conf, box=box(cx, cy, size))

    return _det


@pytest.fixture
def frame_result() -> Callable[..., FrameDetections]:
    """Build a :class:`FrameDetections` as the runner would emit one."""

    def _frame(
        frame_id: int = 0,
        pts: float = 0.0,
        detections: Sequence[Detection] | Iterable[Detection] = (),
        **kwargs,
    ) -> FrameDetections:
        return FrameDetections(
            frame_id=frame_id,
            pts=pts,
            detections=tuple(detections),
            **kwargs,
        )

    return _frame


@pytest.fixture
def model_info() -> ModelInfo:
    """A plausible model identity for round-trip and log tests."""
    return ModelInfo(
        name="yolo11s-fire",
        version="0.3.1",
        weights_sha="9f2c1ab",
        imgsz=640,
        conf_threshold=0.25,
    )


# --------------------------------------------------------------------------
# configuration builders
# --------------------------------------------------------------------------


@pytest.fixture
def temporal_cfg() -> Callable[..., TemporalConfig]:
    """Build a :class:`TemporalConfig`, defaulting to the shipped values."""

    def _cfg(**kwargs) -> TemporalConfig:
        return TemporalConfig(**kwargs)

    return _cfg


@pytest.fixture
def station_config(tmp_path: Path) -> Callable[..., Config]:
    """A :class:`Config` wired to a synthetic source and a temp incident dir.

    Every default that would touch the outside world -- real weights, a real
    camera, the repo's ``incidents/`` directory -- is redirected, so a test
    that forgets to override something still cannot escape ``tmp_path``.
    """

    def _cfg(**overrides) -> Config:
        cfg = Config()
        cfg.station_name = "test station"
        cfg.source = SourceConfig(type="synthetic", uri="synthetic:?fps=10&frames=30")
        cfg.inference = InferenceConfig(max_fps=1000.0)
        cfg.incident_log.dir = str(tmp_path / "incidents")
        cfg.incident_log.record_video = False
        for key, value in overrides.items():
            section, _, leaf = key.partition(".")
            if leaf:
                setattr(getattr(cfg, section), leaf, value)
            else:
                setattr(cfg, section, value)
        return cfg

    return _cfg


# --------------------------------------------------------------------------
# node, for the browser-side unit tests
# --------------------------------------------------------------------------


@pytest.fixture(scope="session")
def node_bin() -> str:
    """Path to the ``node`` executable.

    ``app/js/sync.js`` holds the staleness rule -- the single most
    safety-relevant piece of logic on the tablet -- so its tests are not
    optional in the way an ffmpeg-dependent test is. They are skipped only if
    node genuinely is not installed, and the skip message says so plainly.
    """
    found = shutil.which("node")
    if not found:
        pytest.skip("node is not installed; app/js unit tests cannot run")
    try:
        subprocess.run([found, "--version"], capture_output=True, check=True, timeout=30)
    except (subprocess.SubprocessError, OSError) as exc:  # pragma: no cover
        pytest.skip(f"node is present but not runnable: {exc}")
    return found
