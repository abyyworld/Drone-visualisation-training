"""The Kaggle notebook's dataset discovery, tested against realistic mounts.

The notebook has to find datasets on a filesystem it cannot see until it runs,
under names it does not choose: Kaggle mounts each dataset at
``/kaggle/input/<slug>`` with the slug lowercased and hyphenated, so a folder
uploaded as ``FLAME`` arrives as ``flame-dataset``. Getting this wrong is not a
crash -- the merge simply finds nothing, trains on whatever else was attached,
and the operator discovers it after a GPU session has been spent.

The logic is extracted from the notebook rather than copied, so these tests
exercise the code that actually ships. If the cell is rewritten, they follow it.

Two failure directions are both covered, because only testing one is how a
matcher ends up either useless or dangerous:

* too strict -- a real dataset under a renamed or nested mount is missed;
* too loose  -- an unrelated dataset is merged in as the wrong source. That is
  not hypothetical: ``norm("D-Fire")`` is ``"dfire"``, which is a substring of
  ``norm("wildfire-dataset")``.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

REPO = Path(__file__).resolve().parent.parent
NOTEBOOK = REPO / "training" / "wildfire" / "train_kaggle.ipynb"


def _discovery_namespace() -> dict:
    """Exec the notebook's discovery helpers in an isolated namespace."""
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cell = next(
        "".join(c["source"])
        for c in nb["cells"]
        if c["cell_type"] == "code" and "def locate(" in "".join(c["source"])
    )
    body = cell[cell.index("def norm("):cell.index("resolved = {}")]
    # The cell reads the mount list from disk; tests inject it per case.
    body = body.replace("if IS_KAGGLE:", "if False:")
    body = re.sub(
        r"mounted = sorted\(p for p in \(REPO / \"datasets\" / \"raw\"\).*?else \[\]",
        "mounted = []",
        body,
        flags=re.S,
    )
    cfg = yaml.safe_load((REPO / "training" / "wildfire" / "dataset_config.yaml").read_text())
    ns: dict = {
        "Path": Path, "re": re, "os": os, "sys": sys, "yaml": yaml,
        "REPO": REPO, "IS_KAGGLE": False, "WORK": Path("/tmp"),
        "SOURCES": cfg["sources"], "print": lambda *a, **k: None,
    }
    exec(compile(body, str(NOTEBOOK), "exec"), ns)  # noqa: S102 - the code under test
    return ns


@pytest.fixture(scope="module")
def discovery() -> dict:
    return _discovery_namespace()


def resolve(discovery: dict, tmp_path: Path, layout: list[str]) -> list[str]:
    """Build a fake mount tree, run discovery, return the included source names."""
    root = tmp_path / "input"
    for rel in layout:
        (root / rel).mkdir(parents=True, exist_ok=True)
    discovery["mounted"] = sorted(p for p in root.glob("*") if p.is_dir())

    bridge = tmp_path / "data-root"
    bridge.mkdir(parents=True, exist_ok=True)
    for want in discovery["WANTED"]:
        found = discovery["locate"](want)
        if found is not None:
            os.symlink(found, bridge / want)
    return [
        s["name"]
        for s in discovery["SOURCES"]
        if Path(s["path"].replace("{data_root}", str(bridge))).exists()
    ]


class TestFindsWhatIsThere:
    def test_kaggle_slugs_are_matched_to_config_names(self, discovery, tmp_path):
        # The ordinary case: uploaded as FLAME and Boreal, mounted as slugs.
        found = resolve(discovery, tmp_path, [
            "flame-dataset/FLAME/segmentation",
            "flame-dataset/FLAME/classification/No_Fire",
            "boreal-forest-fire/BorealForestFire/images",
        ])
        assert set(found) == {"flame_seg", "flame_negatives", "boreal_uav"}

    def test_content_directly_under_the_mount(self, discovery, tmp_path):
        found = resolve(discovery, tmp_path, [
            "flame/segmentation", "flame/classification/No_Fire",
            "borealforestfire/images",
        ])
        assert set(found) == {"flame_seg", "flame_negatives", "boreal_uav"}

    def test_content_nested_below_a_matching_name(self, discovery, tmp_path):
        # The bug this test exists for: the mount is called `fasdd` and so is
        # its child, so a name-only match binds one level too high and the
        # merge silently finds nothing.
        found = resolve(discovery, tmp_path, [
            "fasdd/FASDD/FASDD_UAV", "fasdd/FASDD/FASDD_CV",
        ])
        assert set(found) == {"fasdd_uav", "fasdd_cv"}

    def test_content_under_an_unrelated_mount_name(self, discovery, tmp_path):
        found = resolve(discovery, tmp_path, [
            "kaggle-fire-set/FASDD/FASDD_UAV", "kaggle-fire-set/FASDD/FASDD_RS",
        ])
        assert set(found) == {"fasdd_uav", "fasdd_rs"}

    def test_a_source_with_no_declared_subpaths(self, discovery, tmp_path):
        assert resolve(discovery, tmp_path, ["d-fire/images"]) == ["dfire"]


class TestRefusesWhatIsNot:
    def test_a_substring_lookalike_is_not_matched(self, discovery, tmp_path):
        # norm("D-Fire") == "dfire" is a substring of norm("wildfire-dataset").
        # Accepting it would merge an unrelated dataset in as D-Fire and weight
        # it as ground-level imagery.
        assert resolve(discovery, tmp_path, ["wildfire-dataset/images"]) == []

    def test_nothing_relevant_attached(self, discovery, tmp_path):
        assert resolve(discovery, tmp_path, ["unrelated/foo"]) == []

    def test_no_mounts_at_all(self, discovery, tmp_path):
        assert resolve(discovery, tmp_path, []) == []
