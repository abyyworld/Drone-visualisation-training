"""The Python signature in tools/fly_video.py must match the app's.

The tracker recognises somebody who walked behind a van by comparing colour signatures, and
that mechanism is the direct answer to one person collecting several numbers. A harness that
computes a different signature measures a mechanism the app does not have, and reports a
better count than the tablet can achieve.

The port was wrong when it was written: it split the box into bands with int() where the
JavaScript uses floor-per-row, which agrees only when the box height divides by three
exactly. Every person whose height was not a multiple of three had a band boundary one row
out. This is what caught it.
"""
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def test_python_signature_matches_the_app():
    if not (ROOT / "tools" / "check_reid_port.py").exists():
        pytest.skip("the port check is absent")
    result = subprocess.run([sys.executable, "tools/check_reid_port.py"],
                            cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, (
        "tools/fly_video.py and web/js/reid.js build different signatures, so every "
        "re-identification number measured on real footage is about a mechanism the app "
        f"does not have.\n\n{result.stdout}\n{result.stderr}")
