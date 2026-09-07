"""Tests for the vision-model command-line tools.

    python3 -m pytest tests/test_vlm_tools.py -q

No network and no API key: the provider call is the one part that cannot be tested without
spending money, so it is stubbed and everything around it - parsing, box normalisation,
scoring, the dataset build, the split - is tested for real.

The parity block matters most. tools/vlm_inspect.py and web/js/vlm.js parse the same replies
and normalise the same boxes in two languages, and a disagreement between them means the web
app and the batch tool grade the same photograph differently. The expected values here are
the same ones asserted in tests/test_vlm.mjs.
"""

from __future__ import annotations

import json
import struct
import sys
import zlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import vlm_inspect  # noqa: E402
import vlm_to_yolo  # noqa: E402


# ---------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------

def write_png(path: Path, width: int, height: int):
    """A valid single-colour PNG, so image_size() reads a real header rather than a stub."""
    raw = b"".join(b"\x00" + bytes([80, 120, 160] * width) for _ in range(height))

    def chunk(tag, payload):
        body = tag + payload
        return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body))

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


def make_inspection(root: Path, name: str, records, size=(1000, 500)):
    """Lay out one inspection directory the way vlm_inspect.py writes it."""
    session = root / name
    for record in records:
        image = record["image"]
        write_png(session / "images" / image, *size)
        (session / "labels").mkdir(parents=True, exist_ok=True)
        (session / "labels" / f"{Path(image).stem}.json").write_text(json.dumps({
            "image": image,
            "source": str(session / "images" / image),
            "width": size[0], "height": size[1],
            "domain": record.get("domain", "turbine"),
            "provider": "anthropic", "model": "test", "prompt_version": 1,
            "reviewed": record.get("reviewed", True),
            "overall": "",
            "detections": record.get("detections", []),
            "unlocated": record.get("unlocated", []),
        }))
    return session


def detection(label, box, certainty="high"):
    return {"label": label, "class_id": vlm_inspect.colour_index(label),
            "certainty": certainty, "confidence": 0.9, "note": "", "box": box}


# ---------------------------------------------------------------------------------------
# Reply parsing - parity with tests/test_vlm.mjs
# ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ('{"a":1}', 1),
    ('```json\n{"a":1}\n```', 1),
    ('```\n{"a":1}\n```', 1),
    ('Here you go:\n{"a":1}\nHope that helps.', 1),
])
def test_parse_reply_accepts_the_shapes_models_actually_return(text, expected):
    assert vlm_inspect.parse_reply(text)["a"] == expected


def test_parse_reply_keeps_nested_objects_intact():
    assert vlm_inspect.parse_reply('x {"a":{"b":5}} y')["a"]["b"] == 5


@pytest.mark.parametrize("text", ["", "   ", "I cannot help with that."])
def test_parse_reply_refuses_rather_than_returning_an_empty_result(text):
    # A refusal that parsed as "no defects" would render as a clean bill of health for an
    # image the model never looked at. It has to be an error.
    with pytest.raises(ValueError):
        vlm_inspect.parse_reply(text)


# ---------------------------------------------------------------------------------------
# Box normalisation - parity with tests/test_vlm.mjs
# ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("box,expected", [
    ([0.1, 0.2, 0.4, 0.6], [100, 100, 400, 300]),          # fractions, as asked for
    ([100, 200, 400, 600], [100, 100, 400, 300]),          # the 0-1000 convention
    ([0.4, 0.6, 0.1, 0.2], [100, 100, 400, 300]),          # corners the wrong way round
    ([-0.5, -0.5, 1.5, 1.5], [0, 0, 1000, 500]),           # outside the frame
])
def test_to_pixels_matches_the_browser(box, expected):
    got = vlm_inspect.to_pixels(box, 1000, 500)
    assert got is not None
    assert all(abs(g - e) < 0.1 for g, e in zip(got, expected)), got


def test_to_pixels_rescales_raw_pixel_coordinates():
    got = vlm_inspect.to_pixels([1001, 1002, 2000, 2400], 4000, 4000)
    assert all(abs(g - e) < 0.1 for g, e in zip(got, [1001, 1002, 2000, 2400]))


@pytest.mark.parametrize("box", [
    [0.5, 0.5, 0.5001, 0.5001],   # a box too small to draw or to train on
    [0.1, 0.2, 0.3],              # wrong length
    [0.1, 0.2, "x", 0.6],         # not numbers
    None,
    "the whole blade",
])
def test_to_pixels_rejects_what_it_cannot_use(box):
    assert vlm_inspect.to_pixels(box, 1000, 500) is None


# ---------------------------------------------------------------------------------------
# Labels and scoring
# ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("Pressure Against Barrier", "pressure_against_barrier"),
    ("Leading Edge Erosion", "leading_edge_erosion"),
    ("crack -- severe!!", "crack_severe"),
    ("", "defect"),
    (None, "defect"),
])
def test_tidy_label_matches_the_browser(raw, expected):
    assert vlm_inspect.tidy_label(raw) == expected


def test_colour_index_matches_the_browser_hash():
    # Same arithmetic as colourIndex() in web/js/vlm.js: a class must keep its colour
    # whether the annotated image came from the page or from the command line.
    expected = 0
    for char in "blade_severed":
        expected = (expected * 31 + ord(char)) % 4096
    assert vlm_inspect.colour_index("blade_severed") == expected


@pytest.mark.parametrize("score,label", [
    (0, "No defects found"), (1.9, "Minor wear"),
    (2.0, "Moderate damage"), (4.9, "Moderate damage"), (5.0, "Severe damage"),
])
def test_score_bands_match_the_web_app_and_the_report(score, label):
    # web/js/severity.js and report_generator.py use these same two thresholds. A
    # disagreement means one inspection gets two different verdicts.
    assert vlm_inspect.score_to_label(score) == label


def test_a_person_on_the_ground_outranks_dense_packing():
    # Ordering the crowd weights is the whole point of having them: an operator scanning a
    # list of regions has to see the one that needs someone to walk over there.
    weights = vlm_inspect.severity_weights("crowd")
    assert vlm_inspect.weight_for("person_down", weights) > vlm_inspect.weight_for("dense_packing", weights)
    assert vlm_inspect.weight_for("blocked_exit", weights) > vlm_inspect.weight_for("counterflow", weights)


def test_structural_words_outweigh_surface_words():
    weights = vlm_inspect.severity_weights("turbine")
    assert vlm_inspect.weight_for("blade_severed", weights) > vlm_inspect.weight_for("soiling", weights)
    assert vlm_inspect.weight_for("crack_at_root", weights) > vlm_inspect.weight_for("dirt", weights)


def test_an_unknown_label_still_counts_for_something():
    # A vision model invents labels. One that matches no weight must not score zero, or a
    # real finding with an unfamiliar name would render as a clean image.
    assert vlm_inspect.weight_for("blade_tip_anomaly", vlm_inspect.severity_weights("turbine")) == 1.0


# ---------------------------------------------------------------------------------------
# The shared prompt
# ---------------------------------------------------------------------------------------

def test_the_prompt_file_is_the_one_the_browser_loads():
    doc = json.loads(vlm_inspect.PROMPT_FILE.read_text())
    assert vlm_inspect.PROMPT_FILE == ROOT / "web" / "prompts" / "inspection.json"
    assert set(doc["domains"]) == {"turbine", "solar", "crowd", "wildfire", "auto"}
    assert set(doc["certainty"]) == {"high", "medium", "low"}
    assert doc["certainty"]["high"] > doc["certainty"]["medium"] > doc["certainty"]["low"]
    for key in ("asset", "findings", "certainty", "box", "overall"):
        assert key in doc["schema"]


def test_the_crowd_brief_refuses_to_put_a_number_on_people():
    """The one claim a crowd tool is most tempted to make, and least able to support.

    A figure reads as a measurement. From one aerial frame it would be a guess with a
    decimal point on it, and station/core/safety.py bans the phrasing across the repo for
    exactly that reason.
    """
    crowd = json.loads(vlm_inspect.PROMPT_FILE.read_text())["domains"]["crowd"]
    assert "never as a number of\npeople" in crowd
    assert "Box the region, never the individual" in crowd


def test_the_wildfire_brief_refuses_the_claims_that_get_people_killed():
    """The domain station/core/safety.py was written for.

    A drone sees one frame from one angle. Fire behind a smoke column, inside a treeline or
    under canopy does not appear from above, and a person the camera cannot resolve is the
    normal case. So the brief is forbidden from calling anything safe, clear, contained or
    empty, and the whole file is checked against the repository's phrase list below.
    """
    wildfire = json.loads(vlm_inspect.PROMPT_FILE.read_text())["domains"]["wildfire"]
    assert "Do not describe the scene as safe, clear, contained, out, under control" in wildfire
    assert "invisible from above" in wildfire
    assert "people, individually" in wildfire


def test_a_person_at_a_fire_outranks_everything_else():
    weights = vlm_inspect.severity_weights("wildfire")
    for other in ("smoke", "fire", "vehicle", "water_source"):
        assert vlm_inspect.weight_for("person", weights) > vlm_inspect.weight_for(other, weights)


def test_an_empty_result_is_never_described_as_reassurance():
    schema = json.loads(vlm_inspect.PROMPT_FILE.read_text())["schema"]
    assert "not a statement that the asset is sound or that the scene is without risk" in schema


def test_the_prompt_and_the_manifest_carry_no_forbidden_phrasing():
    """The repo-wide safety scan, applied to the two files a new domain adds text to."""
    import sys as _sys
    _sys.path.insert(0, str(ROOT))
    from station.core.safety import find_forbidden_phrases

    for path in (vlm_inspect.PROMPT_FILE, ROOT / "web" / "models" / "manifest.json"):
        assert not list(find_forbidden_phrases(path.read_text())), path


def test_the_solar_brief_refuses_to_guess_at_thermal_faults():
    doc = json.loads(vlm_inspect.PROMPT_FILE.read_text())
    assert "thermal infrared" in doc["domains"]["solar"]
    assert "Do not\nreport them" in doc["domains"]["solar"]


# ---------------------------------------------------------------------------------------
# Image dimensions
# ---------------------------------------------------------------------------------------

def test_image_size_reads_a_png_header(tmp_path):
    write_png(tmp_path / "a.png", 640, 480)
    assert vlm_to_yolo.image_size(tmp_path / "a.png") == (640, 480)


def test_image_size_reads_a_jpeg_header(tmp_path):
    pillow = pytest.importorskip("PIL.Image", reason="JPEG fixture needs Pillow to write")
    path = tmp_path / "a.jpg"
    pillow.new("RGB", (321, 123)).save(path)
    assert vlm_to_yolo.image_size(path) == (321, 123)


def test_image_size_returns_none_for_something_that_is_not_an_image(tmp_path):
    (tmp_path / "a.png").write_bytes(b"not a png")
    assert vlm_to_yolo.image_size(tmp_path / "a.png") is None


# ---------------------------------------------------------------------------------------
# YOLO conversion
# ---------------------------------------------------------------------------------------

def test_to_yolo_line_centres_and_normalises():
    line = vlm_to_yolo.to_yolo_line(2, [100, 100, 400, 300], 1000, 500)
    parts = line.split()
    assert parts[0] == "2"
    assert [round(float(v), 4) for v in parts[1:]] == [0.25, 0.4, 0.3, 0.4]


def test_to_yolo_line_clips_a_box_that_runs_off_the_frame():
    line = vlm_to_yolo.to_yolo_line(0, [-50, -50, 1200, 600], 1000, 500)
    _, cx, cy, w, h = (float(v) for v in line.split())
    assert (w, h) == (1.0, 1.0) and (cx, cy) == (0.5, 0.5)


def test_to_yolo_line_drops_a_box_too_small_to_train_on():
    assert vlm_to_yolo.to_yolo_line(0, [10, 10, 11, 11], 1000, 500) is None


# ---------------------------------------------------------------------------------------
# The dataset build
# ---------------------------------------------------------------------------------------

def test_unreviewed_sidecars_are_excluded_by_default(tmp_path):
    make_inspection(tmp_path, "site-a", [
        {"image": "a.png", "reviewed": False, "detections": [detection("crack", [10, 10, 200, 200])]},
    ])
    groups, skipped = vlm_to_yolo.load_sidecars([tmp_path], "turbine", include_unreviewed=False)
    assert not groups and skipped["not reviewed"] == 1

    groups, _ = vlm_to_yolo.load_sidecars([tmp_path], "turbine", include_unreviewed=True)
    assert sum(len(v) for v in groups.values()) == 1


def test_the_other_domain_is_left_alone(tmp_path):
    make_inspection(tmp_path, "site-a", [
        {"image": "a.png", "domain": "solar", "detections": [detection("soiling", [10, 10, 200, 200])]},
    ])
    groups, skipped = vlm_to_yolo.load_sidecars([tmp_path], "turbine", include_unreviewed=False)
    assert not groups and skipped["other domain (solar)"] == 1


def test_a_sidecar_whose_image_is_gone_is_skipped_not_guessed(tmp_path):
    session = make_inspection(tmp_path, "site-a", [
        {"image": "a.png", "detections": [detection("crack", [10, 10, 200, 200])]},
    ])
    (session / "images" / "a.png").unlink()
    groups, skipped = vlm_to_yolo.load_sidecars([tmp_path], "turbine", include_unreviewed=False)
    assert not groups and skipped["image not found"] == 1


def test_build_produces_a_trainable_set(tmp_path):
    out = tmp_path / "dataset"
    for index in range(6):
        make_inspection(tmp_path / "inspections", f"site-{index}", [
            {"image": f"{index}-a.png", "detections": [
                detection("crack", [100, 100, 400, 300]),
                detection("blade severed", [10, 10, 900, 400]),
            ]},
            {"image": f"{index}-b.png", "detections": []},
        ])

    assert vlm_to_yolo.main([str(tmp_path / "inspections"), "--out", str(out)]) == 0

    config = (out / "data.yaml").read_text()
    assert "crack" in config and "structural_damage" in config

    written = sorted(
        path for split in ("train", "valid", "test")
        for path in (out / split / "labels").glob("*.txt")
    )
    assert len(written) == 12

    # Every box normalised into range, and no class id outside the table.
    class_count = int(next(line for line in config.splitlines() if line.startswith("nc:")).split()[1])
    for path in written:
        for line in path.read_text().strip().splitlines():
            fields = line.split()
            assert 0 <= int(fields[0]) < class_count
            assert all(0.0 <= float(v) <= 1.0 for v in fields[1:])


def test_an_inspection_never_spans_two_splits(tmp_path):
    """The lesson from the first turbine model, encoded as a test.

    Twenty photographs of one turbine on one afternoon are twenty views of one thing.
    Splitting them across train and validation makes validation a memorisation test, which
    is how a useless model came to report mAP50 0.78.
    """
    out = tmp_path / "dataset"
    for index in range(5):
        make_inspection(tmp_path / "inspections", f"site-{index}", [
            {"image": f"{index}-{n}.png", "detections": [detection("crack", [100, 100, 400, 300])]}
            for n in range(4)
        ])

    assert vlm_to_yolo.main([str(tmp_path / "inspections"), "--out", str(out)]) == 0

    seen = {}
    for split in ("train", "valid", "test"):
        for path in (out / split / "images").glob("*.png"):
            session = path.name.split("__")[0]
            assert seen.setdefault(session, split) == split, f"{session} is in two splits"
    assert len(seen) == 5


def test_a_label_with_no_home_stops_the_build(tmp_path):
    make_inspection(tmp_path / "inspections", "site-a", [
        {"image": "a.png", "detections": [detection("iridescent_shimmer", [100, 100, 400, 300])]},
    ])
    # Refusing is the point: guessing a class here is how the wrong boxes get into a
    # training set, which is the failure the whole rebuild exists to fix.
    assert vlm_to_yolo.main([str(tmp_path / "inspections"), "--out", str(tmp_path / "d")]) == 1
    assert not (tmp_path / "d").exists()

    assert vlm_to_yolo.main([
        str(tmp_path / "inspections"), "--out", str(tmp_path / "d"), "--drop-unknown",
    ]) == 0


def test_a_clean_image_becomes_a_background_negative_not_a_dropped_one(tmp_path):
    out = tmp_path / "dataset"
    make_inspection(tmp_path / "inspections", "site-a", [
        {"image": "clean.png", "detections": []},
        {"image": "cracked.png", "detections": [detection("crack", [100, 100, 400, 300])]},
    ])
    assert vlm_to_yolo.main([str(tmp_path / "inspections"), "--out", str(out)]) == 0

    empties = [p for split in ("train", "valid", "test")
               for p in (out / split / "labels").glob("*.txt") if not p.read_text().strip()]
    assert len(empties) == 1, "an image with no defect is a negative, and negatives are training data"


def test_dry_run_writes_nothing(tmp_path):
    make_inspection(tmp_path / "inspections", "site-a", [
        {"image": "a.png", "detections": [detection("crack", [100, 100, 400, 300])]},
    ])
    assert vlm_to_yolo.main([str(tmp_path / "inspections"), "--dry-run"]) == 0
    assert not list(tmp_path.glob("dataset*"))


def test_nothing_to_convert_is_a_failure_not_an_empty_dataset(tmp_path):
    (tmp_path / "inspections").mkdir()
    assert vlm_to_yolo.main([str(tmp_path / "inspections"), "--dry-run"]) == 1


# ---------------------------------------------------------------------------------------
# Provider errors
# ---------------------------------------------------------------------------------------

class _FakeHTTPError(Exception):
    def __init__(self, code, payload):
        self.code = code
        self.reason = "fake"
        self._payload = json.dumps(payload).encode()

    def read(self):
        return self._payload


@pytest.mark.parametrize("code,fragment", [
    (401, "rejected the API key"),
    (403, "rejected the API key"),
    (404, "does not recognise that model"),
    (429, "out of credit"),
    (503, "Provider server error"),
])
def test_http_errors_say_what_to_do_about_them(code, fragment):
    message = vlm_inspect._explain(_FakeHTTPError(code, {"error": {"message": "detail"}}))
    assert fragment in message


def test_a_missing_key_names_the_environment_variable(monkeypatch):
    for name in ("ANTHROPIC_API_KEY",):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(SystemExit) as raised:
        vlm_inspect.resolve_key("anthropic", None)
    assert "ANTHROPIC_API_KEY" in str(raised.value)


def test_a_key_pasted_with_whitespace_is_trimmed(monkeypatch):
    # Every provider answers a trailing newline with a flat 401 that reads like a wrong key.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "  sk-ant-abc\n")
    assert vlm_inspect.resolve_key("anthropic", None) == "sk-ant-abc"


def test_gemini_accepts_either_environment_variable(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "g-key")
    assert vlm_inspect.resolve_key("gemini", None) == "g-key"


# ---------------------------------------------------------------------------------------
# The round trip the whole "train it later" plan depends on
# ---------------------------------------------------------------------------------------

def test_the_web_app_export_feeds_the_dataset_builder(tmp_path):
    """A session exported from the browser must build a training set with no conversion.

    This is the join that makes the plan real. tools/vlm_inspect.py has always written
    sidecars, but nobody runs a Python script on a controller at a wind farm - they use the
    app. If what the app exports needs massaging before vlm_to_yolo.py will read it, the
    data quietly never becomes a dataset.

    The layout below is exactly what exportTrainingData() in web/js/app.js writes into its
    zip. Keep the two in step; this test is what notices when they drift.
    """
    session = tmp_path / "inspections" / "inspection-2026-09-07T12-00-00"
    (session / "images").mkdir(parents=True)
    (session / "labels").mkdir(parents=True)

    for index in range(4):
        name = f"DJI_{index:04d}.png"
        write_png(session / "images" / name, 1000, 500)
        (session / "labels" / f"DJI_{index:04d}.json").write_text(json.dumps({
            "image": name,
            "source": name,          # a bare filename, as the browser has no paths
            "width": 1000, "height": 500,
            "domain": "turbine",
            "provider": "anthropic", "model": "claude-opus-5",
            "generated": "2026-09-07T12:00:00.000Z",
            "reviewed": False,
            "overall": "Leading edge erosion along the outboard third.",
            "detections": [{
                "label": "leading_edge_erosion", "class_id": 12, "certainty": "high",
                "confidence": 0.9, "note": "", "box": [100.0, 100.0, 400.0, 300.0],
            }],
            "unlocated": [],
        }, indent=2))

    inspections = str(tmp_path / "inspections")

    # Unreviewed by default, which is the point: a vision model's output is a first draft.
    assert vlm_to_yolo.main([inspections, "--dry-run"]) == 1

    # Reviewed, it builds.
    for sidecar in (session / "labels").glob("*.json"):
        record = json.loads(sidecar.read_text())
        record["reviewed"] = True
        sidecar.write_text(json.dumps(record))

    out = tmp_path / "dataset"
    assert vlm_to_yolo.main([inspections, "--out", str(out)]) == 0

    labels = [p for split in ("train", "valid", "test")
              for p in (out / split / "labels").glob("*.txt")]
    assert len(labels) == 4
    assert "surface_damage" in (out / "data.yaml").read_text()

    for path in labels:
        fields = path.read_text().strip().split()
        assert len(fields) == 5
        assert all(0.0 <= float(v) <= 1.0 for v in fields[1:])
