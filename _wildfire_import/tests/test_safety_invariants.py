"""Enforcement of safety invariant 1, across the whole repository.

``station/core/safety.py`` states the invariant; ``docs/SAFETY.md`` argues it.
This file is what gives either of them teeth. It walks every text file in the
repository -- station code, the tablet PWA, the docs, the tools, the training
material, the README -- and fails the build if any operator-facing string
asserts an absence of fire.

Why a repo-wide grep rather than a review checklist: RGB fire detection fails
toward silence. Thin smoke on bright sky, smouldering with no flame, fire
under canopy, fire at night -- in every one of those the model returns an
empty list, byte-identical to the result it returns for an empty field. A
human operator knows when they are struggling to see; the model cannot report
that it is struggling. So the one thing the interface may never do is let an
empty result read as an informed all-clear. That is a property of *text*, and
text is added by whoever is in a hurry at the time. A test is the only control
that survives that.

The suite proves three things, in this order:

1. the checker catches violations at all (planted violations, in a temp tree);
2. the walker actually reaches the files it claims to (coverage assertions);
3. the repository is clean.

Without (1) and (2), (3) would pass just as happily on a walker that found
nothing and a regex that matched nothing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from station.core import safety
from station.core.safety import (
    ALLOWED_CONTEXTS,
    FORBIDDEN_UI_PATTERNS,
    PRODUCT_DESCRIPTOR,
    find_forbidden_phrases,
)

# Extensions that carry operator-facing text, plus the source that generates
# it. Deliberately broad: a phrase is just as dangerous in a Python f-string,
# a JS template literal, a YAML comment or a notebook cell as it is in Markdown.
TEXT_SUFFIXES = frozenset(
    {
        ".py", ".js", ".mjs", ".ts", ".json", ".jsonl", ".html", ".css", ".md",
        ".yaml", ".yml", ".txt", ".toml", ".cfg", ".ini", ".ipynb", ".webmanifest",
        ".sh", ".env", ".svg", ".xml",
    }
)

# Extension-less files that are still text we ship.
TEXT_NAMES = frozenset({"Makefile", "makefile", "Dockerfile", ".gitignore", "LICENSE"})

# Directories that hold no authored text: build products, caches, captured
# incidents, credentials, datasets, model weights.
SKIP_DIRS = frozenset(
    {
        ".git", "__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache",
        "node_modules", ".venv", "venv", "build", "dist", "site-packages",
        "incidents", "certs", "data", "datasets", "models", "runs", ".idea",
        "*.egg-info",
    }
)

#: Files that legitimately contain the banned phrases *and are not listed in*
#: ``safety.ALLOWED_CONTEXTS``. Each entry needs a reason, and
#: ``test_no_stale_local_exemptions`` deletes itself from usefulness the
#: moment an entry stops being needed.
#:
#: ``station/core/config.py`` and ``station/core/types.py`` are the frozen wire
#: contract: they define the protocol by describing, in prose, the fields it
#: refuses to have ("no ``all_clear``", "it does not mean the scene is clear").
#: Quoting the banned phrasing is how those docstrings state the ban. The
#: proper home for this is ``safety.ALLOWED_CONTEXTS`` itself -- see the
#: findings note in this suite's docstring companion, ``test_allowed_contexts_
#: covers_the_contract_files`` below.
LOCAL_EXEMPTIONS: dict[str, str] = {
    "tests/test_person_class.py": (
        "carries the banned phrasings as parametrised fixtures, asserting that "
        "each one is refused; the phrases are the test input, not operator text"
    ),
    "station/core/types.py": (
        "the wire contract's module docstring names the fields this protocol "
        "refuses to have ('No all_clear', 'It does not mean the scene is "
        "clear'); it is stating the ban, not making the claim"
    ),
}


def is_text_file(path: Path) -> bool:
    """Whether this path is authored text the invariant applies to."""
    return path.suffix.lower() in TEXT_SUFFIXES or path.name in TEXT_NAMES


def iter_repo_text_files(root: Path):
    """Yield every authored text file under ``root``, deepest-first order.

    Directory pruning is by name rather than by an ignore file: a
    ``.gitignore`` change must not be able to quietly shrink the scope of a
    safety check.
    """
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS or part.endswith(".egg-info") for part in path.parts):
            continue
        if not is_text_file(path):
            continue
        yield path


def scan_tree(root: Path, *, exemptions: dict[str, str] | None = None) -> list[str]:
    """Scan a tree and return one ``file:line: match -- reason`` per violation.

    Args:
        root: Directory to walk.
        exemptions: Repo-relative POSIX paths that may contain the phrases,
            mapped to the reason. ``safety.ALLOWED_CONTEXTS`` is always
            honoured in addition.

    Returns:
        Human-readable violation lines, empty when the tree is clean.
    """
    allowed = set(ALLOWED_CONTEXTS) | set(exemptions or {})
    problems: list[str] = []
    for path in iter_repo_text_files(root):
        rel = path.relative_to(root).as_posix()
        if rel in allowed:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            # Not authored text after all (a mislabelled binary); nothing to
            # enforce, and failing here would only teach people to widen
            # SKIP_DIRS.
            continue
        for match, reason, line_no in find_forbidden_phrases(text):
            problems.append(f"{rel}:{line_no}: {match!r} -- {reason}")
    return problems


@pytest.fixture(scope="module")
def scanned_files(repo_root: Path) -> list[str]:
    return [p.relative_to(repo_root).as_posix() for p in iter_repo_text_files(repo_root)]


# --------------------------------------------------------------------------
# 1. the checker catches things
# --------------------------------------------------------------------------


class TestTheCheckerCatchesViolations:
    """Planted violations, so a silently broken regex cannot pass this suite."""

    PLANTED = [
        ("<div>ALL CLEAR</div>", "all clear"),
        ("<div>All-Clear</div>", "all-clear"),
        ("banner.textContent = 'No fire detected';", "no fire detected"),
        ("print('no smoke found')", "no smoke found"),
        ("label = 'No detections present'", "no detections present"),
        ("caption = '0 fires detected'", "0 fires detected"),
        ("msg = '0 detections found'", "0 detections found"),
        ("<p>The area is clear.</p>", "area is clear"),
        ("status_line = 'Sector secure'", "sector secure"),
        ("toast('Scene clear')", "scene clear"),
        ("h2 = 'Nothing detected'", "nothing detected"),
        ("footer = 'nothing to report'", "nothing to report"),
        ("advice = 'Safe to enter'", "safe to enter"),
        ("advice = 'clear to approach'", "clear to approach"),
        ("chip.title = 'Status: OK'", "status: ok"),
        ("readme = 'a fire detector for drones'", "fire detector"),
        # --- the person class. Higher stakes than the rest of this list: at
        # altitude a person is a few pixels behind canopy, so an empty screen
        # over an occupied hillside is the expected output, not a fault.
        ("banner = 'No one detected'", "no one detected"),
        ("msg = 'zero casualties found'", "zero casualties"),
        ("label = '0 people found'", "0 people found"),
        ("toast('Nobody present')", "nobody present"),
        ("h3 = 'The building is empty'", "building is empty"),
        ("status = 'Sector evacuated'", "sector evacuated"),
        ("line = 'area is clear of people'", "clear of people"),
        ("summary = 'All personnel accounted for'", "personnel accounted"),
        ("done = 'Search complete'", "search complete"),
        ("caption = 'counts the occupants in frame'", "counts occupants"),
        # Note the exact wording: the shipped pattern matches "guarantee
        # detection" but not the plural "guarantees detection" -- see the
        # findings note for this suite.
        ("blurb = 'we guarantee detection of every fire'", "guarantee detection"),
    ]

    @pytest.mark.parametrize("text,label", PLANTED, ids=[p[1] for p in PLANTED])
    def test_each_planted_phrase_is_caught(self, text, label):
        violations = find_forbidden_phrases(text)
        assert violations, f"{label!r} was not caught by find_forbidden_phrases"
        assert all(reason for _match, reason, _line in violations), "a violation carried no reason"

    def test_line_numbers_are_one_indexed_and_correct(self):
        text = "line one\nline two\nthe area is clear\nline four\n"
        violations = find_forbidden_phrases(text)
        assert [v[2] for v in violations] == [3]

    def test_several_violations_in_one_file_are_all_reported(self):
        text = "all clear\nfine\n0 fires detected\nfine\nnothing to report\n"
        assert [v[2] for v in find_forbidden_phrases(text)] == [1, 3, 5]

    def test_matching_is_case_insensitive(self):
        assert find_forbidden_phrases("ALL CLEAR")
        assert find_forbidden_phrases("All Clear")
        assert find_forbidden_phrases("aLl cLeAr")

    def test_innocent_text_is_not_flagged(self):
        # False positives are not harmless: a check that cries wolf gets
        # switched off, and then invariant 1 is enforced by nobody.
        clean = "\n".join(
            [
                "The overlay says look here. It never says there is nothing there.",
                "Absence of boxes is not evidence of absence.",
                "situational-awareness aid, not a certified detection device",
                "clearing the temporal filter resets every track",
                "state: running -- the pipeline is keeping up",
                "detections: [] is a null result and is logged faithfully",
                "the sky is clear blue in this test fixture",  # 'clear' alone is fine
            ]
        )
        assert find_forbidden_phrases(clean) == []

    def test_a_planted_violation_in_a_temp_tree_is_found_by_the_walker(self, tmp_path: Path):
        """End-to-end: the walker, not just the regex.

        This is the test that would have caught a walker skipping the wrong
        directory, or filtering out the extension the violation was written in.
        """
        (tmp_path / "app" / "js").mkdir(parents=True)
        (tmp_path / "docs").mkdir()
        (tmp_path / "app" / "js" / "overlay.js").write_text(
            "export function render() {\n  chip.textContent = 'ALL CLEAR';\n}\n", encoding="utf-8"
        )
        (tmp_path / "docs" / "GUIDE.md").write_text(
            "# Guide\n\nWhen the list is empty the area is clear.\n", encoding="utf-8"
        )
        (tmp_path / "README.md").write_text("A situational-awareness aid.\n", encoding="utf-8")
        problems = scan_tree(tmp_path)
        assert len(problems) == 2, problems
        assert any(p.startswith("app/js/overlay.js:2:") for p in problems)
        assert any(p.startswith("docs/GUIDE.md:3:") for p in problems)

    def test_the_walker_reaches_every_shipped_text_extension(self, tmp_path: Path):
        # A violation must be caught whatever file it is written into.
        for name in ("a.py", "b.js", "c.md", "d.html", "e.css", "f.yaml", "g.json", "h.ipynb", "i.txt"):
            (tmp_path / name).write_text("all clear\n", encoding="utf-8")
        problems = scan_tree(tmp_path)
        assert len(problems) == 9, problems

    def test_allowed_contexts_are_skipped_by_the_walker(self, tmp_path: Path):
        (tmp_path / "docs").mkdir()
        (tmp_path / "docs" / "SAFETY.md").write_text(
            "Banned: 'all clear', 'the area is clear'.\n", encoding="utf-8"
        )
        assert scan_tree(tmp_path) == []

    def test_binary_like_directories_are_not_scanned(self, tmp_path: Path):
        (tmp_path / "incidents").mkdir()
        (tmp_path / "incidents" / "notes.md").write_text("all clear\n", encoding="utf-8")
        assert scan_tree(tmp_path) == []


class TestThePatternSet:
    def test_every_pattern_has_a_reason(self):
        # The reason is shown in the failure, because a contributor deleting a
        # pattern needs to be arguing with the reason, not with a regex.
        for pattern, reason in FORBIDDEN_UI_PATTERNS:
            assert reason and len(reason) > 20, f"{pattern.pattern!r} has a thin reason"

    def test_the_pattern_set_is_not_empty(self):
        assert len(FORBIDDEN_UI_PATTERNS) >= 9

    def test_every_pattern_matches_at_least_one_planted_example(self):
        """No pattern may be dead. A regex that matches nothing is a rule
        nobody is enforcing, and it reads in review as if they were."""
        corpus = "\n".join(text for text, _ in TestTheCheckerCatchesViolations.PLANTED)
        for pattern, _reason in FORBIDDEN_UI_PATTERNS:
            assert pattern.search(corpus), f"no planted example exercises {pattern.pattern!r}"

    def test_product_descriptor_is_the_one_the_docs_use(self, repo_root: Path):
        assert PRODUCT_DESCRIPTOR == "situational-awareness aid"
        readme = (repo_root / "README.md").read_text(encoding="utf-8").lower()
        assert PRODUCT_DESCRIPTOR in readme, "the README must describe the system as the docs do"


# --------------------------------------------------------------------------
# 2. the walker reaches what it claims to
# --------------------------------------------------------------------------


class TestCoverage:
    """A clean result is only meaningful if the walk was real."""

    @pytest.mark.parametrize(
        "expected",
        [
            "README.md",
            "docs/CONTRACT.md",
            "docs/SAFETY.md",
            "station/core/types.py",
            "station/core/config.py",
            "station/inference/temporal.py",
            "station/pipeline.py",
            "app/index.html",
            "app/js/sync.js",
            "app/js/overlay.js",
            "app/css/style.css",
            "app/manifest.webmanifest",
            "tools/evaluate.py",
            "training/prepare_datasets.py",
            "training/train_kaggle.ipynb",
            "config.example.yaml",
            "tests/test_safety_invariants.py",
        ],
    )
    def test_key_files_are_in_scope(self, scanned_files, expected):
        assert expected in scanned_files

    @pytest.mark.parametrize("prefix", ["station/", "app/", "docs/", "tools/", "training/", "tests/"])
    def test_every_shipped_directory_contributes_files(self, scanned_files, prefix):
        assert any(f.startswith(prefix) for f in scanned_files), f"nothing scanned under {prefix}"

    def test_the_walk_is_not_trivially_small(self, scanned_files):
        # A guard against a walker that silently stops after one directory.
        assert len(scanned_files) >= 40, f"only {len(scanned_files)} files scanned"

    def test_no_compiled_or_captured_artefacts_are_scanned(self, scanned_files):
        for name in scanned_files:
            assert "__pycache__" not in name
            assert not name.startswith("incidents/")
            assert not name.startswith("certs/")


# --------------------------------------------------------------------------
# 3. the repository is clean
# --------------------------------------------------------------------------


def test_repository_contains_no_forbidden_phrases(repo_root: Path):
    """The invariant itself. Fails with ``file:line`` and the reason."""
    problems = scan_tree(repo_root, exemptions=LOCAL_EXEMPTIONS)
    assert not problems, (
        "Operator-facing text asserts an absence of fire.\n"
        "An empty result from an unseeing model is byte-identical to an empty result\n"
        "from an empty field, so nothing in this system may render it as reassurance.\n"
        "See docs/SAFETY.md and station/core/safety.py.\n\n" + "\n".join(problems)
    )


def test_no_stale_local_exemptions(repo_root: Path):
    """Every local exemption must still be needed.

    An exemption that is no longer earning its keep is how an allowlist grows
    until it covers the thing it was meant to police.
    """
    for rel, reason in LOCAL_EXEMPTIONS.items():
        path = repo_root / rel
        assert path.exists(), f"{rel} is exempted but does not exist"
        assert find_forbidden_phrases(path.read_text(encoding="utf-8")), (
            f"{rel} no longer contains any banned phrase; remove it from "
            f"LOCAL_EXEMPTIONS (reason recorded was: {reason})"
        )


def test_allowed_contexts_all_exist(repo_root: Path):
    for rel in ALLOWED_CONTEXTS:
        assert (repo_root / rel).exists(), f"safety.ALLOWED_CONTEXTS names a missing file: {rel}"


def test_allowed_contexts_are_only_the_invariants_own_paperwork():
    # The allowlist is for files that *define* the ban. It must never grow to
    # cover a UI file.
    assert set(ALLOWED_CONTEXTS) == {
        "station/core/safety.py",
        "tests/test_safety_invariants.py",
        "docs/SAFETY.md",
    }


def test_safety_module_has_no_side_effects_on_import():
    # Imported by tests, tools and the training scripts. It must stay free of
    # anything that could fail on a machine with no station installed.
    source = Path(safety.__file__).read_text(encoding="utf-8")
    assert "import re" in source
    for heavy in ("import numpy", "import torch", "import yaml", "import cv2"):
        assert heavy not in source


# --------------------------------------------------------------------------
# invariant 1 applied to the wire and to the UI, not just to prose
# --------------------------------------------------------------------------


def test_the_wire_protocol_has_no_negative_assertion_field():
    """Invariant 1 at the schema level: there is no field to misuse."""
    from dataclasses import fields

    from station.core.types import Detection, FrameDetections, ModelInfo, PipelineStatus

    banned_names = {
        "all_clear", "is_safe", "safe", "clear", "no_fire", "fire_count",
        "detection_count", "count", "nothing_found", "status_ok",
    }
    for cls in (FrameDetections, Detection, PipelineStatus, ModelInfo):
        names = {f.name for f in fields(cls)}
        assert names.isdisjoint(banned_names), f"{cls.__name__} exposes {names & banned_names}"


def test_the_overlay_has_no_empty_state_branch(repo_root: Path):
    """Nothing renders on an empty detections list.

    ``app/js/overlay.js`` draws boxes and nothing else -- no "0 detections",
    no green light, no reassuring chip. The check is textual because the
    behaviour is: any of those strings appearing here is the bug.
    """
    text = (repo_root / "app" / "js" / "overlay.js").read_text(encoding="utf-8")
    lowered = text.lower()
    for banned in ("all clear", "no fire", "nothing detected", "0 detections", "area is clear"):
        assert banned not in lowered


def test_the_tablet_palette_contains_no_green(repo_root: Path):
    """A green status light reads as 'no fire' whatever it is labelled.

    The stylesheet documents this at the top; the test is what keeps it true
    when somebody adds a component.
    """
    import re

    css = (repo_root / "app" / "css" / "style.css").read_text(encoding="utf-8")
    # Strip comments: the reasoning above the palette says the word "green".
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    assert "green" not in css.lower()

    for hexcolour in re.findall(r"#([0-9a-fA-F]{6})\b", css):
        r, g, b = (int(hexcolour[i : i + 2], 16) for i in (0, 2, 4))
        assert not (g > r + 40 and g > b + 40), f"#{hexcolour} is a green"

    for h, s, ln in re.findall(r"hsl\(\s*(\d+)[, ]+(\d+)%[, ]+(\d+)%", css):
        if int(s) > 25 and 25 < int(ln) < 85:
            assert not (75 <= int(h) <= 165), f"hsl({h},{s}%,{ln}%) is a green"
