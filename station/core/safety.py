"""The safety invariants, expressed as code so they can be tested.

``docs/SAFETY.md`` explains the reasoning. This module exists so that the
reasoning has teeth: ``tests/test_safety_invariants.py`` walks the whole
repository -- station, PWA, docs -- and fails the build if any of these
phrases appear in operator-facing text.

The threat model, stated once: RGB detection fails toward *silence*.
Thin smoke against bright sky, smouldering with no flame, fire under canopy,
fire at night -- in every one of those the model returns an empty list, which
is byte-identical to the result it returns for an empty field. A human
operator knows when they are struggling to see. The model cannot report that
it is struggling, so the interface must never let an empty result be read as
an informed all-clear. Absence of evidence is displayed as absence of
evidence: nothing at all.

The ``person`` class sharpens all of this rather than adding a footnote to it.
A person seen from a drone is a few pixels, is hidden by canopy and smoke as a
matter of course, and is easily confused with a rock or a stump. The model will
miss people who are plainly there. That is tolerable in a tool that says *look
here* and intolerable in one anybody reads as *nobody is down there* -- so the
patterns covering people are the strictest in this file, and personnel
accountability stays with roll call and crew tracking, where it belongs.

This is not a legal disclaimer bolted on at the end. EN 54 and UL 268 govern
fixed fire-detection installations and do not apply to a drone-feed overlay --
which is precisely why the framing has to hold on its own. The system is a
situational-awareness aid: it says *look here*. It is never permitted to say
*there is nothing there*.
"""

from __future__ import annotations

import re

__all__ = [
    "PRODUCT_DESCRIPTOR",
    "FORBIDDEN_UI_PATTERNS",
    "ALLOWED_CONTEXTS",
    "find_forbidden_phrases",
]

#: How the system describes itself, everywhere, without exception.
PRODUCT_DESCRIPTOR = "situational-awareness aid"

#: Phrases that assert an absence of fire, or that invite that reading.
#:
#: Each entry is (compiled pattern, why it is banned). The "why" is shown in
#: the test failure, because a future contributor deleting one of these needs
#: to be arguing with the reason, not just with a regex.
FORBIDDEN_UI_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"\ball[\s\-_]*clear\b", re.I),
        "asserts the scene is safe; the model cannot know that",
    ),
    (
        re.compile(r"\bno\s+(fire|smoke|flames?|detections?|threats?)\s+(detected|found|present|visible)\b", re.I),
        "reports a negative finding as a finding; an empty result is not evidence of absence",
    ),
    (
        re.compile(
            r"\b(0|zero|no)\s+(fires?|smoke|detections?|people|persons?|casualties|victims)\s+"
            r"(detected|found|present|visible|located)\b",
            re.I,
        ),
        "a zero count reads as a measurement; it is only a null result",
    ),
    (
        re.compile(r"\b(area|zone|sector|scene|site)\s+(is\s+)?(clear|safe|secure)\b", re.I),
        "states the area is safe -- the exact claim this system must never make",
    ),
    (
        re.compile(r"\bnothing\s+(detected|found|to\s+report)\b", re.I),
        "phrases a null result as a reassuring report",
    ),
    (
        re.compile(r"\b(safe|clear)\s+to\s+(enter|proceed|approach|advance)\b", re.I),
        "an operational instruction this system has no basis to give",
    ),
    (
        re.compile(r"\bstatus:\s*(ok|good|normal|clear|safe)\b", re.I),
        "a green status light that will be read as 'no fire' rather than 'pipeline healthy'",
    ),
    (
        re.compile(r"\b(fire|smoke)\s+detector\b", re.I),
        "'detector' implies a certified detection device; this is a "
        + PRODUCT_DESCRIPTOR,
    ),
    (
        re.compile(r"\b(guarantee|guaranteed|ensures?)\s+(detection|safety|coverage)\b", re.I),
        "no detection guarantee exists, and false negatives are the known failure mode",
    ),
    # --- the person class -------------------------------------------------
    # These are the strictest entries here, because the object is a human
    # being and the miss rate is highest. A person at altitude is a few pixels,
    # is routinely hidden by canopy, smoke or terrain, and looks like a rock.
    # An empty screen is the expected output over an occupied hillside, so
    # anything that phrases it as an absence of people is not merely optimistic
    # -- it is the sentence that gets a search called off.
    (
        re.compile(r"\b(no|zero)\s+(one|body|people|persons?|casualties|victims|survivors)\b", re.I),
        "states that nobody is present; a person the model cannot resolve is the normal case, not a rare one",
    ),
    (
        re.compile(r"\b(nobody|no-one)\s+(there|present|detected|found|visible|inside)\b", re.I),
        "asserts an absence of people from a null result",
    ),
    (
        re.compile(r"\b(area|zone|sector|building|structure)\s+(is\s+)?(empty|unoccupied|evacuated|clear\s+of\s+people)\b", re.I),
        "declares a space empty of people -- a search-and-rescue conclusion this system cannot support",
    ),
    (
        re.compile(r"\ball\s+(personnel|crew|firefighters|occupants)\s+(accounted|clear|safe|out)\b", re.I),
        "a personnel accountability claim; accountability comes from roll call and crew tracking, never from a camera",
    ),
    (
        re.compile(r"\b(search|sweep)\s+(is\s+)?(complete|finished|done)\b", re.I),
        "implies the model finishing a pass means the area has been searched",
    ),
    (
        re.compile(r"\bcounts?\s+(the\s+)?(people|persons|occupants|survivors)\b", re.I),
        "a count implies the denominator is known; only the boxes drawn are known",
    ),
)

#: Files where these phrases are legitimate: the invariant's own definition,
#: the tests that enforce it, and the docs that explain what is banned and why.
ALLOWED_CONTEXTS: tuple[str, ...] = (
    "station/core/safety.py",
    "tests/test_safety_invariants.py",
    "docs/SAFETY.md",
)


def find_forbidden_phrases(text: str) -> list[tuple[str, str, int]]:
    """Return ``(matched_text, reason, line_number)`` for each violation.

    Line numbers are 1-indexed so the test output is clickable.
    """
    violations: list[tuple[str, str, int]] = []
    for pattern, reason in FORBIDDEN_UI_PATTERNS:
        for match in pattern.finditer(text):
            line_no = text.count("\n", 0, match.start()) + 1
            violations.append((match.group(0), reason, line_no))
    return sorted(violations, key=lambda v: v[2])
