#!/usr/bin/env python3
"""Find published fire and smoke pictures, and save some to disk to be scored against.

WHY THIS IS A SEPARATE PROGRAM THAT ONLY RUNS ON A RUNNER
    The machine this project is written on cannot reach Hugging Face at all: the host
    returns nothing, which is the same reason .github/workflows/model.yml fetches weights
    on a runner rather than locally. So the one number that matters about a fire model -
    how often it marks real fire - cannot be measured where the code is written. It is
    measured where the pictures are reachable, and the report is committed so the number
    lives in the repository rather than in somebody's terminal.

WHY IT SEARCHES RATHER THAN NAMING A DATASET
    Same lesson as the weights. Every fire dataset address guessed for this project turned
    out not to exist, and names rot faster than code does. So it asks the index what is
    published and reports exactly what it found, including finding nothing, which is an
    answer rather than a failure.

WHAT IT WILL NOT DO
    Invent a positive set. If it cannot get pictures that are labelled as containing fire,
    it says so and exits without writing any, because a recall number computed over
    pictures nobody has confirmed contain fire is worse than no number: it looks like
    evidence.

    The datasets it can reach are mostly ground level rather than aerial. That is a real
    limitation of the measurement and is printed with the result rather than buried: the
    aerial sets (FLAME, FLAME2) sit behind an IEEE DataPort account and cannot be fetched
    by any machine here.

USAGE
    python3 tools/fetch_fire_pictures.py --out datasets/fire --limit 120
"""
from __future__ import annotations

import argparse
import io
import json
import pathlib
import sys
import urllib.parse
import urllib.request

# What a fire dataset tends to be called. Several, because one search's idea of relevance
# is not another's.
SEARCH = [
    "fire smoke detection",
    "wildfire smoke",
    "fire detection images",
    "forest fire",
]

# Column names that carry the picture, and the ones that carry whether it is a fire.
IMAGE_KEYS = ("image", "img", "picture", "jpg", "png")
LABEL_KEYS = ("label", "labels", "class", "target", "category")

# Values that mean "this picture has fire or smoke in it", and the ones that mean it does
# not. Anything else is left out rather than guessed at.
#
# Bare numbers are deliberately absent. A label of 0 means fire in some of these datasets
# and no fire in others, and there is no way to tell which from the number. Guessing would
# silently swap the two sets and produce a recall figure that is not merely wrong but
# backwards, while looking exactly like a measurement. A numeric label is only read when
# the dataset publishes names for its classes, which is what turns it back into a word.
FIRE_TERMS = ("fire", "smoke", "flame")
NEGATIONS = ("no", "non", "not", "without", "neg")
# Only for labels that never mention fire at all: an ordinary picture in a fire dataset.
CLEAR_WORDS = {"neutral", "normal", "default", "none", "background", "empty", "other",
               "negative", "clear", "nothing"}


def published(terms, limit=10):
    """Dataset names matching each search, best first, without duplicates."""
    found = []
    for term in terms:
        query = urllib.parse.quote(term)
        index = (f"https://huggingface.co/api/datasets?search={query}"
                 f"&limit={limit}&sort=downloads&direction=-1")
        try:
            request = urllib.request.Request(index, headers={"User-Agent": "curl/8"})
            with urllib.request.urlopen(request, timeout=60) as response:
                listing = json.load(response)
        except Exception as problem:
            print(f"  could not search for {term!r}: {problem}", flush=True)
            continue
        for entry in listing:
            name = entry.get("id") or entry.get("datasetId")
            if name and name not in found:
                found.append(name)
    return found


def classify(value, names):
    """Does this label mean fire, no fire, or something this cannot interpret?

    Negation is the whole difficulty. "no_fire", "nonfire" and "no_smoke" all mean the
    opposite of the word they contain, and a naive substring test for "no" also fires on
    "normal" and would file every ordinary picture as a fire. So negation is matched on
    whole segments, plus the run-together forms that have no separator to split on.
    """
    if isinstance(value, bool):
        # True and False are unambiguous in a way that 0 and 1 are not.
        return "fire" if value else "clear"
    if isinstance(value, int):
        if not names or not 0 <= value < len(names):
            # A number with nothing to turn it into a word. See the note above FIRE_WORDS.
            return None
        value = names[value]
    if not isinstance(value, str):
        return None

    word = value.strip().lower().replace("-", "_").replace(" ", "_")
    if not word:
        return None

    mentions_fire = any(term in word for term in FIRE_TERMS)
    if mentions_fire:
        segments = [part for part in word.split("_") if part]
        negated = any(part in NEGATIONS for part in segments)
        if not negated:
            # "nofire", "nonfire": no separator to split on, so strip the prefix and see
            # whether what is left is still the fire word.
            for prefix in NEGATIONS:
                rest = word[len(prefix):]
                if word.startswith(prefix) and any(t in rest for t in FIRE_TERMS):
                    negated = True
                    break
        return "clear" if negated else "fire"

    return "clear" if word in CLEAR_WORDS else None


def harvest(name, out, limit):
    """Save up to `limit` pictures from one dataset, split by what its labels say.

    Streaming, so a dataset of thousands of pictures costs the handful actually read
    rather than a full download. Returns how many of each were written.
    """
    from datasets import load_dataset

    counts = {"fire": 0, "clear": 0}
    rows = load_dataset(name, split="train", streaming=True)
    features = getattr(rows, "features", None) or {}

    image_key = next((k for k in features if str(k).lower() in IMAGE_KEYS), None)
    label_key = next((k for k in features if str(k).lower() in LABEL_KEYS), None)
    if image_key is None or label_key is None:
        print(f"    no usable image and label columns in {list(features)}", flush=True)
        return counts

    names = []
    try:
        names = list(features[label_key].names)
    except Exception:
        pass

    for row in rows:
        if counts["fire"] >= limit and counts["clear"] >= limit:
            break
        verdict = classify(row.get(label_key), names)
        if verdict is None or counts[verdict] >= limit:
            continue
        picture = row.get(image_key)
        try:
            if isinstance(picture, (bytes, bytearray)):
                from PIL import Image
                picture = Image.open(io.BytesIO(picture))
            picture = picture.convert("RGB")
        except Exception:
            continue
        folder = out / verdict
        folder.mkdir(parents=True, exist_ok=True)
        picture.save(folder / f"{name.replace('/', '_')}_{counts[verdict]:04d}.jpg",
                     quality=88)
        counts[verdict] += 1
    return counts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="datasets/fire")
    parser.add_argument("--limit", type=int, default=120,
                        help="how many of each of fire and no-fire to save")
    parser.add_argument("--datasets", type=int, default=6,
                        help="how many published datasets to try before giving up")
    args = parser.parse_args(argv)

    out = pathlib.Path(args.out)
    print("searching for published fire and smoke pictures...", flush=True)
    candidates = published(SEARCH)
    print(f"{len(candidates)} dataset(s) to try", flush=True)

    total = {"fire": 0, "clear": 0}
    used = []
    for name in candidates[:args.datasets]:
        if total["fire"] >= args.limit and total["clear"] >= args.limit:
            break
        print(f"  trying {name}", flush=True)
        try:
            got = harvest(name, out, args.limit - min(total.values()))
        except Exception as problem:
            print(f"    no: {problem}", flush=True)
            continue
        if got["fire"] or got["clear"]:
            used.append((name, got))
            total["fire"] += got["fire"]
            total["clear"] += got["clear"]
            print(f"    {got['fire']} with fire, {got['clear']} without", flush=True)

    print()
    if not total["fire"]:
        print("::error::No pictures labelled as containing fire could be fetched. Searched:")
        for term in SEARCH:
            print(f"::error::  {term!r}")
        print("::error::Without them there is no recall to measure, and a recall number "
              "over pictures nobody has confirmed contain fire would look like evidence "
              "while being none.")
        return 1

    print(f"{total['fire']} pictures with fire, {total['clear']} without, from:")
    for name, got in used:
        print(f"  {name}: {got['fire']} with, {got['clear']} without")
    print()
    print("Ground level, almost certainly. The aerial sets that would match how this flies "
          "(FLAME, FLAME2) are behind an IEEE DataPort account and no machine here can "
          "fetch them, so read the result as fire seen from the ground.")
    (out / "sources.json").write_text(json.dumps(
        {"datasets": [{"name": n, **g} for n, g in used], "totals": total}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
