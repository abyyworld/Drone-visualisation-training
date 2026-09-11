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
#
# Three shapes, because published fire datasets come in three and the first version of this
# only read one of them and rejected two perfectly good sets on its way past:
#
#   a label column        one word per picture. The classification shape.
#   an objects column     boxes with categories. The detection shape, and the better one:
#                         a picture with a fire box in it has fire in it.
#   a negative flag       a boolean saying this picture is a counterexample.
IMAGE_KEYS = ("image", "img", "picture", "jpg", "png")
LABEL_KEYS = ("label", "labels", "class", "target", "category")
OBJECT_KEYS = ("objects", "annotations", "boxes", "bbox")
NEGATIVE_KEYS = ("negative", "is_negative", "no_fire")

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


def category_names(feature):
    """The words a dataset publishes for a categorical column, or None."""
    for attribute in ("names",):
        names = getattr(feature, attribute, None)
        if names:
            return list(names)
    # A sequence of categories: the names live on the element rather than the sequence.
    for attribute in ("feature", "_feature"):
        inner = getattr(feature, attribute, None)
        if inner is not None:
            found = category_names(inner)
            if found:
                return found
    if isinstance(feature, dict):
        for inner in feature.values():
            found = category_names(inner)
            if found:
                return found
    return None


def verdict_from_objects(objects, names):
    """Fire, clear or unreadable, from a detection dataset's boxes.

    A picture whose boxes are all fire or smoke has fire in it. A picture with no boxes at
    all is a counterexample, which is exactly what a detection set's empty frames are for.
    Without published names for the categories this refuses, for the same reason a bare
    number is refused: the integers alone do not say which class is which.
    """
    if objects is None:
        return None
    # Either a list of per-object dicts, or a dict of parallel lists. Both are published.
    if isinstance(objects, dict):
        categories = next((v for k, v in objects.items()
                           if str(k).lower() in ("category", "categories", "label",
                                                 "labels", "class", "classes")), None)
        if categories is None:
            return None
    elif isinstance(objects, (list, tuple)):
        categories = []
        for item in objects:
            if not isinstance(item, dict):
                return None
            found = next((v for k, v in item.items()
                          if str(k).lower() in ("category", "label", "class")), None)
            categories.append(found)
    else:
        return None

    if not categories:
        # No boxes: nothing in this picture was worth marking.
        return "clear"
    verdicts = {classify(c, names) for c in categories}
    if "fire" in verdicts:
        return "fire"
    if verdicts == {"clear"}:
        return "clear"
    return None


def verdict_for_row(row, keys, names):
    """What this one row says about itself, by whichever of the three shapes it has."""
    label_key, object_key, negative_key = keys
    if label_key is not None:
        verdict = classify(row.get(label_key), names)
        if verdict is not None:
            return verdict
    if object_key is not None:
        verdict = verdict_from_objects(row.get(object_key), names)
        if verdict is not None:
            return verdict
    if negative_key is not None:
        flag = row.get(negative_key)
        if isinstance(flag, bool):
            # "negative" means a counterexample: no fire in it.
            return "clear" if flag else "fire"
    return None


def harvest(name, out, limit):
    """Save up to `limit` pictures from one dataset, split by what its labels say.

    Streaming, so a dataset of thousands of pictures costs the handful actually read
    rather than a full download. Returns how many of each were written.
    """
    from datasets import load_dataset

    counts = {"fire": 0, "clear": 0}
    rows = load_dataset(name, split="train", streaming=True)
    features = getattr(rows, "features", None) or {}

    def column(wanted):
        return next((k for k in features if str(k).lower() in wanted), None)

    image_key = column(IMAGE_KEYS)
    label_key = column(LABEL_KEYS)
    object_key = column(OBJECT_KEYS)
    negative_key = column(NEGATIVE_KEYS)
    if image_key is None:
        print(f"    no picture column in {list(features)}", flush=True)
        return counts
    if label_key is None and object_key is None and negative_key is None:
        print(f"    nothing that says whether there is fire in it, in {list(features)}",
              flush=True)
        return counts

    names = None
    for key in (label_key, object_key):
        if key is not None and names is None:
            names = category_names(features[key])
    print(f"    reading {[k for k in (label_key, object_key, negative_key) if k]}"
          f"{f', classes {names}' if names else ''}", flush=True)

    keys = (label_key, object_key, negative_key)
    for row in rows:
        if counts["fire"] >= limit and counts["clear"] >= limit:
            break
        verdict = verdict_for_row(row, keys, names)
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
