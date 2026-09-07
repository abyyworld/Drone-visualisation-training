#!/usr/bin/env python3
"""Merge several YOLO datasets into one, without carrying their problems in.

WHY THIS IS NOT A FILE COPY
    Four Roboflow exports of the same subject are not four datasets. They overlap: two of
    the projects behind this tool report exactly 1,730 images, which is what the same
    photographs collected twice looks like. They disagree: one calls a defect `craze`,
    another `crack`, a third `Damage`. And they are individually inflated, since an
    exporter offering augmentation is usually taken up on it.

    Concatenating them produces a dataset that measures well and generalises badly - the
    same failure as before, arrived at from a new direction. So this deduplicates across
    sources by image content, harmonises the class names through a mapping you write, and
    splits by scene so that two versions of one photograph cannot land on opposite sides of
    the split.

DISCOVERY FIRST
    Run without --mapping and it reports what each source contains and writes a mapping
    template. Nothing is merged until you have said what each class means, because a
    guessed mapping relabels the data silently and no metric afterwards would show it.

Usage:
    python3 tools/merge_datasets.py --source raw/a --source raw/b            # discover
    python3 tools/merge_datasets.py --source raw/a --source raw/b \
        --mapping docs/class-map.json --out datasets/turbine                 # merge

Requires Pillow for cross-source deduplication.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.common.imagehash import available, cluster, dhash  # noqa: E402
from training.common.labels import iter_split  # noqa: E402

SPLITS = ("train", "valid", "test")
FRACTIONS = {"train": 0.70, "valid": 0.15, "test": 0.15}
DROP = "__drop__"   # mapping value meaning "this class is not wanted in the merged set"



# Proposed groupings for wind turbine blade defects. Roboflow projects name the same defect
# a dozen ways - craze, crack, cracked, hide_craze - and deciding each by hand is slow and
# inconsistent. This proposes, it does not decide: anything unmatched blocks the merge until
# a person places it.
#
# Four targets rather than eight, deliberately. Six classes of 300 boxes make six weak
# detectors; three of 600 make three usable ones.
#
# Phrases are listed explicitly rather than generated from a word soup. An earlier version
# built them from sliding windows over a string, which quietly turned "crack damage" into a
# rule mapping bare "damage" to crack, and "vg panel with missing teeth" into one mapping
# "panel" to structural damage. Both were wrong and neither was visible without testing.
SYNONYMS = {
    "crack": (
        "crack", "cracks", "cracked", "cracking", "crack damage", "craze", "crazing",
        "hide craze", "hidden craze", "fissure", "split", "fracture",
        "transverse crack", "longitudinal crack",
    ),
    "surface_damage": (
        "erosion", "le erosion", "leading edge erosion", "edge erosion", "peeling",
        "surface peeling", "paint peeling", "delamination", "corrosion", "rust",
        "abrasion", "wear", "surface eye", "injury", "damage", "damaged",
        "surface damage", "pitting", "gelcoat", "blister",
        "discoloration", "discolouration", "browning", "yellowing", "corroded",
        "eroded", "delaminated", "scratch", "scratches", "scuff", "chipped",
    ),
    "structural_damage": (
        "breakage", "broken", "break", "missing", "missing material",
        "missing surface material", "hole", "puncture", "structural",
        "structural damage", "severe", "severe damage", "tear", "torn",
        "missing teeth", "vg panel with missing teeth", "detached",
        # Vocabulary a vision model reaches for that a Roboflow class list never contains.
        # These are the words for the failure the trained detector missed entirely, so they
        # have to map somewhere rather than being reported as unknown on every inspection.
        "severed", "snapped", "shattered", "collapsed", "bent", "deformed",
        "deformation", "buckled", "blade broken", "blade severed", "blade missing",
        "missing blade", "structural failure", "catastrophic",
    ),
    "contamination": (
        "dirt", "dust", "dusty", "soil", "soiling", "soiled", "oil", "surface oil",
        "grease", "bird", "bird drop", "bird drops", "bird droppings", "bird dropping",
        "lightning", "lightning strike", "lightning receptor", "burn", "burnmark",
        "burn mark", "surface attach", "debris", "stain", "snow", "soiling dust",
        "vegetation", "moss", "algae", "water", "standing water", "shading",
    ),
}

# Classes naming the object rather than a defect. Keeping them trains the model to find
# blades, which is not the job, and dilutes every real class. Checked before the defect
# groups so that "panel" drops rather than matching a phrase that happens to contain it.
NOT_A_DEFECT = (
    "panel", "vg panel", "blade", "blades", "turbine", "wind turbine", "background",
    "normal", "healthy", "good", "clean", "ok", "undamaged", "no damage",
)


def normalise(name: str) -> str:
    return " ".join(name.lower().replace("-", " ").replace("_", " ").split())


def propose(names: list[str]) -> tuple[dict, list[str]]:
    """Map each source class name to a target. Returns (mapping, unmatched).

    Exact match first, then longest containing phrase, so "leading edge erosion" beats
    "erosion" and a name containing no known phrase is reported rather than guessed.
    """
    exact = {phrase: DROP for phrase in NOT_A_DEFECT}
    for target, phrases in SYNONYMS.items():
        for phrase in phrases:
            exact.setdefault(phrase, target)
    ordered = sorted(exact.items(), key=lambda pair: -len(pair[0]))

    mapping, unmatched = {}, []
    for name in names:
        clean = normalise(name)
        if clean in exact:
            mapping[name] = exact[clean]
            continue
        hit = next((target for phrase, target in ordered if f" {phrase} " in f" {clean} "), None)
        if hit:
            mapping[name] = hit
        else:
            unmatched.append(name)
    return mapping, unmatched


def load_names(root: Path) -> list[str]:
    """Class names from data.yaml, without requiring a yaml parser."""
    config = root / "data.yaml"
    if not config.exists():
        return []
    for line in config.read_text().splitlines():
        if line.strip().startswith("names:"):
            body = line.split(":", 1)[1].strip()
            if body.startswith("["):
                return [n.strip().strip("'\"") for n in body.strip("[]").split(",") if n.strip()]
    # Block form: names:\n  0: crack\n  1: erosion
    names, collecting = {}, False
    for line in config.read_text().splitlines():
        if line.strip().startswith("names:"):
            collecting = True
            continue
        if collecting:
            if not line.startswith((" ", "\t")):
                break
            if ":" in line:
                index, label = line.split(":", 1)
                try:
                    names[int(index.strip().lstrip("- "))] = label.strip().strip("'\"")
                except ValueError:
                    names[len(names)] = label.strip().strip("- '\"")
    return [names[k] for k in sorted(names)] if names else []


def scan(root: Path) -> dict:
    """Every labelled image in a dataset, with its class ids."""
    names = load_names(root)
    records, per_class = [], Counter()
    for split in SPLITS:
        for image, _label, boxes in iter_split(root, split):
            records.append({"image": image, "boxes": boxes, "source": root.name})
            for box in boxes:
                per_class[box.cls] += 1
    return {"root": root, "names": names, "records": records, "per_class": per_class}


def discover(scans: list[dict], template: Path) -> None:
    print("Sources\n")
    every_name = []
    for data in scans:
        labelled = sum(1 for r in data["records"] if r["boxes"])
        print(f"  {data['root'].name}")
        print(f"    {len(data['records'])} images, {labelled} with boxes")
        if not data["names"]:
            print("    no data.yaml class names found")
        for index, name in enumerate(data["names"]):
            print(f"      {index:>2}  {name:<24}{data['per_class'].get(index, 0):>7} boxes")
            every_name.append(name)
        print()

    mapping, unmatched = propose(sorted(set(every_name)))
    if unmatched:
        print("NOT RECOGNISED - place these by hand before merging:")
        for name in unmatched:
            print(f"    {name!r}")
        print()
        for name in unmatched:
            mapping[name] = "PLACE_ME"
    print("Proposed grouping:")
    for target in sorted(set(mapping.values())):
        members = sorted(k for k, v in mapping.items() if v == target)
        print(f"    {target:<20}{', '.join(members)}")
    print()
    template.parent.mkdir(parents=True, exist_ok=True)
    template.write_text(json.dumps(
        {"$comment": [
            "Source class name -> merged class name. Edit the right-hand side.",
            "Give two source classes the same target to merge them.",
            f"Set a target to {DROP!r} to exclude that class from the merged dataset.",
            "Merging aggressively is usually right: six classes of 300 boxes each make six",
            "weak detectors, three of 600 make three usable ones.",
        ], "classes": mapping}, indent=2) + "\n")
    print(f"Wrote a mapping template to {template}")
    print("Edit the targets, then re-run with --mapping to merge.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", action="append", type=Path, required=True,
                        help="a YOLO dataset root; repeat for each")
    parser.add_argument("--mapping", type=Path, help="class map; omit to discover")
    parser.add_argument("--out", type=Path, default=Path("datasets/turbine"))
    parser.add_argument("--template", type=Path, default=Path("docs/class-map.json"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--copy", action="store_true", help="copy rather than hardlink")
    args = parser.parse_args()

    for source in args.source:
        if not source.exists():
            raise SystemExit(f"no such dataset: {source}")

    scans = [scan(source) for source in args.source]
    if not any(data["records"] for data in scans):
        raise SystemExit("no labelled images found in any source")

    if not args.mapping:
        discover(scans, args.template)
        return 0

    spec = json.loads(args.mapping.read_text())
    mapping = spec["classes"] if "classes" in spec else spec
    placeholders = [k for k, v in mapping.items() if v == "PLACE_ME"]
    if placeholders:
        raise SystemExit("these classes still say PLACE_ME in the mapping:\n  "
                         + "\n  ".join(placeholders)
                         + "\n\nDecide what each one is before merging.")

    targets = sorted({v for v in mapping.values() if v != DROP})
    if not targets:
        raise SystemExit("the mapping drops every class")

    # Remap every box, and note anything the mapping does not cover rather than guessing.
    unmapped, kept = set(), []
    for data in scans:
        for record in data["records"]:
            boxes = []
            for box in record["boxes"]:
                name = data["names"][box.cls] if box.cls < len(data["names"]) else None
                if name is None or name not in mapping:
                    unmapped.add(f"{data['root'].name}:{name or box.cls}")
                    continue
                if mapping[name] == DROP:
                    continue
                boxes.append((targets.index(mapping[name]), box))
            kept.append({**record, "mapped": boxes})
    if unmapped:
        raise SystemExit("these classes are not in the mapping, so the merge would drop them "
                         "silently:\n  " + "\n  ".join(sorted(unmapped)))

    # Deduplicate across sources by image content: one photograph, one scene, whichever
    # dataset it arrived in.
    if not available():
        raise SystemExit("Pillow is required to deduplicate across sources")
    print(f"Hashing {len(kept)} images to find scenes shared between sources ...")
    hashes = {}
    for index, record in enumerate(kept):
        digest = dhash(record["image"])
        if digest is not None:
            hashes[index] = digest
    groups = cluster(hashes)

    # Keep the copy carrying the most boxes: if one source annotated a photograph more
    # thoroughly than another, that is the one worth keeping.
    scenes = []
    for members in groups.values():
        best = max(members, key=lambda i: len(kept[i]["mapped"]))
        scenes.append({"record": kept[best], "copies": len(members),
                       "sources": {kept[i]["source"] for i in members}})

    shared = [s for s in scenes if len(s["sources"]) > 1]
    print(f"  {len(kept)} images -> {len(scenes)} distinct scenes")
    print(f"  {len(shared)} of them appear in more than one source")

    # Split by scene, so no two versions of a photograph straddle the split.
    rng = random.Random(args.seed)
    rng.shuffle(scenes)
    counts = {s: int(len(scenes) * f) for s, f in FRACTIONS.items()}
    counts["train"] = len(scenes) - counts["valid"] - counts["test"]

    out = args.out
    staging = out.with_name(out.name + ".building")
    if staging.exists():
        shutil.rmtree(staging)
    for split in SPLITS:
        (staging / split / "images").mkdir(parents=True)
        (staging / split / "labels").mkdir(parents=True)

    place, cursor = {}, 0
    for split in SPLITS:
        for scene in scenes[cursor:cursor + counts[split]]:
            place[id(scene)] = split
        cursor += counts[split]

    written = {s: 0 for s in SPLITS}
    per_class = {s: Counter() for s in SPLITS}
    for scene in scenes:
        split = place.get(id(scene))
        if split is None:
            continue
        record = scene["record"]
        image = record["image"]
        stem = f"{record['source']}_{image.stem}"
        target = staging / split / "images" / f"{stem}{image.suffix}"
        if args.copy:
            shutil.copy2(image, target)
        else:
            try:
                target.hardlink_to(image)
            except OSError:
                shutil.copy2(image, target)
        lines = []
        for class_id, box in record["mapped"]:
            lines.append(f"{class_id} {box.xc:.6f} {box.yc:.6f} {box.w:.6f} {box.h:.6f}")
            per_class[split][targets[class_id]] += 1
        (staging / split / "labels" / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        written[split] += 1

    (staging / "data.yaml").write_text(
        f"path: {out.resolve()}\ntrain: train/images\nval: valid/images\ntest: test/images\n\n"
        f"nc: {len(targets)}\nnames: {targets}\n")

    if out.exists():
        shutil.rmtree(out)
    staging.rename(out)

    print(f"\n  {'split':<8}{'images':>8}{'boxes':>8}   per class")
    print("  " + "-" * 60)
    for split in SPLITS:
        total = sum(per_class[split].values())
        detail = ", ".join(f"{k} {v}" for k, v in sorted(per_class[split].items()))
        print(f"  {split:<8}{written[split]:>8}{total:>8}   {detail or '-'}")
    print(f"\n  {out}/data.yaml  ({len(targets)} classes: {', '.join(targets)})")
    print("\n  Run tools/audit_dataset.py on it before training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
