#!/usr/bin/env python3
"""Audit a YOLO dataset for the failure modes that produce a high score and a useless model.

Checks, in order of how badly each one burns you:

  1. Shortcut leakage  — does the filename family predict the class? If a model can infer the
     label from the image *domain* it never has to learn the defect.
  2. Split leakage     — are near-duplicate frames (consecutive capture IDs) split across
     train and val/test? Then validation is partly a memorisation test.
  3. Class imbalance   — including per-split instance counts too small to measure AP on.
  4. Box scale spread  — classes whose median box area differs by an order of magnitude
     cannot be balanced by the detection loss.
  5. Format and integrity — mixed polygon/box labels, duplicate images, orphan files.

Usage:
    python3 tools/audit_dataset.py [DATASET_ROOT] [--json OUT] [--markdown OUT]

DATASET_ROOT defaults to the repo root and must contain train/ valid/ test/ and data.yaml.
Pure standard library — no numpy, no torch, no install step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.common.labels import (  # noqa: E402
    Box,
    is_polygon_file,
    iou,
    iter_split,
    source_name,
    split_family_id,
)

SPLITS = ("train", "valid", "test")

# A same-class box at this IoU with the adjacent capture frame means the val image is
# effectively already in train.
LEAK_IOU = 0.5
# Boxes smaller than this fraction of the image are where a 640px detector head is weakest.
TINY_AREA = 0.01
# Above this, a "box" is really a whole-image scene label.
HUGE_AREA = 0.50


def load_class_names(root: Path) -> list[str]:
    """Read `names:` out of data.yaml without a YAML dependency."""
    data_yaml = root / "data.yaml"
    if not data_yaml.exists():
        return []
    for line in data_yaml.read_text().splitlines():
        if line.startswith("names:"):
            inner = line.split(":", 1)[1].strip().strip("[]")
            return [n.strip().strip("'\"") for n in inner.split(",") if n.strip()]
    return []


def collect(root: Path) -> dict:
    """Walk every split once, gathering everything the checks need."""
    records = []
    for split in SPLITS:
        for image, label, boxes in iter_split(root, split):
            source = source_name(image.stem)
            family, capture_id = split_family_id(source)
            records.append(
                {
                    "split": split,
                    "image": image,
                    "stem": image.stem,
                    "source": source,
                    "family": family or "<numeric>",
                    "id": capture_id,
                    "boxes": boxes,
                    "polygon": is_polygon_file(label),
                }
            )
    return {"records": records}


def check_shortcut(records, names) -> dict:
    """Does the source family determine the class? This is the model-killer.

    The dangerous pattern is not "some family is single-class" — it is that the families
    *partition* the classes, so no class is ever seen in more than one visual domain. When
    that holds, a detector can score well by recognising the capture style (aerial wide shot
    vs. defect close-up) without ever learning what the defect looks like. It then behaves
    unpredictably on real frames, where every class shares one domain.

    Two independent signals, both required for a clean bill of health:
      * class sets of different families must overlap, and
      * classes must actually co-occur inside single images.
    """
    per_family = defaultdict(Counter)
    family_files = Counter()
    for rec in records:
        family_files[rec["family"]] += 1
        for box in rec["boxes"]:
            per_family[rec["family"]][box.cls] += 1

    def label(cls: int) -> str:
        return names[cls] if cls < len(names) else str(cls)

    families = []
    for family, counts in sorted(per_family.items()):
        total = sum(counts.values())
        top_cls, top_n = counts.most_common(1)[0]
        families.append(
            {
                "family": family,
                "files": family_files[family],
                "boxes": total,
                "classes": {label(c): n for c, n in sorted(counts.items())},
                "purity": top_n / total if total else 0.0,
                "dominant": label(top_cls),
            }
        )

    # Do any two families share a class? If not, family membership *is* the label.
    # With fewer than two *annotated* families there is nothing to partition, so family
    # membership carries no class information and the check passes vacuously. (Families
    # that contribute only background images are correctly excluded here.)
    class_sets = {f: set(c) for f, c in per_family.items() if c}
    shared = len(class_sets) < 2 or any(
        a & b for (_, a), (_, b) in combinations(sorted(class_sets.items()), 2)
    )

    # Which class pairs are never annotated in the same image, despite both existing?
    present = sorted({b.cls for r in records for b in r["boxes"]})
    seen_together = set()
    mixed = 0
    for rec in records:
        classes = {b.cls for b in rec["boxes"]}
        if len(classes) > 1:
            mixed += 1
            seen_together.update(combinations(sorted(classes), 2))

    isolated = [
        f"{label(a)} + {label(b)}"
        for a, b in combinations(present, 2)
        if (a, b) not in seen_together
    ]

    # A class that never shares a frame with any other is a whole-image scene label in
    # disguise; the detector only has to recognise the scene.
    scene_labels = [
        label(c)
        for c in present
        if not any(c in pair for pair in seen_together)
    ]

    return {
        "families": families,
        "single_class_families": [f for f, c in sorted(class_sets.items()) if len(c) == 1],
        "families_share_a_class": shared,
        "images_with_multiple_classes": mixed,
        "class_pairs_never_co_occurring": isolated,
        "scene_labels": scene_labels,
        "shortcut_present": not shared or bool(scene_labels),
    }


def check_split_leakage(records) -> dict:
    """Find val/test frames whose adjacent capture ID sits in train."""
    by_key: dict[tuple[str, int], list] = defaultdict(list)
    for rec in records:
        if rec["id"] is not None:
            by_key[(rec["family"], rec["id"])].append(rec)

    train_ids = {k for k, v in by_key.items() if any(r["split"] == "train" for r in v)}

    results = {}
    for split in ("valid", "test"):
        ids = {k for k, v in by_key.items() if any(r["split"] == split for r in v)}
        adjacent = {k for k in ids if (k[0], k[1] - 1) in train_ids or (k[0], k[1] + 1) in train_ids}

        # Adjacency alone is circumstantial. Confirm it by comparing annotations.
        overlapping = 0
        for family, cid in adjacent:
            here = [b for r in by_key[(family, cid)] if r["split"] == split for b in r["boxes"]]
            neighbours = [
                b
                for delta in (-1, 1)
                for r in by_key.get((family, cid + delta), [])
                if r["split"] == "train"
                for b in r["boxes"]
            ]
            if any(
                a.cls == b.cls and iou(a, b) > LEAK_IOU for a in here for b in neighbours
            ):
                overlapping += 1

        results[split] = {
            "images": len(ids),
            "with_adjacent_train_frame": len(adjacent),
            "adjacent_fraction": len(adjacent) / len(ids) if ids else 0.0,
            f"confirmed_iou_over_{LEAK_IOU}": overlapping,
            "confirmed_fraction": overlapping / len(ids) if ids else 0.0,
        }

    # Direct leakage: the same source image augmented into two different splits.
    by_source = defaultdict(set)
    for rec in records:
        by_source[rec["source"]].add(rec["split"])
    results["sources_in_multiple_splits"] = sum(1 for s in by_source.values() if len(s) > 1)
    return results


def check_distribution(records, names) -> dict:
    per_split = {}
    for split in SPLITS:
        subset = [r for r in records if r["split"] == split]
        counts = Counter(b.cls for r in subset for b in r["boxes"])
        per_split[split] = {
            "images": len(subset),
            "backgrounds": sum(1 for r in subset if not r["boxes"]),
            "boxes": sum(counts.values()),
            "per_class": {
                names[c] if c < len(names) else str(c): counts.get(c, 0)
                for c in range(max(len(names), max(counts, default=-1) + 1))
            },
        }

    # Fewer than this many instances and the class's AP is dominated by sampling noise.
    thin = {
        cls: n
        for cls, n in per_split.get("test", {}).get("per_class", {}).items()
        if n < 30
    }
    return {"per_split": per_split, "test_classes_too_thin_to_measure": thin}


def check_geometry(records, names) -> dict:
    areas: dict[int, list[float]] = defaultdict(list)
    for rec in records:
        for box in rec["boxes"]:
            areas[box.cls].append(box.area)

    per_class = {}
    for cls, values in sorted(areas.items()):
        name = names[cls] if cls < len(names) else str(cls)
        per_class[name] = {
            "n": len(values),
            "median_area": round(statistics.median(values), 4),
            "tiny_fraction": round(sum(v < TINY_AREA for v in values) / len(values), 4),
            "huge_fraction": round(sum(v > HUGE_AREA for v in values) / len(values), 4),
        }

    medians = [v["median_area"] for v in per_class.values() if v["n"]]
    spread = max(medians) / min(medians) if medians and min(medians) > 0 else 0.0
    return {
        "per_class": per_class,
        "median_area_spread": round(spread, 1),
        # Past ~5x the loss cannot balance the classes; the big boxes dominate the gradient.
        "scale_imbalance": spread > 5,
    }


def check_integrity(records, root: Path) -> dict:
    digests = Counter()
    for rec in records:
        digests[hashlib.md5(rec["image"].read_bytes()).hexdigest()] += 1

    orphans = {}
    for split in SPLITS:
        images = root / split / "images"
        labels = root / split / "labels"
        if not images.is_dir():
            continue
        image_stems = {p.stem for p in images.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
        label_stems = {p.stem for p in labels.iterdir()} if labels.is_dir() else set()
        if image_stems - label_stems or label_stems - image_stems:
            orphans[split] = {
                "images_without_labels": len(image_stems - label_stems),
                "labels_without_images": len(label_stems - image_stems),
            }

    return {
        "total_files": len(records),
        "unique_image_content": len(digests),
        "duplicate_files": sum(n - 1 for n in digests.values() if n > 1),
        "polygon_label_files": sum(1 for r in records if r["polygon"]),
        "box_label_files": sum(1 for r in records if not r["polygon"] and r["boxes"]),
        "orphans": orphans,
    }



# A dataset can carry any number of files and still hold very little. The previous turbine
# export was 7,520 files that collapsed to 3,133 unique sources and 2,281 distinct scenes,
# roughly 750 of them containing a defect - about ten augmented copies per real photograph.
# Nothing in an aggregate metric reveals that, and a model cannot learn variety that is not
# there, so it is checked rather than assumed.
MAX_FILES_PER_SCENE = 2.5    # above this, augmented copies dominate real photographs
MIN_DEFECT_SCENES = 300      # below this, per-class AP is sampling noise whatever the count
HAMMING_NEAR_DUPLICATE = 6   # dHash bits differing; empirically separates scenes from copies


def check_variety(records) -> dict:
    """Distinct photographs behind the file count.

    Two passes. Filename-source dedup is stdlib and catches the common case, since augmenters
    keep a stem and vary a suffix (Roboflow's `_jpg.rf.<hash>` being the obvious example).
    Perceptual hashing catches the rest - copies renamed beyond recognition, and crops or
    tiles of one photograph - but needs Pillow, so it degrades to the first pass rather than
    making this tool require a dependency it otherwise does not.
    """
    files = len(records)
    sources = {r["source"] for r in records}
    defect_sources = {r["source"] for r in records if r["boxes"]}

    scenes, defect_scenes, method = len(sources), len(defect_sources), "filename sources"
    try:
        from PIL import Image  # noqa: F401
    except ImportError:
        pass
    else:
        by_source = {}
        for record in records:
            by_source.setdefault(record["source"], record)
        hashes, labelled = {}, set()
        for source, record in by_source.items():
            digest = _dhash(record["image"])
            if digest is None:
                continue
            hashes[source] = digest
            if record["boxes"]:
                labelled.add(source)
        if hashes:
            groups = _cluster_near_duplicates(hashes)
            scenes = len(groups)
            defect_scenes = len({key for key, members in groups.items()
                                 if members & labelled})
            method = "perceptual hash"

    ratio = files / scenes if scenes else 0.0
    return {
        "files": files, "unique_sources": len(sources),
        "distinct_scenes": scenes, "distinct_defect_scenes": defect_scenes,
        "files_per_scene": round(ratio, 2), "method": method,
        "inflated": ratio > MAX_FILES_PER_SCENE,
        "too_few_scenes": defect_scenes < MIN_DEFECT_SCENES,
    }


def _dhash(path: Path, size: int = 8):
    """64-bit difference hash. Mirror-invariant comparison happens in the clustering."""
    try:
        from PIL import Image

        with Image.open(path) as handle:
            grey = handle.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
        pixels = list(grey.getdata())
        bits = 0
        for row in range(size):
            offset = row * (size + 1)
            for column in range(size):
                bits = (bits << 1) | (pixels[offset + column] > pixels[offset + column + 1])
        return bits
    except Exception:
        return None


def _cluster_near_duplicates(hashes: dict) -> dict:
    """Union-find over Hamming distance. Returns representative -> set of members."""
    parent = {key: key for key in hashes}

    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    items = list(hashes.items())
    for index, (key_a, digest_a) in enumerate(items):
        for key_b, digest_b in items[index + 1:]:
            if bin(digest_a ^ digest_b).count("1") <= HAMMING_NEAR_DUPLICATE:
                root_a, root_b = find(key_a), find(key_b)
                if root_a != root_b:
                    parent[root_a] = root_b

    groups = defaultdict(set)
    for key in hashes:
        groups[find(key)].add(key)
    return groups


def build_report(root: Path) -> dict:
    names = load_class_names(root)
    records = collect(root)["records"]
    if not records:
        raise SystemExit(f"no labelled images found under {root}")

    return {
        "dataset": str(root),
        "classes": names,
        "distribution": check_distribution(records, names),
        "shortcut": check_shortcut(records, names),
        "split_leakage": check_split_leakage(records),
        "geometry": check_geometry(records, names),
        "integrity": check_integrity(records, root),
        "variety": check_variety(records),
    }


def verdicts(report: dict) -> list[tuple[str, bool, str]]:
    """(label, passed, detail) for each thing that would invalidate a training run."""
    shortcut = report["shortcut"]
    leak = report["split_leakage"]
    geom = report["geometry"]
    integrity = report["integrity"]
    variety = report["variety"]

    worst_leak = max(
        (leak[s]["confirmed_fraction"] for s in ("valid", "test") if s in leak), default=0.0
    )
    return [
        (
            "Real variety, not augmented inflation",
            not variety["inflated"] and not variety["too_few_scenes"],
            f"{variety['files']} files -> {variety['distinct_scenes']} distinct scenes "
            f"({variety['files_per_scene']}x, by {variety['method']}); "
            f"{variety['distinct_defect_scenes']} of them contain a defect",
        ),
        (
            "No source-family shortcut",
            not shortcut["shortcut_present"],
            f"families share a class: {shortcut['families_share_a_class']}; "
            f"scene labels: {shortcut['scene_labels'] or 'none'}; "
            f"never co-occurring: {shortcut['class_pairs_never_co_occurring'] or 'none'}",
        ),
        (
            "No cross-split source leakage",
            leak["sources_in_multiple_splits"] == 0,
            f"{leak['sources_in_multiple_splits']} sources span splits",
        ),
        (
            "No near-duplicate frame leakage",
            worst_leak < 0.05,
            f"{worst_leak:.1%} of held-out images match an adjacent train frame at IoU>{LEAK_IOU}",
        ),
        (
            "Box scales comparable across classes",
            not geom["scale_imbalance"],
            f"{geom['median_area_spread']}x spread between class median box areas",
        ),
        (
            "Test split thick enough to measure",
            not report["distribution"]["test_classes_too_thin_to_measure"],
            f"thin classes: {report['distribution']['test_classes_too_thin_to_measure'] or 'none'}",
        ),
        (
            "Label format consistent",
            integrity["polygon_label_files"] == 0 or integrity["box_label_files"] == 0,
            f"{integrity['polygon_label_files']} polygon files, {integrity['box_label_files']} box files",
        ),
        (
            "No duplicate images",
            integrity["duplicate_files"] == 0,
            f"{integrity['duplicate_files']} duplicate files",
        ),
    ]


def to_markdown(report: dict) -> str:
    out = [f"# Dataset audit — `{report['dataset']}`", ""]
    out += ["## Verdict", "", "| Check | Result | Detail |", "|---|---|---|"]
    for label, passed, detail in verdicts(report):
        out.append(f"| {label} | {'PASS' if passed else 'FAIL'} | {detail} |")

    out += ["", "## Distribution", "", "| Split | Images | Backgrounds | Boxes | Per class |", "|---|---|---|---|---|"]
    for split, stats in report["distribution"]["per_split"].items():
        per_class = ", ".join(f"{k} {v}" for k, v in stats["per_class"].items())
        out.append(
            f"| {split} | {stats['images']} | {stats['backgrounds']} | {stats['boxes']} | {per_class} |"
        )

    out += ["", "## Source families", "", "| Family | Files | Boxes | Purity | Classes |", "|---|---|---|---|---|"]
    for fam in report["shortcut"]["families"]:
        classes = ", ".join(f"{k} {v}" for k, v in fam["classes"].items()) or "—"
        out.append(
            f"| `{fam['family']}` | {fam['files']} | {fam['boxes']} | {fam['purity']:.0%} | {classes} |"
        )

    out += ["", "## Split leakage", "", "| Split | Images | Adjacent train frame | Confirmed IoU>0.5 |", "|---|---|---|---|"]
    for split in ("valid", "test"):
        if split in report["split_leakage"]:
            s = report["split_leakage"][split]
            out.append(
                f"| {split} | {s['images']} | {s['with_adjacent_train_frame']} "
                f"({s['adjacent_fraction']:.1%}) | {s[f'confirmed_iou_over_{LEAK_IOU}']} "
                f"({s['confirmed_fraction']:.1%}) |"
            )

    out += ["", "## Box geometry", "", "| Class | n | Median area | <1% area | >50% area |", "|---|---|---|---|---|"]
    for name, stats in report["geometry"]["per_class"].items():
        out.append(
            f"| {name} | {stats['n']} | {stats['median_area']:.4f} | "
            f"{stats['tiny_fraction']:.1%} | {stats['huge_fraction']:.1%} |"
        )
    out.append("")
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", nargs="?", default=".", type=Path)
    parser.add_argument("--json", type=Path, help="write the full report as JSON")
    parser.add_argument("--markdown", type=Path, help="write a human-readable report")
    args = parser.parse_args()

    report = build_report(args.root.resolve())

    failures = 0
    print(f"Dataset: {report['dataset']}")
    print(f"Classes: {', '.join(report['classes']) or '(none declared)'}\n")
    for label, passed, detail in verdicts(report):
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}\n         {detail}")
        failures += not passed

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, default=str))
        print(f"\nJSON report  -> {args.json}")
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(to_markdown(report))
        print(f"Markdown     -> {args.markdown}")

    print(f"\n{failures} of {len(verdicts(report))} checks failed.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
