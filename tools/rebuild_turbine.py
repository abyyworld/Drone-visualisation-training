#!/usr/bin/env python3
"""Rebuild the turbine dataset so a detector has to learn defects instead of capture style.

The v1 export scores mAP50 0.782 and still boxes random things on real photos. `tools/
audit_dataset.py` shows why. This script fixes each cause:

  1. `healthy` is dropped as a detection class. It never co-occurs with a defect and its
     median box covers 44% of the frame, so it is a whole-image scene label wearing an
     object-detection costume — and it dominates the box/DFL loss, drowning out the small
     defects. An image with no boxes *is* the healthy prediction.

  2. Healthy images are kept, as explicit background negatives (empty label files). They are
     the only thing that teaches "blade surface is not a defect", so deleting them would make
     false positives worse, not better. Subsampled, because 5,036 negatives against 2,484
     defect images would just re-create the imbalance from the other side.

  3. The split becomes group-aware. v1 randomly split a contiguous capture sequence, so ~90%
     of val frames have their neighbouring frame sitting in train. Here each family's capture
     IDs are cut into contiguous blocks, and every augmented copy of a source stays with its
     source.

  4. Polygons are converted to boxes up front, so Ultralytics stops discarding segments and
     the mixed-format warning goes away.

  5. Byte-identical duplicate images are dropped.

Val and test keep one copy per source image — offline augmentation belongs in train only,
where it is data; in val it is just the same picture scored three times.

Usage:
    python3 tools/rebuild_turbine.py [--src .] [--out datasets/turbine_v2] [--copy]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.common.labels import iter_split, source_name, split_family_id  # noqa: E402

# v1 class ids -> v2. `healthy` (2) is deliberately absent: those images become backgrounds.
CLASS_REMAP = {0: 0, 1: 1, 3: 2}
V2_NAMES = ["corrosion", "crack", "surface_peeling"]
HEALTHY_CLASS = 2

# Contiguous blocks per family, so neighbouring frames land in the same split.
SPLIT_FRACTIONS = {"train": 0.70, "valid": 0.15, "test": 0.15}

# Background images as a fraction of defect images in the same split. Ultralytics suggests
# ~10%; this is higher because the deployed model will mostly be shown healthy blades and
# false positives are the failure users actually notice.
NEGATIVE_RATIO = 0.25


def block_split(ids: list[int]) -> dict[int, str]:
    """Assign sorted capture IDs to contiguous train/valid/test blocks."""
    ordered = sorted(ids)
    n = len(ordered)
    n_train = int(n * SPLIT_FRACTIONS["train"])
    n_valid = int(n * SPLIT_FRACTIONS["valid"])

    assignment = {}
    for i, cid in enumerate(ordered):
        if i < n_train:
            assignment[cid] = "train"
        elif i < n_train + n_valid:
            assignment[cid] = "valid"
        else:
            assignment[cid] = "test"
    return assignment


def place(src: Path, dst: Path, copy: bool) -> None:
    """Hardlink by default — same bytes, no extra disk, and still a real file when zipped."""
    if dst.exists():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--src", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=Path("datasets/turbine_v2"))
    parser.add_argument("--copy", action="store_true", help="copy files instead of hardlinking")
    parser.add_argument("--negative-ratio", type=float, default=NEGATIVE_RATIO)
    parser.add_argument(
        "--min-sharpness", type=float, metavar="VAR",
        help="drop ANNOTATED images below this Laplacian variance (see tools/image_quality.py "
             "--help-blur before raising it above ~20). Requires numpy and Pillow.",
    )
    parser.add_argument(
        "--keep-augmented", action="store_true",
        help="keep Roboflow's 3 pre-baked variants per source image in train. Off by default: "
             "Ultralytics augments online every epoch with fresh random parameters, which "
             "strictly dominates 3 fixed variants and does not triple epoch time.",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    src, out = args.src.resolve(), args.out.resolve()
    random.seed(args.seed)

    # --- Read v1, deduplicating by image content -------------------------------------
    seen_digests: set[str] = set()
    defects, healthy = [], []
    duplicates = 0

    for split in ("train", "valid", "test"):
        for image, _label, boxes in iter_split(src, split):
            digest = hashlib.md5(image.read_bytes()).hexdigest()
            if digest in seen_digests:
                duplicates += 1
                continue
            seen_digests.add(digest)

            source = source_name(image.stem)
            family, capture_id = split_family_id(source)
            record = {
                "image": image,
                "source": source,
                "family": family or "<numeric>",
                "id": capture_id,
                "boxes": [b for b in boxes if b.cls in CLASS_REMAP],
                "is_healthy": any(b.cls == HEALTHY_CLASS for b in boxes),
            }
            (healthy if record["is_healthy"] else defects).append(record)

    if not defects:
        raise SystemExit(f"no defect-class annotations found under {src}")

    # --- Optional sharpness floor ------------------------------------------------------
    # Only ANNOTATED images are filtered. A blurry background is still a correct negative,
    # and it teaches the model not to hallucinate defects on soft frames — which is exactly
    # what real drone footage looks like. A blurry *annotated* image is different: if the
    # defect is a smear, its box cannot teach localisation, only noise.
    dropped_by_class: Counter = Counter()
    if args.min_sharpness is not None:
        try:
            from training.common.sharpness import laplacian_variance
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise SystemExit(f"--min-sharpness needs numpy and Pillow ({exc})")

        kept = []
        for record in defects:
            if laplacian_variance(record["image"]) >= args.min_sharpness:
                kept.append(record)
            else:
                for cls in {b.cls for b in record["boxes"]}:
                    dropped_by_class[V2_NAMES[CLASS_REMAP[cls]]] += 1
        removed = len(defects) - len(kept)
        defects = kept

        if not defects:
            raise SystemExit(
                f"--min-sharpness {args.min_sharpness} removed every annotated image. "
                "That threshold is far too high — see tools/image_quality.py --help-blur."
            )
        share = removed / (removed + len(defects))
        print(f"Sharpness floor {args.min_sharpness}: dropped {removed} annotated images "
              f"({share:.1%})")
        for name, count in dropped_by_class.most_common():
            print(f"    {name:<20}{count:>6}")
        if share > 0.35:
            print("  ! That is a large share of your only defect data. Real drone footage is\n"
                  "    blurry, so filtering to sharp frames makes training less like deployment.\n"
                  "    Read tools/image_quality.py --help-blur before keeping this threshold.")
        print()

    # --- Group-aware split, computed per family over capture IDs ----------------------
    # Every augmented copy of a source shares that source's id, so copies cannot straddle
    # the split boundary.
    per_family_ids = defaultdict(set)
    for record in defects + healthy:
        if record["id"] is not None:
            per_family_ids[record["family"]].add(record["id"])

    assignment = {
        family: block_split(list(ids)) for family, ids in per_family_ids.items()
    }

    def split_of(record) -> str:
        if record["id"] is None:
            return "train"
        return assignment[record["family"]][record["id"]]

    # --- Choose which files land where ------------------------------------------------
    # One copy per source everywhere by default. Roboflow baked 3 augmented variants of each
    # training image, but Ultralytics applies mosaic, flip, HSV, scale and rotation online
    # each epoch with fresh parameters — strictly more varied than 3 frozen variants. Keeping
    # both means every epoch costs 3x as much to show the model the same scenes.
    chosen: dict[str, list] = {"train": [], "valid": [], "test": []}
    seen_sources: dict[str, set] = {"train": set(), "valid": set(), "test": set()}

    for record in defects:
        split = split_of(record)
        if args.keep_augmented and split == "train":
            chosen["train"].append(record)
        elif record["source"] not in seen_sources[split]:
            seen_sources[split].add(record["source"])
            chosen[split].append(record)
    held_out_sources = seen_sources

    # Negatives: subsample per split, again one copy per source outside train.
    by_split_healthy = defaultdict(list)
    for record in healthy:
        split = split_of(record)
        by_split_healthy[split].append(record)

    negatives_added = {}
    for split, pool in by_split_healthy.items():
        if not (args.keep_augmented and split == "train"):
            # One copy per source, so each scene is counted once.
            unique: dict[str, dict] = {}
            for record in pool:
                unique.setdefault(record["source"], record)
            pool = list(unique.values())

        # A dedicated RNG per split. Sharing the global one meant a change to the TRAIN
        # pool size consumed a different amount of RNG state and silently reshuffled which
        # negatives landed in valid/test — quietly invalidating any A/B against an earlier
        # build. Seeded by split name so each split is reproducible on its own.
        rng = random.Random(f"{args.seed}:{split}")
        budget = int(len(chosen[split]) * args.negative_ratio)
        pool = sorted(pool, key=lambda r: r["image"].name)   # stable order before shuffling
        rng.shuffle(pool)
        picked = pool[:budget]
        chosen[split].extend(picked)
        negatives_added[split] = len(picked)

    # --- Write it out -----------------------------------------------------------------
    if out.exists():
        shutil.rmtree(out)

    stats = {}
    for split, records in chosen.items():
        images_dir = out / split / "images"
        labels_dir = out / split / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        counts: Counter = Counter()
        for record in records:
            place(record["image"], images_dir / record["image"].name, args.copy)
            lines = []
            for box in record["boxes"]:
                remapped = CLASS_REMAP[box.cls]
                counts[remapped] += 1
                lines.append(
                    f"{remapped} {box.xc:.6f} {box.yc:.6f} {box.w:.6f} {box.h:.6f}"
                )
            # An empty file is the explicit "background" signal Ultralytics expects.
            (labels_dir / f"{record['image'].stem}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else "")
            )

        stats[split] = {
            "images": len(records),
            "backgrounds": negatives_added.get(split, 0),
            "boxes": sum(counts.values()),
            "per_class": {V2_NAMES[c]: counts.get(c, 0) for c in range(len(V2_NAMES))},
        }

    # data.yaml with absolute paths, so it works from any working directory. The v1 file
    # used `../train/images`, which resolved correctly from nowhere.
    (out / "data.yaml").write_text(
        f"path: {out}\n"
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n\n"
        f"nc: {len(V2_NAMES)}\n"
        f"names: {V2_NAMES}\n\n"
        "# Built by tools/rebuild_turbine.py. `healthy` is intentionally not a class:\n"
        "# an image with no predicted boxes is the healthy result. Empty label files are\n"
        "# background negatives that teach false-positive suppression.\n"
    )

    summary = {
        "source": str(src),
        "output": str(out),
        "classes": V2_NAMES,
        "dropped_class": "healthy (kept as background negatives)",
        "duplicates_removed": duplicates,
        "min_sharpness": args.min_sharpness,
        "kept_augmented_copies": args.keep_augmented,
        "dropped_by_sharpness": dict(dropped_by_class),
        "splits": stats,
    }
    (out / "build_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"Rebuilt {src} -> {out}")
    print(f"  duplicates removed: {duplicates}\n")
    print(f"  {'split':<8}{'images':>8}{'backgrounds':>13}{'boxes':>8}   per class")
    for split, s in stats.items():
        per_class = ", ".join(f"{k} {v}" for k, v in s["per_class"].items())
        print(
            f"  {split:<8}{s['images']:>8}{s['backgrounds']:>13}{s['boxes']:>8}   {per_class}"
        )
    print(f"\n  data.yaml -> {out / 'data.yaml'}")
    print("\nCaveat: the background negatives come from a different capture domain (aerial")
    print("wide shots) than the defect close-ups, so they only partially teach false-positive")
    print("suppression. Adding real in-domain healthy frames is the stronger fix.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
