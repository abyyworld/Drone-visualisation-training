#!/usr/bin/env python3
"""Turn accumulated vision-model inspections into a YOLO training set.

WHY THIS EXISTS
    This is the point of running inspections through an API at all.

    The turbine detector failed on a blade snapped in half for one reason above all others:
    every image it learned from was a close-up bought from a dataset site, and the
    photograph it was shown was a wide landscape shot taken by an actual drone. No amount
    of retraining on the same data fixes a gap like that. Only photographs from the job
    fix it.

    So every inspection writes a label sidecar, and this turns a pile of those into a
    dataset. In a year the dataset is made of the operator's own imagery, at the operator's
    own framing, in the operator's own weather - and a detector trained on it can replace
    the API, which is the plan.

REVIEWED LABELS ONLY, BY DEFAULT
    A vision model's output is a strong first draft, not ground truth. Its boxes are loose,
    it names the same defect three ways across three images, and it occasionally describes
    something that is not there. Training on unreviewed output teaches a detector to
    reproduce those mistakes with more confidence and less explanation.

    So a sidecar counts only once someone has set "reviewed": true in it, having looked at
    the annotated copy and fixed or deleted what is wrong. --include-unreviewed exists for
    measuring how much data is waiting, not for building a set to ship.

SPLITS ARE BY INSPECTION, NOT BY IMAGE
    Twenty photographs of one turbine on one afternoon are twenty views of one thing. Split
    them across train and validation and the validation score measures memorisation, which
    is exactly how the first turbine model came to report 0.78 while being useless. Whole
    inspection folders go to one split or another, never both.

USAGE
    python3 tools/vlm_to_yolo.py inspections/ --out datasets/turbine_field
    python3 tools/vlm_to_yolo.py inspections/ --domain turbine --dry-run
    python3 tools/vlm_to_yolo.py inspections/ --out datasets/x --include-unreviewed
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import struct
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from merge_datasets import DROP, FRACTIONS, propose  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

# Ultralytics' own production guidance, quoted so the numbers below have a source rather
# than being a feeling. A set under these is worth building and worth knowing is thin.
TARGET_IMAGES = 1500
TARGET_INSTANCES = 10000
MIN_INSTANCES_PER_CLASS = 100


# ---------------------------------------------------------------------------------------
# Image dimensions
# ---------------------------------------------------------------------------------------

def image_size(path: Path):
    """Width and height, from the file header, with no third-party dependency.

    Boxes are stored in pixels, and YOLO wants them normalised, so the dimensions have to
    come from somewhere trustworthy. The sidecar records them, but only when Pillow was
    installed during the inspection - and a dataset builder that silently mis-normalises a
    box because a value was missing is worse than one that will not run. Reading the header
    needs 30 lines and always works.
    """
    try:
        with path.open("rb") as handle:
            head = handle.read(26)

            if head[:8] == b"\x89PNG\r\n\x1a\n" and head[12:16] == b"IHDR":
                return struct.unpack(">II", head[16:24])

            if head[:2] == b"\xff\xd8":  # JPEG: walk the segment chain to a frame header
                handle.seek(2)
                while True:
                    marker = handle.read(2)
                    if len(marker) < 2 or marker[0] != 0xFF:
                        return None
                    length = struct.unpack(">H", handle.read(2))[0]
                    # SOF0..SOF15, excluding the four that are not frame headers.
                    if 0xC0 <= marker[1] <= 0xCF and marker[1] not in (0xC4, 0xC8, 0xCC):
                        handle.read(1)  # sample precision
                        height, width = struct.unpack(">HH", handle.read(4))
                        return width, height
                    handle.seek(length - 2, 1)

            if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
                if head[12:16] == b"VP8X":
                    handle.seek(24)
                    raw = handle.read(6)
                    width = 1 + int.from_bytes(raw[0:3], "little")
                    height = 1 + int.from_bytes(raw[3:6], "little")
                    return width, height
                if head[12:16] == b"VP8 ":
                    handle.seek(26)
                    width, height = struct.unpack("<HH", handle.read(4))
                    return width & 0x3FFF, height & 0x3FFF
    except (OSError, struct.error):
        return None
    return None


# ---------------------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------------------

def load_sidecars(roots, domain, include_unreviewed):
    """Gather label sidecars, grouped by the inspection they came from.

    @returns (groups, skipped) where groups maps an inspection directory to its records.
    """
    groups = defaultdict(list)
    skipped = Counter()

    files = []
    for root in roots:
        if root.is_file() and root.suffix == ".json":
            files.append(root)
        else:
            files.extend(sorted(root.rglob("labels/*.json")))

    for path in files:
        try:
            record = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            skipped["unreadable sidecar"] += 1
            continue

        if domain != "any" and record.get("domain") != domain:
            skipped[f"other domain ({record.get('domain')})"] += 1
            continue
        if not record.get("reviewed") and not include_unreviewed:
            skipped["not reviewed"] += 1
            continue

        # The sidecar stores an absolute source path from the machine that ran the
        # inspection, which will not exist on another one. The annotated copy next to it
        # is not the original. Look for the original beside the inspection first.
        session = path.parent.parent
        image = _find_image(session, record, path)
        if image is None:
            skipped["image not found"] += 1
            continue

        record["_image_path"] = image
        record["_sidecar"] = path
        groups[session].append(record)

    return groups, skipped


def _find_image(session: Path, record: dict, sidecar: Path):
    name = record.get("image") or f"{sidecar.stem}.jpg"
    for candidate in (
        Path(record.get("source", "")),
        session / "images" / name,
        session / name,
        session.parent / "images" / name,
    ):
        if candidate and candidate.is_file():
            return candidate
    # Last resort: the annotated copy carries the boxes burned in, which would teach the
    # detector to find drawn rectangles. Never fall back to it.
    return None


# ---------------------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------------------

def to_yolo_line(class_id: int, box, width: int, height: int):
    """Pixel [x0,y0,x1,y1] -> `class cx cy w h`, normalised, or None if degenerate."""
    x0, y0, x1, y1 = (float(v) for v in box)
    x0, x1 = sorted((max(0.0, x0), min(float(width), x1)))
    y0, y1 = sorted((max(0.0, y0), min(float(height), y1)))
    bw, bh = x1 - x0, y1 - y0
    if bw < 2 or bh < 2:
        return None
    return (
        f"{class_id} {(x0 + bw / 2) / width:.6f} {(y0 + bh / 2) / height:.6f} "
        f"{bw / width:.6f} {bh / height:.6f}"
    )


def split_groups(groups, seed):
    """Assign whole inspections to splits, largest first so the proportions come out close.

    Random assignment of a handful of unequal groups routinely lands 90% of the images in
    train and leaves validation with two. Placing the biggest group into whichever split is
    furthest below its quota keeps the ratios honest with very few groups, which is exactly
    the situation early on.
    """
    keys = sorted(groups, key=lambda k: (-len(groups[k]), str(k)))
    random.Random(seed).shuffle(keys := keys[:])
    keys.sort(key=lambda k: -len(groups[k]))

    total = sum(len(groups[k]) for k in keys)
    quota = {name: fraction * total for name, fraction in FRACTIONS.items()}
    assigned = {name: 0 for name in FRACTIONS}
    placement = {}

    for key in keys:
        target = max(assigned, key=lambda name: quota[name] - assigned[name])
        placement[key] = target
        assigned[target] += len(groups[key])
    return placement, assigned


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("inspections", nargs="+", type=Path,
                        help="inspection directories written by tools/vlm_inspect.py")
    parser.add_argument("--out", type=Path, help="dataset directory to build")
    parser.add_argument("--domain", choices=["turbine", "solar", "any"], default="turbine")
    parser.add_argument("--include-unreviewed", action="store_true",
                        help="count sidecars nobody has checked (for measuring, not for training)")
    parser.add_argument("--drop-unknown", action="store_true",
                        help="discard labels with no place in the class table instead of stopping")
    parser.add_argument("--dry-run", action="store_true", help="report and write nothing")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    if not args.out and not args.dry_run:
        parser.error("give --out, or pass --dry-run to see what is there")

    groups, skipped = load_sidecars(args.inspections, args.domain, args.include_unreviewed)
    images = sum(len(v) for v in groups.values())

    if skipped:
        print("Skipped:")
        for reason, count in skipped.most_common():
            print(f"  {count:5d}  {reason}")
    if not images:
        print("\nNothing to convert.")
        if skipped.get("not reviewed"):
            print(f"{skipped['not reviewed']} sidecars are waiting on review. Open each annotated "
                  f"image, correct the boxes and labels in its sidecar, and set \"reviewed\": true.")
        return 1

    # --- class table -----------------------------------------------------------------
    raw_labels = Counter()
    for records in groups.values():
        for record in records:
            for detection in record.get("detections", []):
                raw_labels[detection["label"]] += 1

    mapping, unmatched = propose(list(raw_labels))
    if unmatched and not args.drop_unknown:
        print("\nThese labels have no place in the class table:")
        for name in sorted(unmatched):
            print(f"  {raw_labels[name]:5d}  {name}")
        print("\nAdd them to SYNONYMS or NOT_A_DEFECT in tools/merge_datasets.py, or rerun "
              "with --drop-unknown to discard them. Guessing a class here would put the wrong "
              "boxes in the training set, which is the failure this whole rebuild exists to fix.")
        return 1

    classes = sorted({target for target in mapping.values() if target != DROP})
    class_id = {name: index for index, name in enumerate(classes)}

    # --- count what would be written ---------------------------------------------------
    per_class = Counter()
    unlocated_total = 0
    usable_images = 0
    for records in groups.values():
        for record in records:
            size = image_size(record["_image_path"])
            if size is None:
                size = (record.get("width"), record.get("height"))
            if not size or not size[0] or not size[1]:
                continue
            record["_size"] = size
            lines = []
            for detection in record.get("detections", []):
                target = mapping.get(detection["label"], DROP)
                if target == DROP:
                    continue
                line = to_yolo_line(class_id[target], detection["box"], size[0], size[1])
                if line:
                    lines.append(line)
                    per_class[target] += 1
            record["_lines"] = lines
            unlocated_total += len(record.get("unlocated", []))
            usable_images += 1

    instances = sum(per_class.values())
    negatives = sum(1 for r in sum(groups.values(), []) if not r.get("_lines"))

    print(f"\n{len(groups)} inspection{'' if len(groups) == 1 else 's'}, "
          f"{usable_images} usable image{'' if usable_images == 1 else 's'}, "
          f"{instances} instance{'' if instances == 1 else 's'} across {len(classes)} classes.")
    for name in classes:
        flag = "  thin" if per_class[name] < MIN_INSTANCES_PER_CLASS else ""
        print(f"  {per_class[name]:6d}  {name}{flag}")
    print(f"  {negatives:6d}  images with no defect (background negatives)")
    if unlocated_total:
        print(f"\n{unlocated_total} finding{'' if unlocated_total == 1 else 's'} had no usable box "
              f"and are not in this set. They are still in the sidecars; box them by hand to "
              f"recover them.")

    print(f"\nAgainst the production target ({TARGET_IMAGES} images, {TARGET_INSTANCES} instances): "
          f"{usable_images / TARGET_IMAGES:.0%} of images, {instances / TARGET_INSTANCES:.0%} of "
          f"instances.")
    if args.include_unreviewed:
        print("These counts include unreviewed labels. Do not train a model you intend to "
              "deploy on this set.")

    if args.dry_run:
        return 0

    # --- write ---------------------------------------------------------------------------
    placement, assigned = split_groups(groups, args.seed)
    out = args.out
    if out.exists():
        shutil.rmtree(out)
    for split in FRACTIONS:
        (out / split / "images").mkdir(parents=True, exist_ok=True)
        (out / split / "labels").mkdir(parents=True, exist_ok=True)

    written = Counter()
    for session, records in groups.items():
        split = placement[session]
        for record in records:
            if "_lines" not in record:
                continue
            # Inspections are named per session, and two sessions can hold DJI_0001.JPG.
            # Prefixing with the session keeps them apart and keeps the provenance visible
            # in the filename, which matters when a bad label has to be traced back.
            stem = f"{session.name}__{record['_image_path'].stem}"
            shutil.copy2(record["_image_path"],
                         out / split / "images" / f"{stem}{record['_image_path'].suffix}")
            (out / split / "labels" / f"{stem}.txt").write_text(
                "\n".join(record["_lines"]) + ("\n" if record["_lines"] else ""))
            written[split] += 1

    names = "\n".join(f"  {index}: {name}" for index, name in enumerate(classes))
    (out / "data.yaml").write_text(
        f"# Built by tools/vlm_to_yolo.py from {len(groups)} field inspections.\n"
        f"# Labels originated from a vision model and were reviewed by hand"
        f"{' - NO, they were not, this set is unreviewed' if args.include_unreviewed else ''}.\n"
        f"# Splits are whole inspections, never split within one, so validation is not a\n"
        f"# memorisation test. See the module docstring for why that matters here.\n"
        f"path: {out.resolve()}\n"
        f"train: train/images\nval: valid/images\ntest: test/images\n"
        f"nc: {len(classes)}\nnames:\n{names}\n"
    )

    print(f"\nWrote {sum(written.values())} images to {out}")
    for split in FRACTIONS:
        share = written[split] / max(1, sum(written.values()))
        print(f"  {split:<6} {written[split]:5d}  ({share:.0%}, target {FRACTIONS[split]:.0%})")
    print(f"\nAudit it: python3 tools/audit_dataset.py {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
