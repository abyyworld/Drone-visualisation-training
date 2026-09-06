#!/usr/bin/env python3
"""Measure image quality across a YOLO dataset, broken down by class and source family.

Why per-class matters more than an overall average: if every image carrying one class is
soft, that class's AP is capped no matter how you train. An aggregate quality number hides
exactly the thing you need to know.

WHAT THE SHARPNESS NUMBER IS
    Variance of the Laplacian, the standard focus metric. High = crisp edges. Low = soft.

WHAT IT IS NOT
    A quality score. It conflates "out of focus" with "genuinely smooth content". A blade
    against clear sky is low-texture and will score low while being a perfectly good image.
    Treat low sharpness as a flag to LOOK at, not a verdict to act on automatically.

    Roboflow also applied 0-1px Gaussian blur as augmentation to this export, so some of the
    softness is from the pipeline rather than the source photograph.

BEFORE YOU DELETE ANYTHING, read `--help-blur`. Deleting blurry training images is usually
the wrong move and this tool will argue with you about it.

Usage:
    python3 tools/image_quality.py [DATASET_ROOT]
    python3 tools/image_quality.py . --list-below 20 > blurry.txt
    python3 tools/image_quality.py . --json docs/quality.json
    python3 tools/image_quality.py --help-blur

Requires numpy and Pillow (`pip install numpy Pillow`).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.common.labels import iter_split, source_name, split_family_id  # noqa: E402

SPLITS = ("train", "valid", "test")

# Below this, an image is soft enough that a small defect may not be localisable. It is a
# flag for inspection, not a delete threshold — see --help-blur.
SOFT = 100
# Below this the image is a smear. A bounding box on it cannot mean much.
UNUSABLE = 20

BLUR_ADVICE = """
SHOULD YOU DELETE BLURRY IMAGES?

Almost certainly not in bulk. Three reasons, in order of how much they matter:

1. REAL DRONE FOOTAGE IS BLURRY. The model will be deployed on frames captured from a moving
   aircraft, often at distance, often in poor light. If you train only on sharp images you
   have made the training distribution LESS like deployment, not more. That is the same class
   of error as the source-family shortcut this project already had: the model scores well on
   a clean set and fails on reality.

2. IT WOULD GUT THE DEFECT DATA. In this dataset the defect close-ups are markedly softer
   than the healthy aerials. A blanket sharpness cut removes most of the only images that
   carry corrosion, crack and surface_peeling labels.

3. THE METRIC IS NOT A QUALITY SCORE. Laplacian variance cannot tell "out of focus" from
   "smooth surface". A blade against sky is legitimately low-texture.

WHAT TO DO INSTEAD

  a) Trim only the extreme tail. Below roughly 20 the image is a smear and its label cannot
     be meaningful. `tools/rebuild_turbine.py --min-sharpness 20` does this, and reports
     exactly what it dropped and from which classes.

  b) Use sharpness as an EVALUATION STRATIFIER, not a training filter. Report mAP separately
     on sharp and soft test images. If the model only works on sharp frames, you have learned
     something that predicts field performance — and no aggregate metric would have told you.

  c) Fix composition, not sharpness. The reason to add data here is that no image in this
     dataset contains a defect and a healthy region together, so the model can win by
     recognising capture style. A perfectly sharp version of this dataset would still have
     that flaw. Sharpness is a second-order problem.
"""


def measure(path: Path):
    """Sharpness, brightness, contrast and size for one image."""
    import numpy as np
    from numpy.lib.stride_tricks import sliding_window_view
    from PIL import Image

    with Image.open(path) as handle:
        grey = handle.convert("L")
        width, height = grey.size
        array = np.asarray(grey, dtype=np.float32)

    # 4-neighbour Laplacian; its variance is the classic focus measure.
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    windows = sliding_window_view(array, (3, 3))
    laplacian = (windows * kernel).sum(axis=(-1, -2))

    return {
        "sharpness": float(laplacian.var()),
        "brightness": float(array.mean()),
        "contrast": float(array.std()),
        "width": width,
        "height": height,
        "kb": path.stat().st_size / 1024,
    }


def summarise(values: list[float]) -> dict:
    ordered = sorted(values)

    def pct(p):
        if not ordered:
            return 0.0
        index = min(len(ordered) - 1, int(len(ordered) * p / 100))
        return ordered[index]

    return {
        "n": len(ordered),
        "p1": round(pct(1), 1),
        "p10": round(pct(10), 1),
        "median": round(statistics.median(ordered), 1) if ordered else 0.0,
        "p90": round(pct(90), 1),
        "soft_pct": round(100 * sum(v < SOFT for v in ordered) / len(ordered), 1) if ordered else 0.0,
        "unusable_pct": round(100 * sum(v < UNUSABLE for v in ordered) / len(ordered), 1) if ordered else 0.0,
    }


def table(title: str, groups: dict[str, list[float]]) -> None:
    print(f"\n{title}")
    header = f"  {'group':<22}{'n':>7}{'p1':>8}{'p10':>8}{'median':>9}{'p90':>8}{'soft%':>8}{'unusable%':>11}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for name, values in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        s = summarise(values)
        print(
            f"  {name:<22}{s['n']:>7}{s['p1']:>8.0f}{s['p10']:>8.0f}{s['median']:>9.0f}"
            f"{s['p90']:>8.0f}{s['soft_pct']:>7.1f}%{s['unusable_pct']:>10.1f}%"
        )


def load_class_names(root: Path) -> list[str]:
    data_yaml = root / "data.yaml"
    if not data_yaml.exists():
        return []
    for line in data_yaml.read_text().splitlines():
        if line.startswith("names:"):
            inner = line.split(":", 1)[1].strip().strip("[]")
            return [n.strip().strip("'\"") for n in inner.split(",") if n.strip()]
    return []


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("root", nargs="?", default=".", type=Path)
    parser.add_argument("--json", type=Path, help="write full per-image measurements")
    parser.add_argument(
        "--list-below", type=float, metavar="SHARPNESS",
        help="print paths below this sharpness (one per line) and nothing else",
    )
    parser.add_argument("--help-blur", action="store_true", help="why bulk-deleting blurry images is a bad idea")
    args = parser.parse_args()

    if args.help_blur:
        print(BLUR_ADVICE)
        return 0

    try:
        import numpy  # noqa: F401
        from PIL import Image  # noqa: F401
    except ImportError:
        raise SystemExit("needs numpy and Pillow — run `pip install numpy Pillow`")

    root = args.root.resolve()
    names = load_class_names(root)
    records = []

    for split in SPLITS:
        for image, _label, boxes in iter_split(root, split):
            family, _ = split_family_id(source_name(image.stem))
            stats = measure(image)
            records.append({
                **stats,
                "path": str(image.relative_to(root)),
                "split": split,
                "family": family or "<numeric>",
                "classes": sorted({b.cls for b in boxes}),
            })

    if not records:
        raise SystemExit(f"no labelled images found under {root}")

    if args.list_below is not None:
        for record in sorted(records, key=lambda r: r["sharpness"]):
            if record["sharpness"] < args.list_below:
                print(record["path"])
        return 0

    print(f"Dataset: {root}\nImages measured: {len(records)}")
    print(f"\nSharpness = variance of the Laplacian. Soft < {SOFT}, unusable < {UNUSABLE}.")
    print("It measures focus, NOT quality — smooth content scores low too. See --help-blur.")

    by_split = defaultdict(list)
    by_family = defaultdict(list)
    by_class = defaultdict(list)
    for record in records:
        by_split[record["split"]].append(record["sharpness"])
        by_family[record["family"]].append(record["sharpness"])
        if not record["classes"]:
            by_class["(background)"].append(record["sharpness"])
        for cls in record["classes"]:
            by_class[names[cls] if cls < len(names) else str(cls)].append(record["sharpness"])

    table("By split", by_split)
    table("By source family", by_family)
    table("By class present in the image  <- the one that matters", by_class)

    resolutions = {(r["width"], r["height"]) for r in records}
    print(f"\nResolutions: {len(resolutions)} distinct"
          + (f" — all {resolutions.pop()}" if len(resolutions) == 1 else ""))

    overall = summarise([r["sharpness"] for r in records])
    unusable = [r for r in records if r["sharpness"] < UNUSABLE]
    print(f"\nOverall: median {overall['median']:.0f}, "
          f"{overall['soft_pct']}% soft, {overall['unusable_pct']}% unusable")

    if unusable:
        affected = defaultdict(int)
        for record in unusable:
            for cls in record["classes"] or [None]:
                affected[names[cls] if cls is not None and cls < len(names) else "(background)"] += 1
        print(f"\n{len(unusable)} images below {UNUSABLE} (a smear — a box on one cannot mean much):")
        for name, count in sorted(affected.items(), key=lambda kv: -kv[1]):
            print(f"    {name:<20}{count:>6}")
        print(f"\n  Drop them with: tools/rebuild_turbine.py --min-sharpness {UNUSABLE}")
        print("  Do NOT raise that threshold to 'clean up' the dataset — run --help-blur first.")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(records, indent=2))
        print(f"\nPer-image measurements -> {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
