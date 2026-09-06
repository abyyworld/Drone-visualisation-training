#!/usr/bin/env python3
"""Count how many genuinely distinct scenes a dataset contains, not how many files.

A file count is the most misleading number in a computer-vision dataset. This export
advertises 7,520 images. It contains roughly 750 distinct defect scenes. Two separate
inflations stack up:

  1. OFFLINE AUGMENTATION. Roboflow wrote 3 variants of each training image (flip, +-15
     rotation, +-25% brightness, 0-1px blur). Those are not new information — the model sees
     the same scene three times. 58% of the files here are copies of another file.

  2. NEAR-DUPLICATE FRAMES. The capture IDs are dense contiguous runs, the signature of
     video frames or burst shots. Consecutive frames of the same blade from the same angle
     are not independent samples, however different their filenames are.

What matters for generalisation is the count after both are collapsed. That number sets a
ceiling on what any model can learn, and it is the number to quote when deciding whether you
need more data.

Method: 64-bit difference hash per unique source image, min-distance against the mirrored
hash too so horizontal flips do not read as distinct, then union-find clustering at a Hamming
threshold. A threshold of 6/64 is conservative — it merges only genuinely near-identical
frames. 10 starts merging distinct scenes and is reported for context, not for quoting.

Usage:
    python3 tools/scene_count.py [DATASET_ROOT] [--threshold 6] [--json OUT]

Requires numpy and Pillow.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.common.labels import iter_split, source_name  # noqa: E402

SPLITS = ("train", "valid", "test")

# Out of 64 bits. 6 merges only near-identical frames; beyond ~10 distinct scenes start
# collapsing into each other and the number stops meaning "distinct scene".
DEFAULT_THRESHOLD = 6
REPORT_THRESHOLDS = (0, 3, 6, 10)


def dhash(path: Path, mirror: bool = False, size: int = 8) -> bytes:
    """64-bit difference hash: compare each pixel with its right neighbour.

    Robust to brightness and compression (which is what we want — those are the
    augmentations), sensitive to content and geometry.
    """
    import numpy as np
    from PIL import Image

    with Image.open(path) as handle:
        image = handle.convert("L")
        if mirror:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        array = np.asarray(image.resize((size + 1, size), Image.BILINEAR), dtype=np.int16)
    return np.packbits(array[:, 1:] > array[:, :-1]).tobytes()


def cluster(bits, mirror_bits, threshold: int) -> int:
    """Union-find over pairs within `threshold` Hamming distance. Returns cluster count."""
    import numpy as np

    n = len(bits)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    # Blocked so the full n^2 bit-comparison never materialises at once.
    block = 400
    for start in range(0, n, block):
        chunk = bits[start:start + block]
        direct = (chunk[:, None, :] != bits[None, :, :]).sum(-1)
        flipped = (chunk[:, None, :] != mirror_bits[None, :, :]).sum(-1)
        distance = np.minimum(direct, flipped)
        for row, column in zip(*np.where(distance <= threshold)):
            a, b = start + row, column
            if a == b:
                continue
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

    return len({find(i) for i in range(n)})


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("root", nargs="?", default=".", type=Path)
    parser.add_argument("--threshold", type=int, default=DEFAULT_THRESHOLD)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    try:
        import numpy as np
    except ImportError:
        raise SystemExit("needs numpy and Pillow — run `pip install numpy Pillow`")

    root = args.root.resolve()

    # One representative per source name; the augmented copies are by construction the same
    # scene, so hashing them would only inflate the clustering cost.
    representative: dict[str, Path] = {}
    classes_of: dict[str, set] = defaultdict(set)
    files = 0

    for split in SPLITS:
        for image, _label, boxes in iter_split(root, split):
            files += 1
            source = source_name(image.stem)
            representative.setdefault(source, image)
            classes_of[source].update(b.cls for b in boxes)

    if not representative:
        raise SystemExit(f"no labelled images found under {root}")

    copies = Counter()
    for split in SPLITS:
        for image, _label, _boxes in iter_split(root, split):
            copies[source_name(image.stem)] += 1

    sources = sorted(representative)
    print(f"Dataset: {root}")
    print(f"  files                 {files:>7}")
    print(f"  unique source images  {len(sources):>7}"
          f"   ({100 * (1 - len(sources) / files):.1f}% of files are augmented copies)")
    print(f"  copies per source     {dict(sorted(Counter(copies.values()).items()))}")

    print(f"\nHashing {len(sources)} unique sources...")
    bits = np.stack([np.unpackbits(np.frombuffer(dhash(representative[s]), dtype=np.uint8))
                     for s in sources])
    mirror = np.stack([np.unpackbits(np.frombuffer(dhash(representative[s], mirror=True), dtype=np.uint8))
                       for s in sources])

    annotated = [i for i, s in enumerate(sources) if classes_of[s]]

    print(f"\n{'threshold':<12}{'distinct scenes':>17}{'of unique':>11}{'of files':>10}"
          f"{'annotated scenes':>19}")
    print("  " + "-" * 67)
    results = {}
    for threshold in REPORT_THRESHOLDS:
        total = cluster(bits, mirror, threshold)
        with_boxes = cluster(bits[annotated], mirror[annotated], threshold) if annotated else 0
        results[threshold] = {"distinct": total, "annotated": with_boxes}
        marker = "  <-" if threshold == args.threshold else ""
        print(f"  <= {threshold:<8}{total:>15}{100 * total / len(sources):>10.1f}%"
              f"{100 * total / files:>9.1f}%{with_boxes:>17}{marker}")

    headline = results[args.threshold]
    print(f"\nAt the conservative threshold ({args.threshold}/64):")
    print(f"  {files} files  ->  {headline['distinct']} distinct scenes"
          f"  ->  {headline['annotated']} carrying annotations")
    inflation = files / headline["annotated"] if headline["annotated"] else 0
    print(f"\n  The file count overstates the annotated training signal by {inflation:.1f}x.")
    print("  That is the number that bounds what a model can learn here, and the one to quote")
    print("  when deciding whether the dataset is big enough.")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({
            "dataset": str(root), "files": files, "unique_sources": len(sources),
            "thresholds": results, "headline_threshold": args.threshold,
        }, indent=2))
        print(f"\n  -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
