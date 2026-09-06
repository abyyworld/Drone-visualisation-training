#!/usr/bin/env python3
"""Find images that do not belong in this dataset, by how far they sit from everything else.

WHAT DID NOT WORK, AND WHY IT IS RECORDED HERE
    The first version of this tool used two hand-built heuristics: a COCO-pretrained
    detector flagging off-domain classes, and an edge-structure score for burnt-in text.
    On 3,090 turbine images it flagged 1,099 (35.6%) — and inspection of the flagged
    samples showed almost all were ordinary blade photographs.

    Both heuristics were mismatched to the domain in the same way. A turbine blade is a
    long thin pale object, so COCO confidently calls it `toothbrush`, `tie` or `bed`. A
    blade against sky is a single strong horizontal edge, which is exactly the structure a
    naive text-overlay detector looks for. The signals were measuring the subject, not
    anything wrong with it.

    Acting on that output would have deleted a third of the training data.

WHAT THIS DOES INSTEAD
    Outlier-ness is defined relative to the dataset rather than against a fixed idea of what
    the images should contain: embed every image with a pretrained backbone, take the
    centroid, and rank by cosine distance from it. Whatever the dataset is mostly made of
    becomes the norm, and genuine intruders — a screenshot, a colour smear, a photo of
    something else entirely — sit far out in the tail regardless of subject.

    This has no opinion about turbines, so it transfers to the solar dataset unchanged.

    A near-uniform frame check is kept, since it is objective: a greyscale standard
    deviation near zero means there is nothing in the image whatever the subject.

Nothing is deleted. The tool ranks and copies images out for review. Look at them. The
previous version is the argument for why.

Usage:
    python3 tools/find_outliers.py [DATASET_ROOT] [--top 100] [--review-dir /tmp/review]

Requires `pip install ultralytics numpy Pillow`.
"""


from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.common.labels import iter_split, source_name  # noqa: E402

SPLITS = ("train", "valid", "test")

# Below this greyscale standard deviation the frame is effectively featureless. Objective
# and subject-independent, unlike anything based on what the image is *of*.
BLANK_STD = 12.0


def embed_all(paths, batch=32):
    """Cosine-normalised embeddings from a pretrained backbone, one row per image."""
    import numpy as np
    from ultralytics import YOLO

    model = YOLO("yolo11n.pt")
    rows = []
    for start in range(0, len(paths), batch):
        chunk = [str(p) for p in paths[start:start + batch]]
        for tensor in model.embed(chunk, imgsz=320, device="cpu", verbose=False):
            vector = tensor.cpu().numpy().reshape(-1)
            norm = np.linalg.norm(vector)
            rows.append(vector / norm if norm else vector)
        print(f"  embedded {min(start + batch, len(paths))}/{len(paths)}", end="\r", flush=True)
    print(" " * 50, end="\r")
    return np.stack(rows)


def blankness(path):
    import numpy as np
    from PIL import Image

    with Image.open(path) as handle:
        return float(np.asarray(handle.convert("L"), dtype=np.float32).std())


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("root", nargs="?", default=".", type=Path)
    parser.add_argument("--limit", type=int, help="scan only the first N unique sources")
    parser.add_argument("--top", type=int, default=100, help="how many of the furthest to report")
    parser.add_argument("--out", type=Path, default=Path("docs/outliers.json"))
    parser.add_argument("--review-dir", type=Path, help="copy the furthest images here to look at")
    parser.add_argument("--annotated-only", action="store_true",
                        help="only scan images carrying labels")
    args = parser.parse_args()

    try:
        import numpy as np
    except ImportError:
        raise SystemExit("needs ultralytics, numpy and Pillow")

    root = args.root.resolve()

    # One representative per source: augmented copies are the same picture, and including
    # them would drag the centroid toward whatever happens to be augmented most.
    seen: dict[str, dict] = {}
    for split in SPLITS:
        for image, _label, boxes in iter_split(root, split):
            key = source_name(image.stem)
            if key in seen or (args.annotated_only and not boxes):
                continue
            seen[key] = {"path": image, "split": split, "labelled": bool(boxes)}

    sources = list(seen.values())[: args.limit] if args.limit else list(seen.values())
    if not sources:
        raise SystemExit(f"no images found under {root}")

    print(f"Embedding {len(sources)} unique source images from {root}\n")
    vectors = embed_all([r["path"] for r in sources])

    centroid = vectors.mean(axis=0)
    centroid /= np.linalg.norm(centroid) or 1.0
    distance = 1.0 - vectors @ centroid          # cosine distance, 0 = typical

    for record, d in zip(sources, distance):
        record["distance"] = float(d)
        record["blank"] = blankness(record["path"]) < BLANK_STD

    ranked = sorted(sources, key=lambda r: -r["distance"])
    blanks = [r for r in sources if r["blank"]]

    print(f"cosine distance from the dataset centroid:")
    for label, value in (("min", distance.min()), ("median", np.median(distance)),
                         ("p95", np.percentile(distance, 95)), ("max", distance.max())):
        print(f"    {label:<8}{value:.4f}")
    print(f"\n  near-blank frames: {len(blanks)}")
    print(f"\n  furthest {min(args.top, len(ranked))} images (most likely not to belong):")
    for record in ranked[: min(12, args.top)]:
        print(f"    {record['distance']:.4f}  {record['path'].name}")

    payload = {
        "scanned": len(sources),
        "distance_median": float(np.median(distance)),
        "distance_p95": float(np.percentile(distance, 95)),
        "blank_frames": [str(r["path"].relative_to(root)) for r in blanks],
        "furthest": [
            {"path": str(r["path"].relative_to(root)), "split": r["split"],
             "labelled": r["labelled"], "distance": round(r["distance"], 4),
             "blank": r["blank"]}
            for r in ranked[: args.top]
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"\n  -> {args.out}")

    if args.review_dir:
        args.review_dir.mkdir(parents=True, exist_ok=True)
        for rank, record in enumerate(ranked[: args.top], 1):
            shutil.copy2(record["path"],
                         args.review_dir / f"{rank:03d}_{record['distance']:.3f}_{record['path'].name}")
        print(f"  -> {min(args.top, len(ranked))} images copied to {args.review_dir}, "
              "named by rank so the worst sort first")

    print("\nNothing was deleted. A ranking is not a verdict - the previous version of this")
    print("tool flagged 36% of the dataset and was wrong about nearly all of it. Look first.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
