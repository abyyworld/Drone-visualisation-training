"""Perceptual hashing, for telling distinct photographs from copies of one.

File counts lie. The previous turbine dataset was 7,520 files holding roughly 750 real
defect photographs; a public set of 701 images is redistributed elsewhere as 13,000 by
tiling it. Both look like large datasets and neither is, and no aggregate metric shows it.

A difference hash compares each pixel with its right-hand neighbour, so it encodes gradient
structure rather than absolute colour. Two exposures of one scene, a re-encode, a crop, a
mild rotation - all land within a few bits of each other. Two genuinely different
photographs do not.

Pillow only, no numpy: this is imported by tools that are otherwise dependency-free.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

# Bits differing between two 64-bit hashes, below which two images are the same scene.
# Six is empirically the gap between "same photograph, altered" and "different photograph";
# raising it starts merging genuinely distinct frames from one flight.
NEAR_DUPLICATE_BITS = 6


def dhash(path: Path, size: int = 8) -> int | None:
    """64-bit difference hash, or None if the image cannot be read."""
    try:
        from PIL import Image

        with Image.open(path) as handle:
            grey = handle.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
    except Exception:
        return None

    pixels = list(grey.getdata())
    bits = 0
    for row in range(size):
        offset = row * (size + 1)
        for column in range(size):
            bits = (bits << 1) | (pixels[offset + column] > pixels[offset + column + 1])
    return bits


def cluster(hashes: dict, threshold: int = NEAR_DUPLICATE_BITS) -> dict:
    """Union-find over Hamming distance. Returns representative key -> set of member keys.

    Quadratic in the number of images. Fine to a few thousand, which is the scale this
    domain offers; beyond that it wants bucketing by hash prefix.
    """
    parent = {key: key for key in hashes}

    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    items = list(hashes.items())
    for index, (key_a, digest_a) in enumerate(items):
        for key_b, digest_b in items[index + 1:]:
            if bin(digest_a ^ digest_b).count("1") <= threshold:
                root_a, root_b = find(key_a), find(key_b)
                if root_a != root_b:
                    parent[root_a] = root_b

    groups = defaultdict(set)
    for key in hashes:
        groups[find(key)].add(key)
    return dict(groups)


def available() -> bool:
    """Whether Pillow is installed, so callers can degrade instead of failing."""
    try:
        import PIL  # noqa: F401

        return True
    except ImportError:
        return False
