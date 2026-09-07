"""Generate YOLO-format datasets with dataset traps deliberately planted.

Purpose: prove tools/audit_dataset.py actually catches the failure that makes
fire datasets dangerous, rather than merely claiming to. Two datasets are
produced -- one that must FAIL the audit and one that must PASS -- so the tool
is tested in both directions. A leak detector that flags everything is as
useless as one that flags nothing.

The trap being reproduced is the one that wasted the sibling project's effort:
fire datasets are cut from video, so a *random* train/val split puts frame 0412
in train and the near-identical frame 0413 in val. Validation then measures
memorisation, every metric is inflated, and the model is useless in the field
while looking excellent on paper.

Writes real PNGs via a minimal stdlib encoder -- PIL is not installed here, and
the audit's structural checks must run without it anyway.
"""

from __future__ import annotations

import argparse
import json
import random
import struct
import zlib
from pathlib import Path

CLASSES = ("fire", "smoke")


def write_png(path: Path, pixels: list[list[tuple[int, int, int]]]) -> None:
    """Minimal RGB8 PNG encoder (stdlib only)."""
    height, width = len(pixels), len(pixels[0])
    raw = b"".join(
        b"\x00" + b"".join(struct.pack("BBB", *px) for px in row) for row in pixels
    )

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )


def frame(seq_id: int, idx: int, size: int = 64) -> tuple[list[list[tuple[int, int, int]]], tuple[float, float, float, float]]:
    """One frame of a synthetic 'video': a blob that drifts slowly.

    Consecutive frames differ only slightly -- which is exactly what makes a
    random split leak, and exactly what a near-duplicate check must catch.
    """
    rng = random.Random(seq_id * 1000)
    # Each sequence must be visually distinct from every other, or the audit
    # legitimately flags cross-sequence duplicates and the "clean" fixture is
    # not clean. The first version used a narrow colour range and a blob path
    # that different sequences collided on, and the audit caught it at
    # perceptual distance 0 -- which is the tool working, not misfiring.
    base = (rng.randrange(15, 210), rng.randrange(15, 210), rng.randrange(15, 210))
    # 2px drift per frame: visually near-identical neighbours, as in real video.
    cx = 12 + (seq_id * 13 + idx * 2) % (size - 26)
    cy = 12 + (seq_id * 19 + idx * 3) % (size - 26)
    r = 5 + (seq_id % 5)

    px = [[base for _ in range(size)] for _ in range(size)]
    # per-sequence texture, so backgrounds are not interchangeable
    for y in range(size):
        for x in range(size):
            n = (rng.randrange(-24, 25))
            px[y][x] = (max(0, min(255, base[0] + n)),
                        max(0, min(255, base[1] + n)),
                        max(0, min(255, base[2] + n)))
    for y in range(max(0, cy - r), min(size, cy + r)):
        for x in range(max(0, cx - r), min(size, cx + r)):
            if (x - cx) ** 2 + (y - cy) ** 2 <= r * r:
                px[y][x] = (240, 120, 30)
    box = ((cx) / size, (cy) / size, (2 * r) / size, (2 * r) / size)
    return px, box


def build(root: Path, *, leaky: bool, n_seq: int = 12, per_seq: int = 10, seed: int = 7) -> dict:
    """Build one dataset.

    leaky=True  -> frames assigned to splits at random (the trap), filename
                   family predicts class, and no image ever holds both classes.
    leaky=False -> split grouped by sequence, mixed filenames, some images
                   containing both classes.
    """
    rng = random.Random(seed)
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (root / sub).mkdir(parents=True, exist_ok=True)

    manifest = {"leaky": leaky, "sequences": n_seq, "frames_per_seq": per_seq, "files": []}

    for seq in range(n_seq):
        # In the leaky set the class is a property of the sequence AND is
        # encoded in the filename -- so a model can score well by reading the
        # filename distribution rather than the image.
        seq_cls = seq % 2
        grouped_split = "train" if seq < int(n_seq * 0.75) else "val"

        for i in range(per_seq):
            px, (cx, cy, bw, bh) = frame(seq, i)
            split = rng.choice(["train", "train", "train", "val"]) if leaky else grouped_split

            if leaky:
                name = f"{CLASSES[seq_cls]}_seq{seq:02d}_frame{i:04d}"
                lines = [f"{seq_cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"]
            else:
                name = f"img_{seq:02d}_{i:04d}"
                lines = [f"{seq_cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"]
                # A clean set has genuine co-occurrence: fire and smoke in one
                # frame. Its total absence means the model may only ever be
                # learning a whole-scene classifier.
                if i % 3 == 0:
                    other = 1 - seq_cls
                    lines.append(f"{other} {min(cx + 0.2, 0.9):.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            write_png(root / "images" / split / f"{name}.png", px)
            (root / "labels" / split / f"{name}.txt").write_text("\n".join(lines) + "\n")
            manifest["files"].append({"name": name, "split": split, "seq": seq, "frame": i})

    (root / "data.yaml").write_text(
        f"path: {root}\ntrain: images/train\nval: images/val\n"
        f"nc: {len(CLASSES)}\nnames: [{', '.join(CLASSES)}]\n"
    )
    (root / "fixture_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="/tmp/fixtures", help="output directory")
    args = ap.parse_args()

    out = Path(args.out)
    for leaky in (True, False):
        root = out / ("leaky" if leaky else "clean")
        m = build(root, leaky=leaky)
        n_files = len(m["files"])
        splits = {s: sum(1 for f in m["files"] if f["split"] == s) for s in ("train", "val")}
        # Count sequences that straddle the split -- the leak, quantified.
        by_seq: dict[int, set] = {}
        for f in m["files"]:
            by_seq.setdefault(f["seq"], set()).add(f["split"])
        straddling = sum(1 for v in by_seq.values() if len(v) > 1)
        print(f"{root}: {n_files} images {splits}, sequences straddling train/val: {straddling}/{len(by_seq)}")


if __name__ == "__main__":
    main()
