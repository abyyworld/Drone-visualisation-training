#!/usr/bin/env python3
"""Audit a YOLO-format fire/smoke dataset for the flaws that inflate scores.

Run this before training and again before believing any number that training
produced. It exits non-zero on a hard failure so it can gate a CI pipeline.

Why this tool is the first thing in ``tools/``: a sibling project trained a
model that scored mAP50 0.782 and was worthless in the field. The dataset had
been assembled so that filename families perfectly predicted the class, and no
image ever contained two classes at once. The model learned the filenames'
statistics -- it was a scene classifier wearing a detector's output layer --
and the validation split could not tell, because the same shortcut worked
there too.

Fire datasets built from video carry the same trap in a nastier form:
**consecutive frames randomly split across train and val**. Frame 411 in train
and frame 412 in val are the same photograph with the sensor noise moved
slightly. Validation then measures memorisation, every number rises, and the
rise is indistinguishable from progress. This is the default state of every
scraped fire dataset the author has looked at, so this tool assumes the trap is
present and reports PASS only once it has actively failed to find it.

The checks are in two tiers:

* **Structural** -- pairing, label syntax, class balance, box geometry,
  filename families, directory purity, class co-occurrence. These need nothing
  but the standard library and numpy, and they always run.
* **Pixel** -- perceptual near-duplicate matching across splits. Needs Pillow
  (and uses ``imagehash`` if it happens to be installed). When Pillow is
  absent the check reports SKIP with the install command and the structural
  checks still run. A skipped pixel check is never counted as a pass, because
  the filename heuristics catch a strict subset of what hashing catches.

Safety note, since this tool feeds a system people make decisions with: none
of the numbers below are a claim about a trained model's field performance.
An audit that finds nothing means the dataset does not contain *these* known
defects. Recall on tiny, distant, thin-smoke targets is measured by
``tools/evaluate.py`` against held-out real footage, and that measurement is
what matters.

Example:
    $ python3 tools/audit_dataset.py --data datasets/fire/data.yaml --json audit.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import struct
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

__all__ = [
    "SIZE_BINS",
    "IMAGE_EXTENSIONS",
    "Box",
    "Sample",
    "Dataset",
    "CheckResult",
    "AuditReport",
    "AuditOptions",
    "size_bucket",
    "load_dataset",
    "run_audit",
    "image_size_from_header",
    "main",
]

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Box-area buckets as a fraction of frame area, shared with ``tools/evaluate.py``
#: so that "tiny" means the same thing in the audit and in the recall report.
#: Fractions rather than pixels because the wire protocol is normalised and
#: because a dataset mixes resolutions; 0.001 of a 1280x720 frame is a 27 px
#: box, which is about the smallest thing an operator could act on.
SIZE_BINS: tuple[tuple[str, float, float], ...] = (
    ("tiny", 0.0, 0.001),
    ("small", 0.001, 0.01),
    ("medium", 0.01, 0.05),
    ("large", 0.05, 1.01),
)

IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
)

STATUS_PASS = "PASS"
STATUS_WARN = "WARN"
STATUS_FAIL = "FAIL"
STATUS_SKIP = "SKIP"

_STATUS_RANK = {STATUS_PASS: 0, STATUS_SKIP: 1, STATUS_WARN: 2, STATUS_FAIL: 3}

#: Path components that carry no information about the subject of an image and
#: would otherwise dominate the filename/class association check.
_STOP_TOKENS: frozenset[str] = frozenset(
    {
        "images", "image", "img", "imgs", "labels", "label", "data", "dataset",
        "datasets", "train", "training", "val", "valid", "validation", "test",
        "testing", "jpg", "jpeg", "png", "bmp", "webp", "frames", "frame",
        "new", "final", "copy", "yolo", "obj", "set", "part", "batch",
    }
)

#: Trailing integer run in a stem: ``clip07_frame_0142`` -> ("clip07_frame_", 142, "").
_SEQUENCE_RE = re.compile(r"^(?P<prefix>.*?)(?P<num>\d+)(?P<suffix>\D*)$")
_TOKEN_RE = re.compile(r"[a-z]{3,}")


# --------------------------------------------------------------------- model


@dataclass(frozen=True, slots=True)
class Box:
    """One YOLO label line: class index plus a normalised centre-form box."""

    cls_id: int
    cx: float
    cy: float
    w: float
    h: float

    @property
    def area(self) -> float:
        """Box area as a fraction of frame area."""
        return max(0.0, self.w) * max(0.0, self.h)

    def xyxy(self) -> tuple[float, float, float, float]:
        """Corner form, normalised 0..1, same convention as the wire protocol."""
        return (
            self.cx - self.w / 2.0,
            self.cy - self.h / 2.0,
            self.cx + self.w / 2.0,
            self.cy + self.h / 2.0,
        )


@dataclass(slots=True)
class Sample:
    """One image and its label file, with everything the checks need."""

    image: Path
    split: str
    label: Path | None = None
    boxes: tuple[Box, ...] = ()
    #: Per-file label syntax problems, already formatted for the report.
    label_errors: tuple[str, ...] = ()
    width: int = 0
    height: int = 0
    #: "header", "pillow" or "unknown" -- how the size above was determined.
    size_source: str = "unknown"
    #: True when the file could not be parsed as an image at all.
    corrupt: bool = False
    content_sha: str = ""
    phash: int | None = None

    @property
    def stem(self) -> str:
        return self.image.stem

    @property
    def class_ids(self) -> frozenset[int]:
        return frozenset(b.cls_id for b in self.boxes)

    def rel(self, root: Path) -> str:
        """Path relative to the dataset root, POSIX-style, for reports and tags."""
        try:
            return self.image.resolve().relative_to(root).as_posix()
        except ValueError:
            return self.image.as_posix()


@dataclass(slots=True)
class Dataset:
    """A loaded YOLO dataset: resolved splits, class names and samples."""

    root: Path
    data_yaml: Path | None
    names: tuple[str, ...]
    samples: list[Sample]
    #: Problems found while resolving data.yaml and the split directories.
    load_notes: list[str] = field(default_factory=list)

    @property
    def splits(self) -> dict[str, list[Sample]]:
        out: dict[str, list[Sample]] = defaultdict(list)
        for s in self.samples:
            out[s.split].append(s)
        return dict(out)

    def class_name(self, cls_id: int) -> str:
        if 0 <= cls_id < len(self.names):
            return self.names[cls_id]
        return f"<id {cls_id}>"


@dataclass(slots=True)
class CheckResult:
    """The outcome of one audit check."""

    name: str
    status: str
    headline: str
    lines: list[str] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "headline": self.headline,
            "detail": list(self.lines),
            "data": self.data,
        }


@dataclass(slots=True)
class AuditOptions:
    """Thresholds for the audit. Every one of these is a judgement call."""

    #: Two frames whose indices differ by no more than this, inside the same
    #: filename family but on opposite sides of the split, are treated as the
    #: same photograph. 3 is deliberately generous: at 25 fps it is 120 ms.
    sequence_gap: int = 3
    #: Hamming distance between 64-bit dHashes below which two images are
    #: "the same shot". Consecutive video frames typically land at 0-4.
    phash_threshold: int = 6
    #: A token or directory must cover at least this many images before its
    #: class purity is worth reporting.
    min_support: int = 5
    #: Purity at or above which a token is treated as predicting the class.
    purity_threshold: float = 0.98
    #: Warn when the most common class outnumbers the rarest by more than this.
    imbalance_ratio: float = 10.0
    #: Warn when fewer than this fraction of boxes fall in the "tiny" bucket.
    min_tiny_fraction: float = 0.02
    #: A label covering more than this fraction of the frame is a scene label.
    whole_frame_area: float = 0.9
    #: Cap on images hashed per split; 0 means hash everything.
    hash_limit: int = 0


#: Defaults for the CLI. A dataclass with ``slots=True`` exposes its class
#: attributes as descriptors rather than values, so argparse has to read the
#: defaults off an instance.
_DEFAULTS = AuditOptions()


@dataclass(slots=True)
class AuditReport:
    """Everything the audit concluded, renderable as text or JSON."""

    dataset: Dataset
    checks: list[CheckResult]
    summary: dict[str, Any]

    @property
    def status(self) -> str:
        """Worst status across all checks."""
        worst = STATUS_PASS
        for check in self.checks:
            if _STATUS_RANK[check.status] > _STATUS_RANK[worst]:
                worst = check.status
        return worst

    def counts(self) -> dict[str, int]:
        counts = Counter(c.status for c in self.checks)
        return {k: counts.get(k, 0) for k in (STATUS_PASS, STATUS_WARN, STATUS_FAIL, STATUS_SKIP)}

    def exit_code(self, strict: bool = False) -> int:
        """0 when the dataset may be trained on, 1 when it must not be."""
        if self.status == STATUS_FAIL:
            return 1
        if strict and self.status in (STATUS_WARN, STATUS_SKIP):
            return 1
        return 0

    def to_json(self) -> dict[str, Any]:
        return {
            "tool": "audit_dataset",
            "version": 1,
            "data_yaml": str(self.dataset.data_yaml) if self.dataset.data_yaml else None,
            "root": str(self.dataset.root),
            "classes": list(self.dataset.names),
            "summary": self.summary,
            "status": self.status,
            "counts": self.counts(),
            "checks": [c.to_json() for c in self.checks],
        }

    def render(self) -> str:
        """The human-readable report, in the order a reader should read it."""
        out: list[str] = []
        out.append("=" * 78)
        out.append("wildfire-watch dataset audit")
        out.append("=" * 78)
        source = self.dataset.data_yaml or self.dataset.root
        out.append(f"dataset : {source}")
        out.append(f"root    : {self.dataset.root}")
        out.append(f"classes : {', '.join(self.dataset.names) or '(none declared)'}")
        counts = self.summary.get("split_counts", {})
        split_desc = ", ".join(f"{k} {v}" for k, v in sorted(counts.items())) or "(none)"
        out.append(f"images  : {self.summary.get('n_images', 0)} ({split_desc})")
        out.append(f"boxes   : {self.summary.get('n_boxes', 0)}")
        for note in self.dataset.load_notes:
            out.append(f"note    : {note}")
        out.append("")

        for check in self.checks:
            out.append(f"[{check.status}] {check.name}")
            out.append(f"        {check.headline}")
            for line in check.lines:
                out.append(f"        {line}".rstrip())
            out.append("")

        c = self.counts()
        out.append("-" * 78)
        out.append(
            f"verdict: {self.status}  "
            f"({c[STATUS_FAIL]} failed, {c[STATUS_WARN]} warned, "
            f"{c[STATUS_SKIP]} skipped, {c[STATUS_PASS]} passed)"
        )
        if self.status == STATUS_FAIL:
            out.append(
                "A FAIL means every metric this dataset produces is unsafe to compare\n"
                "        against anything. Fix the split before training, not after."
            )
        elif self.status in (STATUS_WARN, STATUS_SKIP):
            out.append(
                "No hard failure. The warnings above change how the resulting numbers\n"
                "        should be read; carry them into the model card."
            )
        else:
            out.append(
                "The dataset is free of the defects this tool knows how to look for.\n"
                "        That is a statement about the dataset, not about field recall --\n"
                "        measure that with tools/evaluate.py on held-out real footage."
            )
        out.append("-" * 78)
        return "\n".join(out)


def size_bucket(area_fraction: float) -> str:
    """Bucket a box by its area as a fraction of the frame.

    Args:
        area_fraction: Box area divided by frame area, 0..1.

    Returns:
        One of ``tiny``, ``small``, ``medium``, ``large``. Shared with
        ``tools/evaluate.py`` so recall-by-size and the audit's box-size
        histogram use identical boundaries.
    """
    for name, lo, hi in SIZE_BINS:
        if lo <= area_fraction < hi:
            return name
    return SIZE_BINS[-1][0]


# ------------------------------------------------------------ image geometry


def image_size_from_header(path: Path) -> tuple[int, int] | None:
    """Read ``(width, height)`` from an image header without decoding pixels.

    Pillow is optional in this environment, and the box-size distribution is
    one of the checks that must always run -- a dataset of tiny distant fires
    labelled in a dataset of 4K frames is a different animal from the same
    boxes in 640x360. So the common container headers are parsed directly.

    Args:
        path: Image file to inspect.

    Returns:
        ``(width, height)`` in pixels, or ``None`` if the format is not one of
        PNG/JPEG/GIF/BMP/WEBP or the header is truncated (TIFF falls through to
        the Pillow path).
    """
    try:
        with path.open("rb") as fh:
            head = fh.read(32)
            if len(head) < 16:
                return None
            if head.startswith(b"\x89PNG\r\n\x1a\n"):
                # IHDR is mandated to be the first chunk, at a fixed offset.
                w, h = struct.unpack(">II", head[16:24])
                return int(w), int(h)
            if head[:2] == b"BM":
                fh.seek(18)
                w, h = struct.unpack("<ii", fh.read(8))
                return int(w), abs(int(h))  # negative height = top-down bitmap
            if head[:6] in (b"GIF87a", b"GIF89a"):
                w, h = struct.unpack("<HH", head[6:10])
                return int(w), int(h)
            if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
                return _webp_size(fh, head)
            if head[:2] == b"\xff\xd8":
                return _jpeg_size(fh)
    except OSError:
        return None
    return None


def _webp_size(fh: Any, head: bytes) -> tuple[int, int] | None:
    """Size from the three WebP chunk flavours (lossy, lossless, extended)."""
    chunk = head[12:16]
    fh.seek(20)
    if chunk == b"VP8 ":
        body = fh.read(10)
        if len(body) < 10 or body[3:6] != b"\x9d\x01\x2a":
            return None
        w, h = struct.unpack("<HH", body[6:10])
        return int(w & 0x3FFF), int(h & 0x3FFF)
    if chunk == b"VP8L":
        body = fh.read(5)
        if len(body) < 5 or body[0] != 0x2F:
            return None
        bits = int.from_bytes(body[1:5], "little")
        return int(bits & 0x3FFF) + 1, int((bits >> 14) & 0x3FFF) + 1
    if chunk == b"VP8X":
        body = fh.read(10)
        if len(body) < 10:
            return None
        w = int.from_bytes(body[4:7], "little") + 1
        h = int.from_bytes(body[7:10], "little") + 1
        return w, h
    return None


def _jpeg_size(fh: Any) -> tuple[int, int] | None:
    """Walk JPEG markers to the frame header. Handles progressive and EXIF."""
    fh.seek(2)
    while True:
        byte = fh.read(1)
        if not byte:
            return None
        if byte != b"\xff":
            continue
        marker = fh.read(1)
        while marker == b"\xff":  # fill bytes are legal between markers
            marker = fh.read(1)
        if not marker:
            return None
        code = marker[0]
        if code in (0xD8, 0xD9) or 0xD0 <= code <= 0xD7:
            continue
        raw = fh.read(2)
        if len(raw) < 2:
            return None
        length = struct.unpack(">H", raw)[0]
        # SOF0..SOF15 carry the dimensions; C4/C8/CC are Huffman/arithmetic
        # tables that share the numeric range and must be stepped over.
        if 0xC0 <= code <= 0xCF and code not in (0xC4, 0xC8, 0xCC):
            body = fh.read(5)
            if len(body) < 5:
                return None
            h, w = struct.unpack(">HH", body[1:5])
            return int(w), int(h)
        if code == 0xDA:  # start of scan: no header past here
            return None
        fh.seek(length - 2, os.SEEK_CUR)


def _size_with_pillow(path: Path) -> tuple[int, int] | None:
    """Fall back to Pillow for formats the header parser does not cover."""
    try:
        from PIL import Image  # noqa: PLC0415 -- optional dependency, lazy by design.
    except ImportError:
        return None
    try:
        with Image.open(path) as img:
            return int(img.width), int(img.height)
    except Exception:
        return None


# ---------------------------------------------------------------- dataset io


def _read_yaml(path: Path) -> dict[str, Any]:
    import yaml  # noqa: PLC0415 -- PyYAML is only needed when a data.yaml exists.

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: top level of data.yaml must be a mapping")
    return raw


def _normalise_names(raw: Any) -> tuple[str, ...]:
    """Accept both the list form and the ``{0: fire, 1: smoke}`` mapping form."""
    if raw is None:
        return ()
    if isinstance(raw, dict):
        try:
            ordered = sorted(raw.items(), key=lambda kv: int(kv[0]))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"class names mapping has non-integer keys: {raw!r}") from exc
        return tuple(str(v) for _, v in ordered)
    if isinstance(raw, (list, tuple)):
        return tuple(str(v) for v in raw)
    raise ValueError(f"class names must be a list or a mapping, got {type(raw).__name__}")


def _resolve_split_paths(entry: Any, root: Path, yaml_dir: Path) -> list[Path]:
    """Resolve a data.yaml split entry against the two roots ultralytics allows."""
    if entry is None:
        return []
    entries = entry if isinstance(entry, (list, tuple)) else [entry]
    out: list[Path] = []
    for item in entries:
        text = str(item)
        for base in (root, yaml_dir, Path.cwd()):
            candidate = (base / text).resolve() if not Path(text).is_absolute() else Path(text)
            if candidate.exists():
                out.append(candidate)
                break
        else:
            out.append((root / text).resolve())
    return out


def iter_image_files(target: Path) -> Iterator[Path]:
    """Yield image paths from a directory tree or from a .txt manifest."""
    if target.is_dir():
        for path in sorted(target.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                yield path
        return
    if target.is_file() and target.suffix.lower() == ".txt":
        base = target.parent
        for line in target.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            candidate = Path(line)
            if not candidate.is_absolute():
                candidate = (base / candidate).resolve()
            yield candidate


def label_path_for(image: Path) -> Path:
    """Map an image path to its YOLO label path.

    Follows the ultralytics convention: the last ``images`` path component
    becomes ``labels`` and the extension becomes ``.txt``. Datasets that keep
    labels beside the images are handled by the sibling-directory fallback.
    """
    parts = list(image.parts)
    for index in range(len(parts) - 1, -1, -1):
        if parts[index] == "images":
            parts[index] = "labels"
            return Path(*parts).with_suffix(".txt")
    sibling = image.parent.parent / "labels" / (image.stem + ".txt")
    if sibling.exists():
        return sibling
    return image.with_suffix(".txt")


def parse_label_file(path: Path) -> tuple[list[Box], list[str]]:
    """Parse one YOLO label file.

    Accepts both the 5-token detection form and the polygon segmentation form,
    converting polygons to their bounding box so a segmentation dataset can be
    audited as a detection dataset.

    Args:
        path: The ``.txt`` label file.

    Returns:
        ``(boxes, errors)``. ``errors`` holds one formatted message per
        malformed line; a file with errors still contributes the lines that
        did parse, because dropping the whole file would hide the good boxes.
    """
    boxes: list[Box] = []
    errors: list[str] = []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return [], [f"unreadable: {exc}"]

    for lineno, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        tokens = line.split()
        try:
            cls_id = int(float(tokens[0]))
        except (ValueError, IndexError):
            errors.append(f"line {lineno}: class index is not a number: {line[:60]!r}")
            continue
        try:
            values = [float(t) for t in tokens[1:]]
        except ValueError:
            errors.append(f"line {lineno}: non-numeric coordinate: {line[:60]!r}")
            continue

        if len(values) == 4:
            cx, cy, w, h = values
        elif len(values) >= 6 and len(values) % 2 == 0:
            xs, ys = values[0::2], values[1::2]
            x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
            cx, cy, w, h = (x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1
        else:
            errors.append(f"line {lineno}: expected 4 coords or a polygon, got {len(values)}")
            continue

        if not all(math.isfinite(v) for v in (cx, cy, w, h)):
            errors.append(f"line {lineno}: non-finite coordinate")
            continue
        if w <= 0.0 or h <= 0.0:
            errors.append(f"line {lineno}: degenerate box w={w:g} h={h:g}")
            continue
        # A tolerance rather than an exact bound: many exporters emit 1.0000001.
        if not (-0.01 <= cx <= 1.01 and -0.01 <= cy <= 1.01 and w <= 1.02 and h <= 1.02):
            errors.append(f"line {lineno}: coordinates outside 0..1 (cx={cx:g} cy={cy:g} w={w:g} h={h:g})")
            continue
        boxes.append(Box(cls_id=cls_id, cx=cx, cy=cy, w=min(w, 1.0), h=min(h, 1.0)))
    return boxes, errors


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as fh:
            for block in iter(lambda: fh.read(chunk), b""):
                digest.update(block)
    except OSError:
        return ""
    return digest.hexdigest()


def load_dataset(
    data_yaml: Path | None = None,
    dataset_dir: Path | None = None,
    splits: Sequence[str] = ("train", "val"),
    *,
    hash_content: bool = True,
    read_sizes: bool = True,
) -> Dataset:
    """Load a YOLO dataset from ``data.yaml`` or from a directory layout.

    Args:
        data_yaml: Path to ``data.yaml``. Takes precedence over ``dataset_dir``.
        dataset_dir: A directory holding ``data.yaml``, or one holding
            ``<split>/images`` subdirectories.
        splits: Which splits to load, in report order. ``val`` is aliased to
            ``valid``/``validation`` because all three appear in the wild.
        hash_content: Compute the SHA-256 of every image. Needed for the
            exact-duplicate check; the only expensive part of loading.
        read_sizes: Read image dimensions from headers (cheap) so box areas can
            be expressed in pixels as well as fractions.

    Returns:
        A populated :class:`Dataset`.

    Raises:
        FileNotFoundError: Neither argument points at anything readable.
        ValueError: ``data.yaml`` exists but cannot describe a dataset.
    """
    notes: list[str] = []
    names: tuple[str, ...] = ()
    split_targets: dict[str, list[Path]] = {}

    if data_yaml is None and dataset_dir is not None:
        candidate = dataset_dir / "data.yaml"
        if candidate.is_file():
            data_yaml = candidate

    if data_yaml is not None:
        data_yaml = data_yaml.resolve()
        if not data_yaml.is_file():
            raise FileNotFoundError(f"data.yaml not found: {data_yaml}")
        raw = _read_yaml(data_yaml)
        yaml_dir = data_yaml.parent
        root_entry = raw.get("path")
        root = (yaml_dir / str(root_entry)).resolve() if root_entry else yaml_dir
        names = _normalise_names(raw.get("names"))
        declared_nc = raw.get("nc")
        if declared_nc is not None and names and int(declared_nc) != len(names):
            notes.append(f"data.yaml declares nc={declared_nc} but lists {len(names)} names")
        for split in splits:
            aliases = (split, "valid", "validation") if split == "val" else (split,)
            for alias in aliases:
                if raw.get(alias) is not None:
                    split_targets[split] = _resolve_split_paths(raw[alias], root, yaml_dir)
                    break
    elif dataset_dir is not None:
        root = dataset_dir.resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"dataset directory not found: {root}")
        notes.append("no data.yaml; splits inferred from the directory layout")
        for split in splits:
            for alias in ((split, "valid", "validation") if split == "val" else (split,)):
                for candidate in (root / alias / "images", root / "images" / alias, root / alias):
                    if candidate.is_dir():
                        split_targets[split] = [candidate]
                        break
                if split in split_targets:
                    break
    else:
        raise FileNotFoundError("pass --data pointing at a data.yaml, or --dataset-dir")

    samples: list[Sample] = []
    seen: set[Path] = set()
    for split, targets in split_targets.items():
        found = 0
        for target in targets:
            if not target.exists():
                notes.append(f"split {split!r}: {target} does not exist")
                continue
            for image in iter_image_files(target):
                resolved = image.resolve()
                if resolved in seen:
                    # The same file listed in two splits is itself a finding,
                    # recorded by the duplicate check via content hash; here we
                    # just avoid auditing it twice.
                    notes.append(f"{resolved} appears in more than one split listing")
                    continue
                seen.add(resolved)
                samples.append(_build_sample(resolved, split, hash_content, read_sizes))
                found += 1
        if found == 0:
            notes.append(f"split {split!r} resolved to no images")

    for split in splits:
        if split not in split_targets:
            notes.append(f"split {split!r} is not declared")

    return Dataset(root=root, data_yaml=data_yaml, names=names, samples=samples, load_notes=notes)


def _build_sample(image: Path, split: str, hash_content: bool, read_sizes: bool) -> Sample:
    label = label_path_for(image)
    boxes: tuple[Box, ...] = ()
    errors: tuple[str, ...] = ()
    if label.is_file():
        parsed, errs = parse_label_file(label)
        boxes, errors = tuple(parsed), tuple(errs)
    else:
        label = None  # type: ignore[assignment]

    width = height = 0
    source = "unknown"
    corrupt = False
    if read_sizes:
        size = image_size_from_header(image)
        if size is not None:
            width, height, source = size[0], size[1], "header"
        else:
            size = _size_with_pillow(image)
            if size is not None:
                width, height, source = size[0], size[1], "pillow"
            else:
                corrupt = True
    return Sample(
        image=image,
        split=split,
        label=label,
        boxes=boxes,
        label_errors=errors,
        width=width,
        height=height,
        size_source=source,
        corrupt=corrupt,
        content_sha=_sha256(image) if hash_content else "",
    )


# ------------------------------------------------------------- perceptual hash


def _popcount(values: np.ndarray) -> np.ndarray:
    """Vectorised bit count over uint64, with a fallback for older numpy."""
    counter = getattr(np, "bitwise_count", None)
    if counter is not None:
        return counter(values).astype(np.int16)
    counts = np.zeros(values.shape, dtype=np.int16)
    work = values.copy()
    table = np.array([bin(i).count("1") for i in range(256)], dtype=np.int16)
    for _ in range(8):
        counts += table[(work & np.uint64(0xFF)).astype(np.uint8)]
        work >>= np.uint64(8)
    return counts


def _dhash(path: Path) -> int | None:
    """64-bit difference hash: 9x8 greyscale, compare each pixel to its right.

    Difference hashing rather than average hashing because it is stable under
    the exposure and white-balance drift between consecutive video frames,
    which is exactly the pair we need to catch across a split boundary.
    """
    try:
        from PIL import Image  # noqa: PLC0415 -- optional, lazy by design.
    except ImportError:
        return None
    try:
        with Image.open(path) as img:
            small = img.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
            pixels = np.asarray(small, dtype=np.int16)
    except Exception:
        return None
    bits = (pixels[:, 1:] > pixels[:, :-1]).flatten()
    return int.from_bytes(np.packbits(bits).tobytes(), "big")


def _hash_backend() -> tuple[str, Any]:
    """Pick a hashing backend, preferring imagehash when it is installed."""
    try:
        import imagehash  # noqa: PLC0415 -- optional, lazy by design.
        from PIL import Image  # noqa: PLC0415

        def _hash(path: Path) -> int | None:
            try:
                with Image.open(path) as img:
                    return int(str(imagehash.dhash(img, hash_size=8)), 16)
            except Exception:
                return None

        return "imagehash", _hash
    except ImportError:
        pass
    try:
        import PIL  # noqa: F401, PLC0415
    except ImportError:
        return "unavailable", None
    return "pillow", _dhash


# ------------------------------------------------------------------- checks


def _family_of(stem: str) -> tuple[tuple[str, str] | None, int]:
    """Split a stem into (family key, sequence index).

    ``clip07_frame_0142`` -> ``(("clip07_frame_", ""), 142)``. The family is
    the part that stays constant across a run of frames, so two samples in the
    same family with adjacent indices are consecutive frames of one video.
    """
    match = _SEQUENCE_RE.match(stem)
    if match is None:
        return None, -1
    return (match.group("prefix").lower(), match.group("suffix").lower()), int(match.group("num"))


def _label_key(sample: Sample, dataset: Dataset) -> str:
    """A hashable description of what is labelled in an image."""
    ids = sorted(sample.class_ids)
    if not ids:
        return "<empty>"
    return "+".join(dataset.class_name(i) for i in ids)


def _check_structure(dataset: Dataset, opts: AuditOptions) -> CheckResult:
    """Pairing, readability and label syntax. Everything else assumes this."""
    samples = dataset.samples
    missing_labels = [s for s in samples if s.label is None]
    empty_labels = [s for s in samples if s.label is not None and not s.boxes and not s.label_errors]
    bad_labels = [s for s in samples if s.label_errors]
    corrupt = [s for s in samples if s.corrupt]
    unknown_size = [s for s in samples if not s.corrupt and s.width == 0]

    orphans: list[Path] = []
    for split_dir in {s.label.parent for s in samples if s.label is not None}:
        stems = {s.label.stem for s in samples if s.label is not None and s.label.parent == split_dir}
        for txt in sorted(split_dir.glob("*.txt")):
            if txt.stem not in stems and txt.name != "classes.txt":
                orphans.append(txt)

    lines: list[str] = []
    status = STATUS_PASS
    if not samples:
        return CheckResult(
            "structure",
            STATUS_FAIL,
            "no images were found in any declared split",
            ["Check data.yaml's 'path', 'train' and 'val' keys against the directory layout."],
            {"n_images": 0},
        )

    lines.append(f"{len(samples)} images paired against {len(samples) - len(missing_labels)} label files")
    if corrupt:
        status = STATUS_FAIL
        lines.append(f"{len(corrupt)} file(s) could not be parsed as an image:")
        lines.extend(f"  {s.image}" for s in corrupt[:10])
    if bad_labels:
        status = STATUS_FAIL
        n_errors = sum(len(s.label_errors) for s in bad_labels)
        lines.append(f"{n_errors} malformed label line(s) in {len(bad_labels)} file(s):")
        for s in bad_labels[:10]:
            lines.append(f"  {s.label}: {s.label_errors[0]}")
    if missing_labels:
        # Missing and empty are different: a missing file is usually a broken
        # export, an empty file is a deliberate background image.
        status = max(status, STATUS_WARN, key=lambda v: _STATUS_RANK[v])
        lines.append(f"{len(missing_labels)} image(s) have no label file at all:")
        lines.extend(f"  {s.image}" for s in missing_labels[:5])
    if orphans:
        status = max(status, STATUS_WARN, key=lambda v: _STATUS_RANK[v])
        lines.append(f"{len(orphans)} label file(s) have no matching image:")
        lines.extend(f"  {p}" for p in orphans[:5])
    if unknown_size:
        lines.append(
            f"{len(unknown_size)} image(s) have an unreadable size header "
            "(install Pillow for full coverage); their boxes are excluded from the pixel histogram"
        )
    fraction_empty = len(empty_labels) / len(samples)
    lines.append(
        f"{len(empty_labels)} image(s) carry an empty label file "
        f"({fraction_empty:.1%} background imagery)"
    )
    if not empty_labels:
        status = max(status, STATUS_WARN, key=lambda v: _STATUS_RANK[v])
        lines.append(
            "  No background images at all. Every training image contains a target, so the"
        )
        lines.append(
            "  model is never shown sunlit haze, dust or cloud and told they are not targets."
        )

    headline = {
        STATUS_PASS: "structure is sound",
        STATUS_WARN: "structure is usable, with caveats",
        STATUS_FAIL: "structural problems that will corrupt training",
    }[status]
    return CheckResult(
        "structure",
        status,
        headline,
        lines,
        {
            "n_images": len(samples),
            "missing_labels": len(missing_labels),
            "empty_labels": len(empty_labels),
            "malformed_label_files": len(bad_labels),
            "corrupt_images": len(corrupt),
            "orphan_labels": len(orphans),
        },
    )


def _check_wire_class_order(dataset: Dataset) -> CheckResult:
    """The dataset's class order must equal ``station.core.types.CLASSES``.

    A dataset that lists ``[smoke, fire]`` trains a model whose class 0 is
    smoke. Nothing downstream would notice: the boxes are the right shape, the
    confidences are plausible, and every label on every tablet is inverted.
    """
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    try:
        from station.core.types import CLASSES  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover - only when run outside the repo
        return CheckResult(
            "wire-class-order",
            STATUS_SKIP,
            "could not import station.core.types to compare class order",
            [f"{type(exc).__name__}: {exc}"],
        )

    declared = tuple(n.strip().lower() for n in dataset.names)
    expected = tuple(CLASSES)
    if not declared:
        # Distinct from a wrong order, and it needs a different message: there
        # is nothing to compare, usually because the audit was pointed at a
        # directory rather than a data.yaml.
        return CheckResult(
            "wire-class-order",
            STATUS_WARN,
            "the dataset declares no class names, so the class order is unverified",
            [
                f"The wire contract expects {list(expected)}, in that order.",
                "Add 'names: [fire, smoke]' to data.yaml and re-run. Until then the class",
                "indices in the label files mean whatever the person who wrote them intended,",
                "and a swap between fire and smoke would be invisible everywhere downstream.",
            ],
            {"dataset": [], "wire": list(expected)},
        )
    if declared == expected:
        return CheckResult(
            "wire-class-order",
            STATUS_PASS,
            f"class order matches the wire contract: {list(expected)}",
        )
    if sorted(declared) == sorted(expected):
        return CheckResult(
            "wire-class-order",
            STATUS_FAIL,
            f"class order is permuted: dataset {list(declared)} vs wire {list(expected)}",
            [
                "Training on this yields a model whose class indices are swapped relative to",
                "the wire protocol. Fire would be labelled smoke on every tablet, and no test",
                "downstream of the model can see it. Reorder 'names' in data.yaml and",
                "renumber the label files together -- reordering only one is worse than neither.",
            ],
            {"dataset": list(declared), "wire": list(expected)},
        )
    extra = [n for n in declared if n not in expected]
    absent = [n for n in expected if n not in declared]
    return CheckResult(
        "wire-class-order",
        STATUS_WARN,
        f"dataset classes {list(declared)} differ from the wire classes {list(expected)}",
        [
            line
            for line in (
                f"present here but not on the wire: {extra}" if extra else "",
                f"on the wire but absent here: {absent}" if absent else "",
                "station.inference.runner maps a few known aliases; anything else arrives on",
                "the tablet with a label it has no colour for.",
            )
            if line
        ],
        {"dataset": list(declared), "wire": list(expected)},
    )


def _check_duplicate_content(dataset: Dataset) -> CheckResult:
    """Byte-identical images. Across splits this is unambiguous leakage."""
    by_hash: dict[str, list[Sample]] = defaultdict(list)
    for s in dataset.samples:
        if s.content_sha:
            by_hash[s.content_sha].append(s)

    cross: list[list[Sample]] = []
    within = 0
    for group in by_hash.values():
        if len(group) < 2:
            continue
        if len({s.split for s in group}) > 1:
            cross.append(group)
        else:
            within += len(group) - 1

    lines: list[str] = []
    if cross:
        n_images = sum(len(g) for g in cross)
        lines.append(f"{len(cross)} image(s) appear byte-identically in more than one split:")
        for group in cross[:8]:
            paths = ", ".join(f"{s.split}:{s.image.name}" for s in group[:4])
            lines.append(f"  {paths}")
        lines.append("Every one of these is a validation image the model was trained on.")
        return CheckResult(
            "duplicates-across-splits",
            STATUS_FAIL,
            f"{n_images} images are exact duplicates spanning splits",
            lines,
            {"groups": len(cross), "images": n_images, "within_split_duplicates": within},
        )
    if within:
        return CheckResult(
            "duplicates-across-splits",
            STATUS_WARN,
            f"no cross-split duplicates, but {within} duplicate image(s) inside a split",
            [
                "The dataset is smaller than its file count suggests, and the duplicated",
                "scenes are over-weighted in the loss.",
            ],
            {"groups": 0, "images": 0, "within_split_duplicates": within},
        )
    return CheckResult(
        "duplicates-across-splits",
        STATUS_PASS,
        "no byte-identical images anywhere in the dataset",
        data={"groups": 0, "images": 0, "within_split_duplicates": 0},
    )


def _check_sequence_leak(dataset: Dataset, opts: AuditOptions) -> CheckResult:
    """Consecutive video frames split across train and val. The main trap.

    Two signals, both from filenames alone, so this runs with no image
    decoding at all:

    * a numbered family (``clip3_frame_0141``) whose indices straddle the
      split with a gap of at most ``sequence_gap`` -- these are the same
      photograph;
    * a family that straddles the split at all, which is weaker evidence but
      still means one video contributed to both sides.
    """
    families: dict[tuple[str, str], list[tuple[int, Sample]]] = defaultdict(list)
    unnumbered = 0
    for s in dataset.samples:
        key, index = _family_of(s.stem)
        if key is None:
            unnumbered += 1
            continue
        families[key].append((index, s))

    adjacent: list[tuple[Sample, Sample, int]] = []
    straddling: list[tuple[str, int, set[str]]] = []
    for key, members in families.items():
        splits = {s.split for _, s in members}
        if len(splits) < 2:
            continue
        straddling.append(("".join(key) or "<unnamed>", len(members), splits))
        members.sort(key=lambda item: item[0])
        for i in range(len(members) - 1):
            index_a, sample_a = members[i]
            for j in range(i + 1, len(members)):
                index_b, sample_b = members[j]
                if index_b - index_a > opts.sequence_gap:
                    break
                if sample_a.split != sample_b.split:
                    adjacent.append((sample_a, sample_b, index_b - index_a))

    data = {
        "families": len(families),
        "families_spanning_splits": len(straddling),
        "adjacent_pairs": len(adjacent),
        "unnumbered_images": unnumbered,
        "sequence_gap": opts.sequence_gap,
    }

    if adjacent:
        leaked = {s.image for pair in adjacent for s in pair[:2]}
        lines = [
            f"{len(adjacent)} pair(s) of near-consecutive frames sit on opposite sides of the split,",
            f"involving {len(leaked)} images. Examples:",
        ]
        for a, b, gap in adjacent[:8]:
            lines.append(f"  {a.split}:{a.image.name}  <->  {b.split}:{b.image.name}  (gap {gap})")
        lines += [
            "",
            "These are the same photograph with the noise moved. Validation on them measures",
            "memorisation, and every metric computed from this split is inflated by an unknown",
            "amount -- it cannot be corrected afterwards, only re-split.",
            "Fix: split by video/family, never by image. Assign whole families to one side.",
        ]
        return CheckResult(
            "split-leakage/sequence",
            STATUS_FAIL,
            f"{len(adjacent)} consecutive-frame pairs span the train/val boundary",
            lines,
            data,
        )

    if straddling:
        lines = [
            f"{len(straddling)} filename family/families contribute images to more than one split:",
        ]
        for name, count, splits in sorted(straddling, key=lambda t: -t[1])[:8]:
            lines.append(f"  {name!r}: {count} images across {sorted(splits)}")
        lines += [
            "No pair is within the adjacency window, so this is not proof of duplicated frames,",
            "but one video feeding both sides still shares lighting, terrain and camera.",
            "Prefer splitting by source video.",
        ]
        return CheckResult(
            "split-leakage/sequence",
            STATUS_WARN,
            f"{len(straddling)} filename families span the split",
            lines,
            data,
        )

    note = []
    if unnumbered:
        note.append(
            f"{unnumbered} filenames carry no sequence number, so they were checked by content "
            "and perceptual hash only"
        )
    return CheckResult(
        "split-leakage/sequence",
        STATUS_PASS,
        "no numbered frame family spans the train/val boundary",
        note,
        data,
    )


def _check_perceptual_leak(dataset: Dataset, opts: AuditOptions) -> CheckResult:
    """Near-duplicate images across splits, by perceptual hash.

    Catches what filenames cannot: re-encoded copies, frames renamed on
    import, and the same scene scraped twice from two sources.
    """
    backend, hasher = _hash_backend()
    if hasher is None:
        return CheckResult(
            "split-leakage/perceptual",
            STATUS_SKIP,
            "Pillow is not installed, so no pixels were compared",
            [
                "  pip install pillow        (optionally also: pip install imagehash)",
                "The structural checks above catch renamed-in-sequence frames; they cannot catch",
                "a re-encoded or renamed copy. Treat a clean structural report as partial until",
                "this check has run at least once.",
            ],
            {"backend": backend},
        )

    by_split: dict[str, list[Sample]] = defaultdict(list)
    for s in dataset.samples:
        by_split[s.split].append(s)

    hashed = 0
    for split, members in by_split.items():
        chosen = members if opts.hash_limit <= 0 else members[: opts.hash_limit]
        for s in chosen:
            s.phash = hasher(s.image)
            hashed += 1

    splits = sorted(by_split)
    matches: list[tuple[Sample, Sample, int]] = []
    for i, split_a in enumerate(splits):
        for split_b in splits[i + 1 :]:
            a = [s for s in by_split[split_a] if s.phash is not None]
            b = [s for s in by_split[split_b] if s.phash is not None]
            if not a or not b:
                continue
            matches.extend(_nearest_pairs(a, b, opts.phash_threshold))

    data = {
        "backend": backend,
        "hashed": hashed,
        "threshold_bits": opts.phash_threshold,
        "pairs": len(matches),
    }
    if matches:
        involved = {s.image for pair in matches for s in pair[:2]}
        lines = [
            f"{len(matches)} cross-split image pair(s) are within {opts.phash_threshold} bits of "
            f"64, involving {len(involved)} images.",
            "Examples (distance 0 means visually identical):",
        ]
        for a, b, dist in sorted(matches, key=lambda t: t[2])[:10]:
            lines.append(f"  d={dist:2d}  {a.split}:{a.image.name}  <->  {b.split}:{b.image.name}")
        lines += [
            "",
            "Validation images that the model has effectively already seen. Re-split by source",
            "video or by scene, then re-run this audit before trusting any metric.",
        ]
        return CheckResult(
            "split-leakage/perceptual",
            STATUS_FAIL,
            f"{len(matches)} near-duplicate pairs span the split ({backend} dhash)",
            lines,
            data,
        )
    return CheckResult(
        "split-leakage/perceptual",
        STATUS_PASS,
        f"{hashed} images hashed ({backend}); no cross-split pair within "
        f"{opts.phash_threshold} bits",
        data=data,
    )


def _nearest_pairs(
    left: Sequence[Sample], right: Sequence[Sample], threshold: int
) -> list[tuple[Sample, Sample, int]]:
    """Every pair across two sets whose dHash Hamming distance is <= threshold.

    Chunked so that a 20k x 5k comparison stays inside a few hundred MB. The
    cost is quadratic and unavoidable for an exhaustive answer; ``--hash-limit``
    exists for datasets where that becomes intolerable.
    """
    lhs = np.array([s.phash for s in left], dtype=np.uint64)
    rhs = np.array([s.phash for s in right], dtype=np.uint64)
    out: list[tuple[Sample, Sample, int]] = []
    chunk = max(1, min(len(lhs), 4_000_000 // max(1, len(rhs))))
    for start in range(0, len(lhs), chunk):
        block = lhs[start : start + chunk]
        dist = _popcount(block[:, None] ^ rhs[None, :])
        hits = np.argwhere(dist <= threshold)
        for row, col in hits:
            out.append((left[start + int(row)], right[int(col)], int(dist[row, col])))
    return out


def _check_name_predicts_class(dataset: Dataset, opts: AuditOptions) -> CheckResult:
    """Does the path predict the label without looking at the picture?

    This is the failure that produced mAP50 0.782 on a useless model. Two
    forms are checked: whole directories whose images all share one label set,
    and individual filename tokens that do the same.
    """
    labelled = [s for s in dataset.samples if s.label is not None]
    if not labelled:
        return CheckResult(
            "filename-predicts-class",
            STATUS_SKIP,
            "no label files to correlate against filenames",
        )

    label_of = {id(s): _label_key(s, dataset) for s in labelled}
    distinct_labels = set(label_of.values())
    if len(distinct_labels) < 2:
        return CheckResult(
            "filename-predicts-class",
            STATUS_SKIP,
            f"every image carries the same label set ({next(iter(distinct_labels))}); "
            "nothing to predict",
            [
                "A single-label dataset cannot exhibit this defect, but it also cannot teach a",
                "model to tell the classes apart. See the class-balance check.",
            ],
        )

    dir_counts: dict[str, Counter[str]] = defaultdict(Counter)
    token_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for s in labelled:
        key = label_of[id(s)]
        parent = s.image.parent.as_posix()
        dir_counts[parent][key] += 1
        tokens = set()
        for part in s.image.parent.parts[-2:] + (s.stem,):
            tokens.update(_TOKEN_RE.findall(part.lower()))
        for token in tokens - _STOP_TOKENS:
            token_counts[token][key] += 1

    def _pure(counters: dict[str, Counter[str]]) -> list[tuple[str, int, float, str]]:
        found = []
        for key, counter in counters.items():
            support = sum(counter.values())
            if support < opts.min_support:
                continue
            top_label, top_count = counter.most_common(1)[0]
            purity = top_count / support
            if purity >= opts.purity_threshold:
                found.append((key, support, purity, top_label))
        return sorted(found, key=lambda t: -t[1])

    pure_dirs = _pure(dir_counts)
    pure_tokens = _pure(token_counts)
    # One pure directory proves nothing (a dataset may have one folder per
    # class *and* be perfectly good). Two pure groups with *different* labels
    # is the shortcut: the path alone separates the classes.
    dir_labels = {label for _, _, _, label in pure_dirs}
    token_labels = {label for _, _, _, label in pure_tokens}

    mi = _normalised_mutual_information(
        [label_of[id(s)] for s in labelled],
        [frozenset(_TOKEN_RE.findall(s.stem.lower())) - _STOP_TOKENS for s in labelled],
    )

    lines: list[str] = []
    data = {
        "pure_directories": [
            {"path": p, "support": n, "purity": round(pu, 4), "label": lb}
            for p, n, pu, lb in pure_dirs[:20]
        ],
        "pure_tokens": [
            {"token": t, "support": n, "purity": round(pu, 4), "label": lb}
            for t, n, pu, lb in pure_tokens[:20]
        ],
        "normalised_mutual_information": round(mi, 4),
        "distinct_label_sets": len(distinct_labels),
    }

    status = STATUS_PASS
    if len(dir_labels) > 1:
        status = STATUS_FAIL
        lines.append(
            f"{len(pure_dirs)} directories are class-pure, covering {len(dir_labels)} "
            "different label sets:"
        )
        for path, support, purity, label in pure_dirs[:8]:
            lines.append(f"  {path}  -> {label} ({support} images, {purity:.0%} pure)")
        lines.append(
            "A model can reach a high score here by learning the folder, which it sees as a"
        )
        lines.append("background-statistics prior. Interleave the classes across directories.")
    if len(token_labels) > 1:
        status = STATUS_FAIL
        lines.append(
            f"{len(pure_tokens)} filename tokens are class-pure, covering "
            f"{len(token_labels)} different label sets:"
        )
        for token, support, purity, label in pure_tokens[:8]:
            lines.append(f"  {token!r} -> {label} ({support} images, {purity:.0%} pure)")
        lines.append(
            "Filename families that map one-to-one onto classes usually mean each class came"
        )
        lines.append(
            "from its own source videos, so class and scene are confounded and the model can"
        )
        lines.append("separate them on scene alone.")
    if status == STATUS_FAIL:
        lines.append(f"normalised mutual information between filename tokens and label: {mi:.2f}")
        lines.append(
            "(0 = the filename says nothing about the class, 1 = it says everything.)"
        )
        return CheckResult(
            "filename-predicts-class",
            STATUS_FAIL,
            "the file path predicts the class without reference to the image",
            lines,
            data,
        )

    if mi >= 0.5:
        return CheckResult(
            "filename-predicts-class",
            STATUS_WARN,
            f"filename tokens carry substantial class information (NMI {mi:.2f})",
            [
                "No single token or directory is fully class-pure, but names and labels are",
                "strongly associated. Check how the source videos were assigned to classes.",
            ],
            data,
        )
    return CheckResult(
        "filename-predicts-class",
        STATUS_PASS,
        f"no class-pure directory or token; filename/label NMI {mi:.2f}",
        data=data,
    )


def _normalised_mutual_information(labels: Sequence[str], token_sets: Sequence[frozenset[str]]) -> float:
    """I(token; label) / H(label), maximised over tokens.

    A per-token maximum rather than a joint measure over all tokens: the
    question is whether *some* namepart gives the class away, and a joint
    estimate over thousands of sparse tokens would be dominated by noise.
    """
    n = len(labels)
    if n == 0:
        return 0.0
    label_counts = Counter(labels)
    h_label = -sum((c / n) * math.log2(c / n) for c in label_counts.values())
    if h_label <= 0.0:
        return 0.0

    joint: dict[str, Counter[str]] = defaultdict(Counter)
    token_totals: Counter[str] = Counter()
    for label, tokens in zip(labels, token_sets):
        for token in tokens:
            joint[token][label] += 1
            token_totals[token] += 1

    best = 0.0
    for token, counter in joint.items():
        present = token_totals[token]
        absent = n - present
        if present < 2 or absent < 2:
            continue
        mi = 0.0
        for label, total in label_counts.items():
            for count, marginal in ((counter.get(label, 0), present), (total - counter.get(label, 0), absent)):
                if count == 0:
                    continue
                mi += (count / n) * math.log2((count / n) / ((marginal / n) * (total / n)))
        best = max(best, mi / h_label)
    return best


def _check_cooccurrence(dataset: Dataset) -> CheckResult:
    """Do fire and smoke ever appear in the same image?

    If they never do, the two classes are perfectly separated by scene, and a
    model can score well by classifying the scene and putting a box roughly
    where the interesting pixels are. Real wildfire imagery has both together
    constantly -- a flame front almost always has a plume above it -- so an
    absolute zero here is evidence about the *dataset*, not about fire.
    """
    labelled = [s for s in dataset.samples if s.boxes]
    if not labelled:
        return CheckResult("class-cooccurrence", STATUS_SKIP, "no labelled boxes to compare")

    present_classes = sorted({b.cls_id for s in labelled for b in s.boxes})
    if len(present_classes) < 2:
        return CheckResult(
            "class-cooccurrence",
            STATUS_WARN,
            f"only one class ({dataset.class_name(present_classes[0])}) appears in any label",
            [
                "Co-occurrence cannot be assessed, and the model cannot learn to distinguish",
                "the classes the wire protocol defines.",
            ],
            {"classes_present": present_classes},
        )

    matrix: Counter[tuple[int, int]] = Counter()
    multi = 0
    for s in labelled:
        ids = sorted(s.class_ids)
        if len(ids) > 1:
            multi += 1
        for i, a in enumerate(ids):
            for b in ids[i + 1 :]:
                matrix[(a, b)] += 1

    rate = multi / len(labelled)
    pairs = {
        f"{dataset.class_name(a)}+{dataset.class_name(b)}": n for (a, b), n in sorted(matrix.items())
    }
    data = {"images_with_multiple_classes": multi, "rate": round(rate, 5), "pairs": pairs}

    if multi == 0:
        return CheckResult(
            "class-cooccurrence",
            STATUS_FAIL,
            "no image contains more than one class",
            [
                f"{len(labelled)} labelled images, and not one has both "
                f"{dataset.class_name(present_classes[0])} and "
                f"{dataset.class_name(present_classes[1])} in it.",
                "In aerial wildfire footage a flame front and its plume are almost always in",
                "frame together, so this describes how the dataset was assembled rather than",
                "what fire looks like. Each class comes from its own images, which means the",
                "model can separate them on scene statistics and never learn the objects.",
                "Fix: label both classes on every image where both are visible, and add",
                "images that contain both.",
            ],
            data,
        )
    if rate < 0.01:
        return CheckResult(
            "class-cooccurrence",
            STATUS_WARN,
            f"only {multi} image(s) ({rate:.2%}) contain more than one class",
            [
                "Almost every image is single-class, so scene and class are nearly confounded.",
                f"co-occurrence counts: {pairs}",
            ],
            data,
        )
    return CheckResult(
        "class-cooccurrence",
        STATUS_PASS,
        f"{multi} images ({rate:.1%}) contain more than one class",
        [f"co-occurrence counts: {pairs}"],
        data,
    )


def _check_class_balance(dataset: Dataset, opts: AuditOptions) -> CheckResult:
    """Per-split, per-class box and image counts."""
    splits = dataset.splits
    per_split: dict[str, dict[str, dict[str, int]]] = {}
    for split, members in sorted(splits.items()):
        boxes: Counter[str] = Counter()
        images: Counter[str] = Counter()
        for s in members:
            for b in s.boxes:
                boxes[dataset.class_name(b.cls_id)] += 1
            for cls_id in s.class_ids:
                images[dataset.class_name(cls_id)] += 1
        per_split[split] = {"boxes": dict(boxes), "images": dict(images)}

    lines: list[str] = []
    for split, counts in per_split.items():
        total = sum(counts["boxes"].values())
        detail = ", ".join(f"{k}={v}" for k, v in sorted(counts["boxes"].items())) or "none"
        lines.append(f"{split:<6} {len(splits[split]):>6} images, {total:>6} boxes  ({detail})")

    status = STATUS_PASS
    all_classes = {c for counts in per_split.values() for c in counts["boxes"]}
    val_classes = set(per_split.get("val", {}).get("boxes", {}))
    absent_from_val = sorted(all_classes - val_classes)
    if "val" not in per_split or not per_split["val"]["boxes"]:
        status = STATUS_FAIL
        lines.append("The validation split has no labelled boxes, so recall cannot be measured.")
    elif absent_from_val:
        status = STATUS_FAIL
        lines.append(
            f"class(es) {absent_from_val} have no boxes in val, so their recall is unmeasurable"
        )

    totals: Counter[str] = Counter()
    for counts in per_split.values():
        totals.update(counts["boxes"])
    if len(totals) > 1:
        most, least = max(totals.values()), min(totals.values())
        ratio = most / max(1, least)
        lines.append(f"overall class ratio {ratio:.1f}:1 ({dict(sorted(totals.items()))})")
        if ratio > opts.imbalance_ratio:
            status = max(status, STATUS_WARN, key=lambda v: _STATUS_RANK[v])
            lines.append(
                f"Imbalance above {opts.imbalance_ratio:g}:1. The rare class will be under-"
                "detected, and its recall figure will be noisy on a small val count."
            )

    headline = (
        "class balance is workable" if status == STATUS_PASS else "class balance limits what can be measured"
    )
    return CheckResult("class-balance", status, headline, lines, {"per_split": per_split})


def _check_image_sizes(dataset: Dataset) -> CheckResult:
    """Resolution distribution -- context for the box-size histogram."""
    sizes = Counter((s.width, s.height) for s in dataset.samples if s.width and s.height)
    if not sizes:
        return CheckResult(
            "image-sizes",
            STATUS_WARN,
            "no image dimensions could be read",
            ["Install Pillow to cover formats without a parseable header (TIFF, exotic WebP)."],
        )
    lines = [f"{len(sizes)} distinct resolution(s) across {sum(sizes.values())} images"]
    for (w, h), count in sizes.most_common(6):
        lines.append(f"  {w}x{h}: {count}")
    status = STATUS_PASS
    if len(sizes) == 1 and sum(sizes.values()) > 50:
        status = STATUS_WARN
        lines.append(
            "Every image is the same size. Often that means a single capture chain, so the"
        )
        lines.append(
            "model may not survive a different camera or a different downlink resolution."
        )
    areas = [w * h for (w, h), n in sizes.items() for _ in range(n)]
    lines.append(
        f"median resolution area {int(statistics.median(areas)):,} px "
        f"(min {min(areas):,}, max {max(areas):,})"
    )
    return CheckResult(
        "image-sizes",
        status,
        f"{len(sizes)} distinct resolution(s)",
        lines,
        {"resolutions": {f"{w}x{h}": n for (w, h), n in sizes.most_common(20)}},
    )


def _check_box_sizes(dataset: Dataset, opts: AuditOptions) -> CheckResult:
    """Box-size distribution, with the tiny bucket treated as load-bearing.

    A distant plume on the horizon is the detection that buys the most time,
    and it is the one a model trained only on close-up flame will miss. If the
    dataset has no tiny boxes, the model cannot learn them and
    ``tools/evaluate.py`` cannot measure them.
    """
    boxes = [(s, b) for s in dataset.samples for b in s.boxes]
    if not boxes:
        return CheckResult("box-sizes", STATUS_SKIP, "no boxes to measure")

    buckets: Counter[str] = Counter()
    per_class: dict[str, Counter[str]] = defaultdict(Counter)
    whole_frame: list[Sample] = []
    pixel_areas: list[float] = []
    for sample, box in boxes:
        bucket = size_bucket(box.area)
        buckets[bucket] += 1
        per_class[dataset.class_name(box.cls_id)][bucket] += 1
        if box.area >= opts.whole_frame_area:
            whole_frame.append(sample)
        if sample.width and sample.height:
            pixel_areas.append(box.area * sample.width * sample.height)

    total = len(boxes)
    lines = [f"{total} boxes by area as a fraction of frame:"]
    for name, lo, hi in SIZE_BINS:
        count = buckets.get(name, 0)
        lines.append(f"  {name:<7} [{lo:.3f}, {hi:.3f})  {count:>7}  {count / total:6.1%}")
    if pixel_areas:
        median_side = math.sqrt(statistics.median(pixel_areas))
        lines.append(
            f"median box is about {median_side:.0f}x{median_side:.0f} px at native resolution"
        )
    for cls_name, counter in sorted(per_class.items()):
        spread = ", ".join(f"{k}={counter.get(k, 0)}" for k, _, _ in SIZE_BINS)
        lines.append(f"  {cls_name}: {spread}")

    status = STATUS_PASS
    tiny_fraction = buckets.get("tiny", 0) / total
    if tiny_fraction < opts.min_tiny_fraction:
        status = STATUS_WARN
        lines.append(
            f"Only {tiny_fraction:.2%} of boxes are in the tiny bucket "
            f"(threshold {opts.min_tiny_fraction:.0%})."
        )
        lines.append(
            "Distant, early, small targets are both the highest-value detection and the one"
        )
        lines.append(
            "most often missed. A dataset without them trains a model that only fires once"
        )
        lines.append("the fire is already obvious, and hides that fact from the metrics.")
    if whole_frame:
        status = max(status, STATUS_WARN, key=lambda v: _STATUS_RANK[v])
        lines.append(
            f"{len(whole_frame)} box(es) cover more than {opts.whole_frame_area:.0%} of the frame:"
        )
        lines.extend(f"  {s.image.name}" for s in whole_frame[:5])
        lines.append(
            "A whole-frame box is an image-level tag, not a localisation. It teaches the model"
        )
        lines.append("to answer 'is this a fire picture', which is not the question being asked.")

    return CheckResult(
        "box-sizes",
        status,
        f"{total} boxes; {tiny_fraction:.1%} tiny",
        lines,
        {
            "buckets": dict(buckets),
            "per_class": {k: dict(v) for k, v in per_class.items()},
            "whole_frame_boxes": len(whole_frame),
        },
    )


def run_audit(dataset: Dataset, opts: AuditOptions | None = None) -> AuditReport:
    """Run every check against a loaded dataset.

    Args:
        dataset: The dataset to audit.
        opts: Thresholds; defaults are the ones documented on
            :class:`AuditOptions`.

    Returns:
        An :class:`AuditReport`. Its :meth:`AuditReport.exit_code` is what a CI
        job should propagate.
    """
    opts = opts or AuditOptions()
    checks = [
        _check_structure(dataset, opts),
        _check_wire_class_order(dataset),
        _check_duplicate_content(dataset),
        _check_sequence_leak(dataset, opts),
        _check_perceptual_leak(dataset, opts),
        _check_name_predicts_class(dataset, opts),
        _check_cooccurrence(dataset),
        _check_class_balance(dataset, opts),
        _check_image_sizes(dataset),
        _check_box_sizes(dataset, opts),
    ]
    split_counts = {split: len(members) for split, members in sorted(dataset.splits.items())}
    summary = {
        "n_images": len(dataset.samples),
        "n_boxes": sum(len(s.boxes) for s in dataset.samples),
        "split_counts": split_counts,
    }
    return AuditReport(dataset=dataset, checks=checks, summary=summary)


# ---------------------------------------------------------------------- cli


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        prog="audit_dataset.py",
        description=(
            "Audit a YOLO-format fire/smoke dataset for train/val leakage, filename shortcuts "
            "and label defects. Exits 1 on a hard failure so it can gate training in CI."
        ),
        epilog=(
            "A clean report means the dataset is free of the defects this tool knows about. "
            "It is not a statement about how the trained model will behave on real footage -- "
            "that is measured by tools/evaluate.py, weighted toward false negatives."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    source = parser.add_argument_group("dataset")
    source.add_argument("--data", metavar="DATA_YAML", type=Path, help="Path to the dataset's data.yaml.")
    source.add_argument(
        "--dataset-dir",
        metavar="DIR",
        type=Path,
        help="Dataset root, used when there is no data.yaml (expects <split>/images).",
    )
    source.add_argument(
        "--splits",
        default="train,val",
        help="Comma-separated splits to audit, in report order (default: train,val).",
    )

    thresholds = parser.add_argument_group("thresholds")
    thresholds.add_argument(
        "--sequence-gap",
        type=int,
        default=_DEFAULTS.sequence_gap,
        help="Frames apart, inside one filename family, still counted as the same shot (default: 3).",
    )
    thresholds.add_argument(
        "--phash-threshold",
        type=int,
        default=_DEFAULTS.phash_threshold,
        help="Hamming distance out of 64 below which two images are near-duplicates (default: 6).",
    )
    thresholds.add_argument(
        "--min-support",
        type=int,
        default=_DEFAULTS.min_support,
        help="Images a filename token or directory needs before its class purity counts (default: 5).",
    )
    thresholds.add_argument(
        "--purity-threshold",
        type=float,
        default=_DEFAULTS.purity_threshold,
        help="Class purity at which a token is treated as predicting the class (default: 0.98).",
    )
    thresholds.add_argument(
        "--imbalance-ratio",
        type=float,
        default=_DEFAULTS.imbalance_ratio,
        help="Warn above this most-common-to-rarest box ratio (default: 10).",
    )
    thresholds.add_argument(
        "--min-tiny-fraction",
        type=float,
        default=_DEFAULTS.min_tiny_fraction,
        help="Warn when fewer than this fraction of boxes are tiny (default: 0.02).",
    )
    thresholds.add_argument(
        "--hash-limit",
        type=int,
        default=_DEFAULTS.hash_limit,
        help="Cap images perceptually hashed per split; 0 hashes everything (default: 0).",
    )

    output = parser.add_argument_group("output")
    output.add_argument("--json", metavar="FILE", type=Path, help="Also write the full report as JSON.")
    output.add_argument("--quiet", action="store_true", help="Suppress the text report; set the exit code only.")
    output.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero on warnings and skipped checks as well as failures.",
    )
    output.add_argument(
        "--no-content-hash",
        action="store_true",
        help="Skip SHA-256 of every image (faster; disables the exact-duplicate check).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point.

    Returns:
        0 when the dataset may be trained on, 1 when it must not be, 2 when the
        dataset could not be read at all.
    """
    args = build_parser().parse_args(argv)
    if args.data is None and args.dataset_dir is None:
        build_parser().error("one of --data or --dataset-dir is required")

    opts = AuditOptions(
        sequence_gap=args.sequence_gap,
        phash_threshold=args.phash_threshold,
        min_support=args.min_support,
        purity_threshold=args.purity_threshold,
        imbalance_ratio=args.imbalance_ratio,
        min_tiny_fraction=args.min_tiny_fraction,
        hash_limit=args.hash_limit,
    )
    splits = tuple(s.strip() for s in args.splits.split(",") if s.strip())

    try:
        dataset = load_dataset(
            data_yaml=args.data,
            dataset_dir=args.dataset_dir,
            splits=splits,
            hash_content=not args.no_content_hash,
        )
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"audit_dataset: {exc}", file=sys.stderr)
        return 2

    report = run_audit(dataset, opts)
    if not args.quiet:
        print(report.render())
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report.to_json(), indent=2), encoding="utf-8")
        if not args.quiet:
            print(f"\nJSON report written to {args.json}")
    return report.exit_code(strict=args.strict)


if __name__ == "__main__":
    raise SystemExit(main())
