#!/usr/bin/env python3
"""Merge the public fire/smoke datasets into one YOLO dataset, split by source video.

This script is the place where the leakage trap is *prevented*.
``tools/audit_dataset.py`` exists to catch it afterwards, and it is a
last line of defence, not the plan. The trap, restated:

    A fire dataset built from video contains runs of consecutive frames. Frame
    411 and frame 412 are the same photograph with the sensor noise moved.
    Split those two frames across train and val and validation stops measuring
    generalisation and starts measuring memorisation. Every number rises. The
    rise is indistinguishable from progress, and no downstream test can see it.

So this script never splits images. It splits **groups** -- a source video, a
burn sequence, a directory of frames from one flight -- and assigns each whole
group to exactly one split. There is deliberately no code path that shuffles
images and cuts the list; ``--help`` offers no such flag, and
:func:`_verify_no_group_straddles_split` re-checks the result before anything
is written, because a bug in this file would be invisible in the output.

Two further design choices that look odd until you know why:

*   **Output filenames carry no source or class token.** Every image is written
    as ``wf_<group>_<index>.<ext>``. The audit derives a "filename family" from
    the constant prefix before the trailing number, so this naming makes the
    family the auditor sees *identical* to the group this script split by -- the
    two tools then agree by construction. It also means no filename token can
    predict a label, which is the other half of the shortcut the audit hunts
    for. Provenance is not lost: every output image is recorded in
    ``provenance.jsonl`` with its original path, source dataset and group key.
*   **Negative (background) images get an empty ``.txt``, never a missing one.**
    A missing label file is a broken export; an empty one is a deliberate "there
    is nothing to box here". The two are treated differently downstream and
    conflating them silently discards the most valuable images in the merge.

Viewpoint weighting: ground-level datasets (D-Fire, Corsican) are the wrong
camera angle for a drone and will happily dominate a merge by sheer count. Each
source therefore carries a ``weight`` (fraction of its groups kept) and an
optional separate ``negative_weight``, so ground-level *positives* can be
thinned while their negatives -- which are viewpoint-agnostic confusers -- are
kept in full.

Usage::

    python3 training/prepare_datasets.py --config training/dataset_config.yaml
    python3 training/prepare_datasets.py --config ... --dry-run
    python3 training/prepare_datasets.py --config ... --only fasdd_uav --audit

Exit codes: 0 success, 1 refused (a configuration or safety guard fired),
2 nothing usable was found.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

__all__ = [
    "MANIFEST_VERSION",
    "IMAGE_EXTENSIONS",
    "PrepareError",
    "LeakageError",
    "Box",
    "Sample",
    "SourceSpec",
    "OutputSpec",
    "PrepareConfig",
    "PrepareReport",
    "load_prepare_config",
    "canonical_classes",
    "scan_source",
    "group_key_for",
    "apply_sampling",
    "assign_splits",
    "prepare",
    "main",
]

log = logging.getLogger("prepare_datasets")

#: Bumped when the manifest's shape changes. Written into every manifest so an
#: old manifest is never silently read with new assumptions.
MANIFEST_VERSION = 1

IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
)

#: Same decomposition ``tools/audit_dataset.py`` uses to find frame families:
#: everything before the trailing integer run is the family. Duplicated here
#: (rather than imported) because the two tools must agree on the *concept*
#: while remaining independently runnable -- the audit has to be able to fail
#: this script's output.
_SEQUENCE_RE = re.compile(r"^(?P<prefix>.*?)(?P<num>\d+)(?P<suffix>\D*)$")

#: Output stem. ``wf_000123_000045`` decomposes to family ``wf_000123_`` and
#: index 45 under the audit's rule -- one family per group, exactly.
_OUT_STEM = "wf_{gid:06d}_{idx:06d}"

#: Class-name spellings seen across the public datasets, folded onto the two
#: wire classes. Anything not listed is dropped and counted, never guessed:
#: silently mapping an unknown category onto ``fire`` would put mislabelled
#: boxes into training with no trace.
_CLASS_ALIASES: dict[str, str] = {
    "fire": "fire",
    "fires": "fire",
    "flame": "fire",
    "flames": "fire",
    "wildfire": "fire",
    "forest_fire": "fire",
    "forestfire": "fire",
    "fire_flame": "fire",
    "smoke": "smoke",
    "smokes": "smoke",
    "smoky": "smoke",
    "smoke_plume": "smoke",
    "plume": "smoke",
    "cloud_smoke": "smoke",
}

_VALID_FORMATS = ("yolo", "voc", "coco", "mask", "negatives_only")
_VALID_GROUPING = ("filename_family", "parent_dir", "regex", "per_image")
_VALID_COPY_MODES = ("symlink", "hardlink", "copy")


class PrepareError(RuntimeError):
    """A configuration or input problem that stops the merge."""


class LeakageError(PrepareError):
    """A group would have straddled two splits. Never expected; always fatal."""


# --------------------------------------------------------------------- model


@dataclass(frozen=True, slots=True)
class Box:
    """One label in YOLO centre form, normalised to the image, class canonical."""

    cls: str
    cx: float
    cy: float
    w: float
    h: float

    @property
    def area(self) -> float:
        """Fraction of the frame this box covers."""
        return self.w * self.h

    def to_line(self, class_index: dict[str, int]) -> str:
        """Render as one YOLO label line.

        Args:
            class_index: Canonical class name to integer index. The index order
                is the wire contract's ``CLASSES`` order and nothing else.

        Returns:
            A ``"<id> <cx> <cy> <w> <h>"`` line, 6dp -- sub-pixel at 4K.
        """
        return f"{class_index[self.cls]} {self.cx:.6f} {self.cy:.6f} {self.w:.6f} {self.h:.6f}"


@dataclass(slots=True)
class Sample:
    """One source image plus everything needed to place and label it."""

    source: str
    image: Path
    #: Group this image belongs to, unique within the source. The unit of
    #: splitting -- never subdivided across splits.
    group: str
    boxes: tuple[Box, ...] = ()
    width: int = 0
    height: int = 0
    #: True when this image is deliberately background: it has an annotation
    #: record that contains no objects. Distinct from "we could not find a
    #: label", which is a drop, not a negative.
    negative: bool = False
    sha256: str = ""

    @property
    def classes(self) -> frozenset[str]:
        return frozenset(b.cls for b in self.boxes)


@dataclass(slots=True)
class SourceSpec:
    """One entry in ``dataset_config.yaml``'s ``sources`` list."""

    name: str
    path: Path
    format: str = "yolo"
    enabled: bool = True
    #: Sub-directories, relative to ``path``. Meaning depends on ``format``.
    images: str = "images"
    labels: str = "labels"
    annotations: str = ""
    masks: str = "masks"
    #: For ``format: yolo`` only, and mandatory there: the source's own class
    #: order, so its integer ids can be resolved. Guessing this is how a model
    #: ends up with fire and smoke transposed on every tablet.
    class_names: tuple[str, ...] = ()
    #: Extra name folding on top of :data:`_CLASS_ALIASES`.
    class_map: dict[str, str] = field(default_factory=dict)
    #: For ``format: mask``: which class the mask's foreground represents.
    mask_class: str = "fire"
    mask_threshold: int = 127
    #: Components smaller than this fraction of the frame are mask speckle.
    mask_min_area: float = 0.0004
    #: "uav" | "ground" | "satellite" | "mixed". Recorded in the manifest and
    #: reported per split, because a merge that is 80% ground-level footage
    #: will validate well and fail from a drone.
    viewpoint: str = "unknown"
    #: Fraction of this source's *positive* groups to keep, 0..1.
    weight: float = 1.0
    #: Fraction of its background-only groups to keep. Defaults to ``weight``.
    negative_weight: float | None = None
    #: Keep every Nth frame within a group. The right way to thin dense video:
    #: it removes near-duplicates without removing scenes.
    frame_stride: int = 1
    #: Hard cap on images kept from this source, applied by dropping whole
    #: groups (never by truncating one).
    max_images: int | None = None
    keep_negatives: bool = True
    #: An image whose label file is absent. Default drop: a missing file is
    #: usually a broken export, and importing it as background would teach the
    #: model that a scene with fire in it contains nothing.
    treat_missing_labels_as_negative: bool = False
    grouping: str = "filename_family"
    group_regex: str = ""
    #: Required to enable ``grouping: per_image``. An explicit, recorded claim
    #: that these images are independent stills, not frames of a video.
    independent_images: bool = False
    #: Drop boxes at or above this fraction of the frame. A whole-frame box is
    #: a scene label wearing a detection's clothes; it teaches the model to
    #: answer "is this a fire picture", which is not the question.
    max_box_area: float | None = None
    min_box_area: float = 0.0
    licence: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        if self.format not in _VALID_FORMATS:
            raise PrepareError(
                f"source {self.name!r}: format {self.format!r} is not one of {list(_VALID_FORMATS)}"
            )
        if self.grouping not in _VALID_GROUPING:
            raise PrepareError(
                f"source {self.name!r}: grouping {self.grouping!r} is not one of {list(_VALID_GROUPING)}"
            )
        if self.grouping == "regex" and not self.group_regex:
            raise PrepareError(f"source {self.name!r}: grouping 'regex' needs a group_regex")
        if self.grouping == "per_image" and not self.independent_images:
            # The whole point of the tool. Per-image grouping IS a random split
            # unless the images really are unrelated stills, so it has to be
            # asserted deliberately, by a human, in the config, in writing.
            raise PrepareError(
                f"source {self.name!r}: grouping 'per_image' treats every image as its own "
                "group, which is a random split unless the images are genuinely independent "
                "stills. Set 'independent_images: true' on this source to assert that, or "
                "choose a real grouping (filename_family / parent_dir / regex)."
            )
        if self.format == "yolo" and not self.class_names:
            raise PrepareError(
                f"source {self.name!r}: format 'yolo' requires 'class_names' listing the "
                "source's own class order. Integer ids cannot be resolved without it, and "
                "guessing produces a model whose fire and smoke are transposed."
            )
        if not 0.0 <= self.weight <= 1.0:
            raise PrepareError(f"source {self.name!r}: weight must be in 0..1, got {self.weight}")
        if self.negative_weight is not None and not 0.0 <= self.negative_weight <= 1.0:
            raise PrepareError(
                f"source {self.name!r}: negative_weight must be in 0..1, got {self.negative_weight}"
            )
        if self.frame_stride < 1:
            raise PrepareError(f"source {self.name!r}: frame_stride must be >= 1")

    @property
    def effective_negative_weight(self) -> float:
        return self.weight if self.negative_weight is None else self.negative_weight


@dataclass(slots=True)
class OutputSpec:
    """The ``output`` block of ``dataset_config.yaml``."""

    root: Path = Path("datasets/wildfire-merged")
    splits: dict[str, float] = field(default_factory=lambda: {"train": 0.8, "val": 0.15, "test": 0.05})
    seed: int = 1337
    copy_mode: str = "symlink"
    #: Cap background images as a fraction of the merged set. ``None`` keeps
    #: every negative that survived weighting -- which is usually what we want
    #: here, because the false positives are the field failure mode.
    max_negative_fraction: float | None = None
    dedup: bool = True

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        if self.copy_mode not in _VALID_COPY_MODES:
            raise PrepareError(f"output.copy_mode {self.copy_mode!r} not in {list(_VALID_COPY_MODES)}")
        if "train" not in self.splits or "val" not in self.splits:
            raise PrepareError("output.splits must contain at least 'train' and 'val'")
        for name, frac in self.splits.items():
            if not 0.0 < float(frac) <= 1.0:
                raise PrepareError(f"output.splits[{name}] must be in (0,1], got {frac}")
        total = sum(float(v) for v in self.splits.values())
        if abs(total - 1.0) > 1e-6:
            raise PrepareError(f"output.splits must sum to 1.0, got {total:g}")
        if self.max_negative_fraction is not None and not 0.0 <= self.max_negative_fraction <= 1.0:
            raise PrepareError("output.max_negative_fraction must be in 0..1")


@dataclass(slots=True)
class PrepareConfig:
    """A parsed ``dataset_config.yaml``."""

    output: OutputSpec
    sources: list[SourceSpec]
    #: Prefix substituted for ``{data_root}`` in every source path.
    data_root: Path = Path(".")
    path: Path | None = None
    sha256: str = ""
    version: int = 1


@dataclass(slots=True)
class PrepareReport:
    """Everything the run concluded, renderable as text or JSON."""

    out_root: Path
    classes: tuple[str, ...]
    per_source: dict[str, dict[str, Any]] = field(default_factory=dict)
    split_stats: dict[str, dict[str, Any]] = field(default_factory=dict)
    totals: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    dry_run: bool = False

    def to_json(self) -> dict[str, Any]:
        return {
            "tool": "prepare_datasets",
            "manifest_version": MANIFEST_VERSION,
            "out_root": str(self.out_root),
            "classes": list(self.classes),
            "dry_run": self.dry_run,
            "totals": self.totals,
            "per_source": self.per_source,
            "splits": self.split_stats,
            "warnings": list(self.warnings),
        }

    def render(self) -> str:
        """Human-readable summary, printed at the end of every run."""
        lines: list[str] = []
        head = "PLANNED (dry run)" if self.dry_run else "WRITTEN"
        lines.append(f"{head}: {self.out_root}")
        lines.append("")
        lines.append(f"{'source':<18} {'view':<9} {'kept':>8} {'neg':>7} {'groups':>7}  dropped")
        for name, st in self.per_source.items():
            dropped = ", ".join(f"{k}={v}" for k, v in sorted(st["dropped"].items())) or "-"
            lines.append(
                f"{name:<18} {st['viewpoint']:<9} {st['kept_images']:>8} "
                f"{st['kept_negatives']:>7} {st['kept_groups']:>7}  {dropped}"
            )
        lines.append("")
        lines.append(f"{'split':<8} {'images':>8} {'neg':>7} {'groups':>7} {'boxes':>8}  per-class / viewpoint")
        for split, st in self.split_stats.items():
            per_class = ", ".join(f"{k}={v}" for k, v in sorted(st["boxes_per_class"].items())) or "none"
            views = ", ".join(f"{k}={v}" for k, v in sorted(st["viewpoints"].items()))
            lines.append(
                f"{split:<8} {st['images']:>8} {st['negatives']:>7} {st['groups']:>7} "
                f"{st['boxes']:>8}  {per_class} / {views}"
            )
        lines.append("")
        lines.append(
            f"totals: {self.totals.get('images', 0)} images, "
            f"{self.totals.get('boxes', 0)} boxes, "
            f"{self.totals.get('negatives', 0)} background images "
            f"({self.totals.get('negative_fraction', 0.0):.1%}), "
            f"{self.totals.get('groups', 0)} groups"
        )
        for warning in self.warnings:
            lines.append(f"WARNING: {warning}")
        lines.append("")
        lines.append(
            "Split by group, never by image. Now run tools/audit_dataset.py against "
            "the emitted data.yaml -- this script prevents the leak, the audit proves it."
        )
        return "\n".join(lines)


# ------------------------------------------------------------------- helpers


def canonical_classes() -> tuple[str, ...]:
    """Return the wire contract's class order, imported from the station.

    The dataset's class index order *is* the model's output index order, and
    the tablet colours boxes by that index. Hard-coding the order here would
    let the two drift apart silently, so it is imported from the one file that
    defines it.

    Returns:
        ``station.core.types.CLASSES``, e.g. ``("fire", "smoke")``.

    Raises:
        PrepareError: The station package is not importable, meaning the class
            order cannot be verified. Refusing beats guessing.
    """
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        from station.core.types import CLASSES  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover - only outside the repo
        raise PrepareError(
            "cannot import station.core.types.CLASSES, which defines the class index order "
            f"the trained weights must emit ({type(exc).__name__}: {exc}). Run this script "
            f"from a checkout of the repository (expected root: {repo_root})."
        ) from exc
    return tuple(CLASSES)


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """SHA-256 of a file, or ``""`` when it cannot be read."""
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(chunk), b""):
                digest.update(block)
    except OSError:
        return ""
    return digest.hexdigest()


def image_size(path: Path) -> tuple[int, int] | None:
    """Read image dimensions from the file header, without decoding pixels.

    Covers PNG, JPEG, GIF, BMP and simple WebP from their headers -- enough for
    every public fire dataset -- and falls back to Pillow if it happens to be
    installed. Pillow stays optional so this script runs on a bare Python.

    Args:
        path: Image file.

    Returns:
        ``(width, height)``, or ``None`` when the header is unrecognised.
    """
    try:
        with path.open("rb") as handle:
            head = handle.read(32)
            if head[:8] == b"\x89PNG\r\n\x1a\n" and head[12:16] == b"IHDR":
                return (
                    int.from_bytes(head[16:20], "big"),
                    int.from_bytes(head[20:24], "big"),
                )
            if head[:6] in (b"GIF87a", b"GIF89a"):
                return (
                    int.from_bytes(head[6:8], "little"),
                    int.from_bytes(head[8:10], "little"),
                )
            if head[:2] == b"BM":
                return (
                    int.from_bytes(head[18:22], "little", signed=True),
                    abs(int.from_bytes(head[22:26], "little", signed=True)),
                )
            if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
                if head[12:16] == b"VP8X":
                    return (
                        int.from_bytes(head[24:27], "little") + 1,
                        int.from_bytes(head[27:30], "little") + 1,
                    )
                if head[12:16] == b"VP8 ":
                    return (
                        int.from_bytes(head[26:28], "little") & 0x3FFF,
                        int.from_bytes(head[28:30], "little") & 0x3FFF,
                    )
            if head[:2] == b"\xff\xd8":
                return _jpeg_size(handle)
    except OSError:
        return None
    return _size_with_pillow(path)


def _jpeg_size(handle: Any) -> tuple[int, int] | None:
    """Walk JPEG markers to the first SOF segment."""
    handle.seek(2)
    while True:
        byte = handle.read(1)
        while byte and byte != b"\xff":
            byte = handle.read(1)
        marker = handle.read(1)
        while marker == b"\xff":
            marker = handle.read(1)
        if not marker:
            return None
        code = marker[0]
        # SOF0..SOF15, excluding the non-frame markers DHT/JPG/DAC.
        if 0xC0 <= code <= 0xCF and code not in (0xC4, 0xC8, 0xCC):
            handle.read(3)
            data = handle.read(4)
            if len(data) < 4:
                return None
            return (
                int.from_bytes(data[2:4], "big"),
                int.from_bytes(data[0:2], "big"),
            )
        length = handle.read(2)
        if len(length) < 2:
            return None
        skip = int.from_bytes(length, "big") - 2
        if skip < 0:
            return None
        handle.seek(skip, os.SEEK_CUR)


def _size_with_pillow(path: Path) -> tuple[int, int] | None:
    """Last-resort size read. Pillow is optional, so this may return None."""
    try:
        from PIL import Image  # noqa: PLC0415 - optional, imported only on fallback
    except ImportError:
        return None
    try:
        with Image.open(path) as img:
            return (int(img.width), int(img.height))
    except Exception:  # pragma: no cover - corrupt file
        return None


def iter_images(root: Path) -> Iterator[Path]:
    """Yield every image file under ``root``, sorted for determinism."""
    if not root.is_dir():
        return
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def _family_of(stem: str) -> str:
    """The constant part of a stem around its trailing integer run.

    ``clip07_frame_0142`` -> ``clip07_frame_|``. Matches the decomposition
    ``tools/audit_dataset.py`` performs, so a group defined here is the family
    the auditor will look for.
    """
    match = _SEQUENCE_RE.match(stem)
    if match is None:
        return stem.lower()
    return f"{match.group('prefix').lower()}|{match.group('suffix').lower()}"


def group_key_for(spec: SourceSpec, image: Path, images_root: Path) -> str:
    """Compute the split-unit key for one image.

    Args:
        spec: The source's configuration, which chooses the strategy.
        image: The image file.
        images_root: Directory the source's images were enumerated from;
            keys are made relative to it so they are stable across machines.

    Returns:
        A key that is unique within the source and identical for every image
        that came from the same video or capture session.

    Raises:
        PrepareError: ``grouping: regex`` was configured and the pattern does
            not match a path. Silently falling back would put one video's
            frames into several groups, which is the leak.
    """
    try:
        rel = image.resolve().relative_to(images_root.resolve())
    except ValueError:
        rel = Path(image.name)
    parent = rel.parent.as_posix()

    if spec.grouping == "parent_dir":
        return parent or "."
    if spec.grouping == "per_image":
        return rel.as_posix()
    if spec.grouping == "regex":
        match = re.search(spec.group_regex, rel.as_posix())
        if match is None:
            raise PrepareError(
                f"source {spec.name!r}: group_regex {spec.group_regex!r} does not match "
                f"{rel.as_posix()!r}. Every image must land in a group; an unmatched path "
                "would otherwise be silently grouped with unrelated frames."
            )
        return match.group("group") if "group" in (match.groupdict() or {}) else match.group(0)
    # filename_family: the directory plus the frame family, so two videos that
    # happen to share a naming scheme in different folders stay separate.
    return f"{parent}/{_family_of(rel.stem)}"


def _fold_class(spec: SourceSpec, raw: str, classes: Sequence[str]) -> str | None:
    """Fold a source's class name onto a wire class, or ``None`` to drop it."""
    name = str(raw).strip()
    mapped = spec.class_map.get(name) or spec.class_map.get(name.lower())
    if mapped is None:
        mapped = _CLASS_ALIASES.get(name.lower().replace("-", "_").replace(" ", "_"))
    if mapped is None:
        return None
    if mapped not in classes:
        raise PrepareError(
            f"source {spec.name!r}: class_map sends {name!r} to {mapped!r}, which is not a "
            f"wire class {list(classes)}"
        )
    return mapped


def _clamp_box(spec: SourceSpec, cls: str, cx: float, cy: float, w: float, h: float) -> Box | None:
    """Clamp to the frame and reject degenerate or scene-sized boxes.

    Boxes are clamped rather than dropped when they run off the edge: a fire at
    the edge of frame is the detection we least want to lose.
    """
    if not all(map(_finite, (cx, cy, w, h))):
        return None
    x1, y1 = max(0.0, cx - w / 2.0), max(0.0, cy - h / 2.0)
    x2, y2 = min(1.0, cx + w / 2.0), min(1.0, cy + h / 2.0)
    w2, h2 = x2 - x1, y2 - y1
    if w2 <= 0.0 or h2 <= 0.0:
        return None
    area = w2 * h2
    if area < spec.min_box_area:
        return None
    if spec.max_box_area is not None and area >= spec.max_box_area:
        return None
    return Box(cls=cls, cx=(x1 + x2) / 2.0, cy=(y1 + y2) / 2.0, w=w2, h=h2)


def _finite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))


# ------------------------------------------------------------------ scanning


def scan_source(spec: SourceSpec, classes: Sequence[str], drops: Counter[str]) -> list[Sample]:
    """Read one source dataset into canonical :class:`Sample` objects.

    Args:
        spec: The source configuration.
        classes: The wire class order, used to validate every folded name.
        drops: Mutated with a reason -> count tally of everything skipped, so
            the manifest records what was thrown away and why.

    Returns:
        Samples in deterministic (sorted path) order.

    Raises:
        PrepareError: The source directory is missing, or its annotations
            cannot be interpreted unambiguously.
    """
    if not spec.path.is_dir():
        raise PrepareError(
            f"source {spec.name!r}: path does not exist: {spec.path}\n"
            "  Either download it (see training/README.md), point --data-root at where it "
            "lives, or set 'enabled: false' on this source. A missing source is refused "
            "rather than skipped: silently merging fewer datasets than the config describes "
            "would make the manifest -- and every number derived from it -- untrue."
        )
    reader = {
        "yolo": _scan_yolo,
        "voc": _scan_voc,
        "coco": _scan_coco,
        "mask": _scan_mask,
        "negatives_only": _scan_negatives_only,
    }[spec.format]
    samples = reader(spec, classes, drops)
    log.info(
        "%s: scanned %d image(s) (%d background) from %s",
        spec.name,
        len(samples),
        sum(1 for s in samples if s.negative),
        spec.path,
    )
    return samples


def _images_root(spec: SourceSpec) -> Path:
    root = spec.path / spec.images if spec.images else spec.path
    if not root.is_dir():
        raise PrepareError(f"source {spec.name!r}: images directory not found: {root}")
    return root


def _make_sample(spec: SourceSpec, image: Path, root: Path, boxes: Sequence[Box], negative: bool,
                 width: int = 0, height: int = 0) -> Sample:
    return Sample(
        source=spec.name,
        image=image,
        group=f"{spec.name}::{group_key_for(spec, image, root)}",
        boxes=tuple(boxes),
        width=width,
        height=height,
        negative=negative,
    )


def _scan_yolo(spec: SourceSpec, classes: Sequence[str], drops: Counter[str]) -> list[Sample]:
    """YOLO layout: ``images/**`` mirrored by ``labels/**`` with ``.txt`` files."""
    root = _images_root(spec)
    labels_root = spec.path / spec.labels
    if not labels_root.is_dir():
        raise PrepareError(f"source {spec.name!r}: labels directory not found: {labels_root}")

    # Resolve the source's own index order once, up front, so an unmappable
    # class is a configuration error rather than a per-file surprise.
    index_to_class: dict[int, str | None] = {}
    for index, raw in enumerate(spec.class_names):
        index_to_class[index] = _fold_class(spec, raw, classes)

    out: list[Sample] = []
    for image in iter_images(root):
        rel = image.relative_to(root)
        label = (labels_root / rel).with_suffix(".txt")
        if not label.is_file():
            if spec.treat_missing_labels_as_negative:
                out.append(_make_sample(spec, image, root, (), negative=True))
            else:
                drops["missing_label"] += 1
            continue
        boxes, bad = _parse_yolo_label(spec, label, index_to_class, drops)
        if bad:
            drops["malformed_label_line"] += bad
        out.append(_make_sample(spec, image, root, boxes, negative=not boxes))
    return out


def _parse_yolo_label(
    spec: SourceSpec,
    label: Path,
    index_to_class: dict[int, str | None],
    drops: Counter[str],
) -> tuple[list[Box], int]:
    """Parse one YOLO ``.txt``, accepting the polygon form as its bounding box."""
    boxes: list[Box] = []
    bad = 0
    try:
        text = label.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return [], 1
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        tokens = line.split()
        try:
            cls_id = int(float(tokens[0]))
            values = [float(t) for t in tokens[1:]]
        except (ValueError, IndexError):
            bad += 1
            continue
        if len(values) == 4:
            cx, cy, w, h = values
        elif len(values) >= 6 and len(values) % 2 == 0:
            xs, ys = values[0::2], values[1::2]
            cx, cy = (min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0
            w, h = max(xs) - min(xs), max(ys) - min(ys)
        else:
            bad += 1
            continue
        if cls_id not in index_to_class:
            drops["unknown_class_index"] += 1
            continue
        cls = index_to_class[cls_id]
        if cls is None:
            drops["unmapped_class"] += 1
            continue
        box = _clamp_box(spec, cls, cx, cy, w, h)
        if box is None:
            drops["degenerate_or_oversized_box"] += 1
            continue
        boxes.append(box)
    return boxes, bad


def _scan_voc(spec: SourceSpec, classes: Sequence[str], drops: Counter[str]) -> list[Sample]:
    """Pascal VOC layout: one XML per image with pixel ``<bndbox>`` corners."""
    import xml.etree.ElementTree as ET  # noqa: PLC0415 - stdlib, kept local for symmetry

    root = _images_root(spec)
    ann_root = spec.path / (spec.annotations or "Annotations")
    if not ann_root.is_dir():
        raise PrepareError(f"source {spec.name!r}: VOC annotations directory not found: {ann_root}")

    # VOC exports are usually flat, so index by stem: the XML often sits in one
    # directory while images are nested by scene.
    xml_by_stem: dict[str, Path] = {}
    for xml in sorted(ann_root.rglob("*.xml")):
        xml_by_stem.setdefault(xml.stem, xml)

    out: list[Sample] = []
    for image in iter_images(root):
        xml = xml_by_stem.get(image.stem)
        if xml is None:
            if spec.treat_missing_labels_as_negative:
                out.append(_make_sample(spec, image, root, (), negative=True))
            else:
                drops["missing_label"] += 1
            continue
        try:
            tree = ET.parse(xml)
        except ET.ParseError:
            drops["unparseable_annotation"] += 1
            continue
        node = tree.getroot()
        size = node.find("size")
        width = int(float(size.findtext("width", "0"))) if size is not None else 0
        height = int(float(size.findtext("height", "0"))) if size is not None else 0
        if width <= 0 or height <= 0:
            measured = image_size(image)
            if measured is None:
                drops["unknown_image_size"] += 1
                continue
            width, height = measured
        boxes: list[Box] = []
        for obj in node.findall("object"):
            cls = _fold_class(spec, obj.findtext("name", ""), classes)
            if cls is None:
                drops["unmapped_class"] += 1
                continue
            bnd = obj.find("bndbox")
            if bnd is None:
                drops["malformed_label_line"] += 1
                continue
            try:
                x1 = float(bnd.findtext("xmin", "nan"))
                y1 = float(bnd.findtext("ymin", "nan"))
                x2 = float(bnd.findtext("xmax", "nan"))
                y2 = float(bnd.findtext("ymax", "nan"))
            except ValueError:
                drops["malformed_label_line"] += 1
                continue
            box = _clamp_box(
                spec,
                cls,
                (x1 + x2) / 2.0 / width,
                (y1 + y2) / 2.0 / height,
                abs(x2 - x1) / width,
                abs(y2 - y1) / height,
            )
            if box is None:
                drops["degenerate_or_oversized_box"] += 1
                continue
            boxes.append(box)
        out.append(_make_sample(spec, image, root, boxes, negative=not boxes, width=width, height=height))
    return out


def _scan_coco(spec: SourceSpec, classes: Sequence[str], drops: Counter[str]) -> list[Sample]:
    """COCO layout: one JSON of ``images`` / ``annotations`` / ``categories``.

    Images listed in the JSON with no annotations are genuine negatives -- this
    is how FASDD ships a large part of its background set -- so they are kept
    as background rather than dropped.
    """
    root = _images_root(spec)
    ann_path = spec.path / (spec.annotations or "annotations.json")
    if ann_path.is_dir():
        candidates = sorted(ann_path.glob("*.json"))
        if len(candidates) != 1:
            raise PrepareError(
                f"source {spec.name!r}: 'annotations' points at a directory holding "
                f"{len(candidates)} JSON files; name the exact file so the merge is reproducible"
            )
        ann_path = candidates[0]
    if not ann_path.is_file():
        raise PrepareError(f"source {spec.name!r}: COCO annotation file not found: {ann_path}")

    raw = json.loads(ann_path.read_text(encoding="utf-8"))

    # Real COCO files in the wild deviate from the spec in small ways that fail
    # silently rather than loudly. HIT-UAV keys its box list "annotation" and
    # its filenames "filename"; read strictly and you get zero boxes, every
    # image is filed as a genuine negative, and training proceeds happily on a
    # dataset that has quietly become 2,898 pictures of nothing. Aliases are
    # accepted, and an annotation list that is missing entirely is an error
    # rather than an empty result.
    ann_key = next((k for k in ("annotations", "annotation") if isinstance(raw.get(k), list)), None)
    if ann_key is None:
        raise PrepareError(
            f"source {spec.name!r}: {ann_path} has no 'annotations' (or 'annotation') list. "
            "Reading it as zero boxes would silently turn every image into a negative."
        )
    annotation_list = raw[ann_key]

    cat_to_class: dict[int, str | None] = {}
    for cat in raw.get("categories", ()):
        cat_to_class[int(cat["id"])] = _fold_class(spec, cat.get("name", ""), classes)

    by_image: dict[int, list[Box]] = defaultdict(list)
    meta: dict[int, dict[str, Any]] = {}
    for img in raw.get("images", ()):
        meta[int(img["id"])] = img
    for ann in annotation_list:
        image_id = int(ann["image_id"])
        info = meta.get(image_id)
        if info is None:
            drops["annotation_without_image"] += 1
            continue
        cls = cat_to_class.get(int(ann.get("category_id", -1)))
        if cls is None:
            drops["unmapped_class"] += 1
            continue
        bbox = ann.get("bbox")
        if not bbox or len(bbox) != 4:
            drops["malformed_label_line"] += 1
            continue
        width = float(info.get("width", 0)) or 0.0
        height = float(info.get("height", 0)) or 0.0
        if width <= 0 or height <= 0:
            drops["unknown_image_size"] += 1
            continue
        x, y, bw, bh = (float(v) for v in bbox)
        box = _clamp_box(
            spec, cls, (x + bw / 2.0) / width, (y + bh / 2.0) / height, bw / width, bh / height
        )
        if box is None:
            drops["degenerate_or_oversized_box"] += 1
            continue
        by_image[image_id].append(box)

    # Map file_name (possibly with directories) onto the images actually found.
    on_disk: dict[str, Path] = {}
    for image in iter_images(root):
        rel = image.relative_to(root).as_posix()
        on_disk.setdefault(rel, image)
        on_disk.setdefault(image.name, image)

    out: list[Sample] = []
    for image_id, info in sorted(meta.items()):
        # "filename" is the other common deviation (HIT-UAV again).
        file_name = str(info.get("file_name") or info.get("filename") or "")
        image = on_disk.get(file_name) or on_disk.get(Path(file_name).name)
        if image is None:
            drops["image_listed_but_absent"] += 1
            continue
        boxes = by_image.get(image_id, [])
        out.append(
            _make_sample(
                spec,
                image,
                root,
                boxes,
                negative=not boxes,
                width=int(float(info.get("width", 0) or 0)),
                height=int(float(info.get("height", 0) or 0)),
            )
        )
    return out


def _scan_mask(spec: SourceSpec, classes: Sequence[str], drops: Counter[str]) -> list[Sample]:
    """Segmentation masks (FLAME, Corsican) converted to bounding boxes.

    One box per connected component, which keeps two separate flame fronts as
    two detections instead of one box spanning the gap between them.

    Needs Pillow to decode the masks. That is the only heavy dependency in this
    file and it is imported here, so a merge that uses no mask sources runs on
    a bare Python install.
    """
    try:
        import numpy as np  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415
    except ImportError as exc:
        raise PrepareError(
            f"source {spec.name!r}: format 'mask' needs numpy and Pillow to decode the mask "
            f"images ({exc}). Install them (pip install pillow numpy) or disable this source."
        ) from exc

    cls = _fold_class(spec, spec.mask_class, classes)
    if cls is None:
        raise PrepareError(f"source {spec.name!r}: mask_class {spec.mask_class!r} is not a wire class")

    root = _images_root(spec)
    masks_root = spec.path / spec.masks
    if not masks_root.is_dir():
        raise PrepareError(f"source {spec.name!r}: masks directory not found: {masks_root}")
    mask_by_stem: dict[str, Path] = {}
    for mask in sorted(masks_root.rglob("*")):
        if mask.is_file() and mask.suffix.lower() in IMAGE_EXTENSIONS:
            mask_by_stem.setdefault(mask.stem, mask)

    out: list[Sample] = []
    for image in iter_images(root):
        mask_path = mask_by_stem.get(image.stem)
        if mask_path is None:
            if spec.treat_missing_labels_as_negative:
                out.append(_make_sample(spec, image, root, (), negative=True))
            else:
                drops["missing_label"] += 1
            continue
        try:
            with Image.open(mask_path) as handle:
                mask = np.asarray(handle.convert("L"))
        except Exception:
            drops["unparseable_annotation"] += 1
            continue
        binary = mask >= spec.mask_threshold
        boxes = [
            box
            for rect in _components(binary, np)
            if (box := _clamp_box(spec, cls, *rect)) is not None
        ]
        boxes = [b for b in boxes if b.area >= spec.mask_min_area]
        out.append(
            _make_sample(
                spec,
                image,
                root,
                boxes,
                negative=not boxes,
                width=int(mask.shape[1]),
                height=int(mask.shape[0]),
            )
        )
    return out


def _components(binary: Any, np: Any, max_side: int = 512) -> list[tuple[float, float, float, float]]:
    """Bounding boxes of the connected foreground components of a binary mask.

    Run-length union-find rather than a flood fill: one pass over the rows,
    O(runs) unions, no recursion, and only numpy. A 4K mask neither blows the
    stack nor takes a second.

    Connectivity is 8-way. Flame masks are speckly at their edges, and 4-way
    connectivity splits one flame front into a dozen boxes wherever the mask
    thins to a diagonal.

    Args:
        binary: 2-D boolean array, ``True`` where the target is.
        np: The numpy module (passed in so the import stays lazy).
        max_side: The mask is strided down so its longest side is at most this.
            Costs sub-pixel box accuracy, buys an order of magnitude in time,
            and the boxes are normalised so the downscale is invisible after.

    Returns:
        ``(cx, cy, w, h)`` tuples, normalised 0..1 against the mask.
    """
    height, width = binary.shape[:2]
    if height == 0 or width == 0:
        return []
    step = max(1, int(max(height, width) / max_side))
    small = binary[::step, ::step]
    rows, cols = small.shape
    if rows == 0 or cols == 0:
        return []

    parent: list[int] = []
    spans: list[list[int]] = []  # x1, y1, x2, y2 per provisional label

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    previous: list[tuple[int, int, int]] = []  # (x1, x2, label) of the row above
    for y in range(rows):
        columns = np.flatnonzero(small[y])
        if columns.size == 0:
            previous = []
            continue
        cuts = np.flatnonzero(np.diff(columns) > 1)
        starts = np.concatenate(([0], cuts + 1))
        ends = np.concatenate((cuts + 1, [columns.size]))
        current: list[tuple[int, int, int]] = []
        for start, end in zip(starts.tolist(), ends.tolist()):
            x1, x2 = int(columns[start]), int(columns[end - 1])
            label = len(parent)
            parent.append(label)
            spans.append([x1, y, x2, y])
            for px1, px2, plabel in previous:
                if px1 <= x2 + 1 and x1 <= px2 + 1:  # 8-connectivity
                    union(label, plabel)
            current.append((x1, x2, label))
        previous = current

    merged: dict[int, list[int]] = {}
    for label, (x1, y1, x2, y2) in enumerate(spans):
        root = find(label)
        span = merged.get(root)
        if span is None:
            merged[root] = [x1, y1, x2, y2]
        else:
            span[0] = min(span[0], x1)
            span[1] = min(span[1], y1)
            span[2] = max(span[2], x2)
            span[3] = max(span[3], y2)

    out: list[tuple[float, float, float, float]] = []
    for x1, y1, x2, y2 in merged.values():
        # +1 on the extent so a one-pixel component is not a zero-area box.
        out.append(
            (
                ((x1 + x2 + 1) / 2.0) / cols,
                ((y1 + y2 + 1) / 2.0) / rows,
                (x2 - x1 + 1) / cols,
                (y2 - y1 + 1) / rows,
            )
        )
    return out


def _scan_negatives_only(spec: SourceSpec, classes: Sequence[str], drops: Counter[str]) -> list[Sample]:
    """Import a directory of images as background, with no boxes at all.

    This is how a classification-style set (FLAME's frame folders, a folder of
    smoke-free flight footage) enters the merge. Its *positive* folder is
    deliberately not importable: a whole-frame "this picture contains fire"
    label trains a scene classifier wearing a detector's output layer, which is
    the exact failure ``tools/audit_dataset.py`` was written after.
    """
    del classes  # negatives carry no class by definition
    root = _images_root(spec)
    out = [_make_sample(spec, image, root, (), negative=True) for image in iter_images(root)]
    if not out:
        drops["empty_source"] += 1
    return out


# ------------------------------------------------------------------ sampling


def apply_sampling(
    samples: list[Sample], spec: SourceSpec, seed: int, drops: Counter[str]
) -> list[Sample]:
    """Thin a source down to its configured weight, group by group.

    Order of operations, and each step's reason:

    1. ``frame_stride`` -- keep every Nth frame *within* a group. Removes
       near-duplicate frames while keeping every scene the source covers.
    2. ``keep_negatives`` -- drop background images if this source's are not
       wanted.
    3. ``weight`` / ``negative_weight`` -- keep that fraction of the groups,
       chosen with an RNG seeded from the global seed and the source name so
       the choice is reproducible and independent of the other sources.
    4. ``max_images`` -- drop whole groups, smallest first, until under the cap.

    Groups are the unit throughout. Sampling *within* a group would not cause
    leakage (the whole group still lands on one side) but it would silently
    change what "one video" means between runs, and the manifest would no
    longer describe a reproducible dataset.

    Args:
        samples: Every sample scanned from this source.
        spec: The source configuration.
        seed: The run's global seed.
        drops: Mutated with reason -> count for everything removed.

    Returns:
        The surviving samples, order preserved.
    """
    by_group: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        by_group[sample.group].append(sample)

    if spec.frame_stride > 1:
        thinned: dict[str, list[Sample]] = {}
        for group, members in by_group.items():
            kept = members[:: spec.frame_stride]
            drops["frame_stride"] += len(members) - len(kept)
            thinned[group] = kept
        by_group = thinned

    if not spec.keep_negatives:
        for group, members in list(by_group.items()):
            kept = [s for s in members if not s.negative]
            drops["negatives_disabled"] += len(members) - len(kept)
            if kept:
                by_group[group] = kept
            else:
                del by_group[group]

    # A group counts as negative only when *every* member is background; a
    # mixed group is a video that contains fire and is weighted as a positive.
    positive_groups = sorted(g for g, m in by_group.items() if any(not s.negative for s in m))
    negative_groups = sorted(g for g, m in by_group.items() if all(s.negative for s in m))

    rng = random.Random(f"{seed}:{spec.name}")
    keep: set[str] = set()
    keep |= _sample_groups(positive_groups, spec.weight, rng)
    keep |= _sample_groups(negative_groups, spec.effective_negative_weight, rng)

    for group in by_group:
        if group not in keep:
            drops["weight"] += len(by_group[group])

    if spec.max_images is not None:
        ordered = sorted(keep, key=lambda g: (len(by_group[g]), g))
        total = sum(len(by_group[g]) for g in ordered)
        # Drop the smallest groups first: it takes more of them to get under
        # the cap, which keeps the surviving set as scene-diverse as possible.
        index = 0
        while total > spec.max_images and index < len(ordered):
            group = ordered[index]
            keep.discard(group)
            total -= len(by_group[group])
            drops["max_images"] += len(by_group[group])
            index += 1

    return [s for group in by_group for s in by_group[group] if group in keep]


def _sample_groups(groups: Sequence[str], weight: float, rng: random.Random) -> set[str]:
    """Keep ``weight`` of ``groups``, at least one whenever weight > 0."""
    if not groups:
        return set()
    if weight >= 1.0:
        return set(groups)
    if weight <= 0.0:
        return set()
    target = max(1, round(len(groups) * weight))
    return set(rng.sample(list(groups), target))


# ------------------------------------------------------------------ splitting


def assign_splits(
    samples: Sequence[Sample], splits: dict[str, float], seed: int
) -> dict[str, str]:
    """Assign every group to exactly one split, hitting the target proportions.

    Groups vary wildly in size -- one FLAME flight is thousands of frames, one
    Corsican still is one image -- so a naive "shuffle groups, take the first
    80%" gives wrong proportions. Instead groups are shuffled once (seeded) and
    then placed greedily into whichever split is furthest below its target
    image count. That is deterministic, proportion-accurate to within one
    group, and still never divides a group.

    Args:
        samples: Every sample that survived sampling.
        splits: Split name -> target fraction, summing to 1.
        seed: The run's global seed.

    Returns:
        Group key -> split name.

    Raises:
        PrepareError: There are fewer groups than splits, so at least one split
            would be empty. That is a dataset problem, not something to paper
            over by splitting a group.
    """
    sizes: Counter[str] = Counter()
    for sample in samples:
        sizes[sample.group] += 1
    groups = sorted(sizes)
    if len(groups) < len(splits):
        raise PrepareError(
            f"only {len(groups)} group(s) across {len(splits)} split(s). Splitting by group is "
            "not optional here, so this dataset cannot be split at all -- add more source "
            "videos, or use a grouping that reflects the real capture sessions."
        )

    total = sum(sizes.values())
    targets = {name: total * float(frac) for name, frac in splits.items()}
    current = {name: 0 for name in splits}

    rng = random.Random(f"{seed}:split")
    rng.shuffle(groups)
    # Largest groups first: placing the big ones while every split is still
    # empty is what keeps the final proportions close to target.
    groups.sort(key=lambda g: -sizes[g])

    assignment: dict[str, str] = {}
    for group in groups:
        split = max(splits, key=lambda name: (targets[name] - current[name], name))
        assignment[group] = split
        current[split] += sizes[group]
    return assignment


def ensure_class_coverage(
    samples: Sequence[Sample],
    assignment: dict[str, str],
    classes: Sequence[str],
    warnings: list[str],
) -> None:
    """Move groups until every split that needs boxes of a class has some.

    A class with no boxes in val has an unmeasurable recall, and the audit
    fails the dataset for it -- correctly, because "we never checked whether it
    finds smoke" is not a state to train in. Repaired by moving whole groups,
    never by copying images between splits.
    """
    by_group: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        by_group[sample.group].append(sample)

    split_classes: dict[str, set[str]] = defaultdict(set)
    for group, split in assignment.items():
        for sample in by_group[group]:
            split_classes[split] |= sample.classes

    for cls in classes:
        if cls not in split_classes.get("train", set()):
            warnings.append(
                f"train contains no boxes of class {cls!r}: the model cannot learn it, and a "
                "class it cannot detect is a silent false-negative source in the field"
            )

    for split in sorted(set(assignment.values())):
        if split == "train":
            continue
        for cls in classes:
            if cls in split_classes[split]:
                continue
            donors = [
                g
                for g, s in assignment.items()
                if s == "train" and any(cls in sample.classes for sample in by_group[g])
            ]
            if not donors:
                warnings.append(
                    f"no group anywhere contains class {cls!r}; split {split!r} cannot measure it"
                )
                continue
            # Smallest donor that still has the class: moves the least data.
            donor = min(donors, key=lambda g: (len(by_group[g]), g))
            assignment[donor] = split
            split_classes[split] |= {c for sample in by_group[donor] for c in sample.classes}
            warnings.append(
                f"moved group {donor!r} ({len(by_group[donor])} images) from train to {split} "
                f"so class {cls!r} has boxes there"
            )


def _warn_on_split_drift(
    samples: Sequence[Sample],
    assignment: dict[str, str],
    splits: dict[str, float],
    warnings: list[str],
    tolerance: float = 0.05,
) -> None:
    """Report when the realised split proportions miss their targets.

    Group-aware splitting cannot hit an exact percentage: groups are indivisible
    and vary in size by three orders of magnitude, and the class-coverage repair
    moves whole groups afterwards. That drift is the correct trade -- the
    alternative is a leaking split with beautiful percentages -- but it must be
    visible rather than discovered later as a surprise in the val count.
    """
    counts: Counter[str] = Counter()
    for sample in samples:
        counts[assignment[sample.group]] += 1
    total = sum(counts.values())
    if not total:
        return
    for name, target in splits.items():
        actual = counts[name] / total
        if counts[name] == 0:
            warnings.append(
                f"split {name!r} received no groups at all. With this few groups the split "
                "cannot be honoured; merge more source videos rather than reducing the group size"
            )
            continue
        if abs(actual - float(target)) > tolerance:
            warnings.append(
                f"split {name!r} came out at {actual:.1%} against a {float(target):.0%} target "
                "(groups are indivisible; this is expected when a few groups dominate)"
            )


def _verify_no_group_straddles_split(samples: Sequence[Sample], assignment: dict[str, str]) -> None:
    """Re-derive the split of every sample and fail if a group is divided.

    Defence in depth against a bug in this file. The audit would catch it
    later, but by then the dataset has been written and possibly trained on.
    """
    seen: dict[str, str] = {}
    for sample in samples:
        split = assignment[sample.group]
        previous = seen.setdefault(sample.group, split)
        if previous != split:
            raise LeakageError(
                f"group {sample.group!r} was assigned to both {previous!r} and {split!r}. "
                "This is the leakage the whole script exists to prevent; refusing to write."
            )


# ------------------------------------------------------------- materialising


def _place(src: Path, dst: Path, mode: str) -> None:
    """Put one source image at ``dst`` by symlink, hardlink or copy.

    Symlinks are the default: a 122k-image merge is ~50 GB and the Kaggle input
    mount is read-only but perfectly linkable. Hardlinks and copies exist for
    filesystems where symlinks do not survive (some Windows setups, some
    archive pipelines).
    """
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "symlink":
        os.symlink(os.path.relpath(src.resolve(), dst.parent.resolve()), dst)
        return
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            # Cross-device or unsupported: a copy is correct, just slower.
            pass
    shutil.copy2(src, dst)


def _write_data_yaml(out_root: Path, classes: Sequence[str], splits: Iterable[str]) -> Path:
    """Write ultralytics' ``data.yaml``.

    Emitted by hand rather than via PyYAML so the file carries comments
    explaining the class order, which is the field most likely to be edited by
    someone who does not know it is load-bearing.
    """
    lines = [
        "# Generated by training/prepare_datasets.py -- do not hand-edit.",
        "#",
        "# 'names' order IS the model's output index order and the wire contract's",
        "# CLASSES order (station/core/types.py). Reordering this list without",
        "# renumbering every label file transposes fire and smoke on every tablet,",
        "# and nothing downstream of the model can detect that.",
        f"path: {out_root.resolve().as_posix()}",
    ]
    for split in splits:
        lines.append(f"{split}: images/{split}")
    lines.append(f"nc: {len(classes)}")
    lines.append("names:")
    lines.extend(f"  - {name}" for name in classes)
    path = out_root / "data.yaml"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# ------------------------------------------------------------------ the merge


def prepare(
    cfg: PrepareConfig,
    *,
    only: Sequence[str] = (),
    exclude: Sequence[str] = (),
    dry_run: bool = False,
    force: bool = False,
) -> PrepareReport:
    """Run the whole merge: scan, weight, split, write, manifest.

    Args:
        cfg: Parsed configuration.
        only: If non-empty, restrict to these source names.
        exclude: Source names to skip.
        dry_run: Compute and report everything, write nothing.
        force: Overwrite a non-empty output directory.

    Returns:
        A :class:`PrepareReport`.

    Raises:
        PrepareError: Configuration, input or safety-guard failure.
        LeakageError: A group would have been divided across splits.
    """
    started = time.time()
    classes = canonical_classes()
    class_index = {name: i for i, name in enumerate(classes)}
    out_root = cfg.output.root
    warnings: list[str] = []

    selected = [s for s in cfg.sources if s.enabled]
    if only:
        wanted = set(only)
        unknown = wanted - {s.name for s in cfg.sources}
        if unknown:
            raise PrepareError(f"--only names unknown source(s): {sorted(unknown)}")
        selected = [s for s in selected if s.name in wanted]
    if exclude:
        selected = [s for s in selected if s.name not in set(exclude)]
    if not selected:
        raise PrepareError("no enabled sources selected")

    if not dry_run and out_root.exists() and any(out_root.iterdir()) and not force:
        raise PrepareError(
            f"{out_root} exists and is not empty. Pass --force to overwrite; a merge that "
            "silently mixed with a previous run would produce a manifest that lies."
        )

    per_source: dict[str, dict[str, Any]] = {}
    kept: list[Sample] = []
    for spec in selected:
        drops: Counter[str] = Counter()
        scanned = scan_source(spec, classes, drops)
        survivors = apply_sampling(scanned, spec, cfg.output.seed, drops)
        kept.extend(survivors)
        per_source[spec.name] = {
            "path": str(spec.path),
            "format": spec.format,
            "viewpoint": spec.viewpoint,
            "licence": spec.licence,
            "weight": spec.weight,
            "negative_weight": spec.effective_negative_weight,
            "frame_stride": spec.frame_stride,
            "grouping": spec.grouping,
            "scanned_images": len(scanned),
            "scanned_groups": len({s.group for s in scanned}),
            "kept_images": len(survivors),
            "kept_groups": len({s.group for s in survivors}),
            "kept_negatives": sum(1 for s in survivors if s.negative),
            "kept_boxes": sum(len(s.boxes) for s in survivors),
            "dropped": dict(sorted(drops.items())),
        }

    if not kept:
        raise PrepareError("every source scanned to zero usable images; nothing to merge")

    if cfg.output.dedup:
        kept = _dedupe(kept, per_source, warnings)
    else:
        warnings.append(
            "content de-duplication disabled: the same image appearing in two sources can now "
            "land in two splits, which tools/audit_dataset.py will fail the dataset for"
        )

    kept = _cap_negatives(kept, cfg.output.max_negative_fraction, per_source, warnings)

    # Recomputed after de-duplication and the negative cap so the per-source
    # numbers describe what was actually written, not what was scanned.
    for name, stats in per_source.items():
        survivors = [s for s in kept if s.source == name]
        stats["kept_images"] = len(survivors)
        stats["kept_groups"] = len({s.group for s in survivors})
        stats["kept_negatives"] = sum(1 for s in survivors if s.negative)
        stats["kept_boxes"] = sum(len(s.boxes) for s in survivors)

    assignment = assign_splits(kept, cfg.output.splits, cfg.output.seed)
    ensure_class_coverage(kept, assignment, classes, warnings)
    _verify_no_group_straddles_split(kept, assignment)
    _warn_on_split_drift(kept, assignment, cfg.output.splits, warnings)

    report = PrepareReport(out_root=out_root, classes=classes, per_source=per_source, dry_run=dry_run)
    report.warnings = warnings

    # Group ids are assigned in a stable order so a re-run with the same config
    # and seed produces byte-identical filenames.
    group_ids = {group: gid for gid, group in enumerate(sorted({s.group for s in kept}))}
    by_group: dict[str, list[Sample]] = defaultdict(list)
    for sample in kept:
        by_group[sample.group].append(sample)

    split_stats: dict[str, dict[str, Any]] = {
        split: {
            "images": 0,
            "negatives": 0,
            "groups": 0,
            "boxes": 0,
            "boxes_per_class": Counter(),
            "viewpoints": Counter(),
            "sources": Counter(),
        }
        for split in cfg.output.splits
    }
    viewpoint_of = {spec.name: spec.viewpoint for spec in selected}

    provenance: list[dict[str, Any]] = []
    for group in sorted(by_group):
        split = assignment[group]
        gid = group_ids[group]
        stats = split_stats[split]
        stats["groups"] += 1
        for index, sample in enumerate(sorted(by_group[group], key=lambda s: s.image.as_posix())):
            stem = _OUT_STEM.format(gid=gid, idx=index)
            suffix = sample.image.suffix.lower()
            stats["images"] += 1
            stats["sources"][sample.source] += 1
            stats["viewpoints"][viewpoint_of.get(sample.source, "unknown")] += 1
            if sample.negative:
                stats["negatives"] += 1
            for box in sample.boxes:
                stats["boxes"] += 1
                stats["boxes_per_class"][box.cls] += 1

            if not dry_run:
                image_dst = out_root / "images" / split / f"{stem}{suffix}"
                label_dst = out_root / "labels" / split / f"{stem}.txt"
                image_dst.parent.mkdir(parents=True, exist_ok=True)
                label_dst.parent.mkdir(parents=True, exist_ok=True)
                _place(sample.image, image_dst, cfg.output.copy_mode)
                # Background images get an EMPTY label file, not a missing one.
                # An absent file reads as a broken export; an empty one is the
                # deliberate statement "there is nothing to box in this image",
                # which is the whole value of the negatives.
                label_dst.write_text(
                    "".join(f"{b.to_line(class_index)}\n" for b in sample.boxes), encoding="utf-8"
                )

            provenance.append(
                {
                    "stem": stem,
                    "split": split,
                    "source": sample.source,
                    "group": group,
                    "group_id": gid,
                    "original": sample.image.as_posix(),
                    "sha256": sample.sha256,
                    "negative": sample.negative,
                    "boxes": len(sample.boxes),
                    "classes": sorted(sample.classes),
                }
            )

    for stats in split_stats.values():
        stats["boxes_per_class"] = dict(stats["boxes_per_class"])
        stats["viewpoints"] = dict(stats["viewpoints"])
        stats["sources"] = dict(stats["sources"])
    report.split_stats = split_stats

    total_images = sum(s["images"] for s in split_stats.values())
    total_negatives = sum(s["negatives"] for s in split_stats.values())
    report.totals = {
        "images": total_images,
        "negatives": total_negatives,
        "negative_fraction": (total_negatives / total_images) if total_images else 0.0,
        "boxes": sum(s["boxes"] for s in split_stats.values()),
        "groups": len(group_ids),
        "sources": len(selected),
        "elapsed_s": round(time.time() - started, 2),
    }

    if not dry_run:
        out_root.mkdir(parents=True, exist_ok=True)
        data_yaml = _write_data_yaml(out_root, classes, cfg.output.splits)
        _write_provenance(out_root, provenance)
        _write_manifest(out_root, cfg, report, selected)
        log.info("wrote %s", data_yaml)

    return report


def _dedupe(
    samples: list[Sample], per_source: dict[str, dict[str, Any]], warnings: list[str]
) -> list[Sample]:
    """Drop byte-identical images, keeping the first occurrence.

    Cross-source duplicates are common (the scraped datasets overlap), and two
    copies of one image in two splits is a genuine train/val leak that the
    filename grouping cannot see. Hashing every file is the slow part of a big
    merge and it is worth it.
    """
    seen: dict[str, Sample] = {}
    out: list[Sample] = []
    cross_source = 0
    for sample in samples:
        sample.sha256 = sha256_file(sample.image)
        if not sample.sha256:
            warnings.append(f"unreadable image skipped: {sample.image}")
            per_source[sample.source]["dropped"]["unreadable_image"] = (
                per_source[sample.source]["dropped"].get("unreadable_image", 0) + 1
            )
            continue
        first = seen.get(sample.sha256)
        if first is not None:
            if first.source != sample.source:
                cross_source += 1
            per_source[sample.source]["dropped"]["duplicate_content"] = (
                per_source[sample.source]["dropped"].get("duplicate_content", 0) + 1
            )
            per_source[sample.source]["kept_images"] -= 1
            if sample.negative:
                per_source[sample.source]["kept_negatives"] -= 1
            continue
        seen[sample.sha256] = sample
        out.append(sample)
    if cross_source:
        warnings.append(
            f"{cross_source} image(s) appeared in more than one source dataset and were kept "
            "once; the sources overlap, so their published counts double-count"
        )
    return out


def _cap_negatives(
    samples: list[Sample],
    max_fraction: float | None,
    per_source: dict[str, dict[str, Any]],
    warnings: list[str],
) -> list[Sample]:
    """Enforce ``output.max_negative_fraction`` by dropping whole groups.

    Left uncapped by default. A high background fraction is unusual advice for
    a detector, and it is deliberate here: the field failure this project
    expects is a false positive on a red roof or a dust plume, and background
    imagery is the only thing that teaches the difference.
    """
    if max_fraction is None:
        return samples
    negatives = [s for s in samples if s.negative]
    total = len(samples)
    if not total or len(negatives) / total <= max_fraction:
        return samples

    allowed = int(max_fraction * total)
    by_group: dict[str, list[Sample]] = defaultdict(list)
    for sample in negatives:
        by_group[sample.group].append(sample)
    # Deterministic order, largest first, so the cap removes as few groups
    # (and therefore as few distinct scenes) as possible.
    ordered = sorted(by_group, key=lambda g: (-len(by_group[g]), g))
    dropped: set[str] = set()
    count = len(negatives)
    for group in ordered:
        if count <= allowed:
            break
        dropped.add(group)
        count -= len(by_group[group])
    for group in dropped:
        for sample in by_group[group]:
            per_source[sample.source]["dropped"]["negative_cap"] = (
                per_source[sample.source]["dropped"].get("negative_cap", 0) + 1
            )
            per_source[sample.source]["kept_images"] -= 1
            per_source[sample.source]["kept_negatives"] -= 1
    warnings.append(
        f"negative cap dropped {len(negatives) - count} background image(s) to reach "
        f"{max_fraction:.0%}; background imagery is the main defence against false positives"
    )
    return [s for s in samples if s.group not in dropped]


def _write_provenance(out_root: Path, rows: Sequence[dict[str, Any]]) -> Path:
    """One JSON object per output image: where it came from, and its group."""
    path = out_root / "provenance.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    return path


def _write_manifest(
    out_root: Path, cfg: PrepareConfig, report: PrepareReport, sources: Sequence[SourceSpec]
) -> Path:
    """Record exactly what was merged, so a result can be reproduced.

    A number from a training run means nothing without the dataset that
    produced it, and "the fire datasets" is not a dataset. This manifest plus
    ``provenance.jsonl`` plus the config hash is that dataset's identity.
    """
    manifest = {
        "tool": "prepare_datasets",
        "manifest_version": MANIFEST_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version.split()[0],
        "classes": list(report.classes),
        "class_order_source": "station.core.types.CLASSES",
        "config": {
            "path": str(cfg.path) if cfg.path else None,
            "data_root": str(cfg.data_root),
            "sha256": cfg.sha256,
            "seed": cfg.output.seed,
            "splits": cfg.output.splits,
            "copy_mode": cfg.output.copy_mode,
            "dedup": cfg.output.dedup,
            "max_negative_fraction": cfg.output.max_negative_fraction,
        },
        "split_policy": {
            "unit": "group",
            "rule": "every group is assigned to exactly one split; images are never split",
            "group_strategies": {s.name: s.grouping for s in sources},
            "filename_scheme": "wf_<group_id:06d>_<index:06d>",
            "why": (
                "consecutive video frames on opposite sides of a split turn validation into a "
                "memorisation test; the filename scheme makes the audit's filename family "
                "identical to the group this merge split by"
            ),
        },
        "totals": report.totals,
        "splits": report.split_stats,
        "sources": report.per_source,
        "warnings": report.warnings,
        "verify_with": "python3 tools/audit_dataset.py --data data.yaml --splits train,val",
    }
    path = out_root / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


# ------------------------------------------------------------------- config


def load_prepare_config(path: str | Path, data_root: str | Path | None = None) -> PrepareConfig:
    """Parse ``dataset_config.yaml``.

    Relative source paths resolve against the config file's directory, so a
    config can be moved with its data. A source path may contain the token
    ``{data_root}``, which is replaced by ``data_root`` (or the config's own
    ``data_root:`` key). That is what lets one committed config describe both a
    laptop, where the raw datasets sit under ``datasets/raw``, and Kaggle,
    where they are mounted read-only under ``/kaggle/input``.

    Args:
        path: Path to the YAML file.
        data_root: Overrides the config's ``data_root``.

    Returns:
        A validated :class:`PrepareConfig`.

    Raises:
        PrepareError: The file is unreadable, malformed, or describes a source
            this script refuses to import.
    """
    import yaml  # noqa: PLC0415 - PyYAML only needed when a config is loaded

    config_path = Path(path).resolve()
    try:
        text = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise PrepareError(f"cannot read config {config_path}: {exc}") from exc
    raw = yaml.safe_load(text) or {}
    if not isinstance(raw, dict):
        raise PrepareError(f"{config_path}: top level must be a mapping")

    base = config_path.parent
    root_token = str(data_root if data_root is not None else raw.get("data_root", "../datasets/raw"))
    resolved_data_root = _resolve(base, root_token)

    out_raw = dict(raw.get("output") or {})
    if "root" in out_raw:
        out_raw["root"] = _resolve(base, out_raw["root"])
    known_output = {f for f in OutputSpec.__slots__}
    unknown = set(out_raw) - known_output
    if unknown:
        raise PrepareError(f"unknown key(s) under output: {sorted(unknown)}")
    output = OutputSpec(**out_raw)

    sources: list[SourceSpec] = []
    known_source = {f for f in SourceSpec.__slots__}
    for entry in raw.get("sources") or ():
        if not isinstance(entry, dict):
            raise PrepareError(f"each source must be a mapping, got {type(entry).__name__}")
        item = dict(entry)
        unknown = set(item) - known_source
        if unknown:
            raise PrepareError(
                f"source {item.get('name', '?')!r}: unknown key(s) {sorted(unknown)}; "
                f"valid keys are {sorted(known_source)}"
            )
        if "name" not in item or "path" not in item:
            raise PrepareError(f"source entry needs 'name' and 'path': {item}")
        item["path"] = _resolve(base, str(item["path"]).replace("{data_root}", str(resolved_data_root)))
        if "class_names" in item and item["class_names"] is not None:
            item["class_names"] = tuple(str(v) for v in item["class_names"])
        sources.append(SourceSpec(**item))

    if not sources:
        raise PrepareError(f"{config_path}: no sources defined")
    names = [s.name for s in sources]
    duplicated = [n for n, c in Counter(names).items() if c > 1]
    if duplicated:
        raise PrepareError(f"duplicate source name(s): {duplicated}")

    return PrepareConfig(
        output=output,
        sources=sources,
        data_root=resolved_data_root,
        path=config_path,
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        version=int(raw.get("version", 1)),
    )


def _resolve(base: Path, value: Any) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else (base / path)


# ---------------------------------------------------------------------- cli


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Note what is *not* here: there is no flag to split by image, by percentage
    of files, or at random. Splitting is by group, always.
    """
    parser = argparse.ArgumentParser(
        prog="prepare_datasets.py",
        description=(
            "Merge the public fire/smoke datasets into one YOLO dataset with classes "
            "(fire, smoke), split by source video so that validation measures "
            "generalisation rather than memorisation."
        ),
        epilog=(
            "Always run tools/audit_dataset.py on the result before training. This script "
            "prevents the leak; the audit is the independent proof that it did."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "dataset_config.yaml",
        help="dataset_config.yaml (default: the one beside this script)",
    )
    parser.add_argument("--out", type=Path, help="override output.root from the config")
    parser.add_argument(
        "--data-root",
        type=Path,
        metavar="DIR",
        help="substituted for {data_root} in every source path (e.g. /kaggle/input)",
    )
    parser.add_argument("--seed", type=int, help="override output.seed")
    parser.add_argument("--copy-mode", choices=_VALID_COPY_MODES, help="override output.copy_mode")
    parser.add_argument("--only", action="append", default=[], metavar="SOURCE",
                        help="merge only this source; repeatable")
    parser.add_argument("--exclude", action="append", default=[], metavar="SOURCE",
                        help="skip this source; repeatable")
    parser.add_argument("--max-images", type=int, metavar="N",
                        help="cap every source at N images (quick smoke-test merges)")
    parser.add_argument("--dry-run", action="store_true",
                        help="scan, weight and split, report the result, write nothing")
    parser.add_argument("--force", action="store_true", help="overwrite a non-empty output directory")
    parser.add_argument("--skip-dedup", action="store_true",
                        help="skip content hashing (faster, but cross-source duplicates survive)")
    parser.add_argument("--audit", action="store_true",
                        help="run tools/audit_dataset.py on the result and propagate its exit code")
    parser.add_argument("--json", type=Path, metavar="FILE", help="also write the run report as JSON")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point.

    Returns:
        0 on success, 1 when the merge was refused, 2 when nothing was found.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    try:
        cfg = load_prepare_config(args.config, data_root=args.data_root)
        if args.out:
            cfg.output.root = args.out
        if args.seed is not None:
            cfg.output.seed = args.seed
        if args.copy_mode:
            cfg.output.copy_mode = args.copy_mode
        if args.skip_dedup:
            cfg.output.dedup = False
        if args.max_images is not None:
            for spec in cfg.sources:
                spec.max_images = min(args.max_images, spec.max_images or args.max_images)

        report = prepare(
            cfg,
            only=args.only,
            exclude=args.exclude,
            dry_run=args.dry_run,
            force=args.force,
        )
    except LeakageError as exc:
        log.error("REFUSED (leakage guard): %s", exc)
        return 1
    except PrepareError as exc:
        log.error("REFUSED: %s", exc)
        return 1
    except FileNotFoundError as exc:
        log.error("not found: %s", exc)
        return 2

    print()
    print(report.render())
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report.to_json(), indent=2) + "\n", encoding="utf-8")
        print(f"\nrun report written to {args.json}")

    if args.audit and not args.dry_run:
        return _run_audit(cfg.output.root)
    return 0


def _run_audit(out_root: Path) -> int:
    """Run the repository's dataset audit against the merge we just wrote."""
    import subprocess  # noqa: PLC0415 - only needed for --audit

    audit = Path(__file__).resolve().parent.parent / "tools" / "audit_dataset.py"
    if not audit.is_file():
        log.error("--audit: %s not found", audit)
        return 1
    print(f"\n--- {audit} ---\n", flush=True)
    completed = subprocess.run(
        [sys.executable, str(audit), "--data", str(out_root / "data.yaml")],
        check=False,
    )
    if completed.returncode != 0:
        log.error("audit failed (exit %d): do not train on this dataset", completed.returncode)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
