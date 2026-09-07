#!/usr/bin/env python3
"""Evaluate a fire/smoke model with the weight on what it misses.

Detection benchmarks are built around mAP because mAP ranks models on academic
leaderboards. This system is not on a leaderboard. Its failure mode is a thin
plume on a bright sky that the model returns nothing for, on a frame an
operator then scrolls past. So this tool is organised around **recall** --
per class, at several confidence thresholds, split by box size and by
condition tag -- and its primary work product is the **miss list**: the
specific images the model failed to fire on, so a human can look at them.

mAP@0.5 is computed, and reported last, and treated with suspicion. Published
aerial RGB fire/smoke detectors land around 0.80 mAP@0.5 on honest held-out
data. A number much above that on your own split is far more likely to be
train/val leakage than skill -- run ``tools/audit_dataset.py`` before believing
it. This tool says so in its output rather than leaving it implicit.

Two ways in:

* ``--predictions preds.jsonl`` -- evaluate a file of saved predictions. Needs
  nothing but numpy, so evaluation runs on any laptop and in CI.
* ``--weights best.pt`` -- run ultralytics (lazily imported) over the split
  first, and optionally save the predictions for later re-scoring.

Prediction file format, one JSON object per line (a JSON array also works)::

    {"image": "images/val/clip3_0142.jpg",
     "detections": [{"cls": "fire", "conf": 0.83, "box": [0.41, 0.33, 0.49, 0.40]}]}

``box`` is ``[x1, y1, x2, y2]`` normalised 0..1, origin top-left -- the same
convention as ``station.core.types.BBox``, so predictions captured from a live
incident log can be scored here directly.

A note on what an empty prediction list means, because it governs how this
report reads: it records that this model, on this frame, returned nothing. It
is never evidence that the frame contained nothing. That is why the miss list
exists at all -- it is the population of frames where the two came apart.

Example:
    $ python3 tools/evaluate.py --data datasets/fire/data.yaml \\
          --predictions runs/val_preds.jsonl --conditions tags.json \\
          --miss-list runs/misses.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

_TOOLS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _TOOLS_DIR.parent
for _path in (str(_TOOLS_DIR), str(_REPO_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from audit_dataset import (  # noqa: E402 -- sys.path is set up immediately above.
    SIZE_BINS,
    Dataset,
    load_dataset,
    size_bucket,
)
from station.core.types import BBox  # noqa: E402 -- same geometry as the wire format.

__all__ = [
    "DEFAULT_CONF_THRESHOLDS",
    "GtBox",
    "GtImage",
    "Prediction",
    "Miss",
    "Counts",
    "Evaluation",
    "load_ground_truth",
    "load_predictions",
    "load_conditions",
    "evaluate",
    "main",
]

#: Swept low, because the operating point that matters is wherever recall
#: becomes acceptable, and the station's default (0.25) is already low by
#: detection-benchmark standards for exactly that reason.
DEFAULT_CONF_THRESHOLDS: tuple[float, ...] = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)

#: mAP@0.5 above this is treated as evidence of a leaky split rather than a
#: good model. Published aerial RGB detectors sit near 0.80.
SUSPICIOUS_MAP = 0.85


# --------------------------------------------------------------------- model


@dataclass(frozen=True, slots=True)
class GtBox:
    """One ground-truth box, with everything the breakdowns need."""

    cls: str
    box: BBox
    #: Area as a fraction of the frame; the basis of the size bucket.
    area_fraction: float
    bucket: str
    #: Approximate side length in pixels at the image's native resolution,
    #: ``None`` when the image size could not be read.
    px_side: float | None


@dataclass(frozen=True, slots=True)
class GtImage:
    """One evaluation image: its boxes and its condition tags."""

    key: str
    boxes: tuple[GtBox, ...]
    width: int
    height: int
    tags: tuple[str, ...] = ()
    path: Path | None = None


@dataclass(frozen=True, slots=True)
class Prediction:
    """One predicted box, in the wire geometry convention."""

    cls: str
    conf: float
    box: BBox


@dataclass(frozen=True, slots=True)
class Miss:
    """A ground-truth box the model did not produce a matching box for.

    ``best_conf`` and ``best_iou`` are the crucial fields. A miss with
    ``best_conf = 0.19`` is a threshold problem and is fixed by lowering the
    threshold. A miss with ``best_conf = None`` means nothing the model
    produced overlapped the target at all: it did not see it, and no threshold
    recovers it.
    """

    image: str
    cls: str
    bucket: str
    area_fraction: float
    px_side: float | None
    tags: tuple[str, ...]
    box: tuple[float, float, float, float]
    best_iou: float
    best_conf: float | None
    #: True when the model produced no boxes at all above the threshold on this
    #: image -- the silent frame that the whole design is built to survive.
    silent_image: bool

    def to_json(self) -> dict[str, Any]:
        return {
            "image": self.image,
            "cls": self.cls,
            "bucket": self.bucket,
            "area_fraction": round(self.area_fraction, 6),
            "px_side": round(self.px_side, 1) if self.px_side else None,
            "tags": list(self.tags),
            "box": [round(v, 4) for v in self.box],
            "best_iou": round(self.best_iou, 3),
            "best_conf": round(self.best_conf, 3) if self.best_conf is not None else None,
            "silent_image": self.silent_image,
        }


@dataclass(slots=True)
class Counts:
    """True positives, false negatives and false positives for one slice."""

    tp: int = 0
    fn: int = 0
    fp: int = 0

    @property
    def support(self) -> int:
        """Ground-truth boxes in this slice."""
        return self.tp + self.fn

    @property
    def recall(self) -> float | None:
        """Fraction of ground-truth boxes matched, or ``None`` with no support."""
        return self.tp / self.support if self.support else None

    @property
    def precision(self) -> float | None:
        predicted = self.tp + self.fp
        return self.tp / predicted if predicted else None

    def to_json(self) -> dict[str, Any]:
        return {
            "tp": self.tp,
            "fn": self.fn,
            "fp": self.fp,
            "support": self.support,
            "recall": None if self.recall is None else round(self.recall, 4),
            "precision": None if self.precision is None else round(self.precision, 4),
        }


@dataclass(slots=True)
class ThresholdResult:
    """All the slices at one confidence threshold."""

    conf: float
    overall: Counts = field(default_factory=Counts)
    per_class: dict[str, Counts] = field(default_factory=dict)
    per_bucket: dict[str, Counts] = field(default_factory=dict)
    per_class_bucket: dict[str, dict[str, Counts]] = field(default_factory=dict)
    per_tag: dict[str, Counts] = field(default_factory=dict)
    #: Images with ground truth where the model produced nothing above ``conf``.
    silent_images: int = 0
    images_with_gt: int = 0

    def to_json(self) -> dict[str, Any]:
        return {
            "conf": self.conf,
            "overall": self.overall.to_json(),
            "per_class": {k: v.to_json() for k, v in sorted(self.per_class.items())},
            "per_bucket": {k: v.to_json() for k, v in self.per_bucket.items()},
            "per_class_bucket": {
                cls: {b: c.to_json() for b, c in buckets.items()}
                for cls, buckets in sorted(self.per_class_bucket.items())
            },
            "per_tag": {k: v.to_json() for k, v in sorted(self.per_tag.items())},
            "silent_images": self.silent_images,
            "images_with_gt": self.images_with_gt,
        }


@dataclass(slots=True)
class Evaluation:
    """The full result, renderable as text or JSON."""

    iou_threshold: float
    operating_conf: float
    thresholds: list[ThresholdResult]
    misses: list[Miss]
    ap_per_class: dict[str, float]
    map50: float
    n_images: int
    n_gt: int
    n_pred: int
    unmatched_prediction_files: int
    images_without_predictions: int
    notes: list[str] = field(default_factory=list)

    def at(self, conf: float) -> ThresholdResult:
        """The result closest to a given confidence threshold."""
        return min(self.thresholds, key=lambda t: abs(t.conf - conf))

    def to_json(self) -> dict[str, Any]:
        return {
            "tool": "evaluate",
            "version": 1,
            "iou_threshold": self.iou_threshold,
            "operating_conf": self.operating_conf,
            "n_images": self.n_images,
            "n_ground_truth_boxes": self.n_gt,
            "n_predicted_boxes": self.n_pred,
            "images_without_predictions": self.images_without_predictions,
            "thresholds": [t.to_json() for t in self.thresholds],
            "ap50_per_class": {k: round(v, 4) for k, v in sorted(self.ap_per_class.items())},
            "map50": round(self.map50, 4),
            "map50_is_suspicious": self.map50 > SUSPICIOUS_MAP,
            "misses": [m.to_json() for m in self.misses],
            "notes": list(self.notes),
        }

    # -- rendering ----------------------------------------------------------

    def render(self) -> str:
        """The report, ordered so the most important number is read first."""
        out: list[str] = []
        out.append("=" * 78)
        out.append("wildfire-watch model evaluation -- weighted toward false negatives")
        out.append("=" * 78)
        out.append(
            f"images {self.n_images} | ground-truth boxes {self.n_gt} | "
            f"predicted boxes {self.n_pred} | IoU match {self.iou_threshold:g}"
        )
        if self.images_without_predictions:
            out.append(
                f"{self.images_without_predictions} image(s) had no prediction record at all; "
                "they are scored as producing nothing."
            )
        for note in self.notes:
            out.append(f"note: {note}")
        out.append("")

        out.append("RECALL BY CONFIDENCE THRESHOLD")
        out.append("-" * 78)
        classes = sorted({c for t in self.thresholds for c in t.per_class})
        header = f"{'conf':>6}  {'recall':>7}  {'found/total':>13}"
        header += "".join(f"  {c[:9]:>9}" for c in classes)
        header += f"  {'silent':>7}  {'prec':>6}"
        out.append(header)
        for t in self.thresholds:
            recall = t.overall.recall
            row = f"{t.conf:>6.2f}  " + (f"{recall:>7.3f}" if recall is not None else f"{'n/a':>7}")
            row += f"  {t.overall.tp:>6}/{t.overall.support:<6}"
            for cls in classes:
                counts = t.per_class.get(cls)
                value = counts.recall if counts else None
                row += f"  {value:>9.3f}" if value is not None else f"  {'n/a':>9}"
            row += f"  {t.silent_images:>7}"
            precision = t.overall.precision
            row += f"  {precision:>6.3f}" if precision is not None else f"  {'n/a':>6}"
            marker = "  <- operating point" if abs(t.conf - self.operating_conf) < 1e-9 else ""
            out.append(row + marker)
        out.append("")
        out.append(
            "'silent' counts images that hold at least one labelled target and for which the"
        )
        out.append(
            "model produced no box at all. Those frames are the ones this system is designed"
        )
        out.append("to survive: the overlay stays empty and the operator keeps watching video.")
        out.append("")

        op = self.at(self.operating_conf)
        out.append(f"RECALL BY BOX SIZE  (at conf {op.conf:g})")
        out.append("-" * 78)
        out.append(f"{'bucket':<8} {'area fraction':>16}  {'recall':>7}  {'found/total':>13}")
        for name, lo, hi in SIZE_BINS:
            counts = op.per_bucket.get(name, Counts())
            recall = counts.recall
            value = f"{recall:>7.3f}" if recall is not None else f"{'n/a':>7}"
            out.append(
                f"{name:<8} {f'[{lo:.3f}, {hi:.3f})':>16}  {value}  "
                f"{counts.tp:>6}/{counts.support:<6}"
            )
        out.append(
            "Tiny is the bucket that matters: a distant plume is the detection that buys the"
        )
        out.append(
            "most time, and it is the first thing lost to downscaling, compression and haze."
        )
        if classes:
            out.append("")
            out.append(f"{'class':<10}" + "".join(f"{b:>12}" for b, _, _ in SIZE_BINS))
            for cls in classes:
                buckets = op.per_class_bucket.get(cls, {})
                row = f"{cls:<10}"
                for name, _, _ in SIZE_BINS:
                    counts = buckets.get(name, Counts())
                    if counts.support:
                        row += f"{counts.recall:>8.3f}({counts.support:>2})"
                    else:
                        row += f"{'-':>12}"
                out.append(row)
        out.append("")

        if op.per_tag:
            out.append(f"RECALL BY CONDITION TAG  (at conf {op.conf:g})")
            out.append("-" * 78)
            out.append(f"{'tag':<20} {'recall':>7}  {'found/total':>13}")
            for tag, counts in sorted(op.per_tag.items(), key=lambda kv: (kv[1].recall or 0.0)):
                recall = counts.recall
                value = f"{recall:>7.3f}" if recall is not None else f"{'n/a':>7}"
                out.append(f"{tag:<20} {value}  {counts.tp:>6}/{counts.support:<6}")
            out.append(
                "Sorted worst first. These are the conditions the model degrades in, and the"
            )
            out.append("conditions the operator has to be told about.")
            out.append("")

        out.append(f"MISS LIST  ({len(self.misses)} targets unmatched at conf {op.conf:g})")
        out.append("-" * 78)
        if not self.misses:
            out.append("Every labelled target was matched at this threshold.")
        else:
            unseen = [m for m in self.misses if m.best_conf is None]
            recoverable = [m for m in self.misses if m.best_conf is not None]
            out.append(
                f"{len(unseen)} target(s) had nothing overlapping them at any confidence -- no"
            )
            out.append(
                "threshold change recovers these; they need training data or a bigger imgsz."
            )
            out.append(
                f"{len(recoverable)} target(s) were localised but scored below the threshold."
            )
            if recoverable:
                best = max(m.best_conf or 0.0 for m in recoverable)
                out.append(
                    f"  highest confidence on a sub-threshold match: {best:.3f} -- lowering the"
                )
                out.append("  operating point to just under it would recover at least one.")
            out.append("")
            out.append(f"{'image':<44} {'class':<7}{'bucket':<8}{'iou':>5}{'conf':>7}  tags")
            for miss in self.misses[:40]:
                conf = f"{miss.best_conf:>7.3f}" if miss.best_conf is not None else f"{'-':>7}"
                name = miss.image if len(miss.image) <= 43 else "..." + miss.image[-40:]
                out.append(
                    f"{name:<44} {miss.cls:<7}{miss.bucket:<8}"
                    f"{miss.best_iou:>5.2f}{conf}  {','.join(miss.tags)}"
                )
            if len(self.misses) > 40:
                out.append(f"... and {len(self.misses) - 40} more; use --miss-list for the full set.")
        out.append("")

        out.append("mAP@0.5  (secondary -- read the recall table first)")
        out.append("-" * 78)
        for cls, ap in sorted(self.ap_per_class.items()):
            out.append(f"  AP@0.5 {cls:<10} {ap:.4f}")
        out.append(f"  mAP@0.5             {self.map50:.4f}")
        if self.map50 > SUSPICIOUS_MAP:
            out.append("")
            out.append(
                f"  This is above {SUSPICIOUS_MAP:.2f}, and that is a warning rather than an"
            )
            out.append(
                "  achievement. Published aerial RGB fire/smoke detectors land around 0.80"
            )
            out.append(
                "  mAP@0.5 on honest held-out data. A markedly higher number on your own split"
            )
            out.append(
                "  is much more often train/val leakage -- consecutive video frames on both"
            )
            out.append(
                "  sides of the split -- than genuine skill. Run tools/audit_dataset.py and"
            )
            out.append("  re-split before quoting this figure to anybody.")
        else:
            out.append(
                "  mAP mixes precision into a single number and hides which targets were"
            )
            out.append(
                "  missed. It is here for comparison with the literature, not for deciding"
            )
            out.append("  whether this model is fit to fly.")
        out.append("=" * 78)
        # Trailing spaces come from the fixed-width columns; strip them so the
        # report diffs cleanly when it is committed alongside a model card.
        return "\n".join(line.rstrip() for line in out)


# ----------------------------------------------------------------- loading


def load_ground_truth(
    dataset: Dataset,
    split: str = "val",
    conditions: dict[str, tuple[str, ...]] | None = None,
) -> list[GtImage]:
    """Turn a loaded dataset split into evaluation ground truth.

    Args:
        dataset: A dataset from :func:`audit_dataset.load_dataset`.
        split: Which split to evaluate.
        conditions: Optional map from image key (relative path, filename or
            stem) to condition tags.

    Returns:
        One :class:`GtImage` per image in the split, including images with no
        boxes -- they are what false positives are measured on.
    """
    conditions = conditions or {}
    out: list[GtImage] = []
    for sample in dataset.samples:
        if sample.split != split:
            continue
        key = sample.rel(dataset.root)
        boxes: list[GtBox] = []
        for box in sample.boxes:
            x1, y1, x2, y2 = box.xyxy()
            area = box.area
            px_side: float | None = None
            if sample.width and sample.height:
                px_side = math.sqrt(area * sample.width * sample.height)
            boxes.append(
                GtBox(
                    cls=dataset.class_name(box.cls_id),
                    box=BBox(x1, y1, x2, y2),
                    area_fraction=area,
                    bucket=size_bucket(area),
                    px_side=px_side,
                )
            )
        tags = _lookup_tags(conditions, key, sample.image)
        out.append(
            GtImage(
                key=key,
                boxes=tuple(boxes),
                width=sample.width,
                height=sample.height,
                tags=tags,
                path=sample.image,
            )
        )
    return out


def _lookup_tags(
    conditions: dict[str, tuple[str, ...]], key: str, path: Path
) -> tuple[str, ...]:
    """Match an image against the conditions map by path, name, then stem."""
    for candidate in (key, path.as_posix(), path.name, path.stem):
        if candidate in conditions:
            return conditions[candidate]
    return ()


def load_conditions(path: Path) -> dict[str, tuple[str, ...]]:
    """Load condition tags: ``{"img.jpg": ["night", "thin_smoke"]}``.

    Also accepts a list of ``{"image": ..., "tags": [...]}`` objects, and a
    ``{"tags": {...}}`` wrapper, because all three turn up in hand-written
    annotation exports.

    Args:
        path: JSON file of tags.

    Returns:
        A map from image key to a tuple of tags. Keys may be relative paths,
        filenames or stems; lookup tries each in that order.

    Raises:
        ValueError: The file is not one of the accepted shapes.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "tags" in raw and isinstance(raw["tags"], dict):
        raw = raw["tags"]
    out: dict[str, tuple[str, ...]] = {}
    if isinstance(raw, dict):
        for key, value in raw.items():
            tags = [value] if isinstance(value, str) else list(value)
            out[str(key)] = tuple(str(t) for t in tags)
        return out
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict) or "image" not in item:
                raise ValueError(f"{path}: list entries need an 'image' key")
            tags = item.get("tags", ())
            tags = [tags] if isinstance(tags, str) else list(tags)
            out[str(item["image"])] = tuple(str(t) for t in tags)
        return out
    raise ValueError(f"{path}: expected a mapping or a list of objects")


def load_predictions(path: Path) -> dict[str, list[Prediction]]:
    """Load predictions from JSONL (one object per line) or a JSON array.

    Args:
        path: The predictions file.

    Returns:
        A map from the record's ``image`` value to its predictions.

    Raises:
        ValueError: A record is malformed. Malformed prediction records are
            fatal rather than skipped: a silently dropped record would show up
            as a false negative and be indistinguishable from a real miss,
            which is the one confusion this tool exists to prevent.
    """
    text = path.read_text(encoding="utf-8").strip()
    records: list[Any]
    if text.startswith("["):
        records = json.loads(text)
    else:
        records = []
        for lineno, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: {exc}") from exc

    out: dict[str, list[Prediction]] = {}
    for index, record in enumerate(records):
        if not isinstance(record, dict) or "image" not in record:
            raise ValueError(f"{path}: record {index} has no 'image' key")
        key = str(record["image"])
        preds: list[Prediction] = []
        # 'detections' matches the wire protocol; 'predictions' is accepted
        # because that is what a hand-rolled export usually calls it.
        raw_dets = record.get("detections", record.get("predictions", ()))
        for det in raw_dets:
            try:
                box = det["box"]
                preds.append(
                    Prediction(
                        cls=str(det["cls"]),
                        conf=float(det["conf"]),
                        box=BBox(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                    )
                )
            except (KeyError, TypeError, ValueError, IndexError) as exc:
                raise ValueError(f"{path}: record {index} ({key}) has a bad detection: {exc}") from exc
        out[key] = preds
    return out


def _resolve_predictions(
    gt_images: Sequence[GtImage], predictions: dict[str, list[Prediction]]
) -> tuple[dict[str, list[Prediction]], int, int]:
    """Attach prediction records to ground-truth images.

    Keys are matched on the relative path first, then the filename, then the
    stem, so predictions exported from a different working directory still
    line up.

    Returns:
        ``(by_gt_key, unmatched_prediction_records, images_without_predictions)``.
    """
    by_name: dict[str, str] = {}
    by_stem: dict[str, str] = {}
    for key in predictions:
        name = Path(key).name
        by_name.setdefault(name, key)
        by_stem.setdefault(Path(key).stem, key)

    resolved: dict[str, list[Prediction]] = {}
    used: set[str] = set()
    missing = 0
    for image in gt_images:
        source: str | None = None
        for candidate in (image.key, image.path.name if image.path else "", Path(image.key).name):
            if candidate and candidate in predictions:
                source = candidate
                break
        if source is None:
            source = by_name.get(Path(image.key).name) or by_stem.get(Path(image.key).stem)
        if source is None:
            resolved[image.key] = []
            missing += 1
        else:
            resolved[image.key] = predictions[source]
            used.add(source)
    return resolved, len(set(predictions) - used), missing


# --------------------------------------------------------------- evaluation


def _greedy_match(
    gt: Sequence[GtBox], preds: Sequence[Prediction], iou_threshold: float
) -> tuple[list[int], list[bool]]:
    """Match predictions to ground truth, highest confidence first.

    Args:
        gt: Ground-truth boxes for one image.
        preds: Predictions for the same image, any order.
        iou_threshold: Minimum IoU for a match.

    Returns:
        ``(gt_match, pred_matched)`` where ``gt_match[i]`` is the index of the
        prediction matched to ground-truth box ``i`` or ``-1``, and
        ``pred_matched[j]`` says whether prediction ``j`` was used.

    Greedy by confidence is the COCO/VOC convention. It is not optimal
    assignment, but it is what every published number was computed with, and
    an evaluation that is not comparable to the literature is worth less than
    one that is slightly suboptimal.
    """
    gt_match = [-1] * len(gt)
    pred_matched = [False] * len(preds)
    order = sorted(range(len(preds)), key=lambda j: -preds[j].conf)
    for j in order:
        pred = preds[j]
        best_i, best_iou = -1, iou_threshold
        for i, truth in enumerate(gt):
            if gt_match[i] != -1 or truth.cls != pred.cls:
                continue
            iou = truth.box.iou(pred.box)
            if iou >= best_iou:
                best_i, best_iou = i, iou
        if best_i >= 0:
            gt_match[best_i] = j
            pred_matched[j] = True
    return gt_match, pred_matched


def _best_overlap(truth: GtBox, preds: Sequence[Prediction]) -> tuple[float, float | None]:
    """Best IoU any same-class prediction achieves, and its confidence.

    Ignores the confidence threshold entirely. This is what separates "the
    model saw it and scored it 0.19" from "the model has no idea it is there",
    and those two misses need completely different remedies.
    """
    best_iou, best_conf = 0.0, None
    for pred in preds:
        if pred.cls != truth.cls:
            continue
        iou = truth.box.iou(pred.box)
        if iou > best_iou:
            best_iou, best_conf = iou, pred.conf
    return best_iou, best_conf


def _average_precision(
    gt_images: Sequence[GtImage],
    predictions: dict[str, list[Prediction]],
    cls: str,
    iou_threshold: float,
) -> float:
    """VOC all-point-interpolated AP for one class.

    All-point rather than the 11-point interpolation: it is the modern
    convention (and what ultralytics reports), so the number is comparable
    with the papers this project is measured against.
    """
    n_gt = sum(1 for image in gt_images for b in image.boxes if b.cls == cls)
    if n_gt == 0:
        return float("nan")

    scored: list[tuple[float, bool]] = []
    for image in gt_images:
        truths = [b for b in image.boxes if b.cls == cls]
        preds = [p for p in predictions.get(image.key, ()) if p.cls == cls]
        claimed = [False] * len(truths)
        for pred in sorted(preds, key=lambda p: -p.conf):
            best_i, best_iou = -1, iou_threshold
            for i, truth in enumerate(truths):
                if claimed[i]:
                    continue
                iou = truth.box.iou(pred.box)
                if iou >= best_iou:
                    best_i, best_iou = i, iou
            if best_i >= 0:
                claimed[best_i] = True
                scored.append((pred.conf, True))
            else:
                scored.append((pred.conf, False))

    if not scored:
        return 0.0
    scored.sort(key=lambda item: -item[0])
    hits = np.array([1.0 if ok else 0.0 for _, ok in scored])
    tp = np.cumsum(hits)
    fp = np.cumsum(1.0 - hits)
    recall = tp / n_gt
    precision = tp / np.maximum(tp + fp, 1e-12)
    # Make precision monotonically decreasing from the right, then integrate.
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    recall = np.concatenate(([0.0], recall))
    precision = np.concatenate(([precision[0]], precision))
    return float(np.sum(np.diff(recall) * precision[1:]))


def evaluate(
    gt_images: Sequence[GtImage],
    predictions: dict[str, list[Prediction]],
    *,
    iou_threshold: float = 0.5,
    conf_thresholds: Sequence[float] = DEFAULT_CONF_THRESHOLDS,
    operating_conf: float = 0.25,
    unmatched_prediction_files: int = 0,
    images_without_predictions: int = 0,
    notes: Sequence[str] = (),
) -> Evaluation:
    """Score predictions against ground truth.

    Args:
        gt_images: Ground truth, from :func:`load_ground_truth`.
        predictions: Predictions keyed by :attr:`GtImage.key`, already
            resolved against the ground-truth keys.
        iou_threshold: IoU above which a prediction matches a target.
        conf_thresholds: Confidence thresholds to sweep for the recall table.
        operating_conf: The threshold the miss list and the breakdowns are
            reported at -- normally the station's ``inference.conf_threshold``.
        unmatched_prediction_files: Prediction records that matched no image,
            passed through into the report.
        images_without_predictions: Ground-truth images with no record.
        notes: Extra lines for the report header.

    Returns:
        A populated :class:`Evaluation`.
    """
    conf_list = sorted({round(float(c), 6) for c in conf_thresholds} | {round(float(operating_conf), 6)})
    results: list[ThresholdResult] = []
    misses: list[Miss] = []
    all_classes = sorted({b.cls for image in gt_images for b in image.boxes})

    for conf in conf_list:
        result = ThresholdResult(conf=conf)
        for cls in all_classes:
            result.per_class[cls] = Counts()
            result.per_class_bucket[cls] = {name: Counts() for name, _, _ in SIZE_BINS}
        for name, _, _ in SIZE_BINS:
            result.per_bucket[name] = Counts()

        for image in gt_images:
            preds = [p for p in predictions.get(image.key, ()) if p.conf >= conf]
            gt_match, pred_matched = _greedy_match(image.boxes, preds, iou_threshold)
            if image.boxes:
                result.images_with_gt += 1
                if not preds:
                    result.silent_images += 1

            for i, truth in enumerate(image.boxes):
                hit = gt_match[i] != -1
                slices = [
                    result.overall,
                    result.per_class.setdefault(truth.cls, Counts()),
                    result.per_bucket.setdefault(truth.bucket, Counts()),
                    result.per_class_bucket.setdefault(truth.cls, {}).setdefault(
                        truth.bucket, Counts()
                    ),
                ]
                for tag in image.tags:
                    slices.append(result.per_tag.setdefault(tag, Counts()))
                for slice_counts in slices:
                    if hit:
                        slice_counts.tp += 1
                    else:
                        slice_counts.fn += 1

                if not hit and abs(conf - operating_conf) < 1e-9:
                    best_iou, best_conf = _best_overlap(truth, predictions.get(image.key, ()))
                    misses.append(
                        Miss(
                            image=image.key,
                            cls=truth.cls,
                            bucket=truth.bucket,
                            area_fraction=truth.area_fraction,
                            px_side=truth.px_side,
                            tags=image.tags,
                            box=truth.box.as_tuple(),
                            best_iou=best_iou,
                            best_conf=best_conf,
                            silent_image=not preds,
                        )
                    )

            for j, pred in enumerate(preds):
                if pred_matched[j]:
                    continue
                result.overall.fp += 1
                result.per_class.setdefault(pred.cls, Counts()).fp += 1
                for tag in image.tags:
                    result.per_tag.setdefault(tag, Counts()).fp += 1
        results.append(result)

    ap = {cls: _average_precision(gt_images, predictions, cls, iou_threshold) for cls in all_classes}
    finite = [v for v in ap.values() if not math.isnan(v)]
    map50 = float(sum(finite) / len(finite)) if finite else 0.0

    # Unseen targets first, then by ascending best confidence: the top of the
    # list is the work queue for whoever is collecting more training data.
    misses.sort(key=lambda m: (m.best_conf is not None, m.best_conf or 0.0, m.image))

    return Evaluation(
        iou_threshold=iou_threshold,
        operating_conf=operating_conf,
        thresholds=results,
        misses=misses,
        ap_per_class=ap,
        map50=map50,
        n_images=len(gt_images),
        n_gt=sum(len(image.boxes) for image in gt_images),
        n_pred=sum(len(v) for v in predictions.values()),
        unmatched_prediction_files=unmatched_prediction_files,
        images_without_predictions=images_without_predictions,
        notes=list(notes),
    )


# ------------------------------------------------------------- ultralytics


def predict_with_weights(
    weights: Path,
    gt_images: Sequence[GtImage],
    *,
    imgsz: int = 640,
    conf: float = 0.01,
    iou_nms: float = 0.45,
    device: str = "auto",
    batch: int = 8,
) -> dict[str, list[Prediction]]:
    """Run an ultralytics model over the evaluation images.

    ``conf`` defaults far below any sensible operating point on purpose: the
    recall sweep needs the low-confidence tail, and a model run at 0.25 cannot
    be re-scored at 0.10 afterwards. Predictions are produced once, cheaply,
    and then swept in memory.

    Args:
        weights: Path to a ``.pt`` (or exported) model.
        gt_images: Images to run over; each must carry a real path.
        imgsz: Inference size. Must match what the model was trained at, or
            small targets change size relative to the receptive field and the
            recall-by-size table stops meaning what it says.
        conf: Confidence floor for prediction capture.
        iou_nms: NMS IoU.
        device: ``auto``, or an explicit torch device.
        batch: Images per predict call.

    Returns:
        Predictions keyed by :attr:`GtImage.key`.

    Raises:
        RuntimeError: ultralytics is not installed, or the model will not load.
            Never swallowed: an evaluation that silently produced no
            predictions would report zero recall and look like a terrible
            model rather than a missing package.
    """
    try:
        from ultralytics import YOLO  # noqa: PLC0415 -- lazy by design.
    except ImportError as exc:
        raise RuntimeError(
            "ultralytics is not installed, so --weights cannot be used.\n"
            "  pip install ultralytics\n"
            "To evaluate without it, export predictions elsewhere and pass --predictions."
        ) from exc

    if not weights.is_file():
        raise RuntimeError(f"weights file not found: {weights}")

    sys.path.insert(0, str(_REPO_ROOT))
    from station.inference.runner import resolve_device  # noqa: PLC0415

    model = YOLO(str(weights))
    resolved_device = resolve_device(device)
    names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))

    out: dict[str, list[Prediction]] = {}
    paths = [(image, image.path) for image in gt_images if image.path is not None]
    for start in range(0, len(paths), batch):
        chunk = paths[start : start + batch]
        results = model.predict(
            [str(p) for _, p in chunk],
            imgsz=imgsz,
            conf=conf,
            iou=iou_nms,
            device=resolved_device,
            verbose=False,
        )
        for (image, _), result in zip(chunk, results):
            preds: list[Prediction] = []
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                out[image.key] = preds
                continue
            height, width = result.orig_shape
            for xyxy, confidence, cls_index in zip(
                boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist()
            ):
                preds.append(
                    Prediction(
                        cls=str(names.get(int(cls_index), int(cls_index))).strip().lower(),
                        conf=float(confidence),
                        box=BBox.from_xyxy_pixels(*xyxy, width=width, height=height),
                    )
                )
            out[image.key] = preds
    return out


def save_predictions(path: Path, predictions: dict[str, list[Prediction]]) -> None:
    """Write predictions as JSONL so an expensive run can be re-scored free."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for key, preds in sorted(predictions.items()):
            record = {
                "image": key,
                "detections": [
                    {"cls": p.cls, "conf": round(p.conf, 4), "box": p.box.to_wire()} for p in preds
                ],
            }
            fh.write(json.dumps(record, separators=(",", ":")) + "\n")


# ---------------------------------------------------------------------- cli


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        prog="evaluate.py",
        description=(
            "Evaluate a fire/smoke model, weighted toward false negatives: recall per class, "
            "per box size and per condition tag, plus the explicit list of missed targets. "
            "mAP@0.5 is reported as a secondary, suspect number."
        ),
        epilog=(
            "The miss list is the work product. Look at the images in it. A model's recall "
            "number is an average over a split; the frames it went quiet on are the thing "
            "that decides whether this is safe to put in front of a crew."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    data = parser.add_argument_group("data")
    data.add_argument("--data", metavar="DATA_YAML", type=Path, help="Dataset data.yaml.")
    data.add_argument(
        "--dataset-dir", metavar="DIR", type=Path, help="Dataset root, when there is no data.yaml."
    )
    data.add_argument("--split", default="val", help="Split to evaluate (default: val).")
    data.add_argument(
        "--conditions",
        metavar="FILE",
        type=Path,
        help="JSON of per-image condition tags, e.g. night / thin_smoke / canopy.",
    )

    preds = parser.add_argument_group("predictions (choose one)")
    preds.add_argument(
        "--predictions",
        metavar="FILE",
        type=Path,
        help="JSONL or JSON array of saved predictions; needs no model and no GPU.",
    )
    preds.add_argument(
        "--weights", metavar="PT", type=Path, help="Run this ultralytics model over the split first."
    )
    preds.add_argument(
        "--save-predictions",
        metavar="FILE",
        type=Path,
        help="Write the predictions from --weights here, so re-scoring needs no GPU.",
    )
    preds.add_argument("--imgsz", type=int, default=640, help="Inference size for --weights (default: 640).")
    preds.add_argument("--device", default="auto", help="Torch device for --weights (default: auto).")
    preds.add_argument(
        "--capture-conf",
        type=float,
        default=0.01,
        help="Confidence floor when capturing predictions; must sit below every swept "
        "threshold or the recall sweep is truncated (default: 0.01).",
    )
    preds.add_argument("--iou-nms", type=float, default=0.45, help="NMS IoU for --weights (default: 0.45).")

    scoring = parser.add_argument_group("scoring")
    scoring.add_argument("--iou", type=float, default=0.5, help="IoU for a match (default: 0.5).")
    scoring.add_argument(
        "--conf-thresholds",
        default=",".join(f"{c:g}" for c in DEFAULT_CONF_THRESHOLDS),
        help="Comma-separated confidence thresholds for the recall sweep.",
    )
    scoring.add_argument(
        "--operating-conf",
        type=float,
        default=0.25,
        help="Threshold the miss list and breakdowns use; match inference.conf_threshold "
        "(default: 0.25).",
    )
    scoring.add_argument(
        "--min-recall",
        type=float,
        default=None,
        metavar="R",
        help="Exit non-zero if overall recall at the operating point is below this.",
    )
    scoring.add_argument(
        "--min-tiny-recall",
        type=float,
        default=None,
        metavar="R",
        help="Exit non-zero if recall on tiny boxes at the operating point is below this.",
    )

    output = parser.add_argument_group("output")
    output.add_argument("--json", metavar="FILE", type=Path, help="Write the full result as JSON.")
    output.add_argument(
        "--miss-list",
        metavar="FILE",
        type=Path,
        help="Write every missed target here (.json for structured, anything else for text).",
    )
    output.add_argument("--quiet", action="store_true", help="Suppress the text report.")
    return parser


def _write_miss_list(path: Path, evaluation: Evaluation) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".json":
        path.write_text(
            json.dumps(
                {
                    "operating_conf": evaluation.operating_conf,
                    "iou_threshold": evaluation.iou_threshold,
                    "count": len(evaluation.misses),
                    "misses": [m.to_json() for m in evaluation.misses],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return
    lines = [
        f"# targets unmatched at conf {evaluation.operating_conf:g}, IoU {evaluation.iou_threshold:g}",
        "# image\tclass\tbucket\tbest_iou\tbest_conf\ttags",
    ]
    for miss in evaluation.misses:
        conf = "-" if miss.best_conf is None else f"{miss.best_conf:.3f}"
        lines.append(
            f"{miss.image}\t{miss.cls}\t{miss.bucket}\t{miss.best_iou:.3f}\t{conf}\t"
            f"{','.join(miss.tags)}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point.

    Returns:
        0 on success, 1 when a ``--min-recall`` gate fails, 2 on bad input.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.data is None and args.dataset_dir is None:
        parser.error("one of --data or --dataset-dir is required")
    if args.predictions is None and args.weights is None:
        parser.error("one of --predictions or --weights is required")

    try:
        thresholds = tuple(float(v) for v in args.conf_thresholds.split(",") if v.strip())
    except ValueError:
        parser.error("--conf-thresholds must be a comma-separated list of numbers")

    try:
        dataset = load_dataset(
            data_yaml=args.data,
            dataset_dir=args.dataset_dir,
            splits=(args.split,),
            hash_content=False,
        )
        conditions = load_conditions(args.conditions) if args.conditions else None
        gt_images = load_ground_truth(dataset, args.split, conditions)
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"evaluate: {exc}", file=sys.stderr)
        return 2

    if not gt_images:
        print(f"evaluate: split {args.split!r} contains no images", file=sys.stderr)
        return 2

    notes: list[str] = []
    try:
        if args.weights is not None:
            raw_predictions = predict_with_weights(
                args.weights,
                gt_images,
                imgsz=args.imgsz,
                conf=args.capture_conf,
                iou_nms=args.iou_nms,
                device=args.device,
            )
            notes.append(f"predictions generated from {args.weights} at imgsz {args.imgsz}")
            if args.save_predictions:
                save_predictions(args.save_predictions, raw_predictions)
                notes.append(f"predictions saved to {args.save_predictions}")
        else:
            raw_predictions = load_predictions(args.predictions)
            notes.append(f"predictions read from {args.predictions}")
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"evaluate: {exc}", file=sys.stderr)
        return 2

    if min(thresholds) < args.capture_conf and args.weights is not None:
        notes.append(
            f"swept thresholds go below the capture floor {args.capture_conf:g}; "
            "recall at those points is truncated, not measured"
        )

    resolved, unmatched, missing = _resolve_predictions(gt_images, raw_predictions)
    if unmatched:
        notes.append(f"{unmatched} prediction record(s) matched no image in this split")

    evaluation = evaluate(
        gt_images,
        resolved,
        iou_threshold=args.iou,
        conf_thresholds=thresholds,
        operating_conf=args.operating_conf,
        unmatched_prediction_files=unmatched,
        images_without_predictions=missing,
        notes=notes,
    )

    if not args.quiet:
        print(evaluation.render())
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(evaluation.to_json(), indent=2), encoding="utf-8")
        if not args.quiet:
            print(f"\nJSON report written to {args.json}")
    if args.miss_list:
        _write_miss_list(args.miss_list, evaluation)
        if not args.quiet:
            print(f"Miss list written to {args.miss_list}")

    status = 0
    operating = evaluation.at(args.operating_conf)
    if args.min_recall is not None:
        recall = operating.overall.recall
        if recall is None or recall < args.min_recall:
            print(
                f"\nGATE FAILED: overall recall {recall if recall is not None else float('nan'):.3f} "
                f"< required {args.min_recall:.3f} at conf {operating.conf:g}",
                file=sys.stderr,
            )
            status = 1
    if args.min_tiny_recall is not None:
        tiny = operating.per_bucket.get("tiny", Counts())
        recall = tiny.recall
        if recall is None or recall < args.min_tiny_recall:
            shown = "n/a (no tiny targets in this split)" if recall is None else f"{recall:.3f}"
            print(
                f"\nGATE FAILED: tiny-box recall {shown} < required {args.min_tiny_recall:.3f} "
                f"at conf {operating.conf:g}",
                file=sys.stderr,
            )
            status = 1
    return status


if __name__ == "__main__":
    raise SystemExit(main())
