"""YOLO label parsing that handles the mixed box/polygon format in this dataset.

The Roboflow export stores `healthy` as segmentation polygons and every defect class as
5-value bounding boxes. A naive 5-field parse silently mis-reads the polygons and reports
thousands of bogus out-of-bounds boxes, so every tool here goes through `parse_label_file`.

Pure standard library on purpose - these run on a laptop, on Kaggle, and in CI without
installing anything.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# Roboflow names every file `<source>_jpg.rf.<md5>.jpg`. The part before `_jpg.rf.` is the
# source image; the same source appears up to 3 times because of offline augmentation.
ROBOFLOW_NAME = re.compile(r"^(?P<source>.+?)_jpg\.rf\.(?P<hash>[0-9a-f]+)$")

# Source names are `<family><id>`, e.g. `Healthy_Train774`, `Areial_Healthy913`, `1044`.
FAMILY_ID = re.compile(r"^(?P<family>.*?)(?P<id>\d+)$")


@dataclass(frozen=True)
class Box:
    """A normalised xywh bounding box with its class id."""

    cls: int
    xc: float
    yc: float
    w: float
    h: float

    @property
    def area(self) -> float:
        return self.w * self.h

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        return (
            self.xc - self.w / 2,
            self.yc - self.h / 2,
            self.xc + self.w / 2,
            self.yc + self.h / 2,
        )

    def to_line(self) -> str:
        return f"{self.cls} {self.xc:.6f} {self.yc:.6f} {self.w:.6f} {self.h:.6f}"


def parse_label_line(line: str) -> Box | None:
    """Parse one YOLO label line, converting a polygon to its bounding box.

    Returns None for blank lines. Raises ValueError on genuinely malformed input.
    """
    parts = line.split()
    if not parts:
        return None

    cls = int(float(parts[0]))
    coords = [float(v) for v in parts[1:]]

    if len(coords) == 4:
        xc, yc, w, h = coords
    elif len(coords) >= 6 and len(coords) % 2 == 0:
        # Polygon: x1 y1 x2 y2 ... -> tight bounding box.
        xs, ys = coords[0::2], coords[1::2]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        xc, yc, w, h = (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0
    else:
        raise ValueError(f"unrecognised label geometry ({len(coords)} coords): {line!r}")

    # Clamp to the image. Rotation augmentation can push a corner a hair outside.
    x0, y0 = max(0.0, xc - w / 2), max(0.0, yc - h / 2)
    x1, y1 = min(1.0, xc + w / 2), min(1.0, yc + h / 2)
    return Box(cls, (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0)


def parse_label_file(path: Path) -> list[Box]:
    """Read a YOLO label file. An empty file is a valid background image."""
    boxes = []
    for lineno, line in enumerate(path.read_text().splitlines(), 1):
        try:
            box = parse_label_line(line)
        except ValueError as exc:
            raise ValueError(f"{path}:{lineno}: {exc}") from exc
        if box is not None:
            boxes.append(box)
    return boxes


def is_polygon_file(path: Path) -> bool:
    """True if any line in the file is a polygon rather than a 5-value box."""
    return any(len(line.split()) > 5 for line in path.read_text().splitlines() if line.strip())


def source_name(stem: str) -> str:
    """`Healthy_Train774_jpg.rf.7682c8...` -> `Healthy_Train774`.

    Falls back to the stem unchanged for non-Roboflow filenames.
    """
    match = ROBOFLOW_NAME.match(stem)
    return match.group("source") if match else stem


def split_family_id(source: str) -> tuple[str, int | None]:
    """`Healthy_Train774` -> `("Healthy_Train", 774)`; `1044` -> `("", 1044)`.

    Returns `(source, None)` when there is no trailing integer to key on.
    """
    match = FAMILY_ID.match(source)
    if not match:
        return source, None
    return match.group("family"), int(match.group("id"))


def iou(a: Box, b: Box) -> float:
    """Intersection-over-union of two normalised boxes."""
    ax0, ay0, ax1, ay1 = a.xyxy
    bx0, by0, bx1, by1 = b.xyxy

    ix = min(ax1, bx1) - max(ax0, bx0)
    iy = min(ay1, by1) - max(ay0, by0)
    if ix <= 0 or iy <= 0:
        return 0.0

    inter = ix * iy
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def iter_split(root: Path, split: str):
    """Yield `(image_path, label_path, boxes)` for every labelled image in a split.

    Skips images with no corresponding label file, which YOLO would treat as unlabelled
    rather than as a background.
    """
    images = root / split / "images"
    labels = root / split / "labels"
    if not images.is_dir():
        return

    for image in sorted(images.iterdir()):
        if image.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        label = labels / f"{image.stem}.txt"
        if not label.exists():
            continue
        yield image, label, parse_label_file(label)
