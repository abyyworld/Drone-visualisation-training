"""Wire contract shared by the ground station and the tablet app.

Every value that crosses a process boundary in wildfire-watch is defined here,
and only here. The station serialises these types onto the WebRTC data channel
and into the incident log; ``app/js/wire.js`` parses the same shapes on the
tablet. If you change a field, change it in both places and bump
:data:`WIRE_VERSION`.

Two rules constrain this module, and they are safety requirements rather than
style preferences (see ``docs/SAFETY.md``):

1. **There is no field anywhere in this protocol that asserts the absence of
   fire.** No ``all_clear``, no ``is_safe``, no ``fire_count: 0`` intended to be
   read as reassurance. A frame with no detections serialises to an empty
   ``detections`` list and means exactly one thing: this model, on this frame,
   returned nothing. It does not mean the scene is clear.
2. **Pipeline liveness is reported, and is not a claim about the world.**
   :class:`PipelineStatus` says whether the model is still looking. A stalled
   pipeline must be visibly stalled on the tablet rather than silently showing
   the last boxes it had.

Geometry convention: boxes are normalised to the frame, ``0.0..1.0``, as
``(x1, y1, x2, y2)`` with the origin at top-left. Normalised coordinates
survive every rescale between the model's input size, the encoded stream and
whatever size the tablet happens to render the video at, which is why nothing
in this protocol is ever expressed in pixels.

Time convention: ``pts`` is the frame's presentation timestamp in seconds on
the *media* timeline -- the same timeline the browser exposes as
``video.currentTime`` -- and it is the only field that lets the tablet draw a
box over the frame it was computed from. ``wall_time`` is a human/forensic
clock for the incident log and must never be used for overlay alignment.
See ``docs/CONTRACT.md`` for the synchronisation algorithm.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Iterator, Sequence

__all__ = [
    "WIRE_VERSION",
    "MSG_DETECTIONS",
    "MSG_STATUS",
    "CLASS_FIRE",
    "CLASS_SMOKE",
    "CLASSES",
    "PipelineState",
    "BBox",
    "Detection",
    "ModelInfo",
    "FrameDetections",
    "PipelineStatus",
    "utc_now_iso",
    "parse_message",
]

#: Bumped on any incompatible change to the shapes below. The tablet refuses to
#: render a payload whose version it does not understand rather than guessing.
WIRE_VERSION = 1

MSG_DETECTIONS = "detections"
MSG_STATUS = "status"

CLASS_FIRE = "fire"
CLASS_SMOKE = "smoke"
#: Ordered, and the order is load-bearing: it is the class index order the
#: trained weights emit. ``tools/export.py`` asserts the exported model agrees.
CLASSES: tuple[str, ...] = (CLASS_FIRE, CLASS_SMOKE)


class PipelineState:
    """Liveness of the station pipeline. Never a statement about the scene."""

    STARTING = "starting"
    RUNNING = "running"
    #: Running, but not keeping up (frames dropped, or inference slower than
    #: the configured floor). Overlay is still trustworthy, just sparser.
    DEGRADED = "degraded"
    #: No successful inference within the stall timeout. The tablet must stop
    #: drawing boxes and say so -- stale boxes over live video are the single
    #: most dangerous failure this system has.
    STALLED = "stalled"
    STOPPED = "stopped"

    ALL = (STARTING, RUNNING, DEGRADED, STALLED, STOPPED)


def utc_now_iso() -> str:
    """Wall-clock stamp for the incident log, RFC 3339 with a ``Z`` suffix."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _finite(name: str, value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number, got {value!r}") from exc
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return out


@dataclass(frozen=True, slots=True)
class BBox:
    """A normalised, top-left-origin box: ``0.0 <= x1 < x2 <= 1.0``."""

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        for name in ("x1", "y1", "x2", "y2"):
            object.__setattr__(self, name, _finite(name, getattr(self, name)))
        if self.x2 < self.x1 or self.y2 < self.y1:
            raise ValueError(f"box corners are inverted: {self.as_tuple()}")

    @classmethod
    def from_xyxy_pixels(cls, x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> "BBox":
        """Convert model-space pixels to the normalised wire form.

        Clamped to the frame: a model may predict a box that runs off the edge
        of the image, and a fire at the edge of frame is exactly the case we
        least want to drop.
        """
        if width <= 0 or height <= 0:
            raise ValueError(f"frame size must be positive, got {width}x{height}")
        lo_x, hi_x = sorted((x1 / width, x2 / width))
        lo_y, hi_y = sorted((y1 / height, y2 / height))
        return cls(
            x1=min(max(lo_x, 0.0), 1.0),
            y1=min(max(lo_y, 0.0), 1.0),
            x2=min(max(hi_x, 0.0), 1.0),
            y2=min(max(hi_y, 0.0), 1.0),
        )

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def centre(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def iou(self, other: "BBox") -> float:
        """Intersection over union -- the association metric for tracking."""
        ix1, iy1 = max(self.x1, other.x1), max(self.y1, other.y1)
        ix2, iy2 = min(self.x2, other.x2), min(self.y2, other.y2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        union = self.area + other.area - inter
        return inter / union if union > 0.0 else 0.0

    def to_wire(self) -> list[float]:
        # Rounded to 4dp: ~0.2 px at 1080p, and it roughly halves payload size.
        return [round(v, 4) for v in self.as_tuple()]

    @classmethod
    def from_wire(cls, raw: Sequence[float]) -> "BBox":
        if len(raw) != 4:
            raise ValueError(f"box must have 4 elements, got {len(raw)}")
        return cls(*(float(v) for v in raw))


@dataclass(frozen=True, slots=True)
class Detection:
    """One detected region on one frame, after the temporal filter."""

    cls: str
    conf: float
    box: BBox
    #: Stable across frames while the same region keeps being detected, so the
    #: tablet can draw continuous boxes instead of flickering new ones.
    track_id: int | None = None
    #: How many of the last ``m`` frames this track was detected in. Surfaced
    #: to the operator: a 3-of-5 box and a 5-of-5 box are different evidence.
    persisted: int = 1
    #: Media timestamp at which this track was first *detected* (not first
    #: displayed). Lets after-action review find the earliest visible frame.
    first_seen_pts: float | None = None

    def __post_init__(self) -> None:
        if not self.cls:
            raise ValueError("detection class must be non-empty")
        object.__setattr__(self, "conf", _finite("conf", self.conf))
        if not 0.0 <= self.conf <= 1.0:
            raise ValueError(f"conf must be in 0..1, got {self.conf}")
        if self.persisted < 1:
            raise ValueError(f"persisted must be >= 1, got {self.persisted}")

    def to_wire(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "cls": self.cls,
            "conf": round(self.conf, 3),
            "box": self.box.to_wire(),
            "persisted": self.persisted,
        }
        if self.track_id is not None:
            out["track"] = self.track_id
        if self.first_seen_pts is not None:
            out["first_seen_pts"] = round(self.first_seen_pts, 3)
        return out

    @classmethod
    def from_wire(cls, raw: dict[str, Any]) -> "Detection":
        return cls(
            cls=str(raw["cls"]),
            conf=float(raw["conf"]),
            box=BBox.from_wire(raw["box"]),
            track_id=raw.get("track"),
            persisted=int(raw.get("persisted", 1)),
            first_seen_pts=raw.get("first_seen_pts"),
        )


@dataclass(frozen=True, slots=True)
class ModelInfo:
    """Identifies exactly what produced a detection.

    Logged with every frame. Without this an incident log cannot be replayed
    against the model that generated it, which makes after-action review and
    the false-negative audit in ``docs/VALIDATION.md`` impossible.
    """

    name: str
    version: str
    classes: tuple[str, ...] = CLASSES
    #: Short hash of the weights file. The definitive answer to "which model
    #: was actually running", when ``version`` has been forgotten to be bumped.
    weights_sha: str | None = None
    imgsz: int | None = None
    conf_threshold: float | None = None

    def to_wire(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "name": self.name,
            "version": self.version,
            "classes": list(self.classes),
        }
        for key in ("weights_sha", "imgsz", "conf_threshold"):
            value = getattr(self, key)
            if value is not None:
                out[key] = value
        return out

    @classmethod
    def from_wire(cls, raw: dict[str, Any]) -> "ModelInfo":
        return cls(
            name=str(raw.get("name", "unknown")),
            version=str(raw.get("version", "unknown")),
            classes=tuple(raw.get("classes", CLASSES)),
            weights_sha=raw.get("weights_sha"),
            imgsz=raw.get("imgsz"),
            conf_threshold=raw.get("conf_threshold"),
        )


@dataclass(frozen=True, slots=True)
class FrameDetections:
    """Everything the model concluded about one frame.

    An empty :attr:`detections` list is a normal, frequent, unremarkable
    result. It is recorded faithfully and rendered as nothing at all. It is
    never rendered as reassurance.
    """

    frame_id: int
    #: Media-timeline seconds. The overlay alignment key -- see module docstring.
    pts: float
    wall_time: str = field(default_factory=utc_now_iso)
    detections: tuple[Detection, ...] = ()
    model: ModelInfo | None = None
    inference_ms: float | None = None
    #: 90 kHz RTP timestamp of the frame, when the stream layer can supply it.
    #: This is the exact alignment key: paired with the browser's
    #: ``requestVideoFrameCallback().rtpTimestamp`` it removes overlay drift
    #: entirely, rather than estimating it. Null when unavailable (e.g. file
    #: source, or a relay that does not expose RTP), in which case the tablet
    #: falls back to the pts-offset estimator.
    rtp_ts: int | None = None
    source_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "pts", _finite("pts", self.pts))
        object.__setattr__(self, "detections", tuple(self.detections))

    def __iter__(self) -> Iterator[Detection]:
        return iter(self.detections)

    def __len__(self) -> int:
        return len(self.detections)

    def of_class(self, cls: str) -> tuple[Detection, ...]:
        return tuple(d for d in self.detections if d.cls == cls)

    @property
    def max_conf(self) -> float | None:
        """Highest confidence on this frame, or ``None`` when there are no
        detections. Deliberately ``None`` and not ``0.0``: zero is a number
        that invites being rendered as a gauge reading 'nothing here'."""
        return max((d.conf for d in self.detections), default=None)

    def to_wire(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "v": WIRE_VERSION,
            "type": MSG_DETECTIONS,
            "frame_id": self.frame_id,
            "pts": round(self.pts, 4),
            "wall_time": self.wall_time,
            "detections": [d.to_wire() for d in self.detections],
        }
        if self.model is not None:
            out["model"] = self.model.to_wire()
        if self.inference_ms is not None:
            out["inference_ms"] = round(self.inference_ms, 2)
        if self.rtp_ts is not None:
            out["rtp_ts"] = self.rtp_ts
        if self.source_id is not None:
            out["source_id"] = self.source_id
        return out

    def to_json(self) -> str:
        return json.dumps(self.to_wire(), separators=(",", ":"))

    @classmethod
    def from_wire(cls, raw: dict[str, Any]) -> "FrameDetections":
        return cls(
            frame_id=int(raw["frame_id"]),
            pts=float(raw["pts"]),
            wall_time=str(raw.get("wall_time", "")),
            detections=tuple(Detection.from_wire(d) for d in raw.get("detections", ())),
            model=ModelInfo.from_wire(raw["model"]) if raw.get("model") else None,
            inference_ms=raw.get("inference_ms"),
            rtp_ts=raw.get("rtp_ts"),
            source_id=raw.get("source_id"),
        )


@dataclass(frozen=True, slots=True)
class PipelineStatus:
    """Heartbeat: is the model still looking, and how well is it keeping up.

    Sent on a fixed cadence independent of detections, so that silence on the
    data channel is itself diagnosable. Contains no claim about the scene.
    """

    state: str
    wall_time: str = field(default_factory=utc_now_iso)
    source: str | None = None
    model: ModelInfo | None = None
    #: Frames per second arriving from the source, and frames per second the
    #: model actually processed. The gap is the sampling rate, and the operator
    #: is entitled to see it: it bounds how briefly a fire could appear and
    #: still be missed entirely.
    source_fps: float | None = None
    inference_fps: float | None = None
    #: Media pts of the most recent completed inference. The tablet compares
    #: this with playback position to detect a frozen model behind live video.
    last_inference_pts: float | None = None
    last_inference_wall_time: str | None = None
    dropped_frames: int = 0
    uptime_s: float | None = None
    #: Media pts of the first frame published on this stream session. Seeds the
    #: tablet's pts <-> ``video.currentTime`` offset estimate.
    stream_start_pts: float | None = None
    #: Free-text, operator-facing, for genuinely degraded conditions the
    #: pipeline can self-detect (e.g. "source reconnecting"). Never used to
    #: report an absence of fire.
    note: str | None = None

    def __post_init__(self) -> None:
        if self.state not in PipelineState.ALL:
            raise ValueError(f"unknown pipeline state {self.state!r}; expected one of {PipelineState.ALL}")

    def to_wire(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "v": WIRE_VERSION,
            "type": MSG_STATUS,
            "state": self.state,
            "wall_time": self.wall_time,
            "dropped_frames": self.dropped_frames,
        }
        optional = (
            "source",
            "source_fps",
            "inference_fps",
            "last_inference_pts",
            "last_inference_wall_time",
            "uptime_s",
            "stream_start_pts",
            "note",
        )
        for key in optional:
            value = getattr(self, key)
            if value is not None:
                out[key] = round(value, 3) if isinstance(value, float) else value
        if self.model is not None:
            out["model"] = self.model.to_wire()
        return out

    def to_json(self) -> str:
        return json.dumps(self.to_wire(), separators=(",", ":"))

    @classmethod
    def from_wire(cls, raw: dict[str, Any]) -> "PipelineStatus":
        return cls(
            state=str(raw["state"]),
            wall_time=str(raw.get("wall_time", "")),
            source=raw.get("source"),
            model=ModelInfo.from_wire(raw["model"]) if raw.get("model") else None,
            source_fps=raw.get("source_fps"),
            inference_fps=raw.get("inference_fps"),
            last_inference_pts=raw.get("last_inference_pts"),
            last_inference_wall_time=raw.get("last_inference_wall_time"),
            dropped_frames=int(raw.get("dropped_frames", 0)),
            uptime_s=raw.get("uptime_s"),
            stream_start_pts=raw.get("stream_start_pts"),
            note=raw.get("note"),
        )


def parse_message(raw: str | bytes | dict[str, Any]) -> FrameDetections | PipelineStatus:
    """Parse one data-channel message, rejecting versions we do not understand.

    Refusing an unknown ``v`` is deliberate. A tablet running an old build
    against a new station would otherwise silently drop fields it cannot see --
    and a silently partial overlay is indistinguishable from a quiet scene.
    """
    payload = raw if isinstance(raw, dict) else json.loads(raw)
    version = payload.get("v")
    if version != WIRE_VERSION:
        raise ValueError(f"unsupported wire version {version!r}; this build speaks v{WIRE_VERSION}")
    kind = payload.get("type")
    if kind == MSG_DETECTIONS:
        return FrameDetections.from_wire(payload)
    if kind == MSG_STATUS:
        return PipelineStatus.from_wire(payload)
    raise ValueError(f"unknown message type {kind!r}")
