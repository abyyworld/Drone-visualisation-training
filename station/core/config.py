"""Configuration schema, loaded from YAML.

Central so that every subsystem agrees on key names, and so that defaults are
in one auditable place. ``config.example.yaml`` documents the same fields.

Defaults are chosen for the ground station described in ``docs/HARDWARE.md``:
a laptop with a discrete GPU, one drone feed, a handful of tablets on local
WiFi.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

__all__ = [
    "SourceConfig",
    "InferenceConfig",
    "TemporalConfig",
    "StreamConfig",
    "IncidentLogConfig",
    "Config",
    "load_config",
]


@dataclass(slots=True)
class SourceConfig:
    #: One of: file, rtsp, rtmp, hdmi, synthetic. Resolved by
    #: ``station.ingest.open_source`` -- the adapter seam that lets the whole
    #: pipeline be developed and regression-tested against a recorded file
    #: with no drone in the room.
    type: str = "file"
    uri: str = ""
    #: Seconds between reconnect attempts. A radio link drops; the pipeline
    #: must come back by itself rather than needing an operator at a laptop.
    reconnect_s: float = 2.0
    #: Cap the frames handed downstream. ``null`` means source rate.
    target_fps: float | None = None
    #: Loop a file source forever -- development and demo only.
    loop: bool = False


@dataclass(slots=True)
class InferenceConfig:
    weights: str = "models/yolo11s-fire.pt"
    imgsz: int = 640
    #: Deliberately low. Recall matters more than precision here: a false
    #: positive costs a glance, a false negative costs everything the system
    #: exists to prevent. The temporal filter, not this threshold, is what
    #: suppresses the resulting flicker.
    conf_threshold: float = 0.25
    iou_nms: float = 0.45
    device: str = "auto"
    half: bool = True
    #: Inference frames per second. Fire spreads slowly; 10 is usually ample
    #: and leaves GPU headroom. See open question 4 in docs/OPEN_QUESTIONS.md.
    max_fps: float = 10.0
    model_name: str = "yolo11s-fire"
    model_version: str = "0.0.0-untrained"


@dataclass(slots=True)
class TemporalConfig:
    """N-of-M persistence: the cheapest large accuracy win in the system."""

    #: Require detection in ``n`` of the last ``m`` frames before display.
    n: int = 3
    m: int = 5
    #: IoU above which a detection is considered the same track as last frame.
    iou_match: float = 0.30
    #: Frames a track survives unmatched before it is dropped.
    max_age: int = 5
    #: Emit tracks that have not yet reached ``n``, flagged as unconfirmed.
    #: Off by default: the point of the filter is to not show flicker.
    emit_unconfirmed: bool = False


@dataclass(slots=True)
class StreamConfig:
    bind: str = "0.0.0.0"
    port: int = 8443
    cert: str = "certs/station.crt"
    key: str = "certs/station.key"
    #: Empty by default and that is correct: on a LAN with no internet, host
    #: ICE candidates are sufficient and a STUN server would only add a
    #: timeout to every connection. See docs/DEPLOYMENT.md.
    ice_servers: list[str] = field(default_factory=list)
    #: Media-seconds after which the tablet stops drawing boxes entirely.
    #: Enforced on the tablet; sent here so it is configured in one place.
    max_overlay_age_s: float = 1.0
    #: Seconds of detection payloads the tablet buffers for pts matching.
    overlay_buffer_s: float = 3.0
    status_interval_s: float = 1.0
    #: No successful inference for this long -> state becomes ``stalled``.
    stall_after_s: float = 3.0


@dataclass(slots=True)
class IncidentLogConfig:
    dir: str = "incidents"
    #: Every frame's result is logged, including empty ones. An incident log
    #: that only recorded frames with detections would be a highlight reel,
    #: useless for the false-negative audit that has to happen before anyone
    #: relies on this. See docs/VALIDATION.md.
    enabled: bool = True
    record_video: bool = True
    #: fsync cadence in frames. 1 = never lose a frame to a crash; the write
    #: volume is small enough that this is affordable.
    flush_every: int = 1


@dataclass(slots=True)
class Config:
    station_name: str = "wildfire-watch station"
    source: SourceConfig = field(default_factory=SourceConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    stream: StreamConfig = field(default_factory=StreamConfig)
    incident_log: IncidentLogConfig = field(default_factory=IncidentLogConfig)


def _build(cls: type, raw: Any, path: str = ""):
    """Construct one config section from a mapping, rejecting unknown keys.

    Sections are flat -- every field is a scalar or a list -- so this does not
    recurse. Rejecting unknown keys rather than ignoring them is intentional: a
    typo in a config key on a field laptop would otherwise silently leave a
    safety-relevant default in place, and ``max_overlay_age_s`` misspelt means
    stale boxes over live video.
    """
    if raw is None:
        return cls()
    if not isinstance(raw, dict):
        raise TypeError(f"{path or cls.__name__}: expected a mapping, got {type(raw).__name__}")
    known = {f.name for f in fields(cls)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(
            f"unknown config key(s) {sorted(unknown)} under {path or 'root'}; "
            f"valid keys are {sorted(known)}"
        )
    return cls(**raw)


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> Config:
    """Load YAML config. With no path, returns defaults."""
    raw: dict[str, Any] = {}
    if path is not None:
        import yaml  # PyYAML; imported here so the core stays stdlib-only.

        text = Path(path).read_text(encoding="utf-8")
        raw = yaml.safe_load(text) or {}
        if not isinstance(raw, dict):
            raise TypeError(f"{path}: top level of the config must be a mapping")
    if overrides:
        for key, value in overrides.items():
            section, _, leaf = key.partition(".")
            if leaf:
                raw.setdefault(section, {})[leaf] = value
            else:
                raw[section] = value
    cfg = Config(
        station_name=raw.get("station_name", Config.station_name),
        source=_build(SourceConfig, raw.get("source"), "source"),
        inference=_build(InferenceConfig, raw.get("inference"), "inference"),
        temporal=_build(TemporalConfig, raw.get("temporal"), "temporal"),
        stream=_build(StreamConfig, raw.get("stream"), "stream"),
        incident_log=_build(IncidentLogConfig, raw.get("incident_log"), "incident_log"),
    )
    validate(cfg)
    return cfg


def validate(cfg: Config) -> None:
    """Fail loudly on configurations that are silently unsafe."""
    t = cfg.temporal
    if not 1 <= t.n <= t.m:
        raise ValueError(f"temporal: need 1 <= n <= m, got n={t.n}, m={t.m}")
    if not 0.0 < t.iou_match < 1.0:
        raise ValueError(f"temporal.iou_match must be in (0,1), got {t.iou_match}")
    if not 0.0 < cfg.inference.conf_threshold < 1.0:
        raise ValueError(f"inference.conf_threshold must be in (0,1), got {cfg.inference.conf_threshold}")
    if cfg.inference.max_fps <= 0:
        raise ValueError(f"inference.max_fps must be positive, got {cfg.inference.max_fps}")
    if cfg.stream.max_overlay_age_s <= 0:
        raise ValueError("stream.max_overlay_age_s must be positive; without it stale boxes render over live video")
    if cfg.stream.overlay_buffer_s < cfg.stream.max_overlay_age_s:
        raise ValueError(
            "stream.overlay_buffer_s must be >= max_overlay_age_s, otherwise the tablet "
            "discards payloads it would still be allowed to draw"
        )
