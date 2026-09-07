"""The incident log: what the model saw, frame by frame, kept forever.

Three pieces, one directory per incident::

    incidents/20260906-142231Z-flight-3/
        detections.jsonl   every inferred frame, empty results included
        events.jsonl       pipeline state changes and operator notes
        meta.json          station, config snapshot, model, source, counters
        video/             the frames the detections were computed from

* :class:`~station.incidentlog.writer.IncidentLogWriter` writes it, crash-safely
  and without ever raising into the frame loop.
* :class:`~station.incidentlog.recorder.VideoRecorder` records the source video
  on the same pts timeline the log refers to.
* :class:`~station.incidentlog.reader.IncidentLogReader` reads it back for
  after-action review, filtering, summarising and replaying.

This subsystem is invariant 4 -- *everything is logged* -- and it is the one
part of wildfire-watch whose value is entirely retrospective. Nobody looks at
it during an incident. Afterwards it is the only record of what the model
actually output, and the empty frames in it are not padding: they are the
denominator of every false-negative question anyone will ever ask about this
system, and the raw material of the validation set in ``docs/VALIDATION.md``.

Nothing here imports a codec at module level. The writer and reader are pure
stdlib; the recorder defers PyAV and OpenCV to the moment it opens a file, so
``import station.incidentlog`` works on a laptop with neither installed.

Example:
    >>> from station.core.config import Config
    >>> from station.core.types import FrameDetections
    >>> import tempfile
    >>> cfg = Config()
    >>> cfg.incident_log.dir = tempfile.mkdtemp()
    >>> with IncidentLogWriter.from_config(cfg, slug="example") as writer:
    ...     _ = writer.write(FrameDetections(frame_id=0, pts=0.0))
    ...     incident = writer.dir
    >>> IncidentLogReader(incident).summarise().frames_logged
    1
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - static tools see the real names
    from station.incidentlog.reader import (
        ClassSummary,
        IncidentLogError,
        IncidentLogReader,
        IncidentSummary,
        ReadStats,
        VideoPosition,
        format_frame,
    )
    from station.incidentlog.recorder import Segment, VideoRecorder
    from station.incidentlog.writer import (
        DETECTIONS_FILENAME,
        EMPTY_RESULT_NOTE,
        EVENTS_FILENAME,
        META_FILENAME,
        VIDEO_SUBDIR,
        IncidentCounters,
        IncidentLogWriter,
        new_incident_id,
        redact_uri,
        slugify,
    )

#: Which submodule each exported name lives in. Resolved on first access
#: rather than at import: eagerly importing ``reader`` here would make
#: ``python -m station.incidentlog.reader`` -- the documented after-action
#: command -- emit a runpy double-import warning on every run, and an operator
#: tool that warns about its own plumbing trains people to ignore warnings.
_EXPORTS: dict[str, str] = {
    "IncidentLogWriter": "writer",
    "IncidentCounters": "writer",
    "new_incident_id": "writer",
    "slugify": "writer",
    "redact_uri": "writer",
    "DETECTIONS_FILENAME": "writer",
    "EVENTS_FILENAME": "writer",
    "META_FILENAME": "writer",
    "VIDEO_SUBDIR": "writer",
    "EMPTY_RESULT_NOTE": "writer",
    "VideoRecorder": "recorder",
    "Segment": "recorder",
    "IncidentLogReader": "reader",
    "IncidentSummary": "reader",
    "ClassSummary": "reader",
    "ReadStats": "reader",
    "VideoPosition": "reader",
    "IncidentLogError": "reader",
    "format_frame": "reader",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Import an exported name from its submodule on first access (PEP 562).

    Args:
        name: The attribute being looked up on this package.

    Returns:
        The requested class or constant.

    Raises:
        AttributeError: ``name`` is not part of this package's public API.
    """
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(f"{__name__}.{module}"), name)
    globals()[name] = value  # cached: subsequent lookups skip __getattr__
    return value


def __dir__() -> list[str]:
    return list(__all__)
