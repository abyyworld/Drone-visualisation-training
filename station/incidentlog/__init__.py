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
    EVENTS_FILENAME,
    META_FILENAME,
    VIDEO_SUBDIR,
    EMPTY_RESULT_NOTE,
    IncidentCounters,
    IncidentLogWriter,
    new_incident_id,
    redact_uri,
    slugify,
)

__all__ = [
    "IncidentLogWriter",
    "IncidentCounters",
    "VideoRecorder",
    "Segment",
    "IncidentLogReader",
    "IncidentSummary",
    "ClassSummary",
    "ReadStats",
    "VideoPosition",
    "IncidentLogError",
    "format_frame",
    "new_incident_id",
    "slugify",
    "redact_uri",
    "DETECTIONS_FILENAME",
    "EVENTS_FILENAME",
    "META_FILENAME",
    "VIDEO_SUBDIR",
    "EMPTY_RESULT_NOTE",
]
