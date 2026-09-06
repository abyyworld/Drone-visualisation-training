"""N-of-M temporal persistence with IoU track association.

This is the accuracy lever in wildfire-watch, and it costs nothing to run and
nothing to train. ``inference.conf_threshold`` is deliberately low (0.25) so
that faint smoke is not thresholded away; the price of that recall is a stream
of single-frame false positives. Requiring a region to be detected in ``n`` of
the last ``m`` frames *before it is ever drawn* removes almost all of them,
because uncorrelated noise does not persist in the same place across frames
while a real fire does. Nothing here touches recall on the model side: a region
the model never proposes is never tracked, and this filter cannot invent it.

What this module is not: it is not a detector, and a track is not a fire. It
narrows what the overlay says "look here" about. Its output is still only ever
a positive assertion; the absence of a track is never emitted as anything at
all, in keeping with ``station/core/safety.py``.

Pure stdlib. No numpy, no model, no I/O -- so it imports and unit-tests on any
machine, and runs in tens of microseconds per frame at realistic track counts.
"""

from __future__ import annotations

import dataclasses
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Iterable, Sequence

from station.core.config import TemporalConfig
from station.core.types import BBox, Detection, FrameDetections

__all__ = ["Track", "TemporalFilter"]


@dataclass(slots=True)
class Track:
    """One candidate region followed across frames.

    A track is an association hypothesis, not a conclusion. It exists from the
    first frame a region is proposed on and dies ``max_age`` unmatched frames
    later, whether or not it was ever confirmed.
    """

    track_id: int
    cls: str
    #: Most recently *observed* box. Matching is done against this, never
    #: against the box the track was created with: a smoke plume that grows to
    #: fill the frame has near-zero IoU with its own first box after a few
    #: seconds, but high IoU with its box one frame ago. Chaining frame-to-frame
    #: is what lets a growing region keep one identity.
    box: BBox
    conf: float
    first_seen_pts: float
    last_seen_pts: float
    #: Rolling hit/miss window, newest last, capped at ``m`` by ``maxlen``.
    window: Deque[bool] = field(default_factory=deque)
    #: Frames since the last match. Zero on a frame where the track was seen.
    age: int = 0
    #: Total matched frames over the track's whole life (diagnostics only --
    #: confirmation looks at the window, not at this).
    hits: int = 1
    #: Latches true and never clears while the track lives. See
    #: :meth:`TemporalFilter.update` for why unconfirming would be worse.
    confirmed: bool = False

    @property
    def hits_in_window(self) -> int:
        """Matched frames among the last ``m`` -- the ``n`` in "n of m"."""
        return sum(self.window)


class TemporalFilter:
    """Associates per-frame detections into tracks and gates them on N-of-M.

    Feed every inferred frame to :meth:`update` in media-timeline order,
    including the frames that produced nothing: a miss is evidence about a
    track and the window cannot advance without it.

    Example:
        >>> from station.core.config import TemporalConfig
        >>> filt = TemporalFilter(TemporalConfig(n=2, m=3))
        >>> box = BBox(0.4, 0.4, 0.5, 0.5)
        >>> filt.update([Detection(cls="fire", conf=0.6, box=box)], pts=0.0)
        ()
        >>> emitted = filt.update([Detection(cls="fire", conf=0.7, box=box)], pts=0.1)
        >>> emitted[0].track_id, emitted[0].persisted
        (1, 2)
    """

    def __init__(self, cfg: TemporalConfig) -> None:
        """Create a filter.

        Args:
            cfg: Persistence and association parameters. ``load_config``
                already validates ``1 <= n <= m`` and ``0 < iou_match < 1``;
                they are re-checked here because a ``TemporalConfig`` can also
                be constructed directly, e.g. in tests.

        Raises:
            ValueError: If the configuration could not gate anything sensibly.
        """
        if not 1 <= cfg.n <= cfg.m:
            raise ValueError(f"temporal: need 1 <= n <= m, got n={cfg.n}, m={cfg.m}")
        if not 0.0 < cfg.iou_match < 1.0:
            raise ValueError(f"temporal.iou_match must be in (0,1), got {cfg.iou_match}")
        if cfg.max_age < 0:
            raise ValueError(f"temporal.max_age must be >= 0, got {cfg.max_age}")
        self.cfg = cfg
        self._tracks: list[Track] = []
        #: Never reset, not even across a stream discontinuity. Re-using a
        #: track id after a seek would let the tablet draw a line between two
        #: unrelated regions and call it one continuous fire.
        self._next_id: int = 1
        self._last_pts: float | None = None
        self._frames_seen: int = 0

    # ------------------------------------------------------------------ API

    @property
    def tracks(self) -> tuple[Track, ...]:
        """Live tracks, in creation order. Includes unconfirmed ones."""
        return tuple(self._tracks)

    @property
    def frames_seen(self) -> int:
        """Frames fed to :meth:`update` since construction or last reset."""
        return self._frames_seen

    def reset(self) -> None:
        """Drop all tracks, e.g. after a source reconnect or a file loop.

        Track ids keep counting up: the point of a reset is that nothing
        before it may be associated with anything after it.
        """
        self._tracks.clear()
        self._last_pts = None
        self._frames_seen = 0

    def filter_frame(self, frame: FrameDetections) -> FrameDetections:
        """Return ``frame`` with its raw detections replaced by emitted tracks.

        Args:
            frame: One frame's raw model output.

        Returns:
            A copy of ``frame`` carrying the filtered detections. Every other
            field -- ``pts``, ``rtp_ts``, ``model``, ``inference_ms`` -- is
            preserved, because the incident log and the overlay both key off
            them and this filter is not entitled to alter provenance.
        """
        emitted = self.update(frame.detections, frame.pts)
        return dataclasses.replace(frame, detections=emitted)

    def update(self, detections: Sequence[Detection] | Iterable[Detection], pts: float) -> tuple[Detection, ...]:
        """Advance one frame and return the detections that should be drawn.

        Args:
            detections: Raw detections for this frame, in any order. May be
                empty -- an empty frame is information, and skipping the call
                would freeze every track's window instead of ageing it.
            pts: Media-timeline presentation timestamp, in seconds. Must be
                non-decreasing; a decrease is treated as a stream
                discontinuity and resets the tracker.

        Returns:
            Detections to display, each carrying ``track_id``, ``persisted``
            (hits within the last ``m`` frames) and ``first_seen_pts``. Empty
            when nothing is confirmed -- which means only that nothing was
            confirmed.
        """
        dets = list(detections)

        # A backwards pts means the media timeline restarted (file loop,
        # reconnect, seek). Associating across that would match a box from the
        # end of one pass to a box from the start of the next.
        if self._last_pts is not None and pts < self._last_pts:
            self.reset()
        self._last_pts = pts
        self._frames_seen += 1

        matches = self._associate(dets)

        # Track *ids*, not list indices: the list is rebuilt below when dead
        # tracks are pruned, and an index into the old list would then point at
        # a different track -- i.e. at somebody else's fire.
        seen_now: set[int] = set()
        matched_dets: set[int] = set()
        for track_idx, det_idx in matches.items():
            track = self._tracks[track_idx]
            self._apply_hit(track, dets[det_idx], pts)
            seen_now.add(track.track_id)
            matched_dets.add(det_idx)

        for idx, track in enumerate(self._tracks):
            if idx not in matches:
                self._apply_miss(track)

        # New tracks are appended after ageing, so they are not aged on the
        # frame that created them, and so creation order == list order.
        for det_idx, det in enumerate(dets):
            if det_idx not in matched_dets:
                fresh = self._new_track(det, pts)
                self._tracks.append(fresh)
                seen_now.add(fresh.track_id)

        # age > max_age, so max_age=5 means five unmatched frames survived and
        # the sixth kills the track.
        self._tracks = [t for t in self._tracks if t.age <= self.cfg.max_age]

        return self._emit(seen_now)

    # ------------------------------------------------------------- internals

    def _associate(self, dets: Sequence[Detection]) -> dict[int, int]:
        """Greedy highest-IoU-first matching of tracks to detections.

        Returns:
            ``{track_index: detection_index}``. Each track takes at most one
            detection and each detection at most one track, which is what
            keeps two fires burning side by side from collapsing into one
            track: the higher-IoU pair claims its partner, and the loser is
            left unmatched and becomes (or stays) a track of its own.

        Matching is confined to a single class, deliberately. Fire and smoke
        are the co-located pair by construction -- flame sits at the base of
        its own plume, and the smoke box usually contains the fire box -- so a
        high fire/smoke IoU is the normal geometry of one event, not evidence
        that the two boxes are the same object. Allowing the association would
        (a) let smoke hits confirm a fire track, so the ``persisted`` count on
        screen would no longer be the count of frames that class was actually
        seen, and (b) let a track's label flip frame to frame with the
        classifier's whim. A box that alternates between "fire" and "smoke" is
        unreadable, and a fire being reported under the smoke label understates
        what the operator is looking at.
        """
        candidates: list[tuple[float, int, int]] = []
        for track_idx, track in enumerate(self._tracks):
            for det_idx, det in enumerate(dets):
                if det.cls != track.cls:
                    continue
                iou = track.box.iou(det.box)
                if iou >= self.cfg.iou_match:
                    candidates.append((iou, track_idx, det_idx))

        # Ties broken by track index then detection index: older tracks win,
        # which biases towards keeping an existing identity rather than
        # minting a new one. Also makes the whole filter deterministic, which
        # matters because incident logs get replayed and compared.
        candidates.sort(key=lambda c: (-c[0], c[1], c[2]))

        matches: dict[int, int] = {}
        taken_dets: set[int] = set()
        for _iou, track_idx, det_idx in candidates:
            if track_idx in matches or det_idx in taken_dets:
                continue
            matches[track_idx] = det_idx
            taken_dets.add(det_idx)
        return matches

    def _new_track(self, det: Detection, pts: float) -> Track:
        track = Track(
            track_id=self._next_id,
            cls=det.cls,
            box=det.box,
            conf=det.conf,
            first_seen_pts=pts,
            last_seen_pts=pts,
            window=deque([True], maxlen=self.cfg.m),
        )
        self._next_id += 1
        track.confirmed = track.hits_in_window >= self.cfg.n  # true only when n == 1
        return track

    def _apply_hit(self, track: Track, det: Detection, pts: float) -> None:
        track.box = det.box
        track.conf = det.conf
        track.last_seen_pts = pts
        track.age = 0
        track.hits += 1
        track.window.append(True)
        if track.hits_in_window >= self.cfg.n:
            track.confirmed = True

    def _apply_miss(self, track: Track) -> None:
        track.age += 1
        track.window.append(False)
        # Note what is missing: confirmation is never revoked. See _emit.

    def _emit(self, seen_now: set[int]) -> tuple[Detection, ...]:
        """Build the display detections for this frame.

        A confirmed track keeps being emitted for as long as it is alive, even
        on frames where the model did not re-propose it. That is the
        anti-flicker rule and it is the whole point of the module: at the
        n-of-m boundary a marginal region drops in and out of the model's
        output every other frame, and a box that blinks at 5 Hz is worse than
        useless -- the operator's eye is drawn to the blinking rather than to
        the scene, and it reads as instrument fault rather than as fire. So
        once the evidence bar has been cleared, the box persists until the
        track ages out entirely (``max_age`` unmatched frames, 0.5 s at the
        default 10 fps / max_age 5).

        Coasting like this is safe under the safety invariants because it can
        only ever say "look here" about a place fire was just seen. The
        symmetric mistake -- suppressing a box because evidence dipped -- would
        be the system quietly retracting a warning, which it must not do.
        Staleness of the overlay as a whole is a separate, stricter rule
        enforced on the tablet (``stream.max_overlay_age_s``).
        """
        out: list[Detection] = []
        for track in self._tracks:
            matched_now = track.track_id in seen_now
            if track.confirmed:
                pass
            elif self.cfg.emit_unconfirmed and matched_now:
                # Unconfirmed tracks are emitted only on frames they were
                # actually seen on; coasting an unconfirmed track would be
                # drawing a box for a region with no evidence behind it at all.
                # The wire has no "unconfirmed" flag by design -- the tablet
                # distinguishes them by `persisted` < n, so weak evidence is
                # shown as weak evidence rather than hidden behind a badge.
                pass
            else:
                continue

            # `persisted` is hits within the last m frames, which decays as a
            # coasting track's misses fill the window -- so the number on
            # screen weakens as the evidence does. Floored at 1 because the
            # wire type forbids 0 (a zero would read as a measurement of
            # nothing, exactly the framing this protocol refuses) and because
            # max_age may exceed m, letting the window empty entirely.
            persisted = max(1, track.hits_in_window)
            out.append(
                Detection(
                    cls=track.cls,
                    conf=track.conf,
                    box=track.box,
                    track_id=track.track_id,
                    persisted=persisted,
                    first_seen_pts=track.first_seen_pts,
                )
            )
        return tuple(out)

    def __len__(self) -> int:
        return len(self._tracks)

    def __repr__(self) -> str:
        cfg = self.cfg
        return (
            f"TemporalFilter(n={cfg.n}, m={cfg.m}, iou_match={cfg.iou_match}, "
            f"max_age={cfg.max_age}, tracks={len(self._tracks)})"
        )
