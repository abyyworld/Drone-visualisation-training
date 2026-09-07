# Wire contract

Normative definition of everything crossing a process boundary. The Python
side is `station/core/types.py`; the browser side is `app/js/wire.js`. Those two
files and this document must agree. `WIRE_VERSION` is currently **1**.

Transport: one WebRTC peer connection per tablet, carrying a video track and a
reliable-ordered data channel named `detections`. Messages are UTF-8 JSON, one
object per message.

## Message types

### `detections` — one per inferred frame

```json
{
  "v": 1,
  "type": "detections",
  "frame_id": 4127,
  "pts": 137.4667,
  "wall_time": "2026-09-06T14:22:31.412Z",
  "rtp_ts": 1839472920,
  "inference_ms": 21.7,
  "source_id": "rtsp://192.168.144.25:8554/main.264",
  "model": {
    "name": "yolo11s-fire", "version": "0.3.1",
    "classes": ["fire", "smoke"], "weights_sha": "9f2c1ab",
    "imgsz": 640, "conf_threshold": 0.25
  },
  "detections": [
    {"cls": "fire", "conf": 0.83, "box": [0.412, 0.331, 0.489, 0.402],
     "persisted": 5, "track": 7, "first_seen_pts": 136.9}
  ]
}
```

`detections: []` is normal, frequent and meaningless as reassurance. It is sent
and logged faithfully, and it renders as nothing. See [SAFETY.md](SAFETY.md).

`box` is `[x1, y1, x2, y2]` normalised `0..1`, origin top-left. Never pixels —
the tablet renders the video at whatever size it likes, and the model ran at
640.

### `status` — heartbeat, ~1 Hz, independent of detections

```json
{
  "v": 1, "type": "status", "state": "running",
  "wall_time": "2026-09-06T14:22:31.500Z",
  "source": "rtsp://192.168.144.25:8554/main.264",
  "source_fps": 29.9, "inference_fps": 9.8,
  "last_inference_pts": 137.4667,
  "last_inference_wall_time": "2026-09-06T14:22:31.412Z",
  "stream_start_pts": 0.0, "dropped_frames": 412, "uptime_s": 903.2
}
```

`state` ∈ `starting | running | degraded | stalled | stopped`. It describes the
**pipeline**, never the scene. A heartbeat on a fixed cadence means silence on
the data channel is itself diagnosable — the tablet can distinguish "model
found nothing" from "station fell over", which are the same empty screen
otherwise.

An unknown `v` is a hard parse error on both sides. A tablet that silently
dropped fields it did not understand would render a partial overlay, and a
partial overlay is indistinguishable from a quiet scene.

## Overlay synchronisation

The problem: video and detections travel as separate streams, so a box can be
drawn over a frame it was not computed from. On a moving aerial scene at 15 m/s
even 300 ms of drift puts the box roughly 4.5 m off the ground truth — pointing
a crew at the wrong place, while looking authoritative. **Never render the
newest payload immediately.** Buffer, then match.

Three tiers, best first. The tablet picks the best available and *displays
which one is active*, because the accuracy of the overlay position differs
between them and the operator is entitled to know.

### Tier 1 — RTP timestamp match (exact)

`HTMLVideoElement.requestVideoFrameCallback()` reports, per presented frame,
both `mediaTime` (the `currentTime` timeline) and `rtpTimestamp` (90 kHz, the
sender's clock). When the station can supply `rtp_ts` — the aiortc path can;
a MediaMTX relay generally cannot — the tablet maintains a small mapping of
`rtpTimestamp → mediaTime` from recent frames, converts each payload's
`rtp_ts` into media time by interpolation, and draws it on the frame it
actually belongs to. No estimation, no drift.

`requestVideoFrameCallback` is available in Chrome/Edge/Android WebView and in
Safari 15.4+, so both tablet platforms in scope can reach tier 1.

### Tier 2 — pts offset estimate (default fallback)

Without `rtp_ts`, estimate the constant offset between the two timelines:

```
offset_i = payload.pts − video.currentTime   (sampled at payload arrival)
offset   = median(last N=60 offset_i)        (median, not mean — resists jitter)
target   = video.currentTime + offset
```

Render the buffered payload whose `pts` is nearest `target`. A median over a
rolling window is deliberate: arrival-time jitter and the odd very late
payload are common, and a mean would let one outlier drag every box sideways.
Seed the estimate from `stream_start_pts` in the first `status` message so the
first seconds are not unaligned.

### Tier 3 — unsynchronised (degraded, and labelled)

If neither works, draw the latest payload **and show the overlay as
unsynchronised**. A silently misplaced box is worse than an obviously
approximate one.

### Staleness — the rule that outranks all three

If the best-matching payload is older than `max_overlay_age` (default 1.0 s of
media time) — the pipeline stalled, the data channel dropped, inference fell
behind — **stop drawing boxes** and show the overlay as stale. Live video
under boxes computed from a scene that has already moved on is this system's
most dangerous single failure mode. Dropping the overlay degrades the tool to
plain video, which is a safe state, because the operator is watching the video
anyway; that is the whole premise of the design.

Note the asymmetry with the previous paragraph and take it seriously: tier 3
keeps drawing because it is *labelled* approximate, whereas staleness stops
drawing because the boxes are *wrong*, not merely imprecise.

## Buffer sizing

Hold ~3 s of payloads (at 10 fps inference, ~30 objects, a few KB). Enough to
absorb data-channel jitter and to let the video run slightly behind the
detections; short enough that a genuinely stalled pipeline hits the staleness
rule quickly rather than replaying a stale ring buffer.

## Incident log

The same `detections` objects, one JSON object per line, written to
`incidents/<incident-id>/detections.jsonl` — including the empty ones, which is
what makes the log a usable record of what the model saw rather than a
highlight reel of what it happened to catch. Together with the recorded video
this gives after-action review, and accumulates the real-incident footage that
the false-negative validation in [VALIDATION.md](VALIDATION.md) needs.

## Known gap: an uncorroborated offset can hide a misaligned overlay

`tests/test_sync.js` carries one deliberately failing test,
*"a payload far ahead of the frame on screen also stops the boxes"*.

The case: the tier-2 estimator takes its offset from the samples it has. Give it
a single sample and the median *is* that sample, believed completely. A payload
arriving 40 s from the frame on screen then defines a 40 s offset, the
correction hides the discrepancy, and `ageS` comes out at 0 — the overlay
reports itself perfectly synchronised while every box sits over the wrong
ground. The `ahead` guard cannot see it, because the guard measures age *after*
the offset has been applied.

The obvious fix — refuse to trust an offset built from one sample — is wrong as
stated, because a 40 s difference between the station's media timeline and the
tablet's `video.currentTime` is perfectly legitimate; that is exactly what
`stream_start_pts` exists to seed. With one sample and no seed there is no
information that separates a real timeline difference from one bad measurement.

Two honest routes, neither a one-line change:

1. **Always seed from `stream_start_pts`** and treat a first sample that
   contradicts the seed by more than `max_overlay_age_s` as the outlier rather
   than as the truth. This is the better fix; it needs the station to send
   `stream_start_pts` reliably on the first status of every session.
2. **Require corroboration before tier 2** — but the floor must not be applied
   where a caller has deliberately configured `minOffsetSamples: 1`, so it needs
   to be a separate "unverified" tier rather than a change to the existing gate.

Left failing on purpose: the test states a real defect, and deleting it to get a
green suite would remove the only record that this hole exists. A tier-2 overlay
running on a single offset sample should be treated as unverified until this is
closed.
