# Wire contract

Normative definition of everything crossing a process boundary. The Python
side is `station/core/types.py`; the browser side is `app/js/wire.js`. Those two
files and this document must agree. `WIRE_VERSION` is currently **2**.

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
    "classes": ["fire", "smoke", "person"], "weights_sha": "9f2c1ab",
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

## The `person` class

Added in wire v2, appended so that indices 0 and 1 keep their meaning and
pre-v2 weights and incident logs still decode.

The version was bumped even though the message *shape* did not change. A v1
tablet would have parsed a person payload quite happily and drawn it in the
unknown-class fallback — a thin white dashed box labelled `?`. For a box around
a human being, being silently mislabelled is worse than the tablet refusing to
connect, so the refusal is the intended behaviour.

Three rules attach to this class and not to the others:

* **It is never summarised.** No count, no "N people", no tally. A count
  implies the denominator is known; only the boxes actually drawn are known.
* **It is never allowed to fall through to a default style.** `app/js/overlay.js`
  gives it its own colour, its own dash pattern, its own corner treatment and a
  minimum drawn size, because it is the smallest object on screen and the most
  consequential to overlook.
* **Its temporal window is looser.** `temporal.per_class` defaults person to
  2-of-6 against the global 3-of-5. A person at altitude flickers far more than
  a flame front, and a track that never confirms is a person never drawn. The
  cost is a busier overlay; the alternative is silence about someone who is
  there.

The absence of a person box carries no information about whether anyone is
present. At altitude a person is a handful of pixels, routinely hidden by
canopy, smoke or terrain, and easily confused with a rock. Misses are the
normal case. Personnel accountability comes from roll call and crew tracking,
never from this feed — `station/core/safety.py` fails the build on text that
blurs the two.

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

## An offset must be corroborated before it is trusted

The tier-2 offset is computed as `payload.pts - video.currentTime`, so a single
sample is true by construction and tells you nothing about whether it is
*right*. One absurd payload would otherwise define the offset, the correction
would hide the discrepancy, and `ageS` would come out at 0 — the overlay
reporting itself perfectly synchronised while every box sat over the wrong
ground. The `ahead` guard could not see it, because it measures age *after* the
offset has been applied.

So a large offset is trusted only when something other than itself corroborates
it. Three things can:

1. **`stream_start_pts`** from the first `status` message — the seed exists for
   exactly this purpose.
2. **A second sample** that agrees; the median of a pair moves when the second
   contradicts the first.
3. **A near-zero offset**, which is the subtle one: it asserts that the
   timelines already agree, so no correction is being trusted and there is
   nothing to get wrong.

Failing all three, the estimate is refused, the overlay drops to tier 3 and is
labelled unsynchronised, and the boxes are not drawn — because we know the
detections sit further from the video clock than the overlay limit allows and
cannot tell a legitimate timeline difference from one bad measurement.

The cost is small and bounded. A station joining 28.8 s into its source with no
seed suppresses boxes for exactly one payload — about 100 ms at 10 fps — before
the second sample corroborates the offset. With the seed the station actually
sends, there is no gap at all.
