# `wildfire-640-float.tflite` and `wildfire-640.onnx`

YOLOv8n finetuned on fire and smoke, exported at **640x640** as plain float for the tablet and
to ONNX for the browser. **AGPL-3.0**, from the weights below.

    https://huggingface.co/rabahdev/fire-smoke-yolov8n/resolve/main/best.pt

Both files, and `wildfire-640.json` beside them, are produced by
[`.github/workflows/model.yml`](../../.github/workflows/model.yml) and committed by it.
Nothing here is trained. Two classes, `smoke` and `fire`, in that order, so a box says which
of the two it is rather than hedging between them.

## The one it replaced, and why the gate exists

The first fire model this project shipped did not work. Not badly: at all. It returned a
score between 0.0123 and 0.0155 for black, white, random noise, a flame-coloured block,
sixty real drone photographs and fifty frames of the demo clip alike. It could not mark fire
because it could not mark anything, and every check the conversion workflow made passed it:
the graph ran, the output had a shape the app could decode, the head matched the label list.
The one step that ran the model ran it on a blank image and printed the shape.

[`tools/model_liveness.py`](../../tools/model_liveness.py) is what stands between that and
shipping now. It asks one question that needs to know nothing about the subject: does the
answer change when the picture changes. Over five different pictures:

| | spread |
|---|---|
| this model, ONNX | 0.469 |
| this model, after int8 quantisation | 0.501 |
| the one it replaced | 0.002 |

Both the browser export and the quantised tablet one are gated on it, because int8 throws
away almost all of the precision in every weight and is a far more likely way to arrive at a
dead model than a bad download.

## What it was worth at 320, which is what it USED to ship at

Kept because the section below is a comparison and needs something to compare against. This
is the old export, not the one on disk. Full report in
[`docs/metrics-wildfire.txt`](../../docs/metrics-wildfire.txt).

| confidence | found, of 120 fires | marked, of 150 drone frames | of 120 hard negatives |
|---|---|---|---|
| 0.05 | 65 | 10 | 20 |
| 0.10 | 56 | 4 | 10 |
| **0.15** | **50** | **0** | **7** |
| 0.25 | 43 | 0 | 4 |
| 0.50 | 28 | 0 | 4 |

## The threshold, re-derived at 640

**The old rule does not survive the resolution change, and that is the headline.** 0.15 was
chosen at 320 as the last value with *zero* false alarms on real drone footage. At 640 there
is no such value at all:

| confidence | found, of 120 fires | marked, of 150 fire-free drone frames |
|---|---|---|
| 0.15 | 79 | 17 |
| 0.20 | 72 | 11 |
| **0.25** | **66** | **9** |
| 0.35 | 55 | 5 |
| 0.50 | 40 | 4 |

At 320 the same table read 50 found and 0 marked at 0.15. So 640 finds far more fire and has
lost the property that made the old number defensible. There is no setting that keeps both.

**0.25 is where it ships**, and the reasoning is written down because it is a judgement and
not a knee:

- Against the old 50 of 120, it finds 66. Against the old 15 of 120 distant plumes, about 26.
- It marks 9 of 150 fire-free frames, about one frame in seventeen, against zero before.
- 0.15 would find 79 and mark 17 - one frame in nine, or a spurious box every few seconds at
  the rate this runs. That is the point where an operator stops reading the boxes, and a
  detector nobody reads is worth less than one that misses things.

### The control set does not match the mission, and that matters both ways

The 150 fire-free frames are VisDrone: streets, cars, rooftops. A wildfire flight is over
wildland. On the aerial set's own fire-free pictures the same model at 0.15 marks 4 of 30
rather than one in nine, so the urban number above is the pessimistic end of the range and
the wildland one is milder.

It is quoted anyway, because a wildfire near the urban interface is exactly the flight where
this matters most and is the one the strict number describes.


### It still misses about half, and it does now cry wolf

At 640 and the shipped 0.25, 54 of 120 pictures with fire in them come back empty, and 9 of
150 fire-free drone frames come back marked. Both halves of that are worse than the sentence
this section used to carry, which said the model never marked clear ground - true at 320, and
not true at any threshold at 640.

Two engines is the answer to the first half and the operator is the answer to the second. A
marked frame is a frame to look at, never a frame that has been decided.

Two things this number is not.

It is **not coverage.** A frame with nothing marked has not been cleared of anything. Thin
smoke on a bright sky, smouldering with no visible flame, fire under canopy and fire at night
all return an empty list, and that list is byte-identical to the one an empty field returns.
See [`station/core/safety.py`](../../station/core/safety.py).

It is **not aerial.** The pictures are ground level, because the sets that would match how
this flies are not reachable: FLAME and FLAME2 are behind an IEEE DataPort account, and D-Fire
is a Google Drive link. Fire seen from 100 metres up is a different problem from fire seen
across a room, and nothing here has measured the first one.

## Why the colour scan stays

[`web/js/firescan.js`](../js/firescan.js) and `FireScan.java` still run, on alternate cycles
with this model in a wildfire flight, and neither replaces the other. They fail differently.
The model knows what fire looks like in a single frame and misses more than half of it. The
scan knows what fire *does* over time - it burns in place and churns inside its own outline -
and cannot tell a plume from a painted wall someone walks past. Two engines marking the same
fire twice is a smaller problem than one of them missing it.

The scan also needs no model, so it is what runs on a build with no fire model in it, which
is what the tablet did for fire until this one landed.
