# `wildfire-320-int8.tflite` and `wildfire-320.onnx`

YOLOv8n finetuned on fire and smoke, 320x320, exported int8 for the tablet and to ONNX for
the browser. **AGPL-3.0**, from the weights below.

    https://huggingface.co/rabahdev/fire-smoke-yolov8n/resolve/main/best.pt

Both files, and `wildfire-320.json` beside them, are produced by
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

## What it is worth

Measured on a runner, against 120 published pictures with fire in them, 150 real drone frames
with none, and the fire set's own 120 no-fire pictures. Full report in
[`docs/metrics-wildfire.txt`](../../docs/metrics-wildfire.txt).

| confidence | found, of 120 fires | marked, of 150 drone frames | of 120 hard negatives |
|---|---|---|---|
| 0.05 | 65 | 10 | 20 |
| 0.10 | 56 | 4 | 10 |
| **0.15** | **50** | **0** | **7** |
| 0.25 | 43 | 0 | 4 |
| 0.50 | 28 | 0 | 4 |

**It misses more than half.** That is the headline and it is not buried: at the threshold
this runs at, 70 of 120 pictures with fire in them come back empty.

What it almost never does is cry wolf. Nothing at all on real drone footage, at every
threshold from 0.15 up. That asymmetry is the whole reason the threshold is 0.15: it is the
last row where the middle column is zero, a knee in a measurement rather than a number
somebody liked. The sweep used to stop at 0.15, so the knee sat one row below the bottom of
the table and could not be seen.

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
