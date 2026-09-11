# `person-640-float.tflite` and `person-640.onnx`

YOLO finetuned on **VisDrone**, 640x640, exported as plain float for the tablet and to ONNX for the
browser. **AGPL-3.0**, from the weights below.

    https://huggingface.co/dronefreak/visdrone-yolov26n/resolve/main/best.pt

Both files, and `person-640.json` beside them, are produced by
[`.github/workflows/model.yml`](../../.github/workflows/model.yml) and committed by it.
Nothing here is trained. The weights already existed; the workflow fetches and converts
them, on a runner, because that runner can reach the model hosts this project's own
environment cannot.

## Why this one

The model it replaced on the tablet was EfficientDet-Lite2: eighty everyday classes,
photographs taken by people standing on the ground, 448 px. Three things wrong at once for
this job. A 1920-wide frame squeezed into 448 is a fourfold reduction, so somebody forty
pixels tall in the air arrives nine pixels tall, and nothing it was trained on was shot from
above.

VisDrone is aerial imagery of streets and squares, annotated down to figures a handful of
pixels across, with **pedestrian** and **people** as separate classes. It is the problem
this application has, already solved and published.

The licence is the reason it was not used sooner. Ultralytics releases these weights under
AGPL-3.0, which obliges anyone distributing or serving the work to release their own source
under the same terms. That is a decision to take deliberately rather than walk into. It was
taken deliberately: this is a public repository and a demonstration, not a product, so the
obligation is one this project already meets.

## What it knows

Eleven VisDrone classes, listed in `person-640.json`, which is written from the model itself
rather than typed by hand so the two cannot disagree.

The app keeps only `pedestrian` and `people` and calls both of them **person**. VisDrone
separates a figure standing or walking from a figure in any other pose; for this application
they are the same thing. The vehicle classes are dropped **before** overlapping boxes are
resolved rather than after, so a van parked beside somebody can never suppress the person.

## What it does not know

Fire, smoke, cracks, corrosion, blade damage, soiling. Flame and smoke are computed from
colour and behaviour instead, in [`firescan.js`](../js/firescan.js) and its Java twin;
everything else stays with the provider engine.

## Why 640 and not 320

These are the same weights this project shipped at 320. Only the export size changed, and
with it the number of tiles: two halves of the frame instead of six sixths.

The old shape squeezed a sixth of a 1920-wide frame into 320 pixels. This one squeezes a
half into 640. A person arrives at almost the same size either way, so per look the two are
near enough equal. What differs is how long a person waits to be looked at. Six tiles at the
cycle rate means a patch of ground gets a look once every six cycles, over a second, and
everybody outside the current tile is a prediction rather than an observation. Two tiles
means everybody is looked at every cycle.

Measured over the VisDrone validation set on its four most crowded frames, flying the real
model at the real cadence, with every number attributed to the person it spent the most
frames on, so that finding another person and numbering the same person twice cannot be
confused:

| frame, people in it | 320 at six tiles | 640 at two tiles |
|---|---|---|
| 175 | 45% reached, 1.92 numbers each | **60% reached, 1.69 numbers each** |
| 167 | 34% reached, 1.39 numbers each | **48% reached, 1.37 numbers each** |
| 150 | 25% reached, 1.37 numbers each | **36% reached, 1.30 numbers each** |
| 144 | 39% reached, 1.36 numbers each | **53% reached, 1.42 numbers each** |

Same weights, same 9.4 MB, same 250 ms cycle. Eleven to fifteen more people in every
hundred get a number at all, and the repeat numbering is flat or better in three frames of
four.

Two tiles rather than three or four because recall saturates there. On a still frame, at
this export size, one tile finds 51% of the labelled people, two find 70%, and three, four
and six all find 69 to 70%. Past two tiles the extra inferences buy latency and nothing
else, and latency is what leaves a box sitting where somebody used to be.

### What this does not fix

The stray numbers went the wrong way, 49 to 58 over three flights: numbers that sit on
nobody. More looks at the same ground is more chances to be confidently wrong, and that
trade was taken on purpose against fifteen points of people found.

And a stronger model was measured before this was chosen, because "use the strongest model"
was the obvious answer and it was not the right one. `yolov11s`, four times the file, 2.3x
the time per look, reaches 47% where the shipped nano reaches 45% - two points, against the
fifteen that came from stopping starving the nano of pixels. It is in
[`.github/workflows/model-experiment.yml`](../../.github/workflows/model-experiment.yml) if
anyone wants to re-run it.

## What one frame looks like

Over the twelve most crowded validation frames, 1562 labelled people, at the 320 export:

| | boxes drawn | landed on a labelled person |
|---|---|---|
| the wide pass alone | 179 | 163, or 10% |
| with the close looks merged in | 937 | 730, or 47% |

The gap between those two rows is what tiling is worth, and it is worth restating: reading
the wide frame on its own sees a tenth of a crowd.

Two things that table is not. It counts a single frame, where the live view accumulates over
many and re-identifies people who leave and return, so the running tally climbs above this.
And 207 of those 937 boxes did not land on a labelled person; some of those are people in
regions the dataset marks as ignored and some are the model being wrong, and this has not
separated them.

## Its real limit

Unchanged in kind, only in degree: a person far below a drone is a handful of pixels, and
the higher the camera the more of them go unmarked. Anyone underneath a tree, a canopy or a
vehicle cannot be seen from above at all. The tally is a floor and the app says so.

## The trap this file exists to warn about

The converter emits an input of `[1, 3, 640, 640]`, channels first, where nearly every other
TFLite vision model is channels last. Code that assumes the usual layout reads an input size
of 3, builds a three pixel square, leaves the rest of the tensor at zero, and reports an
empty scene on every frame with no error anywhere to say so.

That is indistinguishable from a frame with nobody in it, which is the failure this whole
application is written against. `NativeDetector.java` reads the layout off the tensor, the
conversion workflow refuses to publish a model whose shape disagrees with `person-640.json`,
and `tests/test_model_contract.py` checks the same wherever a runtime is installed.
