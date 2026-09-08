# `person-320-int8.tflite` and `person-320.onnx`

YOLO finetuned on **VisDrone**, 320x320, exported int8 for the tablet and to ONNX for the
browser. **AGPL-3.0**, from the weights below.

    https://huggingface.co/dronefreak/visdrone-yolov26n/resolve/main/best.pt

Both files, and `person-320.json` beside them, are produced by
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

Eleven VisDrone classes, listed in `person-320.json`, which is written from the model itself
rather than typed by hand so the two cannot disagree.

The app keeps only `pedestrian` and `people` and calls both of them **person**. VisDrone
separates a figure standing or walking from a figure in any other pose; for this application
they are the same thing. The vehicle classes are dropped **before** overlapping boxes are
resolved rather than after, so a van parked beside somebody can never suppress the person.

## What it does not know

Fire, smoke, cracks, corrosion, blade damage, soiling. Flame and smoke are computed from
colour and behaviour instead, in [`firescan.js`](../js/firescan.js) and its Java twin;
everything else stays with the provider engine.

## What it measures

Over the VisDrone validation set, on the twelve most crowded frames, holding 1562 labelled
people between them, run through the pipeline the tablet actually uses:

| | boxes drawn | landed on a labelled person |
|---|---|---|
| the wide pass alone | 179 | 163, or 10% |
| with the close looks merged in | 937 | 730, or 47% |

Between 49 and 132 people boxed per frame. The gap between those two rows is what tiling is
worth, and it is worth restating: reading the wide frame on its own sees a tenth of a crowd.

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

The converter emits an input of `[1, 3, 320, 320]`, channels first, where nearly every other
TFLite vision model is channels last. Code that assumes the usual layout reads an input size
of 3, builds a three pixel square, leaves the rest of the tensor at zero, and reports an
empty scene on every frame with no error anywhere to say so.

That is indistinguishable from a frame with nobody in it, which is the failure this whole
application is written against. `NativeDetector.java` reads the layout off the tensor, the
conversion workflow refuses to publish a model whose shape disagrees with `person-320.json`,
and `tests/test_model_contract.py` checks the same wherever a runtime is installed.
