# `detector.tflite`

EfficientDet-Lite2, int8, 448x448, trained on COCO. **Apache-2.0.**

> **The browser's detector, not the tablet's.** The live view in the APK runs the aerial
> model in [`PERSON-320.md`](PERSON-320.md) instead: this one loses people from altitude,
> which is the whole job there. What follows still describes what the browser runs, and why
> the licence question below was decided the other way for a demonstration.

    https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite2/int8/1/efficientdet_lite2.tflite

## Why this one

It is here because it needs no dataset and no training run, and because its licence permits
commercial use. That second point is not incidental: the obvious alternative is a
YOLOv8/YOLO11 checkpoint, and Ultralytics releases those under **AGPL-3.0**, which obliges
anyone who distributes or serves the work to release their own source under the same terms.
For a product this is a decision to take deliberately with legal advice, not one to walk
into by copying a `.pt` off a model zoo.

int8 at 448 px rather than the smaller 320 px Lite0, because the subjects here are people
seen from a drone and they are small in the frame. 7.5 MB against 4.6 MB is worth it.

## What it knows

The 80 COCO classes. The ones that matter here are **person**, **car**, **truck**, **bus**,
**motorcycle**, **bicycle** and **boat**; the app hides the rest, because a drone inspection
does not need to be told about a toothbrush.

## What it does not know

Fire. Smoke. Cracks, corrosion, blade damage, soiling, delamination. **None of the defect or
hazard classes this project exists to find are in COCO**, and no amount of tuning adds them.

So this covers exactly one half of the job, and it happens to be the half that matters most
in the two live domains: a person at a fire, and people in a crowd. Everything else stays
with the provider API until a model is trained on the accumulated inspections.

## Its real limit

A person forty metres below a drone is a handful of pixels. This will miss them, and it will
miss them more often the higher you fly. It is a screening aid whose recall falls with
altitude, which is a fact about the optics and the input resolution rather than something a
better threshold fixes.
