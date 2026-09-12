# Candidates that were measured and not shipped

[`.github/workflows/model-experiment.yml`](../../../.github/workflows/model-experiment.yml)
exports a candidate detector into this folder, which nothing loads: the manifest does not
name it, the APK excludes it, and the app reads neither. That is the point. "Would a stronger
model be better" gets answered with a number instead of by putting an untested model in front
of an operator between the question and the answer.

The `.json` beside each name is the record of what was tried. The weights themselves are
deleted once a candidate has been judged, because a 37 MB file nothing loads is not worth
carrying; re-run the workflow with the `source` from the JSON to get them back.

## What has been tried

| candidate | why | what it measured |
|---|---|---|
| `visdrone-s-320` | yolov11s, a size up from the shipped nano, at the same 320 | Reached 47% of the people where the nano reached 45%, for 2.3x the time per look and 4x the file. Two points. Rejected. |
| `visdrone-n-640` | the SAME shipped nano weights, exported at 640 instead | Reached 60% where 320 reached 45%, at the same milliseconds and the same file size. **Adopted**, and it is what `person-640.*` now is. |
| `enot-x3-640` | a NAS-selected yolov8s trained on VisDrone: the strongest aerial-person checkpoint published anywhere reachable | Rejected on speed, and the way it was rejected is the point. See below. |
| `mshamrai-n-640` | another yolov8n trained on VisDrone, to see whether the weights rather than the size were the limit | 1.27x the time per look, so a 300 ms cycle. Reached **64%** against the shipped 66%, 1.44 numbers each against 1.43. Slower and no better. Rejected. |

## The candidate that looked better until it was flown at its own speed

`enot-x3-640` was flown over five real VisDrone MOT sequences beside the shipped detector.
At the same 250 ms cycle it reached **70%** of the people against the shipped **66%**, for
the same 1.43 numbers each. On that table it is the better model.

It cannot have that cycle. Measured on one machine over the same frames it is **1.93 times**
the time per look, 77.1 ms against 40.0, which on an MK15 turns a 255 ms cycle into roughly
490. Flown at 500 ms, which is the cycle it would actually achieve there:

| detector | cycle | reached | numbers each | on nobody |
|---|---|---|---|---|
| `person-640` (shipped) | 250 ms | **66%** | 1.43 | 197 |
| `enot-x3-640` | 250 ms | 70% | 1.43 | 220 |
| `enot-x3-640` | 500 ms | **52%** | 1.32 | 146 |

Fourteen points of people worse than what ships. A stronger model that halves the cadence is
not a stronger system: every person is looked at half as often, and the looks are what find
people who are eleven pixels tall. **Rejected**, and the reports are kept at
[`docs/metrics-video-enot-x3-640.txt`](../../../docs/metrics-video-enot-x3-640.txt) and
[`docs/metrics-video-enot-x3-640-500ms.txt`](../../../docs/metrics-video-enot-x3-640-500ms.txt).

That is also the rule for the next candidate: a model is flown at the cycle it would achieve
on the tablet, not at the cycle the shipped one achieves. And the two are timed alternately
in one process on one machine ([`tools/time_models.py`](../../../tools/time_models.py)),
because the per-look figure printed in a flight report is comparable only inside that run -
the same weights measured 20.9 ms on one CI runner and 45.1 ms on another machine the same
day, which is enough to reverse a verdict.

The second row is the whole lesson. The model was never the weak part; it was being shown a
sixth of the frame squeezed into 320 pixels and then asked to remember what it had seen for
five cycles out of six. Numbers and method in
[`../PERSON-640.md`](../PERSON-640.md).
