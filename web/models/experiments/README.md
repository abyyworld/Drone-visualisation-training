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

The second row is the whole lesson. The model was never the weak part; it was being shown a
sixth of the frame squeezed into 320 pixels and then asked to remember what it had seen for
five cycles out of six. Numbers and method in
[`../PERSON-640.md`](../PERSON-640.md).
