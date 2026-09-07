# Archive

Superseded work, kept so the record is complete. **Nothing here is deployed and nothing here
should be deployed.** The live app loads models from `web/models/` and nothing else; this
directory is not on the Pages deploy path.

---

## `turbine-v2/` - the trained detector, obsolete

A YOLOv11s detector trained on a rebuilt three-class turbine dataset, exported to ONNX and
int8-quantised for the browser. It is the second turbine model, and it is the last one
trained before the project moved to vision-model inspection.

| | |
|---|---|
| Architecture | yolo11s, 960 px, int8 dynamic quantisation, opset 12 |
| Size | 10.1 MB |
| Classes | corrosion, crack, surface_peeling |
| Test mAP50 | **0.759** |
| Test mAP50-95 | 0.478 |
| Per class mAP50 | crack 0.938 (100 instances), surface_peeling 0.781 (50), corrosion 0.557 (32) |
| False positives | 2 of 400 healthy blade images at conf 0.25; 0 at conf 0.50 |

Those are real numbers on a held-out test split, produced by `tools/evaluate.py`. Full output
is in `turbine-v2/metrics.json`, and `turbine-v2/manifest-entry.json` is the `web/models/manifest.json`
entry it shipped with, so it can be dropped back in for comparison.

### Why it is not used

It reported **no defects at all** on a photograph of a turbine with a blade severed in half.

Not a low score on the right answer. Zero detections. The reasons, in order of how much each
one mattered:

1. **No class covers structural failure.** The three classes are surface conditions. A blade
   in two pieces is not corrosion, a crack, or peeling, so there was no output for it to
   produce. A detector cannot report a category it was never given.
2. **Every defect image in training is a close-up.** The photograph was a wide landscape
   shot of a whole turbine. The scale was outside anything the model had seen.
3. **The background negatives were wide aerial shots.** This is the one that turns a miss
   into a confident miss. The images with no boxes were framed wide; the images with boxes
   were framed close. The model had every reason to learn *wide framing means nothing here*,
   and it did.
4. **About 750 distinct defect scenes**, after perceptual-hash clustering. Against
   Ultralytics' production guidance of 1,500 images and 10,000 instances, that is thin.
5. **Every number above was measured inside the training distribution**, including the
   false-positive check. 0.759 describes how well it does on more images of the kind it was
   trained on. It says nothing about a photograph taken by a different drone at a different
   distance, and that is the only kind of photograph that matters in the field.

The dataset that produced it has been deleted from this repository. The model is kept
because the numbers are real and the failure is instructive: it is a worked example of a
model that scores well and does not work, which is the single most expensive mistake
available in this field.

### What replaced it

Vision-model inspection through a provider API (`web/js/vlm.js`, `tools/vlm_inspect.py`),
which has no fixed class list and describes a severed blade as a severed blade. Its own
weakness is the opposite one: loose boxes, a per-image cost, and images leaving the device.

The plan is not to stay there. Every inspection writes a label sidecar, and
`tools/vlm_to_yolo.py` accumulates those into a dataset made of the operator's own imagery,
at the operator's own framing. A detector trained on that has the one thing this one never
had - training data that looks like the job - and it can then replace the API.
