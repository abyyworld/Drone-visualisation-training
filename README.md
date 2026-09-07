# Drone Inspection

Browser-based defect detection for wind turbine blades and solar panels. Images are analysed
client-side with ONNX Runtime Web, so nothing is uploaded anywhere, there is no server to pay
for, and GitHub Pages hosts the whole thing as static files.

**Live:** https://abyyworld.github.io/Drone-visualisation-training/

## State: awaiting a dataset

There is no model deployed and no training data in this repo. The site loads, reports that no
detectors are available, and disables upload rather than pretending.

The previous turbine dataset was removed. It scored well — mAP50 0.759 on a split with zero
near-duplicate leakage, 0.5% false positives across 400 healthy blades — and then reported
"no defects" on a photograph of a turbine with a blade severed in half. Both facts are true,
and the second is the one that matters.

Three things caused it, and a replacement dataset has to fix all three:

**The class list did not contain the failure.** `corrosion`, `crack`, `surface_peeling`. A
severed blade is none of them, so a perfect model with those classes still outputs nothing.

**Every defect example was a close-up.** Blade surface at arm's length. A whole turbine across
a field is a different scale entirely, and detectors do not bridge that gap.

**The negatives were the wrong domain.** The "nothing here" examples were wide aerial shots, so
wide framing is precisely what the model learned means *no defect*. That was the framing of the
photograph it failed on. `tools/rebuild_turbine.py` printed this caveat on every run.

## What a replacement needs

- **A class list covering the damage that matters**, structural failure included. Decided
  before collection, not after.
- **Negatives from the same domain as the positives** — healthy blade, same range, same camera,
  same framing. This is the single biggest fix.
- **Range coverage matching the capture hardware**, so the training distribution and the
  deployment distribution are the same one.
- **Out-of-domain images in the test set.** A benchmark that cannot fail measures nothing. Every
  metric quoted above was computed inside the domain the data defined, which is why none of them
  predicted the failure.

## Pipeline

```
new dataset  ->  tools/rebuild_turbine.py   group-aware split, polygons to boxes, negatives
             ->  tools/audit_dataset.py     7 structural checks; do not train on a failure
             ->  tools/train_turbine.py     config lives here, not in the notebook
             ->  tools/evaluate.py          per-class AP, not aggregate mAP
             ->  tools/false_positive_check.py   boxes drawn on healthy frames
             ->  tools/export_onnx.py       ONNX + int8, into web/models/
             ->  tools/publish_results.py   commits the model and its metrics
```

Run it on Kaggle with `training/turbine/turbine_v2_kaggle.ipynb` — **Save Version → Save & Run
All (Commit)**, never a Draft Session. All hyperparameters live in `tools/train_turbine.py`,
which the notebook clones fresh each run, so a stale notebook cannot produce a stale model.

Before the first run, set `CLASS_REMAP` and `V2_NAMES` in `tools/rebuild_turbine.py` from the
new `data.yaml`. Both are empty and the script refuses to run until they are filled: they
encoded the old dataset's class indices, and reusing them would relabel every box.

### The audit

`tools/audit_dataset.py` checks for source-family shortcuts, cross-split leakage,
near-duplicate frames, class-imbalanced box scales, thin test classes, mixed label formats and
duplicate files. It failed 6 of 7 on the original export and passed 7 of 7 after the rebuild.
Run it on anything new before spending GPU time.

## Web app

`web/` is static, no build step. `web/models/manifest.json` is the deploy switch: drop an
`.onnx` beside it, set `labels` and `severityWeights`, and the app picks it up. A missing model
is reported as unavailable rather than failing silently.

```bash
npm run serve          # http://localhost:8080
npm test               # 37 browser checks (needs playwright)
python3 tests/test_export_contract.py   # proves the JS decoder matches Ultralytics
```

Solar detection and the domain gate are scaffolded and unbuilt; both need solar imagery, and
the gate has to learn `solar` as a class, so one dataset unlocks both.
