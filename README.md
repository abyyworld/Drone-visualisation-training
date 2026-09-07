# Drone Inspection

Defect analysis for wind turbine blades and solar panels, from drone photographs or video.
Static site, no server, hosted on GitHub Pages, and installable on a tablet as an app with an
icon.

**Live:** https://abyyworld.github.io/Drone-visualisation-training/

## Two engines

| | On-device model | Provider API |
|---|---|---|
| Where it runs | This browser, via ONNX Runtime Web | Anthropic, Google or OpenAI |
| Cost | Nothing | Per image |
| Offline | Yes, once cached | No |
| Images leave the device | Never | Yes, to the provider you pick |
| What it can find | Only its trained classes | Anything it can see and describe |
| Box precision | Tight | Approximate |
| Needs a dataset first | Yes | No |

Pick one in the app. The header badge stops claiming local processing the moment an API engine
is selected, because that claim would then be false. The key is held in the tab, never written
to disk, and sent only to the provider chosen.

For batch work and for building training data, `tools/vlm_inspect.py` does the same thing on
the command line with the same prompt (`web/prompts/inspection.json`, read by both, so the
page and the tool cannot grade the same photograph differently).

## Video

Drop an MP4 or WebM clip. It is sampled, scored for sharpness by Laplacian variance,
deduplicated by difference hash, and reduced to a couple of dozen distinct in-focus frames.
Five minutes at 25fps is 7,500 frames and most of them are the same photograph.

It selects for coverage, not for interest: a defect visible in exactly one blurred frame can
be dropped. Upload a still of anything suspicious.

## On the tablet

`docs/INSTALL-tablet.md`. Two routes, same code:

- **From the browser.** Add to home screen. Updates itself, needs a current Chrome.
- **APK.** `android/` is a WebView around the same `web/` directory, copied in at build
  time. Built by CI on every push to `main` and downloadable from Actions. Take this one
  when the tablet has no Play Store, no install option, or may never see WiFi - everything
  including the ONNX runtime is inside the file.

Either way the on-device engine works with no signal; the API engines cannot, and the app
says so rather than queueing work that will never send.

## Why the trained model is not the default

`archive/turbine-v2/` holds the previous detector. It scored mAP50 0.759 on a split with zero
near-duplicate leakage and 0.5% false positives across 400 healthy blades, and then reported
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

## The plan: get back to a trained model, with the right data

The API is not the destination. It is what makes the destination reachable.

Every inspection writes a label sidecar. `tools/vlm_to_yolo.py` turns a pile of those into a
YOLO dataset - split by whole inspection so validation is not a memorisation test, reviewed
sidecars only, refusing to run on a label with no place in the class table rather than
guessing one.

That dataset is made of the operator's own photographs, at the operator's own framing, from
the operator's own drone. It is the one thing the failed model never had. When it reaches the
size below, train on it and the API becomes optional.

```bash
python3 tools/vlm_inspect.py photos/ --provider anthropic --domain turbine
# review the annotated copies, correct the sidecars, set "reviewed": true
python3 tools/vlm_to_yolo.py inspections/ --dry-run          # how much is there
python3 tools/vlm_to_yolo.py inspections/ --out datasets/turbine_field
python3 tools/audit_dataset.py datasets/turbine_field
```

## What a replacement dataset needs

- **A class list covering the damage that matters**, structural failure included. Decided
  before collection, not after.
- **Negatives from the same domain as the positives** - healthy blade, same range, same camera,
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

Run it on Kaggle with `training/turbine/turbine_kaggle.ipynb` - **Save Version → Save & Run
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
npm test               # browser checks + the provider adapters (needs playwright)
npm run test:vlm       # provider adapters only, no browser, no network
python3 -m pytest tests/                # 497 checks
python3 tests/test_export_contract.py   # proves the JS decoder matches Ultralytics
```

Solar detection and the domain gate are scaffolded and unbuilt; both need solar imagery, and
the gate has to learn `solar` as a class, so one dataset unlocks both.
