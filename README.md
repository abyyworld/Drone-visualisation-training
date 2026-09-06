# Drone Inspection

Defect detection for **wind turbine blades** and **solar panels** from drone imagery, with a
browser-based inspection tool that runs the models client-side.

One repository, three models, one site. The web app needs both detectors plus a shared domain
classifier, and the export pipeline, severity scoring and report generator are common to both
domains — splitting turbine and solar apart would duplicate all of that and let it drift.

```
web/          static site (GitHub Pages) — inference runs in the visitor's browser
training/     Kaggle notebooks, one per model, plus shared label-parsing helpers
tools/        dataset audit, dataset rebuild, ONNX export, evaluation
tests/        ONNX fixtures + headless-browser end-to-end suite
docs/         audit reports
train/ valid/ test/   original Roboflow export, read-only source for the rebuild
```

---

## Start here: why the v1 model was bad

The v1 turbine checkpoint (`best.pt`, yolo v8m, 58 epochs) reports:

```
mAP50 0.782   mAP50-95 0.520   P 0.782   R 0.733     (validation, best at epoch 38)
```

That score is real and almost meaningless. `tools/audit_dataset.py` says why — run it yourself:

```bash
python3 tools/audit_dataset.py .          # the original export: 6 of 7 checks fail
```

| Finding | Detail |
|---|---|
| **Source-family shortcut** | The three filename families partition the classes. `Healthy_Train*` and `Areial_Healthy*` contain **only** `healthy`; bare-numeric files contain **only** defects. Not one image in 7,520 holds a healthy region and a defect together, so a detector can score 0.78 by recognising capture style — aerial wide shot vs. defect close-up — without learning any defect. |
| **Near-duplicate leakage** | Roboflow randomly split a contiguous capture sequence. **32.5%** of held-out images have a same-class box at IoU>0.5 with their adjacent frame in train. Validation is partly a memorisation test. |
| **Scene label as a class** | Median `healthy` box covers **44%** of the frame (42.7% exceed half the image, 17.1% exceed 80%) against **4.8%** for `surface_peeling`. A 9.2× spread the box/DFL loss cannot balance — small defects are gradient noise. |
| **Mixed label formats** | All 6,051 `healthy` labels are polygons, all 3,673 defect labels are boxes. Ultralytics discards every segment on every run. |
| **Untestable classes** | `test` holds 16 `corrosion` instances. Its AP is sampling noise. |
| **Never evaluated** | The test split was never run, and only aggregate metrics were ever recorded. |

**None of this is fixed by more images or more compute.** It is fixed by changing the task
definition, which is free.

## The fix

```bash
python3 tools/rebuild_turbine.py                        # -> datasets/turbine_v2/
python3 tools/audit_dataset.py datasets/turbine_v2      # 7 of 7 pass
```

- **`healthy` is no longer a class.** An image with no detections *is* the healthy result.
  This removes the shortcut, the 62% majority class and the scale imbalance at once.
- **Those images are kept** as subsampled background negatives. They are the only thing that
  teaches "blade surface is not a defect", so deleting them would make false positives worse.
- **Group-aware split** on contiguous capture-ID blocks per family, so neighbouring frames
  cannot straddle the boundary. Augmented copies stay with their source; val and test keep
  one copy per source.
- **Polygons normalised to boxes**, duplicates dropped.

| | v1 | v2 |
|---|---|---|
| Near-duplicate leakage | 32.5% | **0%** |
| Class median box-area spread | 9.2× | **1.9×** |
| Thinnest test class | 16 instances | 34 |
| Label formats | polygons + boxes | boxes only |
| Audit | 6 of 7 fail | **7 of 7 pass** |

> **Expect the headline number to get worse.** Without the free `healthy` class, aggregate
> mAP50 will likely fall to roughly **0.45–0.60**. That is the model losing a score it was
> cheating for. Judge it on **per-class defect AP** and on real photos, not the aggregate.

### How much data is actually here

```bash
python3 tools/scene_count.py .     # distinct scenes, not file count
```

A file count is the most misleading number in a vision dataset. Two inflations stack:

| | Count |
|---|---|
| Files | 7,520 |
| Unique source images | 3,133 — **58% of files are augmented copies** |
| Distinct scenes (near-duplicates merged) | **2,281** |
| Distinct **defect** scenes | **~750** |

Roboflow baked 3 fixed variants per training image, and the capture IDs are dense contiguous
runs — the signature of video frames, where consecutive frames are not independent samples.
The file count overstates the annotated training signal by **3.3x**.

~750 distinct defect scenes across three classes is thin. That number, not 7,520, is the
ceiling on what a model can learn here, and it is the argument for **adding** data rather
than tuning harder.

The rebuild therefore keeps **one copy per source** by default. This is not data loss:
Ultralytics applies mosaic, flip, HSV, scale and rotation online every epoch with fresh
random parameters, which strictly dominates 3 frozen variants — while 3 frozen variants also
triple epoch time. `--keep-augmented` restores the old behaviour.

### Image sharpness

```bash
python3 tools/image_quality.py .            # per-class focus breakdown
python3 tools/image_quality.py --help-blur  # why not to bulk-delete blurry images
```

Measured with Laplacian variance. The defect images are markedly softer than the healthy ones:

| Class | n | % soft (<100) | % unusable (<20) |
|---|---|---|---|
| `surface_peeling` | 533 | 65.7% | **17.1%** |
| `corrosion` | 351 | 59.3% | **12.3%** |
| `crack` | 1,333 | 57.5% | **14.0%** |
| background (healthy) | 635 | 38.3% | 0.8% |

Every defect class is roughly 15× more likely to be an unusable smear than a healthy image.

**Do not bulk-delete blurry images.** Real drone footage is blurry, so filtering to sharp
frames makes the training set *less* like deployment — the same class of error as the source
shortcut. It would also remove most of the only images carrying defect labels, and Laplacian
variance cannot tell "out of focus" from "smooth surface" anyway.

Trim only the extreme tail, where a box sits on a smear and cannot teach localisation:

```bash
python3 tools/rebuild_turbine.py --min-sharpness 20    # drops 388 images (15%)
```

Recommended for the real training run. It stays opt-in so the choice is deliberate, and it is
recorded in `build_summary.json`. Note the test split is blurrier than train (20.3% vs 10.2%
unusable) — a side effect of block-splitting on capture ID, since conditions correlate with ID
range. Shuffling would fix it and reintroduce leakage, so stratify evaluation by sharpness
instead of trading away a leakage-free split.

---

## Training

Free tiers are sufficient — the v1 bottleneck was session death with a broken resume and a
run twice as long as needed, not GPU power.

| Platform | Free allowance | Use for |
|---|---|---|
| **Kaggle** | 30 GPU-h/week, P100 or 2×T4, 9-h sessions, no card | primary — everything here fits |
| Colab | ~15–30 h/week T4, availability not guaranteed | overflow |
| Lightning AI | ~22 h/month T4, persistent workspace | iterating on the gate classifier |
| Vast.ai / RunPod | ~$0.20–0.40/hr RTX 4090 | optional; ~$1–3 for a full run, buys convenience not accuracy |

Students: the [GitHub Student Developer Pack](https://education.github.com/pack) adds GitHub
Pro, Azure and DigitalOcean credits, JetBrains IDEs and a free domain for a year — so the site
can live somewhere better than `github.io`.

| Notebook | Model | Notes |
|---|---|---|
| `training/turbine/turbine_v2_kaggle.ipynb` | yolo11s @ 960 | 3 defect classes; ~2–3 h for 50 epochs |
| `training/solar/solar_v1_kaggle.ipynb` | yolo11s @ 960 | RGB only; brings its own data from Roboflow |
| `training/gate/gate_kaggle.ipynb` | MobileNetV3-Small @ 224 | turbine / solar / invalid; minutes to train |

Key changes from the v1 run, each traceable to the audit:

| Setting | v1 | v2 | Why |
|---|---|---|---|
| model | yolov8m (25.9M) | **yolo11s (9.4M)** | smaller + higher res beats bigger + low res on small defects, trains faster, and has to fit a browser download |
| `imgsz` | 640 | **960** | 21.6% of `surface_peeling` boxes are under 1% of frame area |
| `epochs` | 100 | **50**, patience 10 | v1 peaked at 38, flat from ~30 |
| `optimizer` | `auto` | **explicit SGD**, warmup 5, `cos_lr` | `auto` chose MuSGD and spiked lr to 0.029, collapsing mAP50 0.411 → 0.183 at epoch 3 |
| `cache` | `True` (7.5 GB RAM) | **`disk`** | Ultralytics flags RAM caching non-deterministic even at `seed=0` |
| `device` | `0,1` | **single GPU** | DDP sync across two T4s cost more than the second GPU returned |

### Evaluate honestly

```bash
# per-class AP + confusion matrix on the untouched test split
python3 tools/evaluate.py runs/turbine_v2/weights/best.pt \
    --data datasets/turbine_v2/data.yaml --split test --imgsz 960

# the check v1 never had: real photos from an unrelated source, and look at the boxes
python3 tools/evaluate.py runs/turbine_v2/weights/best.pt --predict path/to/real_photos
```

v1 and v2 aggregates are **not comparable** — different class sets and different splits.
Compare the three shared defect classes, each model on its own split.

---

## The web app

Live inference in the browser via [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/)
(WebGPU, WASM fallback). GitHub Pages cannot run Python, and this is a better fit than a backend
anyway: no cost, no cold start, and **images never leave the visitor's device**.

```
upload (1..N)
  └─> gate: turbine | solar | invalid          MobileNetV3-Small, ~2 MB
        ├─ invalid / uncertain -> refused, with a reason
        ├─ turbine -> turbine detector          yolo11s int8, ~10 MB
        └─ solar   -> solar detector            yolo11s int8, ~10 MB
              └─> severity scoring -> results -> JSON / annotated PNGs / PDF
```

A detector has no way to say "that's a cat" — shown an out-of-domain image it emits confident
boxes regardless. The gate is what makes refusal possible. Where it is absent, the app asks the
user to pick a type rather than guessing.

### Deploying a model

`web/models/manifest.json` is the source of truth for weights, class names, image size,
thresholds and severity weights. Deploying is a file drop plus a JSON edit — no code change:

```bash
python3 tools/export_onnx.py runs/turbine_v2/weights/best.pt --name turbine --imgsz 960
```

This exports at opset 12 (what WebGPU prefers), int8-quantises, writes to `web/models/`, and
updates the manifest. **Quantisation is not free** — it usually costs a couple of points of
mAP. Measure it before shipping:

```bash
python3 tools/evaluate.py web/models/turbine.onnx --data datasets/turbine_v2/data.yaml
```

Commit `web/models/` and `.github/workflows/pages.yml` publishes the site. Until weights are
present the app shows an explicit banner rather than failing silently.

### Scope: solar is RGB only

**Detects** soiling and dust, bird droppings, cracked or shattered glass, delamination and
discoloration, vegetation shading, missing modules.

**Cannot detect** hotspots, cell defects, bypass-diode failure, offline modules. Those are
*electrical* faults, visible in thermal infrared and essentially invisible in RGB. That is
physics, not a modelling gap — no amount of training data changes it, and the UI says so.

If a thermal camera is added later,
[RaptorMaps InfraredSolarModules](https://github.com/RaptorMaps/InfraredSolarModules) is the
strongest public dataset: 20,000 real IR images, 12 classes, already cropped to single modules.

---

## Tests

```bash
python3 tests/make_fixtures.py     # tiny ONNX models with hand-computed outputs
node    tests/test_web.mjs         # 37 checks in headless Chromium
```

Fixture outputs are calculated by hand, so the suite asserts exact boxes and scores rather
than snapshots: YOLO decoding, NMS collapsing overlapping boxes, letterbox coordinates mapping
back to original pixels, gate routing and rejection, severity arithmetic, JSON export, and both
degraded modes. `onnxruntime-web` is served locally there, so it runs offline.

CI (`.github/workflows/ci.yml`) runs that suite, rebuilds the dataset, audits it, and asserts
the audit **still fails** the original export — a guard against the audit quietly degrading
into something that passes everything.

## Reports

```bash
python3 report_generator.py inspection_summary.json report.pdf \
    --annotated-dir annotated/ --asset-id TRB-042 --model "yolo11s turbine v2"
```

Consumes the JSON the web app exports, so browser and PDF agree on one schema. Rejected and
errored images are excluded from the statistics — counting a refused upload as "healthy" would
inflate the pass rate, which is the one number a report must not overstate.

## Data

Turbine data is the [wind_turbine_healthy](https://universe.roboflow.com/akbars-workspace-hcecg/wind_turbine_healthy-jm5zo)
Roboflow export (v1, Public Domain): 7,520 images at 640×640, 3,133 unique sources tripled by
offline augmentation in the training split.

Worth adding: [*Multiclass Dataset for Intelligent Detection of Wind Turbine Blade Defects
Using Drone Imagery*](https://pmc.ncbi.nlm.nih.gov/articles/PMC12996307/) (Scientific Data,
2026) — 1,065 real UAV blade images across 6 classes. Real drone imagery with defects and
healthy blade in the same frame, which is exactly what the current data lacks. The PDF is in
this repo.

`datasets/` is gitignored; regenerate it with `tools/rebuild_turbine.py`.

## Known gaps

- **No models are trained yet.** The notebooks are ready to run; `best.pt` is the v1 model and
  should not be deployed.
- **Background negatives are out-of-domain.** They come from aerial wide shots, not the defect
  close-up domain, so they only partly teach false-positive suppression. In-domain healthy
  frames would be the stronger fix.
- **No imagery from a real target drone**, so all three models will need fine-tuning before
  field use. The pipeline is built to make that a small step.
- The original 349 MB of images are in git history; removing them from the working tree would
  not shrink a clone, and rewriting history is destructive, so they stay as the read-only
  rebuild source.

## Licence

Dataset is Public Domain via Roboflow — see `README.dataset.txt`.
