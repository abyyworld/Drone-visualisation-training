# Handoff - Turbine & solar inspection (this repo)

State of `Drone-visualisation-training` as of the audit-and-rebuild work. Paste this into a
fresh session to pick the project back up.

Consider renaming the repo to **`drone-inspection`** - it no longer contains only training
data, and the GitHub Pages URL is derived from the repo name.

---

## What was wrong, and what changed

The v1 turbine checkpoint (`best.pt`, yolov8m, 58 epochs) reports **mAP50 0.782 / mAP50-95
0.520** and is useless on real photos. `tools/audit_dataset.py` shows why:

- **Source-family shortcut.** The three filename families partition the classes.
  `Healthy_Train*` and `Areial_Healthy*` contain only `healthy`; bare-numeric files contain
  only defects. **Not one image in 7,520 holds a healthy region and a defect together**, so
  the model scored 0.78 by recognising capture style, not defects.
- **32.5%** of held-out images have a same-class box at IoU>0.5 with their adjacent capture
  frame in train - validation was partly a memorisation test.
- Median `healthy` box covers **44%** of the frame vs **4.8%** for `surface_peeling`. A 9.2×
  spread the loss cannot balance.
- Mixed polygon/box labels; 16 `corrosion` instances in test; the test split was never run.

Fixed by `tools/rebuild_turbine.py`: `healthy` dropped as a class (no detections *is* the
healthy result), those images kept as background negatives, group-aware split on contiguous
capture-ID blocks, polygons normalised to boxes.

| | v1 | v2 |
|---|---|---|
| Near-duplicate leakage | 32.5% | **0%** |
| Class median box-area spread | 9.2× | **1.9×** |
| Audit | 6 of 7 fail | **7 of 7 pass** |

---

## What exists and is verified

- `tools/audit_dataset.py` - catches all of the above, exits non-zero, pure stdlib
- `tools/rebuild_turbine.py` - produces `datasets/turbine_v2` (gitignored, regenerable)
- `tools/export_onnx.py` - opset 12, int8 quantise, rewrites the web manifest
- `tools/evaluate.py` - per-class AP, confusion matrix, `--predict` for unlabelled images
- `tools/image_quality.py` - per-class focus measurement; `--help-blur` for why not to filter
- `tests/test_export_contract.py` - proves a real Ultralytics export matches the browser decoder
- `web/` - complete browser app, inference via ONNX Runtime Web, domain gate, severity
  scoring, JSON/PNG/PDF export
- `tests/` - **37 checks passing** in headless Chromium against ONNX fixtures with
  hand-computed outputs
- `.github/workflows/` - Pages deploy + CI
- `report_generator.py` - CLI, consumes the web app's JSON export
- Three Kaggle notebooks: turbine, solar, gate

## The engine split

The app now has two engines, chosen in the UI.

**On-device** is the ONNX path described above and is still the goal. **Provider API**
(`web/js/vlm.js`, three providers, key entered in the page) sends the image to a vision model
instead. It exists because the trained detector reported zero detections on a turbine with a
blade severed in half, and no amount of retraining on a class list that lacks "structural
failure" fixes that.

The API is a bridge, not the destination:

- `tools/vlm_inspect.py` runs the same prompt from the command line and writes a label
  sidecar per image, plus `inspection_summary.json` in the schema `report_generator.py`
  already consumes.
- `tools/vlm_to_yolo.py` accumulates reviewed sidecars into a YOLO dataset, split by whole
  inspection so validation is not a memorisation test.
- The prompt itself is `web/prompts/inspection.json`, read by both the browser and the CLI so
  they cannot diverge.

That dataset is made of imagery from the actual drone at the actual framing, which is the one
thing the failed model never had. Train on it and the API becomes optional.

Real-time never gets an API. The wildfire and crowd stations decode video at 25fps on a network
with no internet (`docs/DEPLOYMENT.md`); a network round trip cannot be in that loop.

## Deployment on a tablet

`docs/INSTALL-tablet.md`. Web app manifest plus a service worker: installs from the browser
with an icon and its own window, no APK to sign. The on-device engine works offline once the
models are cached; the API engines cannot, and the app says so rather than queueing.

## What does not exist

**No models are trained.** `archive/turbine-v2/` holds the superseded one, with its metrics and
a written account of why 0.759 mAP50 did not survive a real photograph. It must not be
deployed.

---

## Next steps, in order

0. **Rebuild with the sharpness floor**: `python3 tools/rebuild_turbine.py --min-sharpness 20`
   - drops 388 images (15%) where the box sits on a smear. Recommended; see `--help-blur`.
1. **Train the turbine model.** `training/turbine/turbine_v2_kaggle.ipynb`, ~2–3 h on a Kaggle
   P100. Upload this repo as a Kaggle Dataset and attach it first.
   > **Expect aggregate mAP50 to fall to ~0.45–0.60.** That is the model losing a score it was
   > cheating for. Judge per-class defect AP and the `--predict` reality check.
2. **Reality check.** Run `tools/evaluate.py --predict` over turbine photos from any unrelated
   source and *look at the boxes*. This is the check v1 never had.
3. **Source solar data.** Roboflow Universe RGB solar sets. **Audit before training** -
   assume the same defects as the turbine set until proven otherwise.
4. **Train solar**, then **train the gate** (turbine/solar/invalid - minutes to train; the
   `invalid` class needs diverse negatives, including hard ones like metal structures and
   blue rectangles).
5. **Export all three**, commit `web/models/`, and Pages deploys automatically.
6. **Before deploying any model**, run the export contract check - it catches an
   ONNX/manifest mismatch in 30 seconds, versus finding it as subtly-wrong boxes on a live
   site:
   ```bash
   python3 tests/test_export_contract.py --weights runs/turbine_v2/weights/best.pt \
       --imgsz 960 --name turbine
   ```
   Verified already against a real export: output layout `(1, 7, 2100)` matches what
   `detect.js` assumes, and confidences agree with `model.predict()` to 1.6e-05.
7. Optionally add the [Multiclass Wind Turbine Blade Defect dataset](https://pmc.ncbi.nlm.nih.gov/articles/PMC12996307/)
   (1,065 real UAV images, 6 classes) - real drone imagery with defects and healthy blade in
   the same frame, which is exactly what the current data lacks. The PDF is already in the repo.

## Gotchas

- Kaggle: use **GPU P100**, not T4×2. The v1 run used `device=0,1` and DDP sync cost more than
  the second GPU returned.
- Quantisation costs a couple of points of mAP. `tools/evaluate.py` accepts `.onnx`, so
  measure it rather than assuming.
- `datasets/` is gitignored - regenerate with `tools/rebuild_turbine.py`.
- The original 349 MB of images are in git history. Removing them from the working tree would
  not shrink a clone, and rewriting history is destructive, so they stay as the read-only
  rebuild source.
- Background negatives come from aerial wide shots, not the defect close-up domain, so they
  only partly teach false-positive suppression. In-domain healthy frames are the stronger fix.

## Commands

```bash
python3 tools/audit_dataset.py .                     # v1: 6 of 7 fail (expected)
python3 tools/rebuild_turbine.py
python3 tools/audit_dataset.py datasets/turbine_v2   # 7 of 7 pass

python3 tests/make_fixtures.py && node tests/test_web.mjs   # 37 checks
npm run serve                                        # http://localhost:8080
```
