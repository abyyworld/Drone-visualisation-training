# Training on Kaggle — exact steps

Written to be followed literally. Roughly 20 minutes of your attention, then
1.5–4 hours of waiting depending on which run you pick.

Before anything: **do run 0 first.** It is not about accuracy. It proves the
whole chain — train → export → class order survives → weights load in the
station → boxes appear on video. A 1.5-hour run that finds a pipeline bug is
worth far more than a 4-hour one that produces a number you cannot use.

| Run | Data | P100, 50 epochs | Fits the 9 h session cap? |
|---|---|---|---|
| **0** | FLAME + Boreal (~10k, genuine UAV) | ~1.5 h | yes, comfortably |
| **1** | + FASDD UAV & negatives (~30k) | ~4 h | yes |
| **2** | full FASDD (122,634) | ~12–16 h | **no** — rent a 4090, ~$2 |

---

## 1. Get the data onto Kaggle

Download on your own machine, then upload each as a **Kaggle Dataset**
(*Datasets → New Dataset*). Private is fine.

| Source | Where | Notes |
|---|---|---|
| **FLAME** | IEEE DataPort | free account; genuine UAV — start here |
| **Boreal Forest Fire** | via the *Scientific Data* 2025 paper | check the licence |
| FASDD | Zenodo / Science Data Bank | accept terms; run 1 onward |
| D-Fire | GitHub `gaiasd/DFireDataset` | ground-level, weighted down to 0.25 |

Search Kaggle first — several of these are already mirrored there, which saves
the upload entirely.

## 2. Make the folder names match

`training/dataset_config.yaml` expects paths like `{data_root}/FLAME/segmentation`.
Kaggle mounts each dataset at `/kaggle/input/<slug>/`, and slugs are
lowercase-hyphenated — so `/kaggle/input/flame-dataset/...` will **not** match
`FLAME` and the merge will skip it silently.

Fix it in the notebook, in **Cell 4**, before running. Either point `DATA_ROOT`
at a directory you have symlinked into shape, or simplest — add a cell after
Cell 3:

```python
# Bridge Kaggle's slugs to the names dataset_config.yaml expects.
import os
from pathlib import Path
BRIDGE = WORK / "data-root"; BRIDGE.mkdir(exist_ok=True)
for want, slug in {
    "FLAME": "flame-dataset",            # <- replace with YOUR actual slugs,
    "BorealForestFire": "boreal-forest-fire",   #    from `ls /kaggle/input`
}.items():
    src = Path("/kaggle/input") / slug
    if src.exists() and not (BRIDGE / want).exists():
        os.symlink(src, BRIDGE / want)
DATA_ROOT = BRIDGE
print(sorted(p.name for p in BRIDGE.iterdir()))
```

Run `ls /kaggle/input` in a cell first and use what it actually prints.

## 3. Create the notebook

1. *Code → New Notebook*, then **File → Import Notebook** and upload
   `training/train_kaggle.ipynb` from this repo.
2. **Settings → Accelerator → GPU P100** (T4 x2 also works; the notebook uses one).
3. **Settings → Internet → On** — needed to `pip install ultralytics` and clone the repo.
4. **Add Data** → attach the datasets from step 1.

## 4. Point it at this repository

The notebook looks for a checkout, and clones one if you tell it where. Add a
cell **before Cell 3**:

```python
import os
os.environ["WILDFIRE_REPO_URL"] = "https://github.com/abyyworld/wildfire-analysis"
os.environ["WILDFIRE_REPO_REF"] = "claude/wildfire-watch-setup-66paiq"
```

It needs the repo rather than re-implementing anything, because three files own
decisions that must not drift: `station/core/types.py` owns the class order,
`tools/audit_dataset.py` owns the leakage gate, `training/prepare_datasets.py`
owns the merge.

## 5. Set the run scope

In **Cell 4**, `EXCLUDE` lists sources that are *not* attached. Anything left in
the list is skipped; anything not listed must exist on disk or the merge fails.

For **run 0** (FLAME + Boreal only):

```python
EXCLUDE = ["fasdd_uav", "fasdd_cv", "fasdd_rs", "dfire", "corsican",
           "mined_negatives", "mined_positives"]
```

For **run 1**, drop `fasdd_uav` and `fasdd_cv` from that list once FASDD is attached.

## 6. Run it — and stop at the gate

*Run All*. Then **stop and read Cell 5, the audit gate.** It is the one cell
whose output decides whether the rest is worth anything.

If it reports **FAIL**, do not train. Fire datasets are cut from video, so a
random train/val split puts frame 0412 in train and the near-identical 0413 in
val; validation then measures memorisation and every number after it is
fiction. `prepare_datasets.py` splits grouped by source video precisely so this
cannot happen — a FAIL means something upstream is wrong (usually a source
whose frames carry no recoverable video id). Fix the grouping, re-merge, re-run
the gate.

A **WARN** is readable: check which check warned and whether it matters for
your run.

Training then runs to `SESSION_BUDGET_H = 8.25` hours and checkpoints as it
goes, so a session that dies is resumable — re-run the notebook and Cell 7
picks the checkpoint back up.

## 7. Read the validation output correctly

Cell 9 prints recall per class, **per box size**, and per condition tag. Read
that table before anything else, and read the **tiny/small** columns first:
distant early fire is both the highest-value detection and the one most often
missed, and an overall average hides it completely.

Treat mAP with suspicion. Published aerial RGB fire/smoke detectors land around
**0.80 mAP@0.5** on honest held-out data. Markedly above that on your own split
is far more often leakage than skill — the notebook says so itself when it
happens.

You can rehearse this whole reading on the demo clip before any real data
exists:

```bash
python demo/validate_demo.py     # builds a split, scores it, writes the miss list
```

## 8. Bring the weights back to the station

Download `best.pt` from the notebook output, then:

```bash
mkdir -p models && cp ~/Downloads/best.pt models/yolo11s-fire.pt
python tools/export.py --weights models/yolo11s-fire.pt --format onnx
```

`export.py` asserts the exported class order matches `station.core.types.CLASSES`.
Do not skip it: a silent fire/smoke swap is invisible in every metric and
catastrophic in the field.

Then point the station at it:

```yaml
inference:
  weights: models/yolo11s-fire.pt
  model_version: "0.1.0"      # bump this every run; it is written into every
                              # incident log and is how a recording is traced
                              # back to the model that produced it
```

```bash
python -m station check          # should now say the model can run
python -m station -c config.yaml run
```

## 9. Before anyone relies on it

`docs/VALIDATION.md`. The short version: measure **false negatives** on the
department's own footage, broken out by the conditions that break RGB
detection — thin smoke against bright sky, smouldering with no visible flame,
fire under canopy, night, and small distant fire. Nothing in this document
substitutes for that.
