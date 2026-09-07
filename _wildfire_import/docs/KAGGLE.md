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

## 2. Create the notebook

1. *Code → New Notebook*, then **File → Import Notebook** and upload
   `training/train_kaggle.ipynb` from this repo.
2. **Settings → Accelerator → GPU P100** (T4 x2 also works; the notebook uses one).
3. **Settings → Internet → On** — needed to `pip install ultralytics` and clone the repo.
4. **Add Data** → attach the datasets from step 1.

## 3. Run All

That is the whole procedure. There is nothing to edit.

The notebook clones this repository itself, finds whatever datasets you
attached, and trains on those — so the same file works for run 0 and run 1 with
no change beyond which datasets are attached.

Two things it handles that used to need hand-editing, both of which are easy to
get wrong and expensive to get wrong:

* **Kaggle renames your datasets.** A folder uploaded as `FLAME` is mounted at
  `/kaggle/input/flame-dataset`, which does not match the paths in
  `dataset_config.yaml`. The notebook bridges the two by looking for the
  content the config expects, not just a matching name — a dataset can sit at
  the mount, one level below it, or under a completely unrelated slug. It also
  refuses lookalikes: `norm("D-Fire")` is `"dfire"`, which is a substring of
  `norm("wildfire-dataset")`, and merging that in as ground-level imagery
  would quietly corrupt the mix.
* **Sources you did not attach are excluded automatically**, so the merge runs
  on what is present instead of failing on what is not.

Both behaviours are covered by `tests/test_kaggle_discovery.py`, which runs the
notebook's own code against simulated mounts.

Read the cell's output before moving on. It prints:

```
attached: ['boreal-forest-fire', 'flame-dataset']
  FLAME                <- /kaggle/input/flame-dataset/FLAME
  BorealForestFire     <- /kaggle/input/boreal-forest-fire/BorealForestFire
included: ['flame_seg', 'flame_negatives', 'boreal_uav']
excluded (not attached): ['fasdd_uav', 'fasdd_cv', ...]
```

If `included` is empty, or is missing something you attached, the names under
`attached:` tell you what Kaggle actually mounted. Nothing aerial in the list
earns a warning: ground-level data alone trains a model for the wrong
viewpoint, because a drone looks down and D-Fire does not.

## 4. Save Version → Save & Run All (Commit)

Use **Save Version → Save & Run All (Commit)** rather than the interactive
session. A committed run keeps executing after you close the tab, which matters
when the thing takes four hours, and the output is versioned so a later run can
be compared against it.

The interactive session is the right choice only when you are still sorting out
which datasets attached correctly — it is faster to iterate on, and cheaper to
interrupt.

## 5. Stop at the audit gate

When the run reaches **Cell 5, the audit gate**, stop and read it. It is the one cell
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

## 6. Read the validation output correctly

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

## 7. Getting the results back automatically (optional)

The last cell pushes the run to a **`training-runs`** branch — weights, model
card, audit report and metrics, one directory per run — so you do not have to
download and move files by hand. It is skipped silently unless a token is
available, and a failure there never fails the run: by that point the artifacts
already exist and can be downloaded from the notebook's Output tab.

Set it up once:

1. **Create a fine-grained token**:
   github.com/settings/personal-access-tokens/new → *Only select repositories* →
   this repo → **Contents: Read and write**. Nothing else.
2. **Kaggle → Add-ons → Secrets → Add a secret**, named exactly `GITHUB_TOKEN`.
3. That is all. The cell reads it from Kaggle's vault at runtime.

**Never put the token in the notebook, the repo, or a message.** GitHub's secret
scanning revokes tokens that appear in a repository, usually within minutes, so
a pasted token stops working anyway — and everything under `/kaggle/working` is
published as your notebook's output, so a token written there leaks with it. The
push cell clones to `/tmp` for exactly that reason, passes the credential inline
rather than storing it in `.git/config`, scrubs it from any error message, and
deletes the clone afterwards.

Artifacts land on a separate branch, not the code branch, so training output
never mixes with source history. Weights are ~19 MB per run; set
`PUSH_WEIGHTS = False` in that cell to push only the metrics and the model card.

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
