# training/ — building the model the ground station runs

This directory produces one artifact: a `yolo11s` checkpoint that emits two
classes, `fire` and `smoke`, **in that index order**, which is
`station.core.types.CLASSES` and therefore the order the tablet colours boxes
by. Everything else here exists to make that checkpoint trustworthy, or to make
its untrustworthiness visible.

| File | What it is |
|---|---|
| `dataset_config.yaml` | the source table: which datasets, what viewpoint, what weight, what licence to check |
| `prepare_datasets.py` | merges them into one YOLO dataset, **split by source video** |
| `train_kaggle.ipynb` | the training run: audit gate, checkpoint-resume, validation, model card |
| `hard_negatives.md` | the false positives to expect, and how to mine more from our own incident logs |

Read `hard_negatives.md` before you tune anything. It is where the reasoning
about the precision/recall trade lives, and that trade is the whole design.

---

## What this model is, and what it cannot be

A situational-awareness aid. It draws boxes on video an operator is already
watching, and those boxes mean *look here*.

It cannot mean the opposite. RGB fire and smoke models fail toward **silence**:
thin smoke against a bright sky, smouldering with no visible flame, fire under
canopy, fire at night — in every one of those the model returns an empty list,
and that empty list is byte-identical to the one it returns for an empty field.
No metric in this directory changes that, so no metric in this directory may be
quoted as coverage. See `station/core/safety.py`, which enforces the language
this constraint requires across the whole repository.

Consequences that show up as engineering decisions below: the confidence
threshold is low, background imagery is treated as first-class training data,
recall is the metric that gets read first, and the incident log records every
frame including the empty ones.

---

## End to end

### 0. Prerequisites

```bash
python3 -m pip install pyyaml                # prepare_datasets.py: required (reads the config)
python3 -m pip install pillow numpy          # only for mask-format sources, and the audit's pixel checks
```

`prepare_datasets.py` deliberately imports nothing heavy at module level. It
runs on a laptop with no GPU, no ffmpeg and no PyTorch — the same constraint the
rest of the repo's pure-logic modules hold to. Pillow is needed only if you
enable a `format: mask` source (FLAME, Corsican); without it those sources fail
with a clear message and the rest of the merge still works.

Training itself needs `ultralytics` and a GPU, which is what
`train_kaggle.ipynb` is for.

### 1. Get the raw datasets

Download and unpack under `datasets/raw/` (or anywhere — `--data-root` points at
it). Add `datasets/` to your `.gitignore`; none of this belongs in git.

```
datasets/raw/
├── FASDD/{FASDD_UAV,FASDD_CV,FASDD_RS}/{images,annotations}
├── FLAME/{segmentation/{images,masks},classification/No_Fire}
├── BorealForestFire/{images,labels}
├── D-Fire/{images,labels}
└── Corsican/{images,masks}
```

`dataset_config.yaml` documents each one: size, viewpoint, weight and a licence
line. **The licence line is a reminder, not a clearance.** Several of these are
research-use-only, and at least two aggregate web-scraped images whose
provenance nobody can now establish. Check the current terms at the source
before you redistribute anything trained on them, and record what you checked.

The layout above is what the shipped config expects. If yours differs, change
the `images:` / `labels:` / `annotations:` keys rather than moving files around
— the config is the record of what was merged.

### 2. Merge

```bash
python3 training/prepare_datasets.py \
    --config training/dataset_config.yaml \
    --data-root datasets/raw \
    --dry-run                      # scan, weight, split, report. Writes nothing.
```

Read the dry-run report. It shows, per source, how many images survived
weighting and why the rest did not; and per split, the image, box, class and
viewpoint breakdown. If the viewpoint column says the merge is mostly `ground`,
you are about to train a security-camera model and validate it on
security-camera footage.

Then, for real:

```bash
python3 training/prepare_datasets.py \
    --config training/dataset_config.yaml \
    --data-root datasets/raw \
    --json datasets/merge_report.json
```

Output:

```
datasets/wildfire-merged/
├── data.yaml           # path/train/val/test, nc: 2, names: [fire, smoke]
├── manifest.json       # what came from where, with the config hash and seed
├── provenance.jsonl    # one line per output image -> its original path and group
├── images/{train,val,test}/wf_<group>_<index>.jpg
└── labels/{train,val,test}/wf_<group>_<index>.txt
```

Useful flags: `--only NAME` / `--exclude NAME` to select sources, `--max-images N`
for a quick smoke-test merge, `--copy-mode copy` where symlinks will not survive
(packaging the merge as a Kaggle dataset, for instance), `--force` to overwrite,
`--audit` to run the audit immediately afterwards and inherit its exit code.

### 3. Audit — the gate

```bash
python3 tools/audit_dataset.py --data datasets/wildfire-merged/data.yaml
```

Exit 0 means it may be trained on; exit 1 means it must not be. The notebook
runs this itself and refuses to train on a failure. Do not remove that cell.

`WARN` passes deliberately: warnings (class imbalance, no tiny boxes, a single
resolution) change how the resulting numbers should be *read* rather than
invalidating them. They belong in the model card, which the notebook does
automatically.

### 4. Train

Open `train_kaggle.ipynb`. It is written for Kaggle but runs anywhere with a
GPU. In order it: probes the environment, pins `ultralytics`, finds the repo,
builds or locates the merged dataset, **runs the audit gate**, trains inside a
session time budget with checkpoint mirroring, validates with a
recall-focused confidence sweep, and writes `model_card.json`.

Compute reality, `yolo11s` at 640, batch 16, on a Kaggle P100:

| Dataset | Images | 50 epochs | Fits Kaggle's 9 h session? |
|---|---|---|---|
| Full FASDD | 122,634 | ~12–16 h | **No** |
| Weighted merge subset | ~30,000 | ~4 h | Yes |
| FLAME + Boreal (UAV only) | ~10,000 | ~1.5 h | Yes |

**For a full-FASDD run, rent a GPU.** A 4090 on Vast.ai or RunPod is 4–6× a
P100 for this workload and costs about **$2** for the whole run. A 14-hour job
squeezed across two Kaggle sessions is not free — it is a resume mechanism you
have to trust, twice, and a session timeout mid-run is exactly what wasted the
sibling project's first attempt. Kaggle is the right tool for the ~30k merge,
the UAV-only runs and anything where the point is a relative comparison.

Resuming across sessions: `/kaggle/working` is preserved as the notebook's
output, so attach the previous session's output as an input and re-run every
cell. Cell 7 finds `runs/<name>/weights/last.pt`, copies it into the working
tree (resume fails from the read-only input mount) and continues at the exact
epoch it stopped on.

### 5. Before the weights fly

1. Copy `model_card.json` next to the weights and set `inference.weights`,
   `inference.model_name` and `inference.model_version` from it. Every logged
   frame then names the model that produced it — without that an incident log
   cannot be replayed against the model that generated it.
2. Evaluate on **your own drone footage**, not the val split. The val split
   shares cameras and terrain with training; your airframe does not.
3. Run the false-negative review: watch footage containing known fire and count
   what the model did not box. That number never appears in mAP and it is the
   one that matters.
4. Feed the mistakes back through `hard_negatives.md`.

---

## Why the split is by video, and why that is enforced here

A fire dataset built from video contains runs of consecutive frames. Frame 411
and frame 412 are the same photograph with the sensor noise moved. Split those
across train and val and validation stops measuring generalisation and starts
measuring memorisation: every number rises, and the rise is indistinguishable
from progress. This is the default state of every scraped fire dataset anyone
has looked at.

`tools/audit_dataset.py` detects it. `prepare_datasets.py` prevents it, which
is the better place:

* Every source declares a **`grouping`** — `filename_family`, `parent_dir`,
  `regex` or `per_image` — and the merge assigns each whole group to exactly one
  split. There is no flag that splits by image. `per_image` exists for genuinely
  independent stills and requires `independent_images: true` to be written into
  the config as an explicit, recorded claim.
* Output filenames are `wf_<group>_<index>`, which makes the "filename family"
  the audit derives *identical* to the group the merge split by. The two tools
  then agree by construction instead of by coincidence.
* No filename carries a source or class token, so no token can predict a label
  — the other shortcut the audit hunts for. Provenance lives in
  `provenance.jsonl`, where it cannot leak into training.
* Before writing anything, the merge re-derives every sample's split and fails
  hard if one group reached two splits. That guard exists because a bug in the
  splitter would otherwise be invisible until the audit ran, and possibly not
  then.

The cost is that split proportions are approximate — groups are indivisible and
vary in size by three orders of magnitude — so the merge reports the realised
percentages and warns when they drift. Approximate proportions with an honest
split beat exact proportions with a leaking one.

## Why viewpoint weighting exists

This model flies. Ground-level datasets (D-Fire ~21.5k, Corsican) are a
security-camera viewpoint: correct classes, wrong geometry, wrong scale, wrong
background statistics. Merged at face value they outnumber the genuine UAV data
and the model learns their world.

So each source carries a `weight` (the fraction of its **groups** kept) and,
separately, a `negative_weight`. Ground-level *positives* are thinned to
0.15–0.25; their *negatives* are kept at 1.0, because a red roof, a high-vis
jacket and a dust plume are confusing from any angle. `frame_stride` thins dense
video within a group, which removes near-duplicate frames without removing
scenes — a different operation from dropping groups, and the right one for
continuous footage.

## Why the negatives are the point

FASDD ships roughly 52,000 negative images: cloud, fog, sunset, industrial haze,
red-lit scenes. They are preserved through the merge (`max_negative_fraction:
null` by default) and written with **empty** label files rather than missing
ones, because the two mean different things — an empty label is a deliberate
"there is nothing to box here", a missing one is a broken export.

An unusually high background fraction is deliberate. The station runs at
`conf_threshold: 0.25`, low on purpose, so false positives are expected, and
background imagery is the only thing that teaches the model the difference
between a plume and a dust cloud. See `hard_negatives.md` for the catalogue and
the mining loop that grows this set from our own incident logs.

---

## Reproducing a result

A number without its dataset is not a result. Three things pin it down:

* `manifest.json` — every source, its weight, its group strategy, what was
  dropped and why, the split policy, the seed and the SHA-256 of the config
  file that produced it.
* `provenance.jsonl` — every output image mapped back to its original path,
  source dataset, group key and content hash.
* `model_card.json` — the training arguments, the environment versions, the
  repo commit, the audit status and warnings, the metrics, and the weights hash
  that `ModelInfo.weights_sha` reports in every logged frame.

Same config, same seed, same raw data gives byte-identical output filenames and
the same split. Changing the seed reshuffles which whole videos land in which
split; it does not change the policy, and it must be recorded alongside any
number you quote.

## Troubleshooting

**`REFUSED: source 'x': path does not exist`** — the source is in the config but
not on disk. Point `--data-root` at the right place, `--exclude` it, or set
`enabled: false`. A missing source is refused rather than skipped on purpose: a
manifest that describes datasets which were not actually merged is a manifest
that lies.

**`REFUSED: format 'yolo' requires 'class_names'`** — the source's own class
index order has to be stated. Read its `data.yaml` or `classes.txt`. D-Fire is
`[smoke, fire]`, the opposite of ours; guessing produces a model whose fire and
smoke are transposed on every tablet, with nothing downstream able to notice.

**`REFUSED: grouping 'per_image' ...`** — you asked for what is effectively a
random split. If the images really are independent stills, say so with
`independent_images: true`. If they are video frames, they are not.

**`REFUSED: only N group(s) across 3 split(s)`** — too few source videos to
split by video. Add sources, or lower a weight less aggressively. Splitting a
group is not an option offered.

**Audit fails on `split-leakage/perceptual`** — near-duplicate images across
splits that filename grouping could not see, usually the same scene scraped into
two datasets. `dedup: true` catches byte-identical copies; re-encoded ones need
the grouping widened, or the offending source excluded.

**Audit warns `0.0% tiny` boxes** — the dataset has no distant targets, so
small-target recall cannot be measured and probably was not learned. Distant
smoke is the detection that buys the most time. Carry this into the model card
and fix it with data.
