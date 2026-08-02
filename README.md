# Drone Visualisation Training

Wind turbine blade defect detection from drone imagery. YOLO object detection over a
4-class Roboflow dataset (`corrosion`, `crack`, `healthy`, `surface_peeling`).

> **Status: incomplete.** The training run in the notebook died at epoch 7 of 100 and the
> resume cell failed. No trained weights are in this repo. See [Known issues](#known-issues)
> before you rely on anything here.

## Contents

| Path | What it is |
|---|---|
| `drone-pics-training.ipynb` | Kaggle training notebook (YOLOv8m, 2× GPU) |
| `data.yaml` | Dataset config — splits, class names |
| `train/`, `valid/`, `test/` | Images + YOLO-format labels |
| `inspection_report.pdf` | Reference inspection report (6.8 MB) |
| `README.dataset.txt`, `README.roboflow.txt` | Original Roboflow export notes |

## Dataset

Exported from [Roboflow](https://universe.roboflow.com/akbars-workspace-hcecg/wind_turbine_healthy-jm5zo)
(v1, Public Domain). 7,520 images at 640×640, auto-oriented and stretch-resized.

Roboflow generated 3 augmented versions per source image: 50% horizontal flip, ±15° rotation,
±25% brightness, 0–1 px Gaussian blur.

| Split | Images | Annotations | Empty (background) |
|---|---|---|---|
| train | 6,581 | 8,484 | 98 |
| valid | 627 | 831 | 8 |
| test | 312 | 409 | 3 |

### Class distribution (train)

| ID | Class | Annotations | Share |
|---|---|---|---|
| 2 | `healthy` | 5,284 | 62.3% |
| 1 | `crack` | 1,790 | 21.1% |
| 3 | `surface_peeling` | 871 | 10.3% |
| 0 | `corrosion` | 539 | 6.4% |

The three defect classes together are under 38% of annotations, and `corrosion` — arguably the
class you most want to catch — is 6%. Expect weak defect recall without rebalancing.

## Known issues

These are real and mostly unresolved. Read this section rather than trusting the notebook.

**1. Training never finished.** The run stopped at epoch 7/100 when the Kaggle session hit its
time limit. Best validation metrics seen before it died were at epoch 2:

```
mAP50 0.411   mAP50-95 0.245   P 0.475   R 0.416
```

Metrics were still bouncing hard between epochs (mAP50 fell to 0.183 at epoch 3), which is normal
this early but means nothing here is a usable result.

**2. The resume cell fails.** Cell 5 re-runs `yolo train ... resume=True` but errors with
`yolo: command not found`. A fresh Kaggle session doesn't carry over the `pip install ultralytics`
from cell 2 — and `/kaggle/working/turbine_v1/weights/last.pt` doesn't survive either. Re-run the
install cell first, and restore the checkpoint from a Kaggle dataset or output snapshot.

**3. Mixed detect/segment annotations.** All 5,284 `healthy` labels are segmentation polygons;
all defect labels are 5-value bounding boxes. Ultralytics warns and drops the segment masks:

```
WARNING ⚠️ Box and segment counts should be equal, but got
len(segments) = 5284, len(boxes) = 8484. ... only boxes will be used
```

For **detection** this is survivable — polygons get converted to bounding boxes, so no annotations
are lost. But this dataset cannot be used for segmentation as-is, and the mixed format will keep
throwing warnings. Normalise to one format if you touch this again.

**4. Hardcoded Kaggle paths.** The notebook points at
`/kaggle/input/datasets/abyyworld/123456/data.yaml`. Meanwhile `data.yaml` uses relative paths
(`../train/images`). Neither works locally without editing. Fix both before running elsewhere.

**5. `cache=True` loads ~7.5 GB into RAM** and is flagged non-deterministic by Ultralytics. Use
`cache='disk'` if you want reproducible runs.

**6. The keep-alive thread is a workaround, not a fix.** Cell 1 touches a file every 5 minutes to
stop Kaggle idling out. It did not prevent the timeout that killed this run — Kaggle's hard session
cap applies regardless. Checkpoint to a persistent location instead (`save_period=5` writes to
`/kaggle/working`, which is wiped between sessions).

## Running it

```bash
pip install ultralytics

yolo train \
  data=data.yaml \
  model=yolov8m.pt \
  epochs=100 imgsz=640 batch=32 patience=20 \
  device=0 workers=4 cache=disk \
  project=runs name=turbine_v1 save_period=5
```

Point `data=` at an absolute path to `data.yaml`, and make the `train:`/`val:`/`test:` entries
inside it resolve from wherever you run. Drop `device=0,1` to a single GPU unless you have two.

## If you pick this back up

Roughly in order of value:

1. Get a full run to finish — everything else is guesswork until then.
2. Checkpoint somewhere that survives session death, so resume actually works.
3. Address the class imbalance (weighted loss, or oversample corrosion/peeling).
4. Normalise annotations to pure bounding boxes, or commit to segmentation and fix the defect labels.
5. Report per-class metrics, not just aggregate mAP — aggregate is flattered by the 62% `healthy` majority.

## Licence

Dataset is Public Domain via Roboflow. See `README.dataset.txt`.
