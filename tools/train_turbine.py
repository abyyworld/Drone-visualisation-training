#!/usr/bin/env python3
"""Train the turbine defect detector.

WHY THIS IS A SCRIPT AND NOT A NOTEBOOK CELL
    Kaggle stores notebooks on Kaggle. The repo is cloned fresh on every run. So a
    hyperparameter living in a notebook cell is a hyperparameter that silently goes stale
    the moment it is fixed here, and the only symptom is a run that behaves like an older
    one. That cost two training runs. Everything that decides how the model trains now
    lives in this file, which every run re-downloads.

    The notebook's job is reduced to: get a GPU, get the data, call this.

EVERY NON-DEFAULT SETTING BELOW HAS A REASON, RECORDED INLINE. Read them before changing
one - most encode a failure that has already happened once.

Usage:
    python3 tools/train_turbine.py --data /kaggle/working/turbine_v2/data.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

# --- Hyperparameters -------------------------------------------------------------------
# Kept as a plain dict so the run log can print exactly what it used. A number you cannot
# see in the log is a number you will misremember.

CONFIG = {
    "epochs": 100,
    "imgsz": 960,          # 21.6% of surface_peeling boxes are under 1% of frame area;
                           # resolution is the dominant lever for small objects.
    "batch": 8,            # 960px on a 16GB T4. Drop to 4 on OOM.

    # patience MUST outlast warmup by a wide margin. A run with warmup=5/patience=10 peaked
    # at epoch 2 while the LR was still ramping and was killed at epoch 12, having never
    # trained at full learning rate. It scored mAP50 0.194.
    "patience": 30,

    # Explicit SGD: the `auto` optimizer chose MuSGD, drove lr0 to 0.029 during warmup and
    # collapsed mAP50 from 0.411 to 0.183 by epoch 3.
    "optimizer": "SGD",
    # lr0 halved from 0.01: val mAP bounced between 0.015 and 0.155 epoch to epoch while
    # training loss fell smoothly - the step size was overshooting.
    "lr0": 0.005,
    "lrf": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 3,
    "cos_lr": True,

    # NOT cache="disk". That writes .npy caches into /kaggle/working, which is network-backed
    # storage on Kaggle. A stalled read hangs a dataloader worker in a syscall that Python
    # cannot interrupt - the session timer and GPU quota keep advancing while the training
    # loop sits on one iteration forever, and the only way out is a restart that wipes the
    # weights. That deadlocked a run at epoch 47. ~1,832 small JPEGs decode fast enough that
    # caching bought almost nothing.
    "cache": False,
    "workers": 2,
    "device": 0,           # single GPU: DDP sync cost more than the second T4 returned.
    "seed": 0,
    "deterministic": True,

    # Defects are small and orientation-varied; blades appear at any angle from a drone.
    "degrees": 10.0,
    "fliplr": 0.5,
    "flipud": 0.2,
    "scale": 0.5,
    "mosaic": 1.0,
    "close_mosaic": 10,

    "save_period": 10,     # a checkpoint every 10 epochs survives a session dying mid-run
    "plots": True,
}

MODEL = "yolo11s.pt"       # 9.4M params. Smaller + higher resolution beats bigger + lower
                           # resolution for small defects, and has to fit a browser download.


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True, help="path to data.yaml")
    parser.add_argument("--project", default="/kaggle/working/runs")
    parser.add_argument("--name", default="turbine_v2")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--epochs", type=int, help="override epochs (for a smoke test)")
    parser.add_argument("--resume", action="store_true",
                        help="continue an interrupted run from its last checkpoint")
    args = parser.parse_args()

    if not args.data.exists():
        raise SystemExit(f"no data.yaml at {args.data} - run tools/rebuild_turbine.py first")

    try:
        import torch
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("run `pip install ultralytics` first")

    if not torch.cuda.is_available():
        raise SystemExit("no GPU - training will take days on CPU. Set Accelerator to GPU T4 x2.")

    config = dict(CONFIG)
    if args.epochs:
        config["epochs"] = args.epochs

    weights = Path(args.project) / args.name / "weights" / "last.pt"
    if args.resume:
        if not weights.exists():
            raise SystemExit(f"nothing to resume: {weights} does not exist")
        print(f"Resuming from {weights}\n")
        YOLO(str(weights)).train(resume=True)
        return 0

    print("=" * 78)
    print(f"  model   {args.model}")
    print(f"  data    {args.data}")
    print(f"  output  {args.project}/{args.name}")
    for key, value in config.items():
        print(f"  {key:<16}{value}")
    print("=" * 78 + "\n")

    YOLO(args.model).train(
        data=str(args.data), project=args.project, name=args.name, exist_ok=True, **config
    )
    print(f"\nWeights -> {Path(args.project) / args.name / 'weights' / 'best.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
