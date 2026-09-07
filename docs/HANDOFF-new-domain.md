# Handoff: starting a domain chat

Paste the block for your domain into a new chat on `abyyworld/Drone-visualisation-training`.
Written to be read cold.

## The block

> You are working on **<DOMAIN>** in `abyyworld/Drone-visualisation-training`, branch `main`.
>
> Read `docs/PLATFORM.md` and this file first. Do not start by writing code.
>
> **Hard conventions, no exceptions:**
> - Every commit is authored `abyyworld <annolieberto@gmail.com>`. Nothing else, ever.
> - No `Co-Authored-By` trailers. No tool names in commit messages, code or docs.
> - No em dashes anywhere.
>
> **Architecture:** one repo, one `main`, domains are profiles in `profiles/`, not branches
> and not separate repos. `stream/`, `serve/` and `incidentlog/` are domain-agnostic. What
> varies is the class list, the weights, the thresholds and the safety wording.
>
> **You own:** `profiles/<DOMAIN>.yaml`, `training/<DOMAIN>/`, and that domain's dataset.
> **You share:** `tools/`, `web/`, `station/`. Say so before changing a shared file, because
> another domain chat may be in it.
>
> **Before training anything, run `python3 tools/audit_dataset.py <dataset>`.** Eight checks.
> Do not train on a dataset that fails one. This is not ceremony: the check for augmented
> inflation exists because a previous dataset was 7,520 files holding roughly 750 real
> photographs, and nothing else revealed it.

## Where things are

The repo carries several products. Most of it is not yours. Read only your column plus the
shared row, and do not go changing a shared file without saying so first.

```
SHARED - every domain depends on these
  tools/                 dataset audit, merge, train, evaluate, export, publish
  training/common/       label parsing, perceptual hashing, sharpness
  station/core/          domain-agnostic runtime types and the safety invariant
  station/{stream,serve,incidentlog}/   ingest, WebRTC out, incident logging
  profiles/              one yaml per domain: classes, weights, thresholds, wording
  tests/                 python and JS suites for the above

REAL-TIME PRODUCT (wildfire, crowd - video in, detections out)
  station/               the runtime; `python -m station check` reports a laptop's capability
  station/inference/     model loading and the N-of-M temporal filter
  app/                   tablet PWA, video plus Canvas overlay, no build step
  demo/                  offline renderer and a synthetic clip with ground truth
  training/wildfire/     dataset prep and the Kaggle notebook
  Makefile               `make demo`
  config.example.yaml

INSPECTION PRODUCT (turbine, solar - photos in, PDF report out)
  web/                   the browser app, ONNX Runtime Web, client-side inference
  web/models/            manifest.json is the deploy switch; drop an .onnx beside it
  training/turbine/      Kaggle notebooks
  training/solar/
  report_generator.py    PDF output

DOCUMENTS worth reading before code
  docs/PLATFORM.md       the architecture decision. Read this first.
  docs/DATASETS.md       15 sources with licence status. One is confirmed commercial.
  docs/SAFETY.md         the invariant about absence of detection
  docs/KAGGLE.md         how to run training
  docs/CONTRACT.md       the wire format between station and app
  docs/HARDWARE.md
  docs/VALIDATION.md
  docs/OPEN_QUESTIONS.md
```

Two products, one platform. The real-time one streams video and decides in the moment; the
inspection one takes uploads and produces a report. They share the tooling, the audit
discipline and the safety rule, and nothing else. If you are adding a domain, first work out
which of the two it is: wildfire and crowd are real-time, turbine and solar are inspection.

## What this project learned the expensive way

These cost real GPU time and one deployed model. They apply to every domain.

**A held-out split from the same pool proves nothing.** The turbine model scored mAP50
0.759 with zero leakage and 0.5% false positives on healthy blades, then reported "no
defects" on a photograph of a turbine with a blade snapped in half. Every metric was
computed inside the distribution the data defined. Put out-of-domain images in the test set.
A benchmark that cannot fail is not measuring anything.

**Negatives must come from the same domain as the positives.** That model's "nothing here"
examples were wide aerial shots while its defects were all close-ups, so it learned that
wide framing means no defect. The failing photograph was a wide shot. It did exactly what
it was taught.

**The class list must contain the failure you care about.** No amount of training finds a
severed blade when the classes are corrosion, crack and peeling paint. Decide the class list
against the damage that matters, before collecting.

**Count distinct scenes, not files.** Exporters augment, papers tile. `audit_dataset.py`
reports `N files -> M distinct scenes`; the second number is the dataset.

**Merge classes aggressively.** Six classes of 300 boxes make six weak detectors. Three of
600 make three usable ones. DTU collapsed to two with more data than you will have.

**Absence of a detection is never a positive claim.** No box does not mean no fire. No
defect found does not mean the asset is sound. The interface must say what the model cannot
see, or it is lying by omission on every clean result.

**Configuration lives in the repo, not in the notebook.** Kaggle stores notebooks; the repo
is cloned fresh each run. A hyperparameter in a notebook cell goes stale silently and the
only symptom is a run that behaves like an older one. That cost two runs. See
`tools/train_turbine.py`.

**Use Save Version -> Save & Run All (Commit).** A Draft Session dies with the browser and
takes the weights with it. That cost two more.

## Per domain

**wildfire** - classes fire, smoke, person. Station and app already written on
`platform/import-wildfire-station`; merge it before starting. Datasets: FLAME, Boreal.
Nothing public has fire and people in the same frame.

**crowd** - people at events, aerial. VisDrone is the obvious source and is research-licensed
only, so it cannot ship in a product. Check the licence before building on anything here.

**solar** - RGB only. Cannot show hotspots, cell defects, bypass-diode failure or offline
modules; those are electrical faults needing thermal infrared. Say so in the interface.

**turbine** - four Roboflow sources being merged with `tools/merge_datasets.py`.
