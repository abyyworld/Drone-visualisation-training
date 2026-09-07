# Handoff: the merged platform repo

Paste this into a new chat on `abyyworld/Drone-visualisation-training` and it can
pick up cold. Written at the point where the wildfire station was merged in and
this repo became the platform for every drone-vision product.

## Where things stand

This repo now holds two working products and the plan for three more.

**Inspection** (was here already): solar panel and wind turbine defect
detection, `report_generator.py`, `web/`, `training/{solar,turbine,gate}/`, and
the `tools/` that support them.

**Wildfire station** (merged in from `abyyworld/wildfire-analysis`, which is
being deleted): a complete real-time drone detection pipeline.

- `station/` - the runtime: ingest (RTSP/RTMP/HDMI/file), YOLO inference,
  N-of-M temporal filter, WebRTC out, incident logging, HTTPS serving
- `app/` - tablet PWA, video plus Canvas overlay, no build step
- `demo/` - offline renderer and a synthetic clip with ground truth
- `training/wildfire/` - dataset prep, the Kaggle notebook, dataset config
- `tests/` - 442 python tests, 60 JS tests

Classes are `fire, smoke, person`. Wire version 2.

Verified working end to end: a real WebRTC client connects to the running
station and receives video plus detections carrying `rtp_ts`. `make demo`
renders an overlay video. `python -m station check` reports what a given laptop
can do.

## The decision that shapes everything next

`docs/PLATFORM.md` is the architecture record. Short version:

**One repo, one main, domains as profiles. Not branches, not separate repos.**

`stream/`, `serve/` and `incidentlog/` contain zero references to fire. The
pipeline was already domain-agnostic. What varies per domain is the class list,
the weights, the temporal parameters and the safety wording, which is a config
file.

Planned profiles: `wildfire` (fire, smoke, person), `traffic` (car, truck,
person), `solar`, `turbine`, `event` (crowd density).

Person is a class inside wildfire and traffic rather than a profile of its own,
because a crew works at the flame front and a pedestrian stands next to a car,
and the spatial relationship is the thing worth detecting. Unless the airframe
carries thermal, in which case person moves to the thermal stream and becomes
its own model. See PLATFORM.md for the reasoning.

## What is NOT done

1. **The profile extraction has not been started.** `station/core/types.py`
   still hardcodes the wildfire class list. Step 1 of PLATFORM.md is to lift
   that into `profiles/wildfire.yaml` and stop the platform naming a domain.
   Do this before adding traffic; porting solar and turbine into profiles is
   what proves the abstraction works.
2. **Five paths collided in the merge and were parked, not reconciled:**
   - `tools/audit_dataset.py` (inspection's) vs `tools/audit_dataset_wildfire.py`
   - `tools/evaluate.py` (inspection's) vs `tools/evaluate_wildfire.py`
   - the wildfire README is at `docs/README-wildfire-station.md`
   - the wildfire CI workflow is at `.github/workflows/ci-station.yml`
   - `.gitignore` was merged as a union
   Two real implementations of the same idea, written for different failure
   modes. Choosing between them has decisions in it and should not be done
   blind.
3. **No model is trained.** No dataset is in the repo. See below.
4. **Lint is scoped.** The inspection code is in ruff's `extend-exclude` so the
   merge would not drown in unrelated reformatting. Adopt the rules per file as
   each is next touched.
5. **iOS PWA behind a self-signed cert is unverified.** `app/selftest.html`
   answers it in about 30 seconds on a real iPad; the station prints its URL and
   a QR code at startup.

## Datasets

`docs/DATASETS.md` is the full picture, verified where possible. The headline:

- **Of 15 configured sources, one is confirmed usable commercially** (HIT-UAV,
  CC BY 4.0). VisDrone is research-only. Thirteen are `unverified`.
- `prepare_datasets.py --licence-gate strict` refuses to build from anything not
  cleared. Use it for anything destined for a product.
- **Solar and turbine have essentially no usable public data**, which is an
  advantage: you fly those assets, so that dataset is yours to build and nobody
  can buy it.
- Nothing public has fire and people in the same frame. WIT-UAS (CMU) is the
  closest and is thermal.

## Conventions in this repo

- Commits are authored `abyyworld`. History was rewritten to remove all Claude
  trailers; a recovery ref exists at `backup-before-cleanup` locally if anything
  looks wrong.
- No em dashes in documentation.
- `station/core/safety.py` holds the safety invariant as matchable patterns and
  `tests/test_safety_invariants.py` fails the build if banned phrasing appears
  anywhere. The rule: absence of a detection is never rendered as a negative
  claim. It applies per domain with different wording - no box does not mean no
  fire; no defect detected does not mean the asset is sound.

## Suggested first moves

1. Merge `platform/import-wildfire-station` into `main` if you have not.
2. Do the profile extraction (PLATFORM.md step 1 and 2).
3. Reconcile the two `audit_dataset.py` and the two `evaluate.py`.
4. Get FLAME and Boreal onto Kaggle and do training run 0 (docs/KAGGLE.md,
   about 1.5 hours, and the notebook needs no editing).
