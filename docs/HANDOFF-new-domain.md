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
