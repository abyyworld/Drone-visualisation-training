# What to do next

Written to be read cold, in order. Nothing here needs this session to be running.

---

## 1. GitHub — nothing is required of you

Everything is committed and pushed to the branch **`claude/wildfire-watch-setup-66paiq`**
on `abyyworld/wildfire-analysis`, authored as `abyyworld <annolieberto@gmail.com>`.

There is **no open pull request**, because none was asked for. The branch stands on its own
and can be browsed, cloned, or merged whenever you want. If you do want a PR later, open it
from the branch page on GitHub — one click, no local work.

To pick it up on your own machine:

```bash
git clone https://github.com/abyyworld/wildfire-analysis
cd wildfire-analysis
git checkout claude/wildfire-watch-setup-66paiq
```

**Nothing trains itself while you are away.** There is no job running on GitHub and no
process left behind. Training needs a GPU and needs the datasets, and neither exists in the
environment this repo was built in — that work starts when you start it, at step 3 below.

---

## 2. Run the demo — no GPU, no drone, no model, about a minute

The offline render is already committed, so the very first thing you can do is watch it:

```
demo/assets/wildfire_demo_overlay.mp4
```

That is the real pipeline — real ingest, real temporal filter, real incident log — with the
overlay burned in. Beside it sits `demo/assets/incident-demo/detections.jsonl`, one line per
frame including the 78 frames that hold nothing.

To regenerate it, or to run it over **your own** footage:

```bash
pip install av numpy pyyaml
python demo/render_demo.py --input /path/to/your/fire_video.mp4 \
                           --output /tmp/out.mp4 --incident-dir /tmp/incident
```

Any video file works. For anything shown to an audience, real footage beats the synthetic
clip — the clip is a fixture, and it looks like one.

### Showing it live on tablets and phones

```bash
python -m station certs                    # self-signed cert for the LAN
python -m station run --config config.yaml # serves the PWA over local WiFi
```

Then open `https://<laptop-lan-ip>:8443` on the tablet. Expect a certificate warning and
accept it — that is normal for a self-signed cert and is exactly the step that has not yet
been tested on a real iPad.

**Have the offline MP4 ready as a fallback.** If the certificate or the WiFi misbehaves in
the room, you play the video and the demonstration still lands.

---

## 3. Training — this is the part only you can start

The repo cannot reach Zenodo, Hugging Face, Kaggle or IEEE DataPort, and has no GPU. So
downloading and training happen on your side.

**Downloads first — they are the long pole and mostly unattended:**

| Dataset | Where | Friction | Use it for |
|---|---|---|---|
| **FLAME** | IEEE DataPort | free account | genuine UAV fire — start here |
| **Boreal Forest Fire** | via the Sci Data 2025 paper | check licence | genuine UAV, boxes + segmentation |
| **FASDD** | Zenodo / Science Data Bank | accept terms | bulk, and ~52k hard negatives |
| **D-Fire** | GitHub `gaiasd/DFireDataset` | none | ground-level; weight it down |

**Then, in this order — do not skip the audit:**

```bash
python training/prepare_datasets.py --config training/dataset_config.yaml --out data/merged
python tools/audit_dataset.py data/merged/data.yaml        # THE GATE
```

`audit_dataset.py` exits non-zero if the split leaks. Fire datasets are cut from video, so a
random train/val split puts frame 0412 in train and the near-identical 0413 in val;
validation then measures memorisation and every number is inflated. That is how a sibling
project got mAP50 0.782 on a model that was useless in the field. `prepare_datasets.py`
splits grouped by source video so it cannot happen, and the audit proves it did not.

**Then train, smallest run first:**

| Run | Data | Kaggle P100, 50 epochs | Fits the 9h cap? |
|---|---|---|---|
| 0 | FLAME + Boreal (~10k) | ~1.5 h | yes, easily |
| 1 | 30k subset + FASDD negatives | ~4 h | yes |
| 2 | full FASDD (122,634) | ~12–16 h | **no** |

Do run 0 first and **not for the accuracy** — it proves the chain end to end: train → export →
class order survives → weights load in the station → boxes appear on video. A cheap run that
finds a pipeline bug is worth far more than a long one that produces an unusable number.

Only do run 2 if run 1's evaluation says data volume is the bottleneck, and it often is not —
your real problem is false negatives on thin smoke, smouldering, canopy and night, which is a
data *composition* problem, not a volume one. If you do run 2, rent a 4090 for ~$2 rather than
fighting the session cap with checkpoint-resume.

Then point the station at the weights:

```yaml
inference:
  weights: models/your-trained.pt
  model_version: "0.1.0"
```

---

## 4. Before anyone relies on it

Read `docs/VALIDATION.md`. The short version: measure **false negatives** on the department's
own footage, broken out by the conditions that break RGB detection — thin smoke on bright sky,
smouldering with no flame, fire under canopy, night, and small distant fire.

The system is a situational-awareness aid. It says *look here*. It never says *there is
nothing there*, and the interface is built so it cannot start saying that by accident —
`tests/test_safety_invariants.py` fails the build if that language appears anywhere.
