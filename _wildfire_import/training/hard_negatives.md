# Hard negatives: the catalogue, and what to do about each

This model runs at a deliberately low confidence threshold
(`inference.conf_threshold: 0.25`). That is a decision, made once, in
`station/core/config.py`, for a stated reason: a false positive costs an
operator a glance, and a false negative costs everything this system exists to
prevent. The direct consequence is that we will see spurious boxes, and this
file is the plan for them.

The plan is *not* "turn the threshold up". Every spurious box removed that way
also removes the thin plume on a bright sky that the threshold was low to
catch, and you will not see which ones you lost. The plan is, in order of how
much it buys:

1. **Background imagery in the training set.** By far the largest effect. The
   ~52,000 FASDD negatives exist for this, plus FLAME's no-fire aerial frames,
   plus the frames our own system got wrong (mining workflow at the end).
2. **The N-of-M temporal filter** (`station/inference/temporal.py`, default
   3-of-5). Most of the confusers below are momentary: a glint, a flare, a
   brake light at a junction. A box has to survive five frames of a moving
   camera to be drawn.
3. **How the overlay renders.** A box says *look here*. It does not say what
   is there, it does not escalate, and nobody acts on it without seeing the
   video. That is what makes an occasional wrong box a cost rather than a
   hazard, and it is why the box is drawn on a canvas over the video instead of
   being burned into the pixels — the operator can always see what is under it.

Everything below is written as: what it is, what the operator sees, why the
model does it, and what to do. "Mitigation" always means data first.

---

## The catalogue

| # | Confuser | Reads as | Typical situation | First mitigation |
|---|----------|----------|-------------------|------------------|
| 1 | High-vis PPE | fire | crew on the ground, low altitude | mine own footage; FASDD negatives |
| 2 | Red apparatus | fire | staging area, roadside | mine own footage |
| 3 | Brake lights, beacons, strobes | fire | traffic, ingress route | temporal filter + mined data |
| 4 | Golden-hour light | fire (everywhere) | first/last hour of flying | dawn/dusk negatives |
| 5 | Red roofs, terracotta | fire | wildland–urban interface, nadir | nadir negatives from own flights |
| 6 | Autumn foliage | fire | September–November, deciduous | seasonal negatives |
| 7 | Dust plumes | smoke | dirt roads, rotor wash, dozer line | mine own footage; FASDD |
| 8 | Steam, fog, low cloud | smoke | dawn valleys, cooling plant, damp fuel | fog/vapour negatives |
| 9 | Lens flare, sensor bloom | fire | sun in or near frame | flare-through footage; hood |
| 10 | Sunlit water | fire | rivers, ponds, wet roads | specular-glint negatives |

### 1. High-vis PPE

**Operator sees** small boxes tracking with the crew.

**Why** high-vis orange and fluorescent yellow-green are, in RGB, the most
saturated small objects in a wildland scene — brighter and more saturated than
most real flame at distance. Retroreflective striping adds specular highlights
that mimic flame's local contrast. A model trained mostly on "orange blob in
dark vegetation" has learned exactly this feature.

**Why it is the worst one on the list** it puts a marker on the crew. An
operator who learns that boxes appear on their own people learns to discount
boxes, and that habit is the actual risk here — worse than the box itself.

**Mitigation** the only data that fixes this is *your* PPE, at *your* flight
altitude, under *your* camera. Fly a training sortie over a staged crew and
mine every frame (workflow below). No public dataset contains aerial high-vis.
Add these as background images, not as a third class: a "PPE" class would need
its own boxes, its own recall, and its own failure mode, and would buy nothing
the background label does not.

**Confirm in the log** filter `--class fire --min-conf 0.3` and check the boxes
against the recorded video at those pts. PPE detections are small, persistent
and move with a person's gait rather than growing or drifting with wind.

### 2. Red apparatus

**Operator sees** a large, stable box on an engine or tender, usually at a
staging area, often the highest-confidence box on screen.

**Why** a saturated red region several hundred pixels across, with hard
geometric edges. Flame has soft, flickering, fractal edges; a model that has
never been shown a fire truck has no reason to have learned that difference.

**Mitigation** background imagery of apparatus from above. Staging areas are
easy to film and it is worth doing once per vehicle type. Note that the
apparatus is often exactly where the operator is looking, so this one produces
a lot of the perceived false-positive rate for very few actual frames.

### 3. Brake lights, beacons, strobes

**Operator sees** a box that appears and disappears at a road junction.

**Why** small, extremely saturated red/amber, and — the trap — *flickering*.
Flicker is one of the features that distinguishes flame from a red object, so
a beacon is a red object that also has the right temporal signature.

**Note on the temporal filter** N-of-M suppresses things that appear once. It
does **not** suppress a 2 Hz beacon, which recurs reliably and can reach
3-of-5. Do not assume the filter handles this class; it is one of the few that
needs data.

**Mitigation** mine night and dusk ingress footage. Weight it: a handful of
these frames goes a long way because the feature is so distinctive.

### 4. Golden-hour light

**Operator sees** boxes scattered over sunlit slopes, tree crowns and rock,
for the first and last hour of the flying day — which is when much of the
flying happens.

**Why** a global warm colour cast shifts the whole scene toward the model's
fire prior, while long shadows create high local contrast beside every warm
patch. This raises the score of *everything* rather than creating one wrong
box, so it degrades precision broadly and is easy to misread as "the model got
worse today".

**Mitigation** dawn and dusk background imagery, from the air, in the same
terrain. Some of FASDD's negatives are sunset scenes and they help, but they
are ground-level.

**Do not** apply an automatic white balance or colour normalisation to
"correct" this at inference time without measuring recall afterwards. Thin
smoke at low sun is a low-contrast, warm-grey object, and aggressive
normalisation removes it. Measure before and after on real footage.

### 5. Red roofs and terracotta

**Operator sees** persistent boxes on buildings in the wildland–urban
interface, seen from nadir or near-nadir.

**Why** a large uniform red region viewed from above, with no motion and no
plume. The interface is also where the model matters most, so this competes
directly with the detections that count.

**Mitigation** nadir background imagery over housing. This one is
viewpoint-specific enough that ground-level datasets do not help at all — which
is the general argument for `viewpoint` weighting in `dataset_config.yaml`.

### 6. Autumn foliage

**Operator sees** broad, low-confidence boxes over deciduous stands,
seasonally.

**Why** orange-red texture over a large area, moving in wind, with the same
soft fractal edges that distinguish flame from a painted surface. This is the
hardest colour-based confuser on the list.

**Mitigation** seasonal background imagery from your own area of operations.
If you fly deciduous forest in autumn, this must be in the training set or the
model is not usable in that season; there is no runtime trick that fixes it.
Record the seasons your training data covers in the model card, because a model
validated in July has not been validated for October.

### 7. Dust plumes

**Operator sees** a smoke box on a moving vehicle, a dozer line, or under the
aircraft during a low hover.

**Why** the single most likely false smoke detection for a drone. A dust plume
has the right shape, the right motion, the right diffuse edges and often the
right colour. Smoke and dust differ mainly in colour temperature and in how
they disperse with height — dust settles, smoke rises and shears.

**Mitigation** background footage of vehicles on dirt roads and of rotor wash,
from the air. Also worth annotating: the *tail* of a genuine plume is often
dust kicked up by a vehicle at the fire edge, so a box that includes both is
not simply wrong, and mining it needs judgement rather than a bulk export.

**Note** a plume box on a vehicle track is still a box that says *look here*,
and an operator glancing at the video resolves it in under a second. That is
the system working as designed, not a defect to be optimised away at the cost
of recall.

### 8. Steam, fog and low cloud

**Operator sees** smoke boxes over valley floors at dawn, over damp fuel after
rain, over any industrial vapour source.

**Why** white-grey, plume-shaped, rising, diffuse edges. To an RGB model,
steam and thin white smoke are close to the same object. Note the direction of
the danger here: this confuser sits right on top of the detection we most want,
because thin white smoke on a bright sky is already the hardest true positive.

**Mitigation** fog and vapour background imagery — and accept a worse trade
here than elsewhere. Suppressing steam aggressively will suppress early thin
smoke, which is the detection that buys the most time. Prefer to leave this one
noisy and let the operator judge it.

**Do not** add a "steam" class hoping to separate them. The classes are fixed
by the wire contract at (`fire`, `smoke`), the tablet has no colour for a third
class, and the separation is not reliably learnable from RGB anyway.

### 9. Lens flare and sensor bloom

**Operator sees** a box on a bright polygonal artefact, or on a smear that
tracks the sun as the gimbal pans.

**Why** flare is saturated, warm, high-contrast and has soft edges. Rolling
shutter and cheap glass on a downlinked stream make it worse. Bloom around the
sun disc produces a large warm region with no structure.

**Mitigation** partly optical: a lens hood, and a flight pattern that avoids
pointing the gimbal within ~20 degrees of the sun. Then footage flown
deliberately through flare, mined as background. Flare moves with the *camera*,
not with the *scene*, which is also how an operator recognises it instantly.

### 10. Sunlit water

**Operator sees** clusters of small flickering boxes on a river, a pond, wet
tarmac, or a metal roof.

**Why** specular glint is small, extremely bright, saturated toward white-warm,
and flickers frame to frame — a good match for distant flame, which is exactly
what we are trying hardest to detect.

**Mitigation** background footage over water at low sun angle. This is one of
the few cases where the temporal filter genuinely helps, because glint moves
incoherently frame to frame and rarely holds an IoU match for 3 of 5 frames.

---

## Not on this list: real fire that is not your fire

Campfires, barbecues, chimneys, burn barrels, controlled burns and the flare
stack at a plant are all **correct** detections. They are not false positives
and must never be trained away — a model taught that a small managed fire is
background is a model taught to ignore a small fire.

They are an *operational* filtering problem, and the right place to resolve
them is the operator looking at the video, which is the design premise. If a
site has a permanent known heat source, note it in the pre-flight brief. Do not
add a geographic mask to the model: a mask is a region in which the system has
been made permanently unable to report anything, and nobody will remember it is
there.

---

## What you must not do

* **Do not raise `conf_threshold` to make the catalogue go away.** It removes
  distant, thin and early detections first — the ones worth the most — and it
  does so invisibly. If you change it, re-run the false-negative evaluation on
  real footage and record both numbers.
* **Do not suppress small boxes.** The smallest boxes are the earliest
  detections. `tools/audit_dataset.py` warns when a dataset has too few tiny
  boxes for exactly this reason.
* **Do not crop or mask regions of frame** (the sky, the horizon, the corner
  where the flare always is). Fire appears at the edge of frame constantly, and
  a mask is a permanent silent blind spot.
* **Do not train a rejection class** ("PPE", "vehicle", "steam"). The wire
  contract fixes the classes at (`fire`, `smoke`) in that index order; adding a
  third changes the model's output layer, the tablet's colour map and every
  label file. Background images achieve the same thing with none of that.
* **Do not fix a false positive by deleting the true positives near it.**
  Tempting when a plume box also covers a dust tail. Re-box it instead, or drop
  the frame entirely.

---

## Mining hard negatives from our own incident logs

`station/incidentlog` writes exactly the material this needs: every frame's
result, including the empty ones, with pts, model version and confidence, plus
the recorded video and a pts → (segment, offset) index. That means a false
positive seen in the field on Tuesday can be a training image on Wednesday, at
the correct viewpoint, camera and terrain — which no public dataset can offer.

The loop is: **review with the video → export frames → label → merge → audit →
retrain → measure**. Reviewing against the video is not optional. A detection
record on its own cannot tell you whether a box was wrong.

### Step 1 — find the candidates

```bash
# What did this incident actually produce? Read the summary first; it prints
# the empty-result caveat, which is the frame of mind to review in.
python3 -m station.incidentlog.reader incidents/2026-09-06-ridge-road --summary

# Candidate false positives: confident, persisted detections. Use --video so
# every line carries the file and offset to scrub to.
python3 -m station.incidentlog.reader incidents/2026-09-06-ridge-road \
    --class fire --min-conf 0.45 --min-persisted 3 --video --limit 200
```

Watch each one in the recorded video before exporting anything. Roughly a
quarter of what looks like a false positive in a text listing turns out to be a
real detection you had not noticed, which is the system doing its job.

### Step 2 — export the frames

This snippet turns confirmed-wrong detections into images on disk, one
directory per incident, so that `dataset_config.yaml`'s `mined_negatives`
source groups them by incident and never splits one incident across train and
val. Run it on the ground station, which has PyAV; it degrades with a clear
message anywhere else.

```python
#!/usr/bin/env python3
"""Export frames around chosen detections into training/../datasets/raw/mined."""
from pathlib import Path

import av  # PyAV; on the ground station only

from station.incidentlog.reader import IncidentLogReader

INCIDENT = Path("incidents/2026-09-06-ridge-road")
OUT = Path("datasets/raw/mined/negatives") / INCIDENT.name
# pts values you confirmed against the video in step 1.
WRONG_AT_PTS = [137.4, 141.2, 208.9]
STRIDE_S = 0.5  # do not export 25 near-identical frames of one mistake

reader = IncidentLogReader(INCIDENT)
OUT.mkdir(parents=True, exist_ok=True)

wanted, last = [], -1e9
for frame, det in reader.detections(classes=["fire", "smoke"], min_conf=0.4):
    if not any(abs(frame.pts - t) < 1.0 for t in WRONG_AT_PTS):
        continue
    if frame.pts - last < STRIDE_S:
        continue
    last = frame.pts
    wanted.append(frame.pts)

for pts in wanted:
    position = reader.video_position(pts)
    if position is None:
        print(f"no recorded video for pts {pts:.3f}; skipping")
        continue
    with av.open(str(position.path)) as container:
        stream = container.streams.video[0]
        target = int(position.offset_s / float(stream.time_base))
        container.seek(target, stream=stream, backward=True)
        for picture in container.decode(stream):
            if picture.time is not None and picture.time + 1e-3 >= position.offset_s:
                picture.to_image().save(OUT / f"{pts:010.3f}.jpg", quality=92)
                break

print(f"{len(wanted)} frame(s) -> {OUT}")
```

Without PyAV, the same seek is one ffmpeg call per frame:

```bash
ffmpeg -ss "$OFFSET" -i "$SEGMENT" -frames:v 1 -q:v 2 "$OUT/$PTS.jpg"
```

### Step 3 — label honestly

Three outcomes per exported frame, and only you and the video can decide which:

* **Nothing to box.** Write an empty `.txt` beside it (or put it under
  `mined/negatives`, where `prepare_datasets.py` writes the empty label for
  you). This is the common case and the valuable one.
* **Something to box, boxed wrongly.** The box was on the dust tail, or covered
  the plume and half the hillside. Re-box it correctly and put it under
  `mined/positives`. A sloppy box here is worse than no box.
* **Something to box that the model missed.** Box it and file it under
  `mined/positives` too — these are the most valuable images in the whole
  pipeline and the only ones that directly attack the failure mode that
  matters. Note the pts in the incident's notes so the false-negative audit can
  find it again.

### Step 4 — merge, audit, retrain, measure

```bash
# Flip mined_negatives / mined_positives to `enabled: true` in dataset_config.yaml
python3 training/prepare_datasets.py --config training/dataset_config.yaml --force
python3 tools/audit_dataset.py --data datasets/wildfire-merged/data.yaml
```

Then retrain (`training/train_kaggle.ipynb`) and compare against the *previous*
model on the *same* held-out footage. Two numbers, both required:

* the false positives you were trying to remove — did they go?
* **recall on distant and thin targets — did it drop?**

Mining hard negatives is a recall-for-precision trade whether or not you
measure it. Measure it. A model that stopped boxing the fire truck and also
stopped boxing the plume behind it is a worse model, and it will look better on
every summary statistic except the one that matters.

### Keeping the loop honest

* Mine from **many** incidents, not one. Fifty frames from one afternoon teach
  the model that afternoon.
* Keep mined frames grouped by incident (`grouping: regex`, group = incident
  id). Frames from one incident are consecutive video frames; splitting them
  across train and val is the leak, and it is doubly tempting here because the
  set is small.
* Record in the model card which incidents fed which model version. The
  `ModelInfo` in every log line (`name`, `version`, `weights_sha`) is what lets
  you answer "which model produced this box" a year later.
* Re-mining the same incident after retraining is fine and expected. Keep the
  incident id as the group key and `prepare_datasets.py` will de-duplicate the
  frames you exported twice.
