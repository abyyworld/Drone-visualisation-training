# Validation

What has to be measured, and by whom, **before anyone relies on this system**.

The short version: measure what it **misses**, on the department's **own**
footage, broken out by the conditions that break RGB fire detection, and put
the result in front of the people who would be acting on it. Everything else
in this document is detail on how to do that at a scale a student project can
actually finish.

Read `docs/SAFETY.md` first. This document is the empirical half of the
argument that document makes.

---

## 1. Why false negatives, and not mAP

Object-detection work reports mAP because mAP ranks models against each other
on a leaderboard. This system is not on a leaderboard, and mAP is the wrong
instrument here for three separate reasons.

**It averages away the thing that matters.** mAP is a mean over classes and
IoU thresholds. A model that is excellent on a fully-developed flame front and
useless on thin smoke scores respectably. In the field those are different
events with different consequences, and the second one is the early one.

**Box precision is not the operational quantity.** The overlay says *look
here*. A box that is 40% off but on the right plume does its whole job; the
operator's eye lands on the plume and the human decides. IoU 0.5 versus IoU
0.75 is nearly meaningless for that task, and mAP@[.5:.95] spends most of its
dynamic range there.

**Published fire numbers are inflated, systematically.** Fire datasets are cut
from video. A random train/val split puts frame 0412 in train and the
near-identical frame 0413 in val, so validation measures memorisation and
every number rises. `tools/audit_dataset.py` exists to catch exactly that, and
`training/prepare_datasets.py` splits grouped by source video so it cannot
happen in the first place. Assume any fire-detection mAP you read in a paper
or a repo README is optimistic until you can see how it was split.

So the primary metric is **event-level recall**: of the fire and smoke events
a human reviewer can see in the footage, how many did the model produce a
confirmed box for, and how quickly. The miss list -- the actual frames it
failed on -- is the primary *artefact*, more useful than any single number,
because a human can look at it and understand the failure.

`tools/evaluate.py` is organised around this: per-class recall at several
thresholds, split by box size and by condition tag, with the miss list as its
main output. mAP is computed, reported last, and treated with suspicion.

---

## 2. Why the department's own footage

A model validated on public datasets and deployed on a different fleet, in
different terrain, in a different season, has been validated against a
different problem. The things that shift:

- **Camera and codec.** The drone's sensor, its colour processing, its
  compression artefacts at the bitrate the radio link actually sustains.
- **Altitude and angle.** Plume size in pixels is a function of how they fly,
  and pixel size is the single strongest predictor of whether a small fire is
  found at all.
- **Terrain and vegetation.** Chaparral smoke, conifer canopy, grass, peat --
  and the local false-positive population: dust from vehicles on dirt roads,
  glint off polytunnels and metal roofs, morning valley fog, chimney smoke.
- **Season and light.** Angle of the sun, haze, the colour of dry vegetation.

Public datasets are for **training**. The department's own footage is for
**validation**. Do not reuse a single frame of the validation footage in
training, and keep them in separate directories so it cannot happen by
accident.

### Getting footage without waiting for a wildfire

In rough order of how quickly you can get it:

1. **The department's archive.** Most services already hold drone or
   helicopter video from past incidents. This is the highest-value material
   and asking for it costs one conversation.
2. **Prescribed burns and training burns.** Scheduled, safe, legal to fly, and
   they generate exactly the early-stage smoke that matters. Ask to fly one.
3. **Negative footage.** Ordinary patrol flights over the same terrain with
   nothing burning. Cheap to collect, and the only way to measure the
   false-positive burden honestly.
4. **The incident log itself.** Once the station is running alongside normal
   operations -- with the video recorded and nobody relying on the overlay --
   `incidents/` accumulates real footage paired with what the model produced
   for it, frame by frame, including the empty frames. This is the long-term
   validation corpus and it is a large part of why the log records everything.

---

## 3. The conditions that must each be measured

Each row below is a separate measurement with its own sample count. A single
pooled number across all of them hides precisely what needs to be seen.

| # | Condition | Why it is on this list | How to obtain it |
|---|---|---|---|
| 0 | **Control: daytime flaming, unobstructed** | The best case. Sets the ceiling; if this is weak, nothing else is worth measuring yet | Any archive incident footage, prescribed burn |
| 1 | **Thin smoke on bright sky** | Low contrast against a bright background; the weakest class and often the earliest signal | Prescribed burn shot upward or across a ridge line |
| 2 | **Smouldering, no visible flame** | Little colour signature in RGB; the model's dominant cue is gone | Burn pile hours after ignition; post-suppression hotspots |
| 3 | **Fire under canopy** | Frequently no line of sight; tests whether a partial visual cue is enough | Forested prescribed burn, oblique angles |
| 4 | **Night** | An RGB sensor at night is a different imaging problem from the daytime footage the model trained on | Evening burn, or archive night footage |
| 5 | **Small / distant fire** | Recall falls off hard below roughly 20 px on the model's input | Same burn flown at 2-3 altitudes; label by plume pixel height |

Two crosscutting sets to collect alongside them:

- **Negative footage** (nothing burning): at least 30 minutes over the same
  terrain, for the false-positive burden.
- **Known-hard false positives**: dust plumes, fog banks, glint, chimney
  smoke. Not a pass/fail gate -- an inventory of what will show up on screen,
  so operators are briefed on what they will actually see.

---

## 4. Protocol

### 4.1 Freeze what you are testing

Record, in the validation report: `model_name`, `model_version`,
`weights_sha`, `imgsz`, `conf_threshold`, the temporal filter `n`/`m`, and
`inference.max_fps`. Every one of those changes the answer. The station writes
all of them into `incidents/<id>/meta.json` and into every logged frame, so a
run is self-describing -- but copy them into the report as well, because a
report that cannot be tied to a specific model is not evidence of anything.

### 4.2 Label the footage

Work in **events**, not frames. An event is one fire or smoke source, visible
to a human reviewer, over a span of time.

For each clip record: `clip_id`, condition tag (0-5 above), and for each
event, `t_start` (first second a human reviewer can see it), `t_end`, class
(`fire` / `smoke`), and approximate plume height as a fraction of frame
height. Frame-accurate boxes are **not** needed for this measurement and cost
an order of magnitude more time; a start time and a rough location is enough
to decide whether the model found the thing.

Practical rules that keep this honest:

- **Two reviewers on at least 20% of the clips**, labelled independently, and
  report their agreement. If two humans disagree about whether a plume is
  visible in a clip, the model's answer on that clip carries no information
  either way, and you need to know how large that grey zone is.
- **Label before you look at any model output.** Labelling with the overlay
  visible is how a validation set quietly turns into a confirmation exercise.
- **A human reviewer may scrub, pause and zoom.** That is the right reference:
  the question is whether the information was present in the video, not
  whether it was easy.

Cost estimate: a 30-second clip takes about 5-10 minutes to label this way.
Sixty clips is roughly two days of work. That is the size of the job.

### 4.3 Run the model over the footage

Offline, through the real pipeline, with the real temporal filter:

```bash
# Per clip: writes incidents/<id>/detections.jsonl -- every frame, including
# the empty ones, which is what the miss analysis reads.
python -m station run --source-type file --source validation/clips/0007.mp4 \
                      --no-stream -c config.yaml
```

Use the same config the department would deploy. If you are validating a
configuration you would not ship, you are validating nothing.

`tools/evaluate.py` runs the equivalent measurement against a labelled YOLO
dataset and produces the miss list directly; use it for frame-level work on
still images. For clip footage, the incident log is the record to analyse,
because it is exactly what the deployed system produces.

### 4.4 Score it

**Event recall.** An event counts as **found** if at least one *confirmed*
detection (after the N-of-M filter -- the same boxes the tablet would draw) of
the right class overlaps the event's location within `t_start + T` seconds,
where `T` is decided in advance and written down. `T = 10 s` is a reasonable
starting point for a fire that a crew is going to be dispatched to; justify
whatever you pick.

An event that is never found in its whole visible span is a **miss**. Misses
are the output that matters.

**Report per condition:**

```
condition            events   found   recall   95% CI (Wilson)
0 control                24      23    0.958   [0.796, 0.992]
1 thin smoke / sky       20      12    0.600   [0.386, 0.781]
...
```

Use a **Wilson score interval**, not recall alone. With 20 events and 18
found, recall is 0.90 and the 95% interval is roughly [0.70, 0.97] -- and that
interval is the honest content of the measurement. Reporting "90% recall" from
20 events, with no interval, overstates what was learned by a lot.

**Also report, per condition:** median and 90th-percentile **time from
`t_start` to first confirmed box** (how much of the early window the system
actually buys), and on the negative footage, **confirmed boxes per minute**
with a note on what they were.

### 4.5 Build the miss gallery, and hold a review

For every miss, cut a short clip around `t_start` with the model's output
(empty) alongside. That gallery is the deliverable that changes decisions.
Sit down with the officers who would be using this and watch it together.

Two things come out of that meeting, and both matter more than the numbers:

1. **Whether the misses are the kind they can live with**, given that the
   overlay never subtracts from their own scan of the video.
2. **What goes in the operator briefing.** Every condition where recall is
   weak has to be something operators are told about explicitly, before first
   use, in the words they will remember.

---

## 5. What a pass looks like

A pass is **not** a single number. It is this set of conditions, together:

1. **Every condition in section 3 has been measured**, with its sample count
   stated. A condition that was not tested is reported as untested, never
   folded into an average. This is the most important criterion on the list.
2. **The control condition is strong**: event recall >= 0.90 with a lower 95%
   bound >= 0.75, over at least 20 events. If the best case is weak, stop and
   go back to training; the rest of the table is not informative yet.
3. **Every weak condition is named.** Any condition whose recall point
   estimate is below 0.50 is written into the operator briefing as a
   *condition in which the overlay should be expected to stay empty even when
   there is fire*. It is not a blocker -- it is a disclosure. Night and canopy
   may well land here, and that is an acceptable outcome provided it is said
   out loud.
4. **The false-positive burden is tolerable to the people watching the
   screen**: on negative footage, confirmed boxes are rare enough that
   operators do not learn to ignore the overlay. Roughly one per minute is
   already annoying; ten per minute makes the system worse than nothing.
   Judged by the operators, not by the developer. Never traded against recall
   by raising `conf_threshold` -- fix it with training data instead.
5. **Latency is bounded**: 90th-percentile time to first confirmed box, from
   the first frame a human can see the event, is a few seconds at most, and
   the end-to-end budget in `docs/HARDWARE.md` has been measured on the real
   hardware rather than estimated.
6. **The safety invariants hold on the real device.** `make safety` passes,
   and a human has confirmed on an actual tablet that: an empty result draws
   nothing at all; a stalled pipeline visibly stops drawing boxes; and the
   overlay reports which synchronisation tier it is using.
7. **It is signed off** (section 6).

Anything less is a work in progress, and should be described that way to
everyone who sees a screen.

---

## 6. Who signs it off

Not the person who wrote the code, alone. Three signatures on one page:

- **A named officer of the fire service** who watched the miss gallery, in
  full, and who is willing to state that the measured behaviour -- including
  the weak conditions -- is understood by the people who will be using it.
  This is the signature that matters; the rest is supporting.
- **The person who ran the validation**, attesting to the protocol: how the
  footage was collected, that labelling happened before any model output was
  seen, and that validation footage never appeared in training.
- **The project supervisor** (academic or technical), attesting that the
  method and the intervals are sound.

The signed artefact is one page and contains: model version and weights hash,
config values from 4.1, the per-condition table with intervals, the list of
disclosed weak conditions, and the date. Keep it with the code. An unsigned,
undated validation gets treated in six months as though it covered whatever
the model is doing then.

### Re-validate when any of these change

- New weights, of any kind, including "just a few more epochs".
- `conf_threshold`, `imgsz`, `n`/`m`, or `inference.max_fps`.
- A different drone, camera or controller.
- A new season, or terrain the original footage did not cover.

A full re-run is not needed every time. Keep a **fixed regression set** -- 20
to 30 labelled clips spanning all six conditions -- and re-run that on every
model version. It takes an afternoon and it catches the case where an
improvement in one condition quietly cost recall in another, which is the
normal way a retrain goes wrong.

---

## 7. Doing this as a student project

The full protocol above is roughly three weeks of part-time work. A minimum
version that is still worth something:

| Stage | Effort | Output |
|---|---|---|
| Collect footage: one prescribed burn + archive clips + 30 min negative | 2 days, mostly waiting on other people | ~40 clips |
| Label events (section 4.2), double-label 20% | 2-3 days | `labels.csv` |
| Run the pipeline over every clip | half a day, unattended | `incidents/` |
| Score, with Wilson intervals | 1 day of scripting | the table |
| Build the miss gallery | 1 day | the clips that matter |
| Review with the department, write and sign the page | 1 meeting | the signed page |

**Start collecting footage in week one**, before the model is finished. It is
the only part of this that depends on other people's calendars, and it is
always the reason a validation does not get done.

If you are short of time, cut sample sizes -- 10 events per condition with an
honestly wide interval -- before you cut conditions. Six conditions measured
roughly is far more informative than one condition measured precisely, because
the question being answered is *where does this fail*, not *how good is it*.

---

## 8. What this validation does not establish

State these limits in the report, so nobody else has to infer them:

- It says nothing about conditions that were not sampled. Different terrain,
  season, camera or altitude is a different measurement.
- It bounds nothing about rare events. Twenty events per condition cannot
  characterise a tail.
- It is not certification. No standard applies to this system: EN 54 and
  UL 268 cover fixed installations in buildings, not a drone-feed overlay.
  See `docs/SAFETY.md`.
- It does not license reliance. Even a good result changes nothing about the
  invariants: the overlay still says *look here* and never the opposite, and
  the operator is still watching the video.
