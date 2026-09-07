# One platform, five products

A decision record for how wildfire, inspection, traffic and events share code.
Written when the fourth domain appeared, which is roughly the last moment this
is cheap.

## Not branches

Branches are for divergence that merges back. These domains never merge back —
they are permanent parallel products, and a long-lived branch per product rots
in a specific, predictable way: a fix to the shared pipeline has to be
cherry-picked into every branch, forever, and the first time one is missed a
tablet ships that draws stale boxes over live video. You also cannot run two
domains from one checkout, CI has no single trunk to protect, and there is
nowhere that "the platform" actually lives.

Not separate repositories either, for the same reason at larger scale. The
original handoff said to *copy* the overlay code between wildfire and
inspection rather than abstract it — right when two repos shared two hundred
lines, wrong now that five products share the whole pipeline.

**One repository. One `main`. Domains are configuration and weights.**

That repository is **`drone-visualisation-training`**. `wildfire-analysis`
becomes a profile inside it, not the other way round — the general name should
own the platform, and a repo called "wildfire" would be a bad home for turbine
inspection.

**Move the code with its history, not by copying files.** The reasoning behind
this pipeline lives in its commit messages — why the overlay drops boxes rather
than dimming them, why frames are dropped rather than queued, why the offset
estimator refuses an uncorroborated sample. A copy-paste migration throws all
of that away and the next person re-introduces the bugs. Use a subtree merge:

```bash
cd drone-visualisation-training
git remote add wildfire https://github.com/abyyworld/wildfire-analysis
git fetch wildfire claude/wildfire-watch-setup-66paiq
git merge -s ours --no-commit --allow-unrelated-histories \
    wildfire/claude/wildfire-watch-setup-66paiq
git read-tree --prefix=/ -u wildfire/claude/wildfire-watch-setup-66paiq
git commit -m "Bring in the wildfire-watch station as the platform"
```

Then move the wildfire specifics into `profiles/wildfire.yaml` and the repo is
the platform.

The measurement that justifies this: `stream/`, `serve/` and `incidentlog/`
contain zero references to fire or smoke, and `ingest/` has one in seven files.
The pipeline was already domain-agnostic before anyone planned for it. What is
domain-specific is the class list, the weights, the temporal parameters, the
safety wording and the validation criteria — a config file, not a codebase.

```
station/            the platform. Never names a domain again.
profiles/
  wildfire.yaml     fire, smoke, person
  traffic.yaml      car, truck, person
  solar.yaml        panel defect classes
  turbine.yaml      blade defect classes
  event.yaml        crowd density  (see below: not a detector)
models/             weights per profile — released as artifacts, not committed
```

`python -m station --profile wildfire run`

## Which classes go in which model

The rule that decides every case: **train classes together only when they
appear in the same frame at the same moment and are seen by the same sensor.**
Anything else costs accuracy at both ends for no transfer benefit.

| Profile | Classes | Why grouped this way |
|---|---|---|
| `wildfire` | fire, smoke, person | A crew works *at* the flame front. Fire and person co-occur in one frame and their spatial relationship is the whole point — a person 20 m from a fire is the most important box on the screen. Splitting them into two models means two sets of boxes to fuse and no model that ever sees the relationship. |
| `traffic` | car, truck, person | Same argument. A pedestrian near a vehicle is the case that matters, and a car-only model cannot express it. |
| `solar` | panel defects | Never flown at the same time as a wind farm. No visual structure shared with turbines, so training them together helps neither. |
| `turbine` | blade defects | As above. Separate, smaller, better at its one job. |
| `event` | crowd density | Not a detector at all — see below. |

**So: person is a class inside wildfire and traffic, not a profile of its own.**
That answers the question directly. A person means something different in each
context, but it is the same *detection* task, and the context is exactly what
the model needs to see.

### The one thing that changes this answer: sensor

If the airframe carries thermal, person detection should move to the thermal
stream, and then it *must* be a separate model — one model cannot take two
sensors as input.

- **RGB-only airframe** → person is a class inside `wildfire` and `traffic`.
  One model per profile. Simple, and what is built today.
- **RGB + thermal airframe** → `wildfire` stays fire/smoke on RGB, and a
  `wildfire-thermal` profile does person on the thermal stream. Two models, two
  streams, fused at the overlay — which costs nothing, because the wire format
  carries class-labelled boxes and does not care how many models produced them.

Thermal is materially better for finding people: it works at night and sees
through smoke, and from altitude a person is a bright signature rather than a
few pixels the colour of a rock. If the payload exists, use it. See
`docs/DATASETS.md`.

## Events are a different technique, not a fifth class list

At concert density, people occlude each other and box detectors undercount
badly — a YOLO trained on crowds will confidently report four hundred people in
a crowd of three thousand. The correct approach is **density-map regression**
(CSRNet-style, trained on ShanghaiTech / UCF-QNRF / JHU-CROWD++), which outputs
a density field rather than boxes.

That has two consequences:

1. The `event` profile does not share the detection model interface. It shares
   ingest, streaming, logging and the tablet — most of the platform — but its
   inference stage and its overlay are its own.
2. Its safety wording is its own, and it matters more than the others. A
   density estimate is not a headcount and not a capacity judgement.
   Undercounting is the failure mode behind crowd-crush disasters, so a number
   on a screen that a steward reads as authoritative is the hazard. The
   repository already fails the build on counting language for the person
   class; the event profile needs its own version of that rule rather than an
   exemption from it.

Crowd footage is also personal data in a way a solar farm is not. Settle the
data-protection position before collecting, not after.

## Per-domain safety rules, one enforcement mechanism

The invariant is the same everywhere and the sentence is different everywhere:

- wildfire — no box does not mean no fire
- inspection — no defect detected does not mean the asset is sound
- traffic — no pedestrian box does not mean the road is clear
- event — a density estimate is not a capacity judgement

So `station/core/safety.py` keeps the machinery and each profile supplies its
own patterns and its own operator-facing copy. One test walks the repository
for all of them.

## Release cadence, which was the original worry

The concern that argued for separate repos was that these products move at
different speeds and carry different risk. Handle that by versioning profiles
independently of the platform, not by forking code:

- the platform has one version and one `main`;
- each profile pins the platform version it was validated against, and carries
  its own `model_version`;
- a wildfire profile can sit frozen on platform 1.2 while traffic moves to 1.4,
  and the pin is what makes that safe rather than accidental.

Re-validating a frozen profile against a newer platform is then a deliberate
act with a checklist, which is what "different risk profiles" should have meant
all along.

## Order of work

1. **Extract the profile loader** and move wildfire's specifics into
   `profiles/wildfire.yaml`. The platform stops naming a domain.
2. **Port the inspection repo** in as `solar.yaml` and `turbine.yaml`. This is
   the step that proves the abstraction — if it does not fit two real domains
   it is wrong, and better to learn that now than at domain five.
3. **Add traffic** as a config file plus a training run. If that is all it
   takes, the split worked.
4. **Event last**, because it needs a second inference interface and a separate
   safety review, and doing it before the other three would design the platform
   around its least representative case.
