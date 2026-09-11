# Drone Inspection

Analysis of wind turbines, solar panels, crowds and wildfires, from drone photographs or
video.
Static site, no server, hosted on GitHub Pages, and installable on a tablet as an app with an
icon.

### [⬇ Download the app for the tablet (.apk)](https://github.com/abyyworld/Drone-visualisation-training/releases/latest/download/drone-inspection.apk)

**Or open it in a browser:** https://abyyworld.github.io/Drone-visualisation-training/

The download link above always serves the newest build and never changes. Copy the file to
the tablet, open it with the file manager, and allow installs from that source when Android
asks. Full instructions, and the reasons to prefer one route over the other, are in
[`docs/INSTALL-tablet.md`](docs/INSTALL-tablet.md).

## Three engines

| | On device | Trained defect models | Provider API |
|---|---|---|---|
| Where it runs | This device: a detector plus a computed flame scan | This device, ONNX Runtime Web | Anthropic, Google or OpenAI |
| Ready now | **Yes** | **Crowd, solar and wildfire.** Not turbine | With a key |
| Cost | Nothing | Nothing | Per image |
| Offline | Yes | Yes | No |
| Images leave the device | Never | Never | Yes |
| Speed | 5-15 per second | Similar | One every few seconds |
| **Tracks across frames** | **Yes** | Possible | No, and never will |
| Finds | people, and vehicles in the browser, plus flame and smoke regions | its trained classes | anything it can describe |
| Cannot find | cracks, corrosion, soiling | anything outside its classes | - |

None of the trained models here were trained here. People, fire and solar panel defects have
all been done already by somebody else and published, so
[`.github/workflows/model.yml`](.github/workflows/model.yml) searches for the weights,
fetches them and converts them on a runner, which can reach the hosts this project's own
environment cannot. Blade damage is the one subject that search comes back empty on, so it is
the one subject still needing a key.

**On device** is the default because it is the only one ready without a key or a dataset.
It is two engines at once, and the detector half is not the same model in both places.

On the tablet it is YOLO finetuned on **VisDrone**, which is aerial footage full of people a
few pixels tall seen from overhead. That is this job rather than an approximation of it, and
somebody else had already done the work and published the weights. 2.8 MB, 320 px, int8,
driven through LiteRT so that NNAPI can reach the Snapdragon's DSP. Ultralytics releases
these weights under **AGPL-3.0**, which is a deliberate choice made possible by this being a
public repository and a demonstration rather than a product. See
[`web/models/PERSON-640.md`](web/models/PERSON-640.md).

In the browser the live view still runs a COCO-trained EfficientDet-Lite2 through MediaPipe,
which finds people and vehicles at ground level and loses people from altitude. See
[`web/models/DETECTOR.md`](web/models/DETECTOR.md). The **Crowd** inspection type in the
analysis screen runs the aerial model instead, so a drone still can be compared between the
two on one machine.

Flame and smoke are found twice over. There is a trained model, and there is a scan that
needs no model at all, and the second one is what runs on the tablet.

The model took two attempts. The first returned the same score for black, white, noise, a
flame-coloured block, sixty real drone photographs and fifty frames of the demo clip alike:
it could not mark fire because it could not mark anything, and every check the conversion
workflow made passed it. [`tools/model_liveness.py`](tools/model_liveness.py) now stands
between that and shipping, and the workflow keeps trying candidates until one gets past it.
The licence was never the obstacle, whatever this used to say here: this project takes
AGPL-3.0 deliberately and every detector it ships is already a YOLO derivative.

**Recall on real fire is measured now, and it is not good.** Scored on a runner against 120
published pictures with fire in them and 150 real drone frames with none
([`docs/metrics-wildfire.txt`](docs/metrics-wildfire.txt)):

| confidence | found, of 120 fires | marked, of 150 drone frames |
|---|---|---|
| 0.10 | 56 | 4 |
| **0.15** | **50** | **0** |
| 0.25 | 43 | 0 |
| 0.50 | 28 | 0 |

It misses more than half of them. What it almost never does is cry wolf: nothing at all on
drone footage at every threshold from 0.15 up. That asymmetry is why the threshold is 0.15,
which is the last row where the middle column is zero - a knee in a measurement rather than
a number somebody liked.

Two things that number is not. It is not coverage: a frame with nothing marked has not been
cleared of anything. And it is not aerial. The pictures are ground level, because the sets
that would match how this flies (FLAME, FLAME2) are behind an IEEE DataPort account no
machine here can reach.

The computed scan stays, because it is the only fire capability on the tablet and needs no
model. Colour finds the candidates; time decides. Fire burns in place and
churns inside its own outline, so the flame fraction of a patch changes on nearly every
frame, while a red van crossing the shot changes it further but only twice and holds
perfectly still in between. The test is how often a patch changes, not how far, which is
what keeps a sunset, a tiled roof and a parked red car out of the results. What it produces
is a region to look at, not a verdict, and a single photograph carries no motion at all, so
a still is capped at 0.60 confidence and says why. See
[`web/js/firescan.js`](web/js/firescan.js).

## Live tracking

**Start camera** on the site. Real time, on the device, no key: each person or vehicle keeps
a number that follows it, trails behind it, survives a frame the detector missed, and is
counted once rather than once per frame. Record and it saves the video with the boxes on it.

### What a flight leaves behind

Press record and three kinds of file land in the app's own Movies directory, all named after
the same timestamp, which the tablet's file manager and the analysis screen's file picker
can both see:

| | |
|---|---|
| `flight-<time>.mp4` | the video, with the boxes drawn into it |
| `flight-<time>.csv` | one row per detection: seconds in, how many were in view, how many different people had been seen by then. The last line is the total. |
| `flight-<time>-<n>-people.jpg` | a still with its boxes, saved as the total passes each ten, up to twenty of them |

The two numbers in the table are different questions and it keeps them apart: a running
total that goes down is nonsense, and an in-view count that only ever climbs is a lie. The
stills are capped because a busy square would otherwise write a full-resolution photograph
every few seconds for the whole flight.

Nothing here can interrupt a flight. A log that will not open is reported once and the
recording carries on, because losing the numbers is a nuisance and losing the footage is the
flight.

Detection and drawing are decoupled - drawing runs every animation frame so boxes move with
the video, detection runs as fast as the device manages, and the tracker coasts between.
Nothing queues behind itself, so a slow device degrades instead of spiralling.

**People are counted on every subject**, not just crowd - someone at the base of a turbine
or near a fire is the most important thing in the frame. A model that reports people answers
for them itself; one that does not, such as a defect model on a turbine blade, has no idea
what a person is, so the count comes from the on-device detector instead. Either way one
frame gets one answer rather than two that can disagree. It appears as a footnote on each card, under the
batch summary, in the JSON export and in the PDF.

**People count** is a toggle on the live view. Two numbers, because they answer different
questions: how many are in view now, and how many distinct people have been seen since the
camera opened - counted once each by their track, not once per frame. It counts what the
detector found, which is a floor and not a measurement: anyone small, distant, overlapping
or turned away is missed, and more are missed the higher the camera is. The readout says so
under the number.

**Trails** are a toggle too, and off by default.

`web/js/track.js` is IoU association with a centre-distance fallback and a short memory. The
fallback is what makes it survive a person walking: detection runs a few times a second, so
between two looks someone can move most of their own width and the boxes then do not overlap
at all. IoU alone lost them and issued a new number, which is what "person #1, then #2, then
#3" looked like from the outside.

Not SORT, which is GPL-3.0 and not something to inherit by accident.

An API cannot do any of this at any price: one round trip takes seconds, so there is nothing
to associate between frames. That is the difference between watching something and
describing a photograph of it.

The browser cannot open RTSP, so this is the tablet's or laptop's own camera. The drone's
own feed is RTSP and is handled natively by the Android app's camera screen.

Pick one in the app. The header badge stops claiming local processing the moment an API engine
is selected, because that claim would then be false. The key is held in the tab, never written
to disk, and sent only to the provider chosen.

For batch work and for building training data, `tools/vlm_inspect.py` does the same thing on
the command line with the same prompt (`web/prompts/inspection.json`, read by both, so the
page and the tool cannot grade the same photograph differently).

## Video

Drop an MP4 or WebM clip. It is sampled, scored for sharpness by Laplacian variance,
deduplicated by difference hash, and reduced to a couple of dozen distinct in-focus frames.
Five minutes at 25fps is 7,500 frames and most of them are the same photograph.

It selects for coverage, not for interest: a defect visible in exactly one blurred frame can
be dropped. Upload a still of anything suspicious.

## On the tablet

`docs/INSTALL-tablet.md`. Two routes, same code:

- **From the browser.** Add to home screen. Updates itself, needs a current Chrome.
- **APK.** `android/` is a WebView around the same `web/` directory, copied in at build
  time. Built by CI on every push to `main` and downloadable from Actions. Take this one
  when the tablet has no Play Store, no install option, or may never see WiFi - everything
  including the ONNX runtime is inside the file.

Either way the on-device engine works with no signal; the API engines cannot, and the app
says so rather than queueing work that will never send.

## Why the trained model is not the default

`archive/turbine-v2/` holds the previous detector. It scored mAP50 0.759 on a split with zero
near-duplicate leakage and 0.5% false positives across 400 healthy blades, and then reported
"no defects" on a photograph of a turbine with a blade severed in half. Both facts are true,
and the second is the one that matters.

Three things caused it, and a replacement dataset has to fix all three:

**The class list did not contain the failure.** `corrosion`, `crack`, `surface_peeling`. A
severed blade is none of them, so a perfect model with those classes still outputs nothing.

**Every defect example was a close-up.** Blade surface at arm's length. A whole turbine across
a field is a different scale entirely, and detectors do not bridge that gap.

**The negatives were the wrong domain.** The "nothing here" examples were wide aerial shots, so
wide framing is precisely what the model learned means *no defect*. That was the framing of the
photograph it failed on. `tools/rebuild_turbine.py` printed this caveat on every run.

## The plan: get back to a trained model, with the right data

The API is not the destination. It is what makes the destination reachable.

Every inspection writes a label sidecar. `tools/vlm_to_yolo.py` turns a pile of those into a
YOLO dataset - split by whole inspection so validation is not a memorisation test, reviewed
sidecars only, refusing to run on a label with no place in the class table rather than
guessing one.

That dataset is made of the operator's own photographs, at the operator's own framing, from
the operator's own drone. It is the one thing the failed model never had. When it reaches the
size below, train on it and the API becomes optional.

From the app: analyse a batch, then **Save as training data**. That downloads a zip laid
out exactly as an inspection, so it drops straight into `inspections/`.

From the command line, for a folder of photographs:

```bash
python3 tools/vlm_inspect.py photos/ --provider anthropic --domain turbine
```

Either way, the rest is the same:

```bash
# review the annotated copies, correct the sidecars, set "reviewed": true in each
python3 tools/vlm_to_yolo.py inspections/ --dry-run          # how much is there
python3 tools/vlm_to_yolo.py inspections/ --out datasets/turbine_field
python3 tools/audit_dataset.py datasets/turbine_field
```

## What a replacement dataset needs

- **A class list covering the damage that matters**, structural failure included. Decided
  before collection, not after.
- **Negatives from the same domain as the positives** - healthy blade, same range, same camera,
  same framing. This is the single biggest fix.
- **Range coverage matching the capture hardware**, so the training distribution and the
  deployment distribution are the same one.
- **Out-of-domain images in the test set.** A benchmark that cannot fail measures nothing. Every
  metric quoted above was computed inside the domain the data defined, which is why none of them
  predicted the failure.

## Pipeline

```
new dataset  ->  tools/rebuild_turbine.py   group-aware split, polygons to boxes, negatives
             ->  tools/audit_dataset.py     7 structural checks; do not train on a failure
             ->  tools/train_turbine.py     config lives here, not in the notebook
             ->  tools/evaluate.py          per-class AP, not aggregate mAP
             ->  tools/false_positive_check.py   boxes drawn on healthy frames
             ->  tools/export_onnx.py       ONNX + int8, into web/models/
             ->  tools/publish_results.py   commits the model and its metrics
```

Run it on Kaggle with `training/turbine/turbine_kaggle.ipynb` - **Save Version → Save & Run
All (Commit)**, never a Draft Session. All hyperparameters live in `tools/train_turbine.py`,
which the notebook clones fresh each run, so a stale notebook cannot produce a stale model.

Before the first run, set `CLASS_REMAP` and `V2_NAMES` in `tools/rebuild_turbine.py` from the
new `data.yaml`. Both are empty and the script refuses to run until they are filled: they
encoded the old dataset's class indices, and reusing them would relabel every box.

### The audit

`tools/audit_dataset.py` checks for source-family shortcuts, cross-split leakage,
near-duplicate frames, class-imbalanced box scales, thin test classes, mixed label formats and
duplicate files. It failed 6 of 7 on the original export and passed 7 of 7 after the rebuild.
Run it on anything new before spending GPU time.

## Web app

`web/` is static, no build step. `web/models/manifest.json` is the deploy switch: drop an
`.onnx` beside it, set `labels` and `severityWeights`, and the app picks it up. A missing model
is reported as unavailable rather than failing silently.

```bash
npm run serve          # http://localhost:8080
npm test               # browser checks + the provider adapters (needs playwright)
npm run test:vlm       # provider adapters only, no browser, no network
python3 -m pytest tests/                # 497 checks
python3 tests/test_export_contract.py   # proves the JS decoder matches Ultralytics
```

Solar detection and the domain gate are scaffolded and unbuilt; both need solar imagery, and
the gate has to learn `solar` as a class, so one dataset unlocks both.
