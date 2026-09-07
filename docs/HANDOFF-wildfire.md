# Handoff — Real-time drone fire detection for firefighters

**This is the plan for a NEW repository.** Paste this whole file into a fresh Claude Code
session as the opening context. It is written to be read cold, with no prior conversation.

Suggested repo name: `wildfire-watch`. Nothing is shared with the sibling inspection repo
(`drone-inspection`) except a few hundred lines of overlay-drawing code, which should be
copied rather than abstracted — the two have different release cadences and very different
risk profiles.

---

## 1. What this is

A drone flies over an incident. Its video reaches a ground-station laptop, which runs fire
and smoke detection on every frame and pushes both the video and the detections to
firefighters' tablets over local WiFi, with sub-second latency. Tablets are Android **and**
iOS, and the build must not require Xcode or an App Store.

### Scope decision that governs everything else

**Firefighters may genuinely use this.** That is not a README concern — it changes the
architecture. The specific danger with RGB fire detection is not false positives (annoying,
survivable) but **false negatives**: thin smoke against bright sky, smouldering with no
visible flame, fire under canopy, night. A human knows when they are struggling to see. The
model does not — it simply returns nothing, and nothing looks exactly like "all clear".

So these are **architectural invariants, not disclaimers**. Do not let them erode:

1. **Absence of a detection never renders as a negative claim.** No "ALL CLEAR", no
   "0 fires detected", no green status light. Detections appear when present; nothing appears
   when absent. The UI never makes an affirmative safety statement.
2. **The overlay augments a feed the operator is already watching.** It says *look here*. It
   never says *there is nothing there*. Nobody should be looking at boxes instead of video.
3. **No automated alerting that anyone acts on without seeing the video themselves.**
4. **Everything is logged** — every frame's detections, timestamped, with model version and
   confidence. This gives after-action review, and quietly builds the real-incident
   validation set that any eventual deployment will need.
5. **Validation before reliance.** Before a department leans on this, test against *their*
   footage, measuring false negatives specifically. That is achievable for a student project,
   but it comes before deployment, not after.

EN 54 and UL 268 cover fixed fire-detection installations and do not apply to a drone-feed
overlay. That is exactly why the framing matters: it is not a certified detector, so it must
never be presented as one. Call it a **situational-awareness aid** consistently, in the UI
and in any writeup.

---

## 2. Hardware context

The ground station is confirmed available — assume a laptop with a discrete GPU.

Video source is a handheld ground-station controller, most likely a **SIYI MK15** (the user
said "MK16", which is not a product; the near matches are SIYI MK15/MK32, Skydroid/CUAV H16
Pro, and MX16). **Confirm which.** It matters less than it sounds, because all of them expose
the same thing:

> **RTSP over Ethernet at a fixed IP.** SIYI MK15 serves `rtsp://192.168.144.25:8554/main.264`
> at 1080p and quotes ~180 ms end-to-end video latency.

Build the ingest layer as a **swappable adapter** — RTSP, RTMP, HDMI capture device, and a
local video file must all be valid sources. The file source is what you develop against
before any drone is involved, and what you use for regression tests forever after.

---

## 3. Architecture

```
  Drone camera
      │  ~180 ms  (radio link + controller, SIYI MK15 spec)
      ▼
  Ground-station controller  ──RTSP over Ethernet──┐
                                                    ▼
  ┌──────────────────── Ground station (laptop) ────────────────────┐
  │  ingest: RTSP/RTMP/HDMI/file  →  decode                          │
  │       ↓                                                          │
  │  inference: YOLO fire+smoke, GPU, 10-30 fps                      │
  │       ↓                                                          │
  │  temporal filter: N-of-M frame persistence  ← the accuracy lever │
  │       ↓                                                          │
  │  ├─ WebRTC video track ─────────────────┐                        │
  │  ├─ detections over data channel ───────┤  (small JSON/frame)    │
  │  ├─ incident log (JSONL + video record) │                        │
  │  └─ serves the PWA over local WiFi ─────┤                        │
  └─────────────────────────────────────────┼────────────────────────┘
                                             ▼
                          Tablets (Android + iOS, any browser)
                          <video> element + Canvas 2D overlay
```

### Why this shape

**Inference once at the ground station, not on each tablet.** N firefighters watching means N
identical inferences on weak, battery-powered hardware that will thermally throttle. One GPU
pass gives identical results to every viewer and keeps tablets cool.

**Detections travel on a data channel, not burned into the video.** Burning boxes into pixels
forces a re-encode (adds latency and CPU), makes overlays blurry, and prevents the tablet from
doing anything smart with them. A frame's detections are a few hundred bytes of JSON.

> **Known risk, flagged for verification:** video and data channel are separate streams, so
> boxes can drift out of sync with the frames they describe. On a moving aerial scene that
> could put a box over the wrong ground. **Mitigation: stamp every detection payload with the
> frame's presentation timestamp and have the tablet buffer detections to match
> `video.currentTime`, rather than drawing the newest payload immediately.** Build this in
> from the start; retrofitting it is painful.

**WebGL is the wrong question.** It is a rendering API, not an architecture. Canvas 2D draws a
dozen boxes fine. WebGPU would only matter for on-tablet inference, which this design avoids.

### Transport

WebRTC, sub-500 ms, works in every current browser including iOS Safari. HLS/LL-HLS (2–30 s)
is useless here. Self-hosted server options, easiest first: **MediaMTX** (single Go binary,
speaks RTSP in and WebRTC out — likely all you need), then Pion, mediasoup, LiveKit.

**The no-internet problem is the one to solve early.** An incident scene may have no
connectivity. Verify on real hardware, early:
- WebRTC on a LAN should work on host ICE candidates alone, with no STUN/TURN. Confirm it.
- Browsers require a **secure context** for WebRTC and service workers. On a LAN with no
  public CA this means a self-signed certificate — and it is genuinely unclear whether iOS
  will install a PWA or run a service worker behind one. **Test this before building the app
  around it.** If it fails, options are a trusted cert for a `.local` name via a private CA
  installed on the tablets, or falling back to a plain non-installed browser tab.

---

## 4. Model and data

### Datasets

| Dataset | Size | Viewpoint | Why it matters |
|---|---|---|---|
| **FASDD** | 122,634 samples — 70,581 positive, **52,073 negative** | mixed: UAV, surveillance, satellite | The negatives are "confusing non-fire" images. That is exactly the hard-negative set you need. YOLO/VOC/COCO formats. |
| **FLAME** (IEEE DataPort) | drone pile burns, Arizona | **genuine UAV** | Frames, masks, video, plus thermal palettes. Real aerial fire. |
| **Boreal Forest Fire** (Sci Data 2025) | bbox + segmentation + video | **genuine UAV** | Finnish boreal, human + foundation-model co-annotated. |
| D-Fire | ~21k | ground-level | Useful bulk, wrong viewpoint. Weight it down. |
| Corsican Fire DB | — | ground-level | Classic benchmark, mostly ground. |

**Audit every one of them before training.** The sibling repo has `tools/audit_dataset.py`,
which is directly reusable — copy it over. It exists because that project's turbine model
scored mAP50 0.782 and was useless: filename families perfectly predicted the class and no
image contained two classes together. Fire datasets built from video have the *same* trap in
a different form — **consecutive frames randomly split across train/val**, which makes
validation a memorisation test. Assume it is present until the audit says otherwise.

### Model

- **yolo11s at 640** for real-time. Start here. Measure fps before considering anything bigger.
- Two classes: `fire`, `smoke`. Consider a third for hard negatives if the data supports it.
- **Smoke is a poor fit for bounding boxes** — it is amorphous with no crisp boundary.
  Detection is the pragmatic start; note the limitation, and evaluate segmentation for smoke
  later if box quality is bad.
- Published aerial RGB detectors land around **80% mAP@0.5** (YOLOv5x on FASDD). Treat
  anything much above that on your own test split as evidence of leakage, not skill.

### The single biggest accuracy lever: temporal consistency

Per-frame detection on video is noisy. **Require a detection to persist across N of the last
M frames before it is displayed.** This is cheap, needs no retraining, and kills most
transient false positives — sun glint, a passing red vehicle, a single bad frame. It is the
difference between a jittery demo and something watchable. Build it into the pipeline, not
the UI.

### Expected false positives — design for these

Firefighters wear high-vis. Apparatus is red. At golden hour everything looks like fire.
Also: brake lights, red roofs, autumn foliage, dust plumes, steam, lens flare. FASDD's
negative set is your best defence, plus hard-negative mining from your own footage.

---

## 5. Tablet app

**PWA, installed to the home screen.** No Xcode, no App Store, no $99/yr Apple Developer
account. Served by the ground station over local WiFi so it works with no internet.

**Verify these on a real iPad before committing** — if Wake Lock or service workers fail
behind a self-signed cert, the tablet story needs rework:

- [ ] Home-screen PWA sustains a WebRTC video session on iPadOS
- [ ] **Screen Wake Lock API** — a tablet that sleeps mid-incident is useless
- [ ] Service worker registration behind a self-signed LAN certificate
- [ ] Behaviour when the app backgrounds or the tablet locks
- [ ] Storage eviction between incidents

Fallback if PWA install fails on iOS: a plain browser tab still does WebRTC and Canvas. You
lose offline caching and home-screen launch, not core function.

**UI rules** (field conditions: sunlight, gloves, urgency, nobody reads a manual):
- Maximum contrast. Assume direct sun on the screen.
- Do not rely on red/green alone — colour-blind users, and red means something else here.
- Big touch targets. Gloves.
- Show detection confidence, not just a box. A 0.3 and a 0.9 must not look alike.
- Show the model's last-inference timestamp, so a frozen pipeline is visible rather than
  silently showing stale boxes.
- **Nothing that could be read as "the area is clear".**

---

## 6. Build order

1. **File-source spike.** Ingest a recorded fire video → run a pretrained YOLO → draw boxes →
   serve over WebRTC to a browser. No drone, no real model. Proves the whole pipeline.
2. **Verify the platform assumptions** (§3 no-internet, §5 iOS checklist) on real hardware.
   These are the two things that can force a redesign — find out now, not after training.
3. **Temporal filter + logging + the safety invariants** from §1. Before any real model, so
   they are structural.
4. **Train the model.** See §7.
5. **RTSP ingest** from the actual controller. Measure real end-to-end latency.
6. **Field test** with recorded footage from the department, measuring false negatives.

Steps 1–3 need no GPU, no drone, and no trained model. Do them first.

---

## 7. Compute — and where free stops being enough

Kaggle's free 30 GPU-h/week is right for most of this, but **fire is the one place it breaks
down**:

| | Images | Est. on Kaggle P100, 50 epochs | Verdict |
|---|---|---|---|
| Full FASDD | 122,634 | **~12–16 h** | ✗ exceeds the 9-hour session limit |
| FASDD subset (~30k) | 30,000 | ~4 h | ✓ fits |
| FLAME + Boreal only | ~10k | ~1.5 h | ✓ fits |

Three ways through: subsample, checkpoint-resume across sessions, or **rent a 4090 on
Vast.ai/RunPod for ~$2** (4–6× a P100, so one 2–3 h sitting). For the full-FASDD run the $2 is
genuinely worth it — a session timeout mid-run is exactly what wasted the sibling project's
first attempt.

Everything else — the server, the PWA, the streaming — is free. Students: the
[GitHub Student Developer Pack](https://education.github.com/pack) adds credits, JetBrains,
and a free domain for a year.

---

## 8. Suggested repo layout

```
wildfire-watch/
├── station/                ground station (Python)
│   ├── ingest/             RTSP / RTMP / HDMI / file adapters
│   ├── inference/          model runner, temporal filter
│   ├── stream/             WebRTC server (MediaMTX or aiortc)
│   ├── logging/            incident JSONL + video recording
│   └── serve/              serves the PWA over LAN
├── app/                    tablet PWA
│   ├── index.html, sw.js, manifest.webmanifest
│   └── js/  video + timestamp-matched overlay + status
├── training/               Kaggle notebooks
├── tools/                  audit_dataset.py (copied), export, evaluate
├── tests/                  pipeline tests against recorded video
└── docs/
```

---

## 9. Open questions to resolve first

1. **Which controller exactly?** SIYI MK15 / MK32 / Skydroid H16 / MX16 — confirm the RTSP URL
   and resolution.
2. **Does iOS install a PWA and run a service worker behind a self-signed LAN cert?** This is
   the single biggest unknown. If no, the tablet plan changes.
3. **How badly do overlays drift** against video on a moving aerial scene, and is
   timestamp-matching enough?
4. **What fps is actually needed?** Fire spreads slowly — 5 fps may be entirely sufficient,
   which would relax every hardware constraint downstream. Worth deciding deliberately rather
   than defaulting to 30.
5. **Who validates it**, against what footage, before anyone relies on it?

---

## 10. Sources

- [FASDD (100k-level flame and smoke detection dataset)](https://essd.copernicus.org/preprints/essd-2023-73/)
- [FLAME — aerial pile burn detection using UAVs](https://ieee-dataport.org/open-access/flame-dataset-aerial-imagery-pile-burn-detection-using-drones-uavs)
- [Boreal Forest Fire — UAV wildfire detection and smoke segmentation (Scientific Data, 2025)](https://www.nature.com/articles/s41597-025-05634-0)
- [Detecting Wildfire Flame and Smoke through Edge Computing](https://arxiv.org/pdf/2501.08639)
- [SIYI MK15 — RTSP and 180 ms latency](https://shop.siyi.biz/products/siyi-mk15-enterprise)
- [WebRTC vs LL-HLS vs CMAF latency comparison (2026)](https://floatleftinteractive.com/guides/low-latency-streaming-protocols-webrtc-vs-ll-hls-vs-cmaf-2026-guide/)
- [WebRTC latency figures](https://www.nanocosmos.net/blog/webrtc-latency/)
