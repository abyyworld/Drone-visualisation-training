# wildfire-watch

A real-time overlay for drone video at a wildfire. One laptop on the ground
runs a fire/smoke model over the drone's feed and highlights what it finds on
tablets the crew are already watching.

## Read this before anything else

**This is a situational-awareness aid. It says *look here*. It never says
there is nothing there.**

RGB fire detection fails toward silence. Thin smoke on a bright sky,
smouldering with no flame, fire under canopy, fire at night -- in every one of
those the model returns an empty result, and that empty result is
byte-identical to the one it returns for an empty field. A human knows when
they are struggling to see. The model cannot report that it is struggling.

So the empty result is rendered as **nothing at all** -- no reassuring panel,
no count, no green light -- and the operator keeps watching the video, which
they were doing anyway. The overlay only ever adds a hint on top. It never
subtracts from the human's own scan of the picture.

That principle is enforced in code, not by convention:
`station/core/safety.py` lists the phrasings that are banned from
operator-facing text and why, and `tests/test_safety_invariants.py` fails the
build if any of them appear anywhere in the repository. The full reasoning,
including why EN 54 and UL 268 do not apply and why that makes the framing
*more* important rather than less, is in **[docs/SAFETY.md](docs/SAFETY.md)**.

It is not a certified device and no standard covers it. Before anyone relies
on it, its false-negative behaviour has to be measured on the department's own
footage: **[docs/VALIDATION.md](docs/VALIDATION.md)**.

---

## Architecture

```
   ┌──────────┐
   │  drone   │  camera + H.264 encode
   └────┬─────┘
        │  radio link
        v
   ┌──────────────────────┐
   │ controller ground    │  fixed IP, Ethernet
   │ unit (SIYI MK15 /    │
   │ MK32 / H16 -- TBC)   │
   └────┬─────────────────┘
        │  RTSP   rtsp://192.168.144.25:8554/main.264
        v
╔═══════════════════════════════════════════════════════════════════╗
║  GROUND-STATION LAPTOP  (discrete GPU)                            ║
║                                                                   ║
║   ingest ──▶ decode ──▶ YOLO inference ──▶ N-of-M temporal filter ║
║  (rtsp|file|                (once, here)      (3 of last 5)       ║
║   rtmp|hdmi)                                        │             ║
║                                    ┌────────────────┼───────────┐ ║
║                                    v                v           v ║
║                            (a) WebRTC video   (b) detections  (c) ║
║                                track            as JSON on a  log ║
║                                                  data channel     ║
║                                    (d) serves the PWA over HTTPS  ║
╚═══════════════════════════════════╤═══════════════════════════════╝
                                    │  LAN WiFi, no internet
                    ┌───────────────┴───────────────┐
                    v                               v
            ┌───────────────┐               ┌───────────────┐
            │   tablet      │               │   tablet      │
            │  <video>      │               │  <video>      │
            │  + Canvas 2D  │               │  + Canvas 2D  │
            │    overlay    │               │    overlay    │
            └───────────────┘               └───────────────┘
```

Two properties of that diagram are load-bearing:

- **Inference happens once, on the ground station.** Tablets receive pixels
  and JSON; they never run a model. Adding a tablet costs a WebRTC connection,
  not a GPU.
- **Detections are never burned into the video pixels.** They travel as JSON
  on a data channel and are drawn on a canvas over the video, so the overlay
  can be *withdrawn* -- when the pipeline stalls or the data goes stale, the
  boxes stop and the operator is left with plain video, which is a safe state.
  Boxes painted into the picture could not be taken back.

The wire format, and the three-tier algorithm that keeps a box on the frame it
was computed from, are specified in **[docs/CONTRACT.md](docs/CONTRACT.md)**.

---

## Quickstart

### 1. Install (no GPU, no drone, no model needed)

```bash
git clone <this repo> && cd wildfire-analysis
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'          # core + pytest + ruff
make test                        # the whole pure-logic suite
```

The core installs and tests on any machine. Every heavy dependency
(ultralytics, torch, av, cv2, aiortc, aiohttp) is imported inside the function
that needs it, never at module import time, so the logic runs where the
runtime does not.

### 2. Run the pipeline over a video file

The file source is the development path and it exercises everything except the
radio link: ingest, decode, the temporal filter, the incident log, WebRTC and
the PWA.

```bash
pip install -e '.[dev,stream,cv]'        # decode + WebRTC + HTTPS. Still no GPU.

# The whole pipeline over the committed demo clip, served to the LAN.
python -m station certs                                  # self-signed cert
python -m station run --source-type file \
                      --source demo/assets/wildfire_demo.mp4
```

Open `https://<laptop-lan-ip>:8443` on a tablet or a phone on the same WiFi
and accept the certificate warning. That warning, and what it costs on iOS, is
the subject of [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).

Useful variants:

```bash
make run-stub          # transport + overlay only: NO MODEL RUNS, boxes are generated
make demo              # render the overlay offline to an MP4 -- the stage-proof fallback
python -m station check    # what can this laptop actually do? Run it on the bench.
```

With trained weights in place, `--weights models/your.pt` runs the real model.
Without them the station says so and stops, rather than pretending.

### 3. Then, and only then, RTSP from the real controller

```bash
# Laptop needs an address on the controller's subnet (SIYI family shown).
sudo ip addr add 192.168.144.100/24 dev eth0
ffprobe -rtsp_transport tcp rtsp://192.168.144.25:8554/main.264   # confirm first

python -m station run --source-type rtsp \
                      --source rtsp://192.168.144.25:8554/main.264
```

Or put it in the config:

```bash
cp config.example.yaml config.yaml     # every key is documented in place
python -m station -c config.yaml run
```

The exact controller model is unconfirmed -- see
[docs/HARDWARE.md](docs/HARDWARE.md) §2, which also explains why it matters
less than it sounds, and how to find the stream on whatever unit turns up.

---

## Build order

Plainly, in this order, and each step is useful on its own:

1. **File-source spike, end to end.** Video file in, boxes on a tablet out,
   through the real transport. This is the step that proves the architecture,
   and it is the one most projects skip in favour of training first.
2. **Verify the platform assumptions on real hardware.** The controller's
   actual RTSP stream ([docs/HARDWARE.md](docs/HARDWARE.md)) and the iPad
   checklist ([docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) §5). Half a day each,
   and each can invalidate a plan that would otherwise be built on for weeks.
3. **Temporal filter, incident logging and the safety invariants.** N-of-M
   persistence, every frame logged including the empty ones, and the test that
   enforces the language rules.
4. **Train.** Datasets, the leakage audit, then the smallest useful run first.
   See `training/` and `tools/audit_dataset.py`.
5. **RTSP from the real controller.** One config change, if steps 1-3 were
   done properly.
6. **Field test on the department's own footage, measuring false negatives.**
   [docs/VALIDATION.md](docs/VALIDATION.md). This is the step that decides
   whether anyone may rely on it.

**Steps 1-3 need no GPU, no drone and no trained model.** That is deliberate.
Roughly two-thirds of this system is transport, synchronisation, honest
failure handling and record-keeping, and all of it can be built, tested and
demonstrated on a laptop before a single epoch is trained -- against a
recorded clip, with a stub runner, and with a synthetic source whose ground
truth is known exactly.

---

## Repository map

| Path | What is in it |
|---|---|
| `station/core/` | **The contract.** Wire types, config schema, safety invariants. Stdlib only |
| `station/ingest/` | Frame sources: file, rtsp, rtmp, hdmi, synthetic -- one adapter seam |
| `station/inference/` | Model runner, the stub runner, and the N-of-M temporal filter |
| `station/stream/` | WebRTC video track, the detections data channel, signalling |
| `station/incidentlog/` | Writer, reader and video recorder for `incidents/<id>/` |
| `station/serve/` | HTTPS server for the PWA, and self-signed certificate issuance |
| `station/cli.py` | `run`, `certs`, `check`, `replay` |
| `app/` | The tablet PWA: video element, Canvas 2D overlay, sync, service worker |
| `training/` | Dataset merge, split by source video so the split cannot leak |
| `tools/` | Dataset audit, recall-first evaluation, export with a class-order check |
| `demo/` | Synthetic clip generator and the offline overlay render |
| `docs/` | The documents linked below |

## Documents

| | |
|---|---|
| **[docs/SAFETY.md](docs/SAFETY.md)** | The five invariants and the full reasoning. Read first |
| **[docs/CONTRACT.md](docs/CONTRACT.md)** | Wire protocol, overlay synchronisation, the staleness rule |
| **[docs/VALIDATION.md](docs/VALIDATION.md)** | How to measure false negatives before anyone relies on this |
| **[docs/HARDWARE.md](docs/HARDWARE.md)** | Laptop, controller, tablets, network, latency budget |
| **[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)** | No-internet LAN, TLS without a public CA, the field runbook |
| **[docs/OPEN_QUESTIONS.md](docs/OPEN_QUESTIONS.md)** | The five things that are not known, and what each one changes |

## Development

```bash
make help              # every target
make install           # core + dev
make test              # pytest
make lint              # ruff
make verify            # what CI runs
make safety            # only the safety-invariant test
make config-check      # validate config.example.yaml against the schema
```

CI (`.github/workflows/ci.yml`) installs the dev extra **only** -- no GPU, no
ffmpeg, no ML stack -- and runs ruff and pytest. That is not a limitation
being tolerated; it is the constraint that keeps the heavy imports lazy, and
CI is what notices when one of them drifts back to module scope.

## Status

Early. What is unconfirmed is listed honestly in
[docs/OPEN_QUESTIONS.md](docs/OPEN_QUESTIONS.md): the controller model, iOS PWA
behaviour behind a self-signed certificate, how much overlay drift the
timestamp estimator really has, what frame rate is actually needed, and who
validates it against which footage.

Until the validation in [docs/VALIDATION.md](docs/VALIDATION.md) has been done
and signed, this is a demonstrator, and it should be described that way to
everyone who sees a screen.
