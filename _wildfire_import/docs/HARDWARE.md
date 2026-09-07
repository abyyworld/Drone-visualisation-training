# Hardware

Everything the system runs on, what is confirmed, what is not, and where the
milliseconds go.

Two things in this document are unresolved and both are flagged in
`docs/OPEN_QUESTIONS.md`: the exact ground-station controller model, and
whether iPads will do what the tablet plan needs (`docs/DEPLOYMENT.md`).

---

## 1. The ground station laptop

One laptop runs ingest, decode, inference, the WebRTC server, the PWA and the
incident log. Inference happens here **once**, for all tablets. That is the
central architectural decision: tablets get pixels and JSON, never a model.

### Requirements

| Part | Requirement | Why |
|---|---|---|
| GPU | Discrete NVIDIA, >= 6 GB VRAM (RTX 4060 mobile / 3060 or better) | YOLO11s at 640 is ~10-20 ms per frame on a card like this and ~200-500 ms on a laptop CPU. The difference is between 10 fps of headroom and a slideshow |
| CPU | 6+ modern cores | H.264 decode when the GPU decoder is busy, plus the WebRTC encode if you take that path |
| RAM | 16 GB, 32 GB comfortable | Decoder, model, encoder, and a browser for the operator's own view |
| Storage | NVMe, >= 256 GB free | Incident video at 1080p/8 Mbps is ~3.6 GB per hour. `detections.jsonl` is ~7 MB per hour and irrelevant by comparison |
| Network | **Wired Ethernet port**, plus WiFi | The controller is an Ethernet device. Thin laptops need a USB-C gigabit adapter -- buy a known-good one and test it, cheap adapters drop link under load |
| Display | Readable outdoors, >= 400 nits | The operator is often in a vehicle with the doors open |
| Power | 12 V inverter or vehicle power | Sustained GPU inference gives 1-1.5 h of battery on a typical gaming laptop. Plan for mains, not for the battery |

### Software

Linux or Windows both work. Install the CUDA build of PyTorch **before**
ultralytics -- installing ultralytics first pulls the CPU wheel of torch as a
dependency, and the station will then run inference at a few frames per second
with no error message explaining why. `python -m station check` reports the
device that was actually selected; run it on the bench.

Thermals matter more than they sound. A GPU laptop in a hot vehicle throttles
within minutes; inference time doubles and the pipeline reports `degraded`.
Airflow and shade are part of the deployment, not an afterthought.

---

## 2. The controller -- **this must be confirmed**

> The model was given as **"MK16"**. There is no product by that name.

The near matches, any of which could have been meant:

| Candidate | What it is | Video out |
|---|---|---|
| **SIYI MK15** | 5.5" handheld, "Mini HD" enterprise link, ~15 km | RTSP over Ethernet, 1080p |
| **SIYI MK32** | 7" handheld, same family, brighter screen, ~15 km | RTSP over Ethernet, 1080p |
| **Skydroid / CUAV H16 Pro** | 7" Android handheld with integrated video link | RTSP over Ethernet (and an Android app on the unit itself) |
| **MX16** | Handheld ground station in the same class | RTSP over Ethernet |

**Confirm the model before ordering anything, and before quoting a latency
number to anybody.** Ask for a photo of the label on the back of the unit.

### Why it matters less than it sounds

Every candidate above solves the same problem the same way: the ground unit
puts an H.264 stream on a **fixed IP address over Ethernet**, and serves it as
**RTSP**. The station takes a URI:

```yaml
source:
  type: rtsp
  uri: "rtsp://192.168.144.25:8554/main.264"
```

So the model determines a string in a config file, plus the numbers in the
latency budget. `station/ingest/rtsp_source.py` does not care which box is at
the other end. This is exactly why the ingest layer is an adapter seam with a
file source behind it: build steps 1-3 in the README need no controller at
all, and the controller becomes a one-line config change at step 5.

### What is actually known

**SIYI MK15** -- from the manufacturer's documentation:

- RTSP endpoint `rtsp://192.168.144.25:8554/main.264`
- 1080p
- **~180 ms quoted end-to-end video latency** (camera glass to the
  controller's own screen -- see the budget below for what that does and does
  not include)
- Ground unit on the `192.168.144.0/24` subnet; the laptop needs a static
  address on that subnet

For **MK32**, **H16 Pro** and **MX16**, treat the address and path as
**unverified** until you have run `ffprobe` against the actual unit. The SIYI
units share the `192.168.144.x` family; the Skydroid/CUAV units use their own
scheme. Do not copy an address out of a forum post into a config file and
assume it is right.

### How to find the stream on any of them

```bash
# 1. Ethernet from the controller to the laptop. Give yourself an address on
#    the controller's subnet (SIYI family shown).
sudo ip addr add 192.168.144.100/24 dev eth0
sudo ip link set eth0 up
ping 192.168.144.25

# 2. If you do not know the address, sweep the subnet the manual names.
nmap -sn 192.168.144.0/24
nmap -p 554,8554,1935 192.168.144.0/24

# 3. Confirm the stream and read its real parameters.
ffprobe -rtsp_transport tcp rtsp://192.168.144.25:8554/main.264

# 4. Then hand the same URI to the station.
python -m station run --source-type rtsp --source rtsp://192.168.144.25:8554/main.264
```

Record what `ffprobe` reports: resolution, frame rate, profile, and **GOP
length**. A long GOP (say 4 seconds between keyframes) adds latency at every
reconnect and makes a dropped link expensive to recover from; if the
controller lets you shorten it, do.

### Confirmation checklist for the real unit

- [ ] Exact model, from the label on the hardware
- [ ] Video out: Ethernet RTSP / USB / HDMI only?
- [ ] RTSP URI, verified with `ffprobe`, written into `config.yaml`
- [ ] Resolution, frame rate and bitrate actually delivered over the radio at
      range -- not the brochure figure
- [ ] Whether a second, lower-resolution stream is available (useful: infer on
      the small one, relay the large one)
- [ ] GOP length, and whether it can be changed
- [ ] Measured glass-to-controller-screen latency, with a stopwatch on screen
- [ ] Behaviour when the link drops and recovers: does the RTSP server keep
      the port open? (`source.reconnect_s` exists for this)
- [ ] Power draw and how the unit is powered in the field

### If it turns out there is no RTSP

Two documented fallbacks, both already in the ingest layer:

- **HDMI capture** (`source.type: hdmi`). A USB capture device on the
  controller's HDMI output. Always works, because it is downstream of
  everything. Costs one extra decode/encode round trip (~30-60 ms), gives you
  the controller's on-screen overlay burned into the picture, and needs
  `opencv-python`.
- **RTMP push** (`source.type: rtmp`). If the controller can push to a relay
  rather than serve, run one and point the station at it.

---

## 3. Tablets

| | Minimum | Notes |
|---|---|---|
| iOS | iPad, iOS/iPadOS 15.4+ | 15.4 is where Safari gained `requestVideoFrameCallback`, which is what tier-1 overlay synchronisation needs (`docs/CONTRACT.md`) |
| Android | Android 10+, Chrome 90+ | `requestVideoFrameCallback` since Chrome 83 |
| Screen | >= 500 nits, matte or a hood | Sunlight is the real constraint. A tablet nobody can read is not a tablet |
| Battery | A day, or a power bank per tablet | Video decode plus a wake lock is a heavy load |
| Case | Rugged, usable with gloves | Obvious in the field, easy to forget when ordering |

Nothing is installed on the tablets except a browser. The PWA is served by the
station, over the LAN, from `app/`. Whether iOS will install it to the home
screen and run a service worker behind a self-signed certificate is
**unverified** -- see `docs/DEPLOYMENT.md`, which also gives the fallback
(a plain browser tab, which keeps full core function).

---

## 4. Network

```
   drone
     |  radio link (the controller's own protocol)
     v
  controller ground unit
     |  Ethernet, fixed IP (e.g. 192.168.144.25)
     v
  ground-station laptop  --- WiFi AP (5 GHz) --->  tablets
     serves https://<station>:8443
```

Two separate networks on the laptop, and keep them separate:

1. **Ethernet to the controller.** Static address on the controller's subnet.
2. **WiFi to the tablets.** Either the laptop's own hotspot, or -- better -- a
   small battery-powered travel router on 5 GHz. A dedicated router is more
   reliable than a laptop hotspot, keeps working while the laptop reboots, and
   lets tablets stay associated across a station restart.

There is **no internet on this network and there is not supposed to be**. That
constraint drives the whole of `docs/DEPLOYMENT.md`: no STUN, no TURN, no
public CA, no CDN, no app store.

---

## 5. Latency budget

What the operator sees on the tablet is older than the world by the sum of
these. Figures are for 1080p H.264 with the hardware above; measure them on
the real kit rather than trusting this table.

| # | Stage | Typical | Worst | Notes |
|---|---|---|---|---|
| 1 | Camera capture + encode on the air unit | *included* | | Part of the controller's quoted figure |
| 2 | Radio link | *included* | | Grows at range and with interference |
| 3 | Controller decode + serve RTSP | *included* | | |
| | **(1)-(3) combined** | **~180 ms** | 300 ms+ | SIYI MK15's quoted glass-to-its-own-screen figure. The RTSP served to Ethernet is not guaranteed to be the same path -- verify with a stopwatch |
| 4 | Ethernet + RTSP client buffering | 20-80 ms | 500 ms | The biggest avoidable cost here. Use TCP transport, minimal buffering, and a short GOP |
| 5 | H.264 decode on the laptop | 5-15 ms | 30 ms | |
| 6 | Preprocess + inference + NMS | 15-30 ms | 60 ms | YOLO11s @640 FP16 on an RTX 4060 mobile |
| 7 | Encode for WebRTC | 10-25 ms | 50 ms | **Zero if you relay the original stream** instead of re-encoding -- see the trade-off below |
| 8 | WebRTC transport + tablet jitter buffer | 30-100 ms | 300 ms | WiFi. Dominated by the jitter buffer, not by the air time |
| 9 | Tablet decode, composite, display | 16-33 ms | 50 ms | One or two frames at 60 Hz |
| | **Glass to tablet, video** | **~270-460 ms** | ~1.2 s | |

**Detections take a different path.** They are computed at stage 6 and go
straight onto the data channel, skipping stages 7-9's encode and jitter
buffer, so a payload usually arrives at the tablet **before** the frame it
describes. This is why the tablet buffers payloads and matches them to frames
rather than drawing the newest one on arrival (`docs/CONTRACT.md`), and why
`stream.overlay_buffer_s` defaults to 3 s.

**Time from a plume first being visible to a box appearing** is a different,
longer number:

```
  video latency                       ~0.3-0.5 s
+ inference sampling wait             up to 1 / inference.max_fps   = 0.1 s at 10 fps
+ N-of-M confirmation ((n-1) frames)  (3-1) / 10                    = 0.2 s
= roughly 0.6-0.8 s
```

That is the honest figure for "how quickly does a box appear", and against a
fire's own timescale it is nothing. Which is the point of open question 4: the
system is nowhere near needing 30 fps, and saying so out loud relaxes every
constraint downstream.

### The re-encode trade-off (stage 7)

Two paths exist in the codebase and they trade latency against overlay
precision:

- **aiortc re-encode** (`station/stream/webrtc.py`): decode, infer, re-encode,
  send. Costs stage 7 and one generation of quality loss, but the station owns
  the RTP timestamps, so `rtp_ts` is populated and the tablet gets **tier-1**
  (exact) overlay synchronisation.
- **MediaMTX relay** (`station/stream/mediamtx.py`): relay the controller's
  original stream to the tablets untouched, and decode a separate copy for
  inference. Saves the re-encode, but a relay generally cannot expose the RTP
  timestamp, so the tablet drops to **tier-2** (median pts-offset estimate).

Neither is wrong. Decide it with a measurement, and note that open question 3
-- how much overlay drift there actually is in tier 2 -- is the measurement
that decides it.

### If the budget blows out

In the order worth trying:

1. Shorten the GOP on the controller, and force RTSP over TCP with minimal
   client buffering (stage 4 is usually where the surprise is).
2. Drop `inference.max_fps`. It costs less than it feels like it should.
3. Relay instead of re-encoding (stage 7), accepting tier-2 sync.
4. 720p instead of 1080p to the tablets. The model runs at 640 regardless, so
   this costs the operator's picture, not the model's input.
5. Only then, a bigger GPU.
