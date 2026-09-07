# Open questions

Five things that are not known, that change the design, and that should be
answered deliberately rather than by default. Each one says why it matters,
how to answer it, and what changes depending on the answer.

None of them blocks build steps 1-3 in the README. That is not an accident:
the build order was chosen so that the work which can proceed under
uncertainty proceeds, and each question gets answered before the step that
actually depends on it.

| # | Question | Blocks | Cost to answer |
|---|---|---|---|
| 1 | Which controller, exactly? | step 5 (RTSP) | one photo of the label |
| 2 | iOS PWA behind a self-signed certificate? | the tablet plan | half a day with an iPad |
| 3 | Is timestamp matching enough for overlay alignment? | trusting box positions | one afternoon, measurable |
| 4 | What frame rate is actually needed? | every downstream constraint | one conversation + one measurement |
| 5 | Who validates it, against what footage? | anyone relying on it | one conversation, then weeks of work |

---

## 1. Which ground-station controller is it, exactly?

**The stated model is "MK16", and there is no such product.** The near matches
are SIYI MK15, SIYI MK32, Skydroid/CUAV H16 Pro and MX16.

### Why it matters

Less than it first appears, and that is worth stating plainly. Every candidate
puts H.264 on a fixed IP over Ethernet and serves it as RTSP, and the station
consumes a URI:

```yaml
source: { type: rtsp, uri: "rtsp://192.168.144.25:8554/main.264" }
```

So for the software, the answer is a string in a config file. `station/ingest/`
is an adapter seam precisely so that this is true.

Where it does matter: the **exact address and path** (guessing wastes an
afternoon on site), the **real latency** (the ~180 ms figure is SIYI MK15's,
and quoting it for a different unit would be dishonest), whether Ethernet is
available at all or only HDMI, whether a **second lower-resolution stream**
exists (infer on the small one, relay the large one -- a free win), and the
**GOP length**, which sets how expensive every reconnect is.

### How to answer

Photograph the label on the back of the unit. Then work through the
confirmation checklist in `docs/HARDWARE.md` §2: `nmap` the subnet the manual
names, `ffprobe` the stream, record resolution, frame rate and GOP, and
measure glass-to-screen latency with a stopwatch on camera.

### What changes

- **A SIYI unit**: `192.168.144.x`, laptop gets a static address on that
  subnet, config as above. Nothing else changes.
- **A Skydroid/CUAV unit**: same shape, different addressing. Confirm from the
  manual rather than from a forum post.
- **No RTSP at all**: fall back to `source.type: hdmi` with a capture card
  (costs ~30-60 ms and the controller's own on-screen overlay is burned into
  the picture) or `rtmp` with a relay. Both already exist in the ingest layer.
- **A second stream exists**: relay the high-resolution stream to the tablets
  untouched and infer on the low-resolution one. Saves the re-encode, but see
  question 3 -- it costs tier-1 overlay synchronisation.

---

## 2. Will iOS install this as a PWA, with a service worker, behind a self-signed certificate?

**Unverified. Do not commit to the tablet plan until it has been tested on a
real iPad.**

### Why it matters

The incident network has no internet, so there is no public CA, so the station
serves TLS from a self-signed certificate or a private CA
(`docs/DEPLOYMENT.md` §4). Chrome refuses to register a service worker behind
a certificate error; Safari's behaviour differs between a normal tab and a
home-screen app, has changed across releases, and is poorly documented.

What is at stake is narrower than it sounds. If the PWA does not install, the
fallback is a **plain browser tab**, which keeps video, the data channel, the
canvas overlay, synchronisation and the staleness rule -- every
safety-relevant behaviour. What is lost is the home-screen icon, the
chrome-free display and the offline app shell, and the app shell matters
little when the station serving it is ten metres away on the same LAN.

The reason to answer it early is that it decides **what the project promises**
and what the tablets are for.

### How to answer

The iPad verification checklist in `docs/DEPLOYMENT.md` §5. Half a day, on the
actual iPad model, on the actual access point. Eight items: plain tab,
home-screen install, service worker, a 20-minute WebRTC session, wake lock,
backgrounding and lock, storage eviction between incidents, two tablets at
once.

### What changes

- **It all works**: ship the PWA as planned; document the CA install as a
  one-time per-tablet step.
- **Service worker refused, everything else fine**: use the tab. Drop the
  offline-shell claim from the README. No architectural change.
- **WebRTC drops when backgrounded** (plausible on iOS): the tablet must
  detect it and the staleness rule must fire -- a frozen last frame under
  live-looking boxes is the failure mode that must never happen. Wake lock
  and an explicit "reconnecting" state become mandatory rather than nice.
- **The whole thing is unworkable on iOS**: standardise on Android tablets.
  Cheaper hardware, better browser behaviour for this use case.

---

## 3. Is timestamp matching enough to keep the overlay on the right frame?

### Why it matters

Video and detections travel as separate streams, so a box can land on a frame
it was not computed from. On a moving aerial scene at 15 m/s, 300 ms of drift
puts the box about 4.5 m off -- pointing a crew at the wrong place while
looking completely authoritative. A confidently misplaced box is worse than no
box, because it converts a hint into misdirection.

`docs/CONTRACT.md` specifies three tiers: exact RTP-timestamp matching (tier
1), a median pts-offset estimate (tier 2), and unsynchronised-but-labelled
(tier 3), with an absolute staleness rule on top. **What is not known is how
much residual drift tier 2 actually has in practice**, and therefore whether
tier 1 is worth the constraints it imposes on the streaming path.

### How to answer

It is directly measurable, and cheaply:

1. `tools/make_test_video.py` generates a clip whose target position is known
   exactly, frame by frame -- the sidecar is the geometry the renderer drew
   from, not an annotation somebody made.
2. Run it through the real pipeline to a real tablet.
3. Screen-record the tablet, and compare drawn box positions against the
   sidecar. Report the distribution of the error in **milliseconds of drift**
   and in **fraction of frame width**, not just a mean.
4. Repeat under load: WiFi congestion, a second tablet, and a deliberately
   stalled pipeline to confirm the staleness rule fires when it should.

Do it in both configurations: aiortc re-encode (tier 1 available) and MediaMTX
relay (tier 2 only).

### What changes

- **Tier 2 drift is small** (say under 100 ms, i.e. ~1.5 m at 15 m/s): relay
  the stream instead of re-encoding. Lower latency, no quality loss, simpler.
  Tier 1 becomes an optimisation nobody needs.
- **Tier 2 drift is large or heavy-tailed**: the aiortc re-encode path becomes
  the deployment path, because `rtp_ts` is worth the milliseconds it costs.
- **Even tier 1 drifts**: the assumption that a constant offset exists is
  wrong -- look for variable-latency queueing in the pipeline before blaming
  the algorithm.
- **Any tier is unreliable**: shrink `stream.max_overlay_age_s` so the overlay
  withdraws sooner, and make the tier indicator more prominent. Degrading to
  plain video is always available and always safe.

---

## 4. What frame rate does this actually need?

**Currently `inference.max_fps: 10`. The question is whether even that is
generous, and 5 fps would do.**

### Why it matters

This is the highest-leverage unanswered question in the project, because the
frame rate is upstream of nearly every other constraint:

- **GPU**: 5 fps instead of 30 is a sixth of the inference load. It is the
  difference between needing a discrete GPU and possibly not, or between one
  feed and three on the same laptop.
- **Latency budget**: sampling wait is `1 / max_fps`. At 30 fps it is 33 ms;
  at 5 fps it is 200 ms. Both are irrelevant next to a fire's timescale, which
  is precisely the point.
- **Temporal filter**: N-of-M is measured in *inferred* frames. 3-of-5 at 30
  fps spans 0.17 s -- barely longer than the flicker it is meant to suppress.
  At 5 fps it spans 1 s of genuinely independent evidence, which is a far
  stronger filter. Lower frame rate makes the filter *better*, not worse.
- **Thermals, power, incident-log size, and how much hardware a department has
  to buy.**

The counter-argument, which deserves a fair hearing: fire spreads slowly but
the *camera* does not. A drone panning at speed can bring a plume into frame
and out again in under a second, and at 5 fps that is five chances, some of
them motion-blurred. The question is not really "how fast does fire move" but
"how fast does the *scene* move", and that is a question about how they fly.

### How to answer

1. **Ask the operators how they fly.** Orbit at altitude? Fast transects? A
   slow scan of a ridge line? Ten minutes of conversation, and it settles most
   of the argument.
2. **Measure it on their footage.** Take validation clips, run the pipeline at
   30, 15, 10 and 5 fps, and compare **event recall** and **time to first
   confirmed box** (`docs/VALIDATION.md` §4.4). The metric that matters is
   whether events are found at all, not how many frames they are found in.
3. Re-tune `n`/`m` at each rate -- comparing rates at a fixed 3-of-5 compares
   two things at once and tells you nothing.

### What changes

- **5 fps is enough**: halve the GPU requirement, or run a second feed.
  Lengthen the N-of-M window in wall-clock terms for a stronger filter.
  Consider whether a smaller station machine becomes viable.
- **10 fps is the sweet spot** (the current default): confirm it and record
  the measurement, so the number stops being an assumption.
- **Fast panning genuinely needs 20-30 fps**: the GPU requirement in
  `docs/HARDWARE.md` becomes firm, `n`/`m` need re-tuning, and it is worth
  asking whether the flying pattern can change instead -- slowing a scan is
  free, and a bigger GPU is not.

Whatever the answer, **write it down as a decision with its evidence**. The
failure mode here is not choosing wrong, it is defaulting to 30 because video
is 30 and never revisiting it.

---

## 5. Who validates this, and against what footage?

### Why it matters

Nothing else in this repository matters until this is answered. A model whose
false-negative behaviour has never been measured on the footage it will
actually see is a model with unknown recall, and every design decision in
`docs/SAFETY.md` exists because unknown recall is the normal state of an RGB
fire-detection model and the interface must be honest about it.

There is also a subtler reason. Validation is where the department stops being
a recipient and becomes a participant: the officers who watch the miss gallery
are the ones who will brief their own people on what the overlay will not
show. Nobody outside the service can do that for them.

### How to answer

`docs/VALIDATION.md` is the whole protocol. The two things to settle **now**,
because they gate everything else:

1. **Who signs.** A named officer of the fire service, the person who ran the
   validation, and the project supervisor. The officer's signature is the one
   that matters; the others support it.
2. **What footage.** The department's own: archive incident video, a
   prescribed or training burn, and at least 30 minutes of ordinary flying
   with nothing burning. Public datasets are for training and cannot serve
   here -- different camera, altitude, terrain and season.

**Start the footage conversation in week one of the project.** It is the only
part that depends on other people's calendars, and it is the usual reason a
validation never happens.

### What changes

- **The department engages and supplies footage**: run the protocol, measure
  per-condition recall with intervals, disclose the weak conditions, sign the
  page. This is the good path and it is what makes the system deployable.
- **Footage exists but nobody will sign**: the system stays a demonstrator.
  Say so in the README, on the screen, and to anybody who asks. That is an
  acceptable outcome for a student project and a dishonest one to hide.
- **No footage is available at all**: validate on public datasets, report the
  numbers as *not transferable*, and treat every deployment as unvalidated.
  The safety invariants are what make even that defensible -- but only just,
  and only because the overlay never makes a negative claim.
