# Deployment

How this runs at an incident, on a network with **no internet**, and what has
to be tested before the tablet plan can be committed to.

Solve the no-internet problem early. It is not a packaging detail at the end of
the project -- it decides whether the tablet story works at all, and it is the
one part of the system that cannot be developed against a file source on a
desk with WiFi.

---

## 1. The problem in one paragraph

At an incident there is a laptop, a controller, some tablets and a WiFi
access point, and nothing else. Everything a browser normally gets from the
internet is unavailable: public DNS, a public certificate authority, STUN and
TURN servers, app stores, CDNs, NTP. Meanwhile browsers have spent a decade
making the features this system needs -- WebRTC, service workers, wake lock --
conditional on a **secure context**, which normally means a certificate from a
public CA, which normally means the internet. The whole of this document is
about closing that gap without weakening anything.

---

## 2. WebRTC on a LAN with no STUN and no TURN

The station and the tablets are on the same subnet, so ICE needs nothing
external. `stream.ice_servers` is empty by default and that is the correct
value:

```yaml
stream:
  ice_servers: []   # host candidates only
```

A STUN server exists to discover a public address behind NAT. Here there is no
NAT between the peers and no public address to discover, and a STUN server
that cannot be reached adds a gathering timeout -- several seconds -- to every
tablet that connects. Do not paste in a public STUN address to "be safe": on
this network it can only make connection slower and failure modes murkier.
TURN is a relay for peers that cannot reach each other directly; the same
argument applies twice over.

### The trap: mDNS-obfuscated ICE candidates

Chromium hides local IP addresses in ICE candidates behind randomised
`.local` hostnames (`a1b2c3d4-....local`) unless the page has been granted
camera or microphone permission. This is a privacy feature and it is on by
default. It works fine **provided multicast DNS resolution works on the WiFi**
-- and consumer access points frequently break it:

- **Client isolation / "AP isolation"** stops clients talking to each other at
  all. Turn it off.
- **Multicast filtering / IGMP snooping** can drop mDNS. Allow multicast.

Symptom: ICE gathers candidates, the connection stays in `checking`, and it
eventually fails with nothing obviously wrong in the logs. Test this on the
actual access point before an incident, not during one.

---

## 3. The secure-context requirement

Browsers gate the following behind a secure context (HTTPS, or `localhost`):

| Feature | Needed for | Behaviour on plain HTTP |
|---|---|---|
| `RTCPeerConnection` | video + the detections data channel | blocked in current Chrome and Safari |
| Service workers | installable PWA, offline app shell | `registration` rejected outright |
| Screen Wake Lock | keeping the tablet awake on a long incident | unavailable |

`localhost` is exempt, which is why development on the laptop itself works
over HTTP (`python -m station run --insecure`). Tablets are not localhost.
**The station therefore needs TLS even though it never touches the internet.**

---

## 4. Certificates: the two workable routes

### Route A -- self-signed, per-station (`python -m station certs`)

```bash
python -m station -c config.yaml certs        # issue, covering this machine's IPs
python -m station certs --show                # inspect what is installed
```

The certificate must carry the station's **IP addresses in the SAN**, not just
a hostname. iOS rejects a certificate that does not name the address that was
actually typed, and it does not fall back to the Common Name. `station/serve/certs.py`
enumerates the machine's addresses for exactly this reason; add more with
`--ip` and `--host`.

What happens on a tablet:

- **Browsing** to `https://<station-ip>:8443` shows an interstitial. Accepting
  it gets you the page, the video and the data channel. Core function works.
- **Service worker registration fails** on Chrome behind a certificate error,
  even after the interstitial is accepted. So on route A you get a working
  browser tab and, on Android at least, **no installable PWA**.

Route A is the right choice for development, for the first field trial, and as
the fallback if route B is more ceremony than the department will accept.

### Route B -- a private CA (what to do if the PWA install matters)

Generate one CA, keep the CA key on the laptop, and issue station certificates
from it. Install the **CA certificate** once per tablet; after that the
station's certificate is genuinely valid to that device, with no interstitial
and no service-worker refusal.

- **iOS/iPadOS**: transfer the CA `.crt`, install the resulting profile in
  *Settings > General > VPN & Device Management*, and then -- the step everyone
  misses -- switch it on in *Settings > General > About > Certificate Trust
  Settings*. Without that second step the certificate is installed and still
  not trusted.
- **Android**: *Settings > Security > Encryption & credentials > Install a
  certificate > CA certificate*. The device will warn that the network may be
  monitored; on a department-owned tablet on an isolated LAN that is an
  accurate description and an acceptable one. Note that Android's user CA
  store is not trusted by every app, but Chrome does honour it.

With a private CA you can also issue for an **mDNS name** such as
`https://station.local:8443`, which is friendlier than typing an IP:

- iOS and macOS resolve `.local` natively (Bonjour).
- Android's support is inconsistent across versions. **Treat the name as a
  convenience and always keep the IP in the SAN as well**, so there is a
  working URL when the name does not resolve.

### Route C -- a public certificate (usually not worth it)

You can obtain a real certificate for a domain you control using a DNS-01
challenge, point a record at the LAN address, and ship it to the station.
Nothing needs internet at *use* time -- but the tablets do need to resolve the
name, so you also need local DNS; the certificate expires every 90 days and
renewal needs internet at the base; and a certificate that quietly expires
mid-deployment fails in a way nobody on scene can fix. Mentioned for
completeness. Route A or B.

### Whichever route: pin the address

The certificate names IP addresses, so a station that changes network changes
address and the certificate stops matching. Reserve a fixed address for the
laptop on the access point (DHCP reservation or static), and reissue with
`certs --force` if the network really does change -- remembering that every
tablet then has to re-trust it.

---

## 5. UNVERIFIED: iOS, PWAs and self-signed certificates

**This has not been tested on a real iPad, and the tablet plan must not be
committed to until it has been.**

The specific uncertainty: whether iOS/iPadOS will add a site served over a
**self-signed certificate** to the home screen as a PWA, register and run its
**service worker**, and keep a **WebRTC** session alive from that home-screen
context. Safari's behaviour here has changed repeatedly across releases, is
poorly documented, and differs between a normal tab and a standalone
home-screen app. There is no substitute for trying it on the device.

**The fallback, if it does not work, is a plain browser tab**, and it is a
perfectly good fallback:

| | Home-screen PWA | Plain Safari tab |
|---|---|---|
| Live video | yes | yes |
| Detections data channel | yes | yes |
| Canvas overlay + synchronisation | yes | yes |
| Staleness handling | yes | yes |
| Full-screen without browser chrome | yes | mostly (add to home screen or hide chrome) |
| Offline app shell after first load | yes | no -- but the station serves it on the LAN anyway |
| Wake lock | yes, if supported | same, if supported |

Every safety-relevant behaviour survives the fallback. **Nothing in the
architecture depends on the PWA installing.** Build steps 1-3 in the README
proceed regardless, which is why this is an open question rather than a
blocker -- but it decides what the tablet plan promises, so answer it before
promising anything.

### iPad verification checklist

Do this once, on the actual iPad model the department will use, on the actual
access point, with the station on route A (self-signed). Half a day.

- [ ] **1. Plain tab.** Open `https://<station-ip>:8443`, accept the
      interstitial. Video plays; boxes appear over it; the overlay reports its
      synchronisation tier. *Pass: works. Fail: nothing else matters, fix this
      first.*
- [ ] **2. Home-screen install.** Share > Add to Home Screen. Launch from the
      icon. *Record: does it open standalone or bounce into Safari? Does it
      re-prompt about the certificate?*
- [ ] **3. Service worker.** From the home-screen app, check the worker
      registered (`app/sw.js`) and that a reload works with the station
      briefly stopped. *Record the exact error if registration is refused --
      this is the single most likely thing to fail.*
- [ ] **4. WebRTC from the home-screen app.** A **20-minute** continuous
      session. Watch for: connection surviving, overlay staying aligned,
      memory growth, thermal throttling. Twenty minutes, not two -- WebRTC
      failures on iOS are usually slow.
- [ ] **5. Screen Wake Lock.** Request it and leave the tablet untouched for
      15 minutes with auto-lock set to 2. *Pass: screen stays on. Fail: note
      it -- operators will otherwise be tapping the screen every two minutes,
      and that is a real operational cost worth knowing about.*
- [ ] **6. Backgrounding.** Switch apps for 30 s and come back. Lock the
      screen for 60 s and come back. *Record: does the video resume, does the
      peer connection recover by itself, or does it need a reload? A tablet
      that silently shows a frozen last frame after unlock is dangerous -- the
      staleness rule must fire and the boxes must stop.*
- [ ] **7. Storage eviction.** Use the app, leave it a day, use it again.
      iOS evicts storage for sites it considers unused after 7 days. *Record:
      does the app shell survive between incidents, or does it re-download
      from the station? Re-downloading is fine on a LAN -- but it must not be
      surprising.*
- [ ] **8. Two tablets at once**, on the same station, for the whole 20
      minutes. Confirms the one-peer-connection-per-tablet path and the
      station's CPU headroom.

Write the answers into `docs/OPEN_QUESTIONS.md` and delete question 2.

---

## 6. Field runbook

### On the bench, before going out

```bash
python -m station -c config.yaml check     # config + what this laptop can do
python -m station -c config.yaml certs     # issue the certificate
```

- Install the CA (route B) or accept the certificate (route A) on **every**
  tablet, and confirm each one loads the page.
- Label the tablets. Confirm the station's address is reserved on the AP.
- Set the laptop clock. There is no NTP on this network, and while the overlay
  aligns on media timestamps rather than wall clock -- so a wrong clock will
  not misplace a box -- every incident log record is stamped with
  `wall_time`, and a log whose times are an hour out is much harder to
  reconcile with a radio log afterwards.
- Confirm the disk has room: ~3.6 GB per hour of incident video.

### At the incident

1. Ethernet from the controller to the laptop; confirm the RTSP stream with
   `ffprobe` before starting the station.
2. Bring up the access point; confirm the laptop's address.
3. `python -m station -c config.yaml run`.
4. Read the banner: source, model name **and version**, incident log path.
5. Tablets connect. Confirm on each one that video is playing and that the
   status shows the pipeline running.
6. Brief anyone using a tablet, out loud, every time: *boxes appear when the
   model produces them; their absence means nothing at all; keep watching the
   video.*

### Firewall

- **TCP 8443** inbound for the PWA and signalling (`POST /webrtc/offer`).
- **UDP, ephemeral range** inbound for the ICE host candidates. This is what
  gets forgotten: the page loads, the video never starts.
- Windows will prompt on first run -- allow it on the **private** profile, and
  make sure the incident WiFi is classified private, not public.

### When something breaks

| Symptom | First thing to check |
|---|---|
| Page loads, video never starts | UDP blocked by the firewall, or AP client isolation (section 2) |
| Certificate warning that cannot be dismissed | Certificate does not name the IP you typed. Reissue with `--ip` |
| "Add to Home Screen" missing or service worker refused | Certificate error (route A). Expected -- use the tab, or move to route B |
| Video plays, no boxes ever | Weights not loaded, or the pipeline is `stalled`. Check the station banner and the status line. **Do not read an empty overlay as information** |
| Boxes visibly lag the picture | Overlay synchronisation tier -- the tablet displays which one is active. See `docs/CONTRACT.md` and open question 3 |
| Everything stops when the drone goes behind a ridge | Expected. `source.reconnect_s` retries; the pipeline reports `stalled` and the tablets stop drawing |

---

## 7. What is deliberately not deployed

- **No cloud.** No account, no upload, no telemetry. The incident log stays on
  the laptop; see `docs/SAFETY.md` on handling and retention.
- **No app store.** The tablets get a URL on the LAN. That is also why the
  no-internet problem had to be solved rather than avoided.
- **No inference on the tablets.** One model run per frame, on the station,
  for every tablet. Tablets receive pixels and JSON.
- **No automated alerting.** Invariant 3 in `docs/SAFETY.md`.
