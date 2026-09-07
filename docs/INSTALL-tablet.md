# Installing on the MK15 tablet

The inspection app installs as a home-screen app with an icon, no APK and no app store. It
is a Progressive Web App: the tablet's browser downloads it once, keeps it, and from then on
it opens in its own window with no address bar. To Android it looks and behaves like an
installed app.

**Why not an APK.** An APK has to be built, signed, sideloaded past Android's unknown-sources
warning, and rebuilt and re-sideloaded for every change. A PWA updates itself the next time
the tablet has a connection, and the whole delivery mechanism is a URL. The app is HTML,
JavaScript and models - there is nothing in it that needs native Android, so an APK would be
a wrapper around the same web page with more steps.

---

## Install it

1. Put the tablet on WiFi with internet. This is needed **once**, for the install.
2. Open **Chrome** and go to:

   ```
   https://abyyworld.github.io/Drone-visualisation-training/
   ```

3. Menu (three dots) -> **Add to Home screen** or **Install app**.
4. Confirm. The icon appears on the home screen.
5. Open it from the icon. It should fill the screen with no address bar. If you can still see
   the address bar, it was added as a bookmark rather than installed - remove it and repeat
   from step 3.

### If the MK15's built-in browser will not do it

The MK15 ships a stock Android browser that is old and often has no install option.

- Install Chrome from the Play Store if the unit has it, and use that.
- No Play Store: download the Chrome APK from `google.com/chrome` on the tablet itself and
  install it, allowing installs from unknown sources when prompted.
- Neither is possible: the site still works as a normal page in whatever browser exists.
  You lose the icon and the full-screen window; you do not lose any function.

### It must be HTTPS

An installable app and an offline cache both require a secure context. The GitHub Pages URL
above is HTTPS, so this is already satisfied. A copy served over plain `http://192.168.x.x`
for testing runs fine but cannot be installed, and that is a browser rule, not a bug.

---

## What works without a connection

| | Offline |
|---|---|
| Opening the app, uploading, browsing results, PDF report | Yes |
| **On-device engine** (ONNX in the browser) | Yes, once the models have downloaded once |
| **Anthropic / Gemini / OpenAI engines** | **No.** They are a network call by definition |
| Video frame extraction | Yes, it is all local decoding |

So: for a site with no signal, use the on-device engine, and open the app once on WiFi
beforehand so the models are cached. For an API engine, the tablet needs a connection at the
moment of analysis. There is no queue-and-send-later - a result that cannot be produced is
reported as an error rather than silently deferred.

**Warm the cache before going out.** On WiFi, open the app and analyse one image with the
on-device engine. That pulls the runtime and the model files into the cache. Skipping this
and arriving with no signal gives an app that opens and then cannot analyse anything.

---

## Updating

Open it on WiFi. The service worker fetches the new version in the background and it is live
next launch. Nothing to reinstall.

To force it: Android **Settings -> Apps -> Drone Inspection -> Storage -> Clear cache**, then
reopen on WiFi.

---

## Using it in the field

1. Fly the inspection and record video, or take stills.
2. Copy them to the tablet, or shoot on the tablet.
3. Open the app, pick the engine, drop the files in. A video is sampled into distinct,
   in-focus frames automatically - `Frames per video` sets how many.
4. Review the cards. **Print / save PDF report** produces the report.
5. **Export JSON** saves the machine-readable record. Keep it: with
   `tools/vlm_inspect.py` and `tools/vlm_to_yolo.py`, those records accumulate into the
   training set for a detector that will eventually replace the API.

### The API key on a shared tablet

The key is held in the tab and never written to disk, so closing the app forgets it. That is
deliberate, and it means it has to be pasted again each session. On a tablet several people
use, that is the right trade: a key saved on a shared device is a key everyone has.

For repeated unattended runs, do not use the tablet. Run `tools/vlm_inspect.py` on a laptop
with the key in an environment variable.

---

## Rebuilding the icon

`python3 tools/make_icons.py` redraws `web/icons/` from a script, so the mark is 25 lines of
code rather than a binary nobody can edit. Change the colours or the shape there and rerun.
