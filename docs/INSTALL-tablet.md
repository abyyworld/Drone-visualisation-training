# Installing on the MK15 tablet

Two routes to the same app with an icon on the home screen. Both run identical code - the
APK is a WebView around the very same `web/` directory the site deploys, copied in at build
time, so there is no second implementation to keep in step.

| | Install from the browser | APK |
|---|---|---|
| Needs | Chrome, and internet once | A file, and unknown-sources enabled |
| Updates | Itself, next time it has signal | You sideload a new APK |
| Works on a locked-down tablet | Only if the browser supports installing | Yes |
| Works with no connection ever | No: the first load is a download | Yes: everything is inside the file |
| Where the models live | Downloaded and cached by the browser | Inside the APK |

**Take the APK if** the MK15 has no Play Store, its browser has no install option, the
tablet may never see WiFi, or you are setting up several units from one file.

**Take the browser route if** the tablet has a current Chrome and internet now and then. It
is less to maintain, because it updates itself.

---

## Route A: the APK

### Get it

Every push to `main` builds one. Repository -> **Actions** -> **Build Android APK** -> the
most recent run -> **Artifacts** -> `drone-inspection-apk`. It is a zip; the `.apk` is
inside.

### Install it

1. Copy the `.apk` onto the tablet (USB, or a memory card).
2. Open it with the tablet's file manager.
3. Android asks to allow installs from that source. Allow it - this is Android's standard
   warning for any app that did not come from a store, not a sign that anything is wrong.
4. The icon appears in the app drawer.

### What is in it

The web app, the ONNX runtime, and the icon. **Not** any detection model, because none is
trained yet - the APK ships with the API engines working and the on-device engine reporting
that it has no model, which is the truthful state. When a model lands in `web/models/`, the
next APK contains it and the on-device engine starts working offline.

### It needs a current Android System WebView

The app is a web page, so it runs in the tablet's WebView, and it uses ES modules, dynamic
import and `createImageBitmap`. An MK15 whose WebView has never been updated may be too old
for those, and the app will open to a blank screen or an error rather than a broken half.

That component updates through the Play Store separately from the Android version, so an
old tablet with a fresh WebView is fine. If the app opens blank: update **Android System
WebView** and **Chrome** from the Play Store, then reopen it.

### Signing, and upgrading in place

With no signing key configured, CI builds a **debug** APK. It installs and runs normally.
The catch is upgrades: Android only replaces an app with one signed by the same key, so a
debug APK from a later build may have to be uninstalled first (which loses nothing - the app
stores no data).

To get in-place upgrades, create a key once and give it to the repository:

```bash
keytool -genkeypair -v -keystore release.jks -keyalg RSA -keysize 4096 \
        -validity 10000 -alias drone-inspection

base64 -w0 release.jks    # paste the output as the secret below
```

Repository -> Settings -> Secrets and variables -> Actions, add four secrets:
`ANDROID_KEYSTORE_BASE64`, `ANDROID_KEYSTORE_PASSWORD`, `ANDROID_KEY_ALIAS`,
`ANDROID_KEY_PASSWORD`. CI then builds a signed release APK instead.

**Keep `release.jks` and back it up somewhere safe.** Lose it and every future build is a
different app to Android, and every tablet has to uninstall before it can update. It never
goes in the repository - `*.jks` and `*.keystore` are gitignored for that reason.

### Building it yourself

```bash
npm install --no-save onnxruntime-web@1.23.0   # optional; bundles the runtime
cd android && ./gradlew assembleDebug
# android/app/build/outputs/apk/debug/app-debug.apk
```

Needs a JDK 17 or newer and the Android SDK. The Gradle wrapper fetches everything else.

---

## Route B: install from the browser

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

- **Use the APK instead.** This is exactly the case it exists for.
- Or install Chrome from the Play Store, if the unit has one, and use that.
- Or open the site in whatever browser exists. You lose the icon and the full-screen
  window; you do not lose any function.

### It must be HTTPS

An installable app and an offline cache both require a secure context. The GitHub Pages URL
above is HTTPS, so this is already satisfied. A copy served over plain `http://192.168.x.x`
for testing runs fine but cannot be installed, and that is a browser rule, not a bug.

---

## What works without a connection

Applies to both routes, except where the table above says otherwise.

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

**Browser install:** open it on WiFi. The service worker fetches the new version in the
background and it is live next launch. Nothing to reinstall. To force it: Android
**Settings -> Apps -> Drone Inspection -> Storage -> Clear cache**, then reopen on WiFi.

**APK:** download the newer one from Actions and install it over the top. If Android refuses
because the signature differs, uninstall first - see Signing above for how to stop that
happening again.

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

`python3 tools/make_icons.py` redraws both the web icons and the Android launcher icons from
one script, so the browser tab, the home-screen shortcut and the APK are the same mark
rather than three slightly different ones. Change the colours or the shape there and rerun.
