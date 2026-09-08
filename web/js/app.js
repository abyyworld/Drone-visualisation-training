/**
 * Application orchestration: uploads -> gate -> detector -> severity -> report.
 *
 * Images are processed one at a time rather than in parallel. Two 960px models competing for
 * one WebGPU device is slower than running them in sequence and makes progress reporting
 * meaningless, and on the WASM fallback it will exhaust memory on a large batch.
 */

import { probeBackend, activeBackend, configureRuntime } from './runtime.js';
import { loadImage } from './preprocess.js';
import { classify, rejectionMessage, VERDICT } from './gate.js';
import { detect } from './detect.js';
import { assess, summarise, severityLabel } from './severity.js';
import { drawDetections, toBlob, colorFor } from './render.js';
import { PROVIDERS, ENGINE_LOCAL, inspect, listModels } from './vlm.js';
import { extractFrames, VIDEO_DEFAULTS } from './video.js';
import { classify as classifyFile, ACCEPT_ATTRIBUTE } from './formats.js';
import { zip } from './zip.js';
import { detectOnDevice, configureOnDevice, handles as onDeviceHandles } from './ondevice.js';
import { FireScan } from './firescan.js';
import { Tracker } from './track.js';
import { LiveView } from './live.js';
import { configureHeic } from './heic.js';

const MODELS_BASE = 'models/';
const MAX_FILES = 100;

/**
 * The subjects, in the order they are offered.
 *
 * Written once. Adding wildfire meant editing this same list in six places, and the gate's
 * rejection message was missed - so it told people to upload a turbine or a solar panel
 * for a while after it had learned two more subjects.
 */
const DOMAINS = ['turbine', 'solar', 'crowd', 'wildfire'];

const state = {
  manifest: null,
  available: {},
  results: [],
  busy: false,
  // The engine the next batch will run on. `apiKey` lives here and nowhere else - not in
  // localStorage, not in the URL, not in an exported file - so closing the tab discards it.
  engine: { provider: 'ondevice', model: null, apiKey: '' },
  // Set once, the first time we find there is no on-device model. Without it, choosing
  // "On-device model" from the picker would bounce straight back to a provider, which is
  // the page overruling a deliberate choice rather than helping with an unmade one.
  steeredToApi: false,
  // One tracker per batch of video frames. Photographs are unrelated to each other, so it
  // is reset before every batch and only consulted for frames.
  tracker: new Tracker(),
  fire: new FireScan(),
};

const el = {};

// ---------------------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------------------

async function init() {
  for (const id of [
    'drop', 'file-input', 'results', 'summary', 'status-banner', 'backend',
    'domain-override', 'override-row', 'export-json', 'export-images', 'print-report',
    'empty-state', 'progress', 'progress-bar', 'progress-label', 'clear', 'export-training',
    'drop-blocked', 'engine-key-link',
    'live-start', 'live-stop', 'live-record', 'live-camera', 'live-stage', 'live-video',
    'live-overlay', 'live-status', 'live-count', 'live-trails', 'live-count-readout',
    'live-fire', 'live-subject',
    'engine-provider', 'engine-model', 'engine-model-field', 'engine-model-hint',
    'engine-refresh', 'engine-key', 'engine-key-field', 'engine-key-label',
    'engine-key-hint', 'engine-key-toggle', 'engine-warning', 'privacy-pill',
    'video-frames',
  ]) {
    el[id] = document.getElementById(id);
  }

  el.backend.textContent = (await probeBackend()) === 'webgpu' ? 'WebGPU' : 'WASM (CPU)';

  // Set from the format table rather than written into the HTML, so the picker and the
  // classifier can never disagree about what is offered. A picker that greys out a HEIC
  // is the same bug one step earlier.
  el['file-input'].setAttribute('accept', ACCEPT_ATTRIBUTE);

  await loadManifest();
  wireEngine();
  wireEvents();
  wireLive();
  registerServiceWorker();
}

/**
 * Register the service worker, so the page can be installed to a tablet home screen.
 *
 * Deliberately not awaited and deliberately silent on failure. Registration needs a secure
 * context, which GitHub Pages provides and a plain-HTTP local server does not, and a page
 * opened over http://192.168.x.x for testing must still work in full - it simply is not
 * installable. Nothing in the analysis pipeline depends on this succeeding.
 */
function registerServiceWorker() {
  if (!('serviceWorker' in navigator)) return;
  navigator.serviceWorker.register('sw.js').catch(() => {});
}

async function loadManifest() {
  try {
    const response = await fetch(`${MODELS_BASE}manifest.json`);
    if (!response.ok) throw new Error(`${response.status}`);
    state.manifest = await response.json();
    configureRuntime(state.manifest.runtime ?? {});
    configureOnDevice(state.manifest.runtime ?? {}, state.manifest.ondevice ?? {});
    configureHeic(state.manifest.runtime ?? {});
  } catch (error) {
    showBanner(
      'error',
      'Could not load models/manifest.json. The site is deployed but not configured.',
    );
    return;
  }

  // A manifest entry is a promise, not a fact - check each file is actually there. A HEAD
  // request is enough and costs nothing next to downloading the weights.
  const keys = ['gate', ...DOMAINS];
  await Promise.all(
    keys.map(async (key) => {
      const spec = state.manifest[key];
      if (!spec?.file) return;
      try {
        const response = await fetch(`${MODELS_BASE}${spec.file}`, { method: 'HEAD' });
        state.available[key] = response.ok;
      } catch {
        state.available[key] = false;
      }
    }),
  );

  reportModelStatus();
}

function reportModelStatus() {
  const detectors = DOMAINS.filter((k) => state.available[k]);
  const missing = ['gate', ...DOMAINS].filter((k) => !state.available[k]);

  if (!detectors.length) {
    // No local model, but the API engines need none, so this is a setup step rather than a
    // dead end. Select one for them: the alternative is a page that looks ready, accepts a
    // file and does nothing, which is what it used to do.
    showBanner(
      'warning',
      'No trained defect model is deployed yet. People and vehicles are found on this '
      + 'device with no key; for damage, fire or smoke choose a provider under Analysis '
      + 'engine and enter a key.',
    );
    return;
  }

  // Without the gate we cannot refuse an out-of-domain image, so the user has to say what
  // they uploaded. Silently guessing would be the exact failure the gate exists to prevent.
  if (!state.available.gate) {
    el['override-row'].classList.remove('hidden');
    showBanner(
      'warning',
      'The domain gate model is not deployed, so invalid images cannot be rejected '
      + 'automatically. Select the inspection type manually below.',
    );
  } else if (missing.length) {
    showBanner('warning', `Not yet deployed: ${missing.join(', ')}.`);
  }

  // Only offer domains that actually have a model behind them. This applies to the
  // on-device engine alone: a provider decides the subject from the picture, so every
  // option is re-enabled when one is selected.
  for (const option of el['domain-override'].options) {
    if (option.value !== 'auto' && !state.available[option.value]) option.disabled = true;
  }
  if (detectors.length === 1) el['domain-override'].value = detectors[0];
}



// ---------------------------------------------------------------------------------------
// Live tracking
// ---------------------------------------------------------------------------------------

function wireLive() {
  const view = new LiveView(el['live-video'], el['live-overlay'], {
    onStatus: (message, stats) => renderLiveStatus(message, stats),
  });
  state.live = view;

  el['live-start'].addEventListener('click', async () => {
    el['live-start'].disabled = true;
    try {
      await view.start(el['live-camera'].value || undefined);
      el['live-stage'].hidden = false;
      el['live-stop'].hidden = false;
      el['live-record'].hidden = false;
      el['live-start'].hidden = true;

      // Labels only exist once permission has been granted, so the picker is filled after
      // the camera opens rather than before, when every entry would read "Camera 2".
      const cameras = await LiveView.cameras();
      if (cameras.length > 1) {
        el['live-camera'].innerHTML = '';
        for (const camera of cameras) {
          const option = document.createElement('option');
          option.value = camera.id;
          option.textContent = camera.label;
          el['live-camera'].appendChild(option);
        }
        el['live-camera'].hidden = false;
      }
    } catch (error) {
      renderLiveStatus(error.message);
    } finally {
      el['live-start'].disabled = false;
    }
  });

  el['live-stop'].addEventListener('click', () => {
    const recording = view.stopRecording();
    view.stop();
    if (recording?.size) saveLiveRecording(recording);
    el['live-stage'].hidden = true;
    el['live-stop'].hidden = true;
    el['live-record'].hidden = true;
    el['live-camera'].hidden = true;
    el['live-start'].hidden = false;
    el['live-record'].classList.remove('button--recording');
    el['live-record'].textContent = 'Record';
    renderLiveStatus('Camera stopped.');
  });

  el['live-record'].addEventListener('click', () => {
    if (el['live-record'].classList.contains('button--recording')) {
      const recording = view.stopRecording();
      el['live-record'].classList.remove('button--recording');
      el['live-record'].textContent = 'Record';
      if (recording?.size) saveLiveRecording(recording);
      return;
    }
    try {
      view.startRecording();
      el['live-record'].classList.add('button--recording');
      el['live-record'].textContent = 'Stop recording';
    } catch (error) {
      renderLiveStatus(error.message);
    }
  });

  // A camera left running in a hidden tab keeps the light on and drains a tablet to draw
  // boxes nobody can see.
  document.addEventListener('visibilitychange', () => {
    if (document.hidden && !el['live-stop'].hidden && !view.recorder) {
      el['live-stop'].click();
    }
  });

  view.countPeople = el['live-count'].checked;
  el['live-count-readout'].hidden = !view.countPeople;

  el['live-count'].addEventListener('change', () => {
    view.countPeople = el['live-count'].checked;
    el['live-count-readout'].hidden = !view.countPeople;
  });

  el['live-trails'].addEventListener('change', () => {
    view.showTrails = el['live-trails'].checked;
  });

  // The subject decides which engines run. Applied immediately, including mid-stream: an
  // operator who has just realised the flame scan is marking a wall should be able to
  // switch it off without stopping the camera.
  view.setSubject(el['live-subject'].value);
  el['live-subject'].addEventListener('change', () => {
    view.setSubject(el['live-subject'].value);
    el['live-fire'].hidden = true;
  });

  el['live-camera'].addEventListener('change', async () => {
    if (el['live-stop'].hidden) return;
    view.stop();
    await view.start(el['live-camera'].value);
  });
}

function saveLiveRecording(blob) {
  const stamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
  download(blob, `live-${stamp}.webm`);
}

function renderLiveStatus(message, stats) {
  if (message) {
    el['live-status'].textContent = message;
    return;
  }
  if (!stats) return;

  const parts = [`${stats.fps.toFixed(1)} detections per second`];
  if (stats.inferenceMs) parts.push(`${Math.round(stats.inferenceMs)} ms each`);
  parts.push(stats.onScreen.length ? `on screen: ${stats.onScreen.join(', ')}` : 'nothing on screen');
  if (stats.recording) parts.push('RECORDING');
  el['live-status'].textContent = parts.join('  ·  ');

  renderFireBanner(stats);

  if (!el['live-count-readout'].hidden) renderPeopleCount(stats);
}

/**
 * The flame and smoke line.
 *
 * Present only while there is something to say. It counts regions, not fires: one fire seen
 * as two regions is two boxes and one fire, and the scanner has no way to tell those apart,
 * so it does not claim to. The wording is "candidate" throughout, because that is what a
 * colour and motion rule can honestly produce - the operator, or a provider model pointed
 * at the frame, decides what it is.
 */
function renderFireBanner(stats) {
  const banner = el['live-fire'];
  if (!banner) return;

  const said = [];
  if (stats.flame) said.push(`${stats.flame} flame region${stats.flame === 1 ? '' : 's'}`);
  if (stats.smoke) said.push(`${stats.smoke} smoke region${stats.smoke === 1 ? '' : 's'}`);

  banner.hidden = said.length === 0;
  if (!said.length) return;
  banner.textContent = `${said.join(' and ')} marked. Candidates from colour and motion. Look before acting.`;
}

/**
 * The people readout.
 *
 * Two numbers, because they answer different questions and are constantly confused. "In
 * view" is how many are on screen at this instant. "Seen so far" is how many distinct
 * people have appeared since the camera opened, counted once each by their track rather
 * than once per frame, so it keeps climbing as people walk through and never goes down.
 *
 * The caveat under them is not boilerplate. This counts what the detector found, which is
 * fewer than the number of people present whenever anyone is small, distant, behind
 * something or in a group - and that gap grows with altitude. Presented without saying so,
 * the number would be read as a measurement of the crowd, which it is not.
 */
function renderPeopleCount(stats) {
  el['live-count-readout'].innerHTML = '';

  const line = document.createElement('span');
  line.append(inView(stats.people), ' in view  ·  ');
  line.append(inView(stats.peopleTotal), ' seen so far');
  el['live-count-readout'].appendChild(line);

  const caveat = document.createElement('span');
  caveat.className = 'live__caveat';
  caveat.textContent =
    'People the detector found, which is fewer than the people there. Anyone small, '
    + 'distant, overlapping someone else or turned away is missed, and more of them are '
    + 'missed the higher the camera is. Treat it as a floor, not a measurement.';
  el['live-count-readout'].appendChild(caveat);
}

function inView(value) {
  const strong = document.createElement('strong');
  strong.textContent = String(value);
  return strong;
}

// ---------------------------------------------------------------------------------------
// Engine selection
// ---------------------------------------------------------------------------------------

const ENGINE_ONDEVICE = 'ondevice';

function usingApi() {
  return state.engine.provider !== ENGINE_LOCAL && state.engine.provider !== ENGINE_ONDEVICE;
}

function usingOnDevice() {
  return state.engine.provider === ENGINE_ONDEVICE;
}

/**
 * Why an upload cannot be analysed right now, or null when it can.
 *
 * One function, consulted both when files arrive and whenever the engine changes, so the
 * drop zone can never look ready while the pipeline behind it has nothing to run. The bug
 * this replaces: a warning banner above the fold, easy to scroll past, and a drop zone that
 * looked like a working uploader and silently produced nothing.
 */
function blockedReason() {
  // The on-device detector needs no key and no deployed .onnx. It is the only engine that
  // is ready the moment the page loads, which is why it is the default.
  if (usingOnDevice()) return null;

  if (usingApi()) {
    if (!state.engine.apiKey) {
      const provider = PROVIDERS[state.engine.provider];
      return `Enter your ${provider.keyLabel} above before uploading. `
        + 'Nothing can be analysed without it.';
    }
    if (!state.engine.model) return 'Choose a model above before uploading.';
    return null;
  }
  if (!DOMAINS.some((k) => state.available[k])) {
    return 'The on-device engine has no model to run yet. Choose a provider under '
      + 'Analysis engine above and enter an API key.';
  }
  return null;
}

/** Show or clear that reason on the drop zone itself, where the files are going. */
function renderDropState() {
  const reason = blockedReason();
  el['drop-blocked'].textContent = reason ?? '';
  el['drop-blocked'].hidden = !reason;
  el.drop.classList.toggle('drop--blocked', Boolean(reason));
}

function wireEngine() {
  el['engine-provider'].addEventListener('change', () => {
    state.engine.provider = el['engine-provider'].value;
    renderEngine();
  });

  el['engine-model'].addEventListener('change', () => {
    state.engine.model = el['engine-model'].value;
    renderDropState();
  });

  el['engine-key'].addEventListener('input', () => {
    // Keys are routinely pasted with a trailing newline or a stray space out of a password
    // manager, and every provider then returns a flat 401 that reads like a wrong key.
    state.engine.apiKey = el['engine-key'].value.trim();
    el['engine-refresh'].disabled = !state.engine.apiKey;
    renderDropState();
  });

  el['engine-key-toggle'].addEventListener('click', () => {
    const shown = el['engine-key'].type === 'text';
    el['engine-key'].type = shown ? 'password' : 'text';
    el['engine-key-toggle'].textContent = shown ? 'Show' : 'Hide';
    el['engine-key-toggle'].setAttribute('aria-pressed', String(!shown));
  });

  el['engine-refresh'].addEventListener('click', refreshModels);

  renderEngine();
}

/** Redraw the engine controls for the currently selected provider. */
function renderEngine() {
  const provider = PROVIDERS[state.engine.provider];
  const onDevice = usingOnDevice();

  // Reset it here rather than in each branch. Every engine has its own answer to whether
  // the operator picks the subject, and the row was previously only ever shown - so once
  // any engine had revealed it, switching to one that decides for itself still left it on
  // screen offering a choice that no longer did anything.
  el['override-row'].classList.add('hidden');

  el['engine-model-field'].hidden = !provider;
  el['engine-key-field'].hidden = !provider;
  el['engine-warning'].hidden = !provider;

  if (el['privacy-pill']) {
    el['privacy-pill'].textContent = provider ? `Sent to ${provider.label}` : 'Runs in your browser';
    el['privacy-pill'].classList.toggle('pill--privacy', !provider);
    el['privacy-pill'].classList.toggle('pill--offsite', Boolean(provider));
    el['privacy-pill'].title = provider
      ? `Each image is uploaded to ${provider.label} for analysis.`
      : 'Inference runs locally via ONNX Runtime Web. No image is uploaded to any server.';
  }

  el.backend.textContent = provider
    ? 'Provider API'
    : (activeBackend() === 'webgpu' ? 'WebGPU' : 'WASM (CPU)');

  if (!provider) {
    state.engine.model = null;
    if (onDevice) {
      // The detector ships with the app, so there is never a missing model to report. It
      // has no gate either - it finds people, not defects, and cannot mistake a cat for a
      // turbine - so the subject is the operator's to choose.
      el['status-banner'].classList.add('hidden');
      el.drop.removeAttribute('aria-disabled');
      el['override-row'].classList.remove('hidden');
    } else {
      // The trained-detector engine decides both of those from what is deployed.
      reportModelStatus();
    }
    renderDropState();
    return;
  }

  el['engine-key-label'].textContent = provider.keyLabel;
  el['engine-key-hint'].textContent = provider.keyHint;
  el['engine-key-link'].href = provider.keyUrl;
  el['engine-key-link'].textContent = `Get a ${provider.label} key`;
  el['engine-key'].value = '';
  state.engine.apiKey = '';
  el['engine-refresh'].disabled = true;

  fillModelOptions(provider.models);
  el['engine-model-hint'].textContent =
    'Built-in list. Enter a key and press Refresh to load the models your account can reach.';

  // An API engine needs no .onnx, so a missing model file is no longer a reason to refuse
  // uploads. Clear the warning the local path may have raised.
  el['status-banner'].classList.add('hidden');
  el.drop.removeAttribute('aria-disabled');
  el['override-row'].classList.remove('hidden');

  // A provider is not limited to the subjects a local model was trained for.
  for (const option of el['domain-override'].options) option.disabled = false;

  renderDropState();
}

function fillModelOptions(models) {
  el['engine-model'].innerHTML = '';
  for (const model of models) {
    const option = document.createElement('option');
    option.value = model.id;
    option.textContent = model.label;
    el['engine-model'].appendChild(option);
  }
  state.engine.model = models[0]?.id ?? null;
  el['engine-model'].value = state.engine.model ?? '';
}

/**
 * Replace the built-in model list with what the key can actually reach.
 *
 * Model IDs are retired and released faster than a hardcoded list in this file can track,
 * and a stale ID surfaces as a 404 that reads like a bug in the app. Asking the provider
 * is both current and a free check that the key works before a batch is started.
 */
async function refreshModels() {
  const provider = PROVIDERS[state.engine.provider];
  if (!provider || !state.engine.apiKey) return;

  const previous = state.engine.model;
  el['engine-refresh'].disabled = true;
  el['engine-model-hint'].textContent = 'Asking the provider which models this key can reach...';

  try {
    const models = await listModels(state.engine.provider, state.engine.apiKey);
    fillModelOptions(models);
    if (models.some((m) => m.id === previous)) {
      state.engine.model = previous;
      el['engine-model'].value = previous;
    }
    el['engine-model-hint'].textContent =
      `${models.length} model${models.length === 1 ? '' : 's'} available to this key. `
      + 'Vision support varies - if one refuses the image, pick another.';
  } catch (error) {
    fillModelOptions(provider.models);
    el['engine-model-hint'].textContent = `Could not load the list: ${error.message}`;
  } finally {
    el['engine-refresh'].disabled = false;
  }
}



/**
 * How many people the on-device detector finds, whatever engine is doing the main analysis.
 *
 * WHY THIS RUNS EVERY TIME
 *     People matter in all four subjects, not just the crowd one. Someone at the base of a
 *     turbine, on a solar array, or anywhere near a fire is the most important thing in the
 *     frame, and which engine happens to be selected for defects has nothing to do with it.
 *
 *     So the count always comes from the same place - the detector that ships with the app -
 *     rather than from whichever engine is running. That is what makes it comparable between
 *     a batch analysed by a provider and one analysed on the device: same model, same
 *     threshold, same meaning. A provider's own mention of people is prose and varies with
 *     the wording of the reply; this is a number that means one thing.
 *
 * IT IS A FLOOR
 *     It counts what the detector found. Anyone small, distant, overlapping someone else or
 *     turned away is missed, and more of them are missed the higher the camera was. Every
 *     place this number is shown says so.
 */
async function countPeople(image, existing) {
  // The on-device engine has already looked; counting its own findings again would be a
  // second inference for an answer already on the table.
  if (existing) return existing.filter((d) => d.label === 'person').length;

  try {
    const found = await detectOnDevice(image);
    return found.filter((d) => d.label === 'person').length;
  } catch {
    // A failed count must never fail the analysis. null renders as "not counted" rather
    // than as zero, because those are very different claims.
    return null;
  }
}

/**
 * Look for flame and smoke in one image.
 *
 * Sequential frames share one scanner, so a clip gets the time evidence the method depends
 * on; a photograph on its own does not, and comes back with a capped confidence saying so.
 * Never allowed to throw: a failure here loses the fire regions, and losing the people and
 * vehicles as well because of it would be the worse outcome.
 */
function scanForFire(image, sequential) {
  if (!sequential) state.fire.reset();
  try {
    return state.fire.scan(image, image.width, image.height).map((region) => ({
      label: region.label,
      classId: region.classId,
      confidence: Number(region.confidence.toFixed(3)),
      box: region.box.map((v) => Math.round(v)),
      note: fireNote(region),
    }));
  } catch {
    return [];
  }
}

/** What the scanner actually saw, in a sentence, so the box can be argued with. */
function fireNote(region) {
  if (!region.temporal) {
    return `${region.label} colour in a single frame. With one frame there is no motion to `
      + 'measure, which is the evidence that separates fire from anything else this colour, '
      + 'so this is a region to look at rather than a reading. Confirm it before acting.';
  }
  if (region.label === 'smoke') {
    return 'A desaturated region drifting across the detail behind it, which it is veiling '
      + `(${Math.round(region.evidence.edgeDrop * 100)}% of the texture that was there). `
      + 'Confirm it before acting.';
  }
  return 'Flame colour that changes on '
    + `${Math.round(region.evidence.flicker * 100)}% of frames rather than holding still, `
    + 'which is what separates it from something merely this colour. Confirm it before acting.';
}

/**
 * Analyse one image with the detector that ships with the app.
 *
 * No gate is consulted. The gate exists to stop a defect detector emitting confident boxes
 * on an image it has no business seeing, and this detector has no such failure mode: shown
 * a cat it finds a cat, which is not a person and is therefore reported as nothing. The
 * subject is whatever the operator selected, because a person is a person at a fire and in
 * a crowd alike.
 */
async function analyseOnDevice(image, base) {
  let detections = await detectOnDevice(image, (label) => setProgress(50, label));

  if (base.track) {
    // Boxes become tracks: each keeps a number across frames, survives a frame the model
    // missed, and carries how many frames it has been seen for. The label shown is the
    // identity, because "person 4" through a clip says something "person" cannot.
    detections = state.tracker.update(detections).map((t) => ({
      label: t.label,
      classId: t.classId,
      confidence: t.confidence,
      box: t.box.map((v) => Math.round(v)),
      trackId: t.id,
      seenFrames: t.seen,
      coasted: t.missed > 0,
      note: t.missed > 0
        ? `#${t.id}, predicted - not seen for ${t.missed} frame${t.missed === 1 ? '' : 's'}`
        : `#${t.id}, seen in ${t.seen} frame${t.seen === 1 ? '' : 's'}`,
    }));
  }

  // Flame and smoke, computed rather than detected - see firescan.js. Appended after the
  // tracker rather than through it: a fire has no identity to follow. It is not one thing
  // moving through the shot, it is a region that grows, splits and dies, and giving it a
  // number would say something about it that is not true.
  for (const region of scanForFire(image, base.track)) detections.push(region);

  const requested = el['domain-override'].value;
  const domain = onDeviceHandles(requested) && requested !== 'auto' ? requested : 'crowd';
  const spec = state.manifest?.ondevice ?? {};
  const { score, severity } = assess(detections, spec.severityWeights ?? {});

  return {
    ...base, image, status: 'analysed',
    domain,
    displayName: spec.displayName ?? 'People and vehicles',
    notes: spec.notes,
    zeroDetectionNote: spec.zeroDetectionNote,
    gate: null,
    engine: { provider: ENGINE_ONDEVICE, model: spec.file ?? 'detector.tflite' },
    detections,
    unlocated: [],
    people: await countPeople(image, detections),
    score, severity,
  };
}

/**
 * Run one image through the selected vision model.
 *
 * The API engine does its own domain check - it is asked what the asset is and answers
 * "neither" for anything that is not a turbine or a solar array - so the ONNX gate is not
 * consulted here. Two gates disagreeing would be worse than one.
 */
async function analyseWithApi(image, base) {
  const requested = el['domain-override'].value;
  const outcome = await inspect({
    provider: state.engine.provider,
    model: state.engine.model,
    apiKey: state.engine.apiKey,
    image,
    domain: requested,
  });

  if (outcome.asset === 'neither') {
    return {
      ...base, image, status: 'rejected',
      engine: { provider: outcome.provider, model: outcome.model },
      message: outcome.assetReason
        || 'This does not look like a wind turbine or a solar installation.',
    };
  }
  if (requested !== 'auto' && outcome.asset !== requested) {
    return {
      ...base, image, status: 'rejected',
      engine: { provider: outcome.provider, model: outcome.model },
      message: `You selected ${requested}, but this image looks like a ${outcome.asset}. `
        + 'Switch the inspection type, or set it to detect automatically.',
    };
  }

  const spec = state.manifest?.[outcome.asset] ?? {};
  const { score, severity } = assess(outcome.detections, spec.severityWeights ?? {});

  return {
    ...base, image, status: 'analysed',
    domain: outcome.asset,
    displayName: spec.displayName ?? outcome.asset,
    // The model's own prose about the asset is more informative than the manifest's fixed
    // note, so it replaces it. The solar RGB caveat is kept because it is a limit of the
    // photograph, not of the model, and no model can talk its way past it.
    notes: outcome.overall || spec.notes,
    zeroDetectionNote: outcome.asset === 'solar'
      ? (outcome.overall ? `${outcome.overall} ${spec.zeroDetectionNote ?? ''}`.trim() : spec.zeroDetectionNote)
      : (outcome.overall || spec.zeroDetectionNote),
    gate: null,
    engine: { provider: outcome.provider, model: outcome.model },
    detections: outcome.detections,
    unlocated: outcome.unlocated,
    people: await countPeople(image),
    score, severity,
  };
}

// ---------------------------------------------------------------------------------------
// Events
// ---------------------------------------------------------------------------------------

function wireEvents() {
  el.drop.addEventListener('click', () => el['file-input'].click());
  el.drop.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      el['file-input'].click();
    }
  });

  el['file-input'].addEventListener('change', (event) => {
    handleFiles([...event.target.files]);
    event.target.value = ''; // let the same file be re-selected
  });

  for (const type of ['dragenter', 'dragover']) {
    el.drop.addEventListener(type, (event) => {
      event.preventDefault();
      el.drop.classList.add('dragging');
    });
  }
  for (const type of ['dragleave', 'drop']) {
    el.drop.addEventListener(type, (event) => {
      event.preventDefault();
      if (type === 'dragleave' && el.drop.contains(event.relatedTarget)) return;
      el.drop.classList.remove('dragging');
    });
  }
  el.drop.addEventListener('drop', (event) => {
    handleFiles([...(event.dataTransfer?.files ?? [])]);
  });

  el.clear.addEventListener('click', reset);
  el['export-json'].addEventListener('click', exportJson);
  el['export-images'].addEventListener('click', exportImages);
  el['export-training'].addEventListener('click', exportTrainingData);
  el['print-report'].addEventListener('click', () => window.print());

  // Stamp the report as it goes to paper rather than at page load, so a tab left open
  // overnight cannot print yesterday's date onto today's inspection.
  window.addEventListener('beforeprint', stampReport);
}

// ---------------------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------------------

async function handleFiles(files) {
  // Classified by extension when the operating system supplied no MIME type, which is the
  // normal case for a file copied off a phone. Judging on file.type alone dropped iPhone
  // stills and clips before anything tried to open them.
  const images = files.filter((f) => classifyFile(f) === 'image');
  const videos = files.filter((f) => classifyFile(f) === 'video');
  const skipped = files.length - images.length - videos.length;

  if (skipped) {
    showBanner('warning', skipped === 1
      ? 'Skipped 1 file that is neither an image nor a video.'
      : `Skipped ${skipped} files that are neither images nor videos.`);
  }
  if ((!images.length && !videos.length) || state.busy) return;

  const blocked = blockedReason();
  if (blocked) {
    // Never a silent return. That is what this was, and a silent refusal is
    // indistinguishable from a broken app.
    showBanner('warning', blocked);
    renderDropState();
    el['engine-key'].focus();
    return;
  }

  const batch = images.slice(0, MAX_FILES);
  if (images.length > MAX_FILES) {
    showBanner('warning', `Processing the first ${MAX_FILES} of ${images.length} images.`);
  }

  state.busy = true;
  el['empty-state'].classList.add('hidden');
  el.progress.classList.remove('hidden');

  const engineName = usingApi() ? PROVIDERS[state.engine.provider].label : 'on-device model';

  // Videos are turned into frames before anything is analysed, so the progress bar counts
  // real work rather than jumping when a clip expands into twenty images halfway through.
  const frames = [];
  for (const file of videos) {
    try {
      const maxFrames = Math.max(1, Number(el['video-frames'].value) || VIDEO_DEFAULTS.maxFrames);
      const outcome = await extractFrames(file, { maxFrames }, (done, total, label) => {
        setProgress((done / Math.max(1, total)) * 100, label);
      });
      frames.push(...outcome.frames.map((frame) => ({ ...frame, from: file.name })));
      showBanner(
        'info',
        `${file.name}: scanned ${outcome.scanned} points across ${outcome.duration.toFixed(0)}s `
        + `and kept ${outcome.frames.length} distinct, in-focus frame`
        + `${outcome.frames.length === 1 ? '' : 's'}. Blurred and near-identical views were `
        + 'dropped, so a defect visible only in a blurred frame can be missed - upload a '
        + 'still of anything suspicious.',
      );
    } catch (error) {
      state.results.push({ file: file.name, size: file.size, status: 'error', message: error.message });
      appendResultCard(state.results.at(-1));
    }
  }

  const work = [
    ...batch.map((file) => ({ kind: 'file', file, name: file.name })),
    ...frames.map((frame) => ({ kind: 'frame', frame, name: frame.name })),
  ];

  // Frames from one clip are one sequence, so they get a tracker rather than being graded
  // as unrelated photographs. That is what turns a box per frame into a thing with an
  // identity that can be counted once and followed.
  state.tracker.reset();
  // The flame scanner keeps its own window of frames, and it measures how a region changes
  // over that window. Carrying one clip's window into the next would have it comparing a
  // frame against something filmed somewhere else.
  state.fire.reset();

  for (const [index, item] of work.entries()) {
    setProgress(
      (index / work.length) * 100,
      `Analysing ${index + 1} of ${work.length} with the ${engineName} - ${item.name}`,
    );
    const result = item.kind === 'file'
      ? await analyse(item.file)
      : await analyseImage(item.frame.bitmap, {
        track: true,
        file: item.frame.name,
        size: null,
        source: { video: item.frame.from, time: Number(item.frame.time.toFixed(2)) },
      });
    state.results.push(result);
    appendResultCard(result);
    renderSummary();
  }

  setProgress(100, 'Complete');
  el.progress.classList.add('hidden');
  if (!usingApi()) {
    el.backend.textContent = activeBackend() === 'webgpu' ? 'WebGPU' : 'WASM (CPU)';
  }
  state.busy = false;
  updateExportButtons();
}

async function analyse(file) {
  const base = { file: file.name, size: file.size };

  let image;
  try {
    image = await loadImage(file);
  } catch (error) {
    return { ...base, status: 'error', message: error.message };
  }
  // The original file is kept so the training-data export can carry the pixels the
  // findings describe. A label without its image trains nothing.
  return analyseImage(image, { ...base, sourceFile: file });
}

/**
 * Analyse a decoded image.
 *
 * Split out from analyse() because a video frame arrives as an ImageBitmap with no File
 * behind it. Both paths converge here, so a frame is graded by exactly the same code as an
 * uploaded photograph rather than by a parallel implementation that can drift.
 */
async function analyseImage(image, base) {
  if (usingOnDevice()) {
    try {
      return await analyseOnDevice(image, base);
    } catch (error) {
      return { ...base, image, status: 'error', message: error.message };
    }
  }

  if (usingApi()) {
    try {
      return await analyseWithApi(image, base);
    } catch (error) {
      return { ...base, image, status: 'error', message: error.message };
    }
  }

  try {
    const domain = await resolveDomain(image);
    if (domain.rejected) {
      return {
        ...base, image, status: 'rejected',
        message: domain.message, gate: domain.gate,
      };
    }

    const spec = state.manifest[domain.key];
    const detections = await detect(
      domain.key,
      spec,
      `${MODELS_BASE}${spec.file}`,
      image,
      (loaded, total) => setProgress(
        (loaded / total) * 100,
        `Downloading ${domain.key} model - ${(loaded / 1e6).toFixed(1)} of ${(total / 1e6).toFixed(1)} MB`,
      ),
    );

    const { score, severity } = assess(detections, spec.severityWeights ?? {});
    return {
      ...base, image, status: 'analysed',
      domain: domain.key, displayName: spec.displayName ?? domain.key,
      notes: spec.notes, zeroDetectionNote: spec.zeroDetectionNote,
      gate: domain.gate, detections, score, severity,
      people: await countPeople(image),
    };
  } catch (error) {
    return { ...base, image, status: 'error', message: error.message };
  }
}

/** Decide which detector an image goes to, or refuse it. */
async function resolveDomain(image) {
  const override = el['domain-override'].value;
  if (override !== 'auto') {
    return { key: override, rejected: false, gate: null };
  }

  if (!state.available.gate) {
    return { rejected: true, message: 'Select an inspection type to analyse this image.' };
  }

  const gateSpec = state.manifest.gate;
  const gate = await classify(gateSpec, `${MODELS_BASE}${gateSpec.file}`, image);

  if (gate.verdict === VERDICT.INVALID || gate.verdict === VERDICT.UNCERTAIN) {
    const deployed = DOMAINS
      .filter((key) => state.available[key])
      .map((key) => (state.manifest?.[key]?.displayName ?? key).toLowerCase());
    return {
      rejected: true,
      message: rejectionMessage(gate.verdict, gate.confidence, deployed),
      gate,
    };
  }
  if (!state.available[gate.verdict]) {
    return {
      rejected: true,
      gate,
      message: `Recognised as ${gate.verdict}, but that detection model is not deployed yet.`,
    };
  }
  return { key: gate.verdict, rejected: false, gate };
}

// ---------------------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------------------

function appendResultCard(result) {
  const card = document.createElement('article');
  card.className = `card card--${result.status}`;

  const media = document.createElement('div');
  media.className = 'card__media';

  if (result.image) {
    const canvas = document.createElement('canvas');
    canvas.className = 'card__canvas';
    drawDetections(canvas, result.image, result.detections ?? []);
    media.appendChild(canvas);
    result.canvas = canvas;
  }
  card.appendChild(media);

  const body = document.createElement('div');
  body.className = 'card__body';

  const title = document.createElement('h3');
  title.className = 'card__title';
  title.textContent = result.file;
  body.appendChild(title);

  if (result.status === 'analysed') {
    const badge = document.createElement('span');
    badge.className = `badge badge--${result.severity.key}`;
    badge.textContent = `${severityLabel(result.severity, result.domain)} · score ${result.score.toFixed(2)}`;
    body.appendChild(badge);

    const meta = document.createElement('p');
    meta.className = 'card__meta';
    meta.textContent = `${result.displayName} · ${result.detections.length} detection${
      result.detections.length === 1 ? '' : 's'
    }${result.gate ? ` · gate ${(result.gate.confidence * 100).toFixed(0)}%` : ''}`
      + (result.engine ? ` · ${result.engine.model}` : '');
    body.appendChild(meta);

    if (result.detections.length) {
      const list = document.createElement('ul');
      list.className = 'detections';
      for (const detection of result.detections) {
        const item = document.createElement('li');
        const swatch = document.createElement('span');
        swatch.className = 'swatch';
        swatch.style.background = colorFor(detection.classId);
        item.appendChild(swatch);
        // A vision model reports a certainty band, not a calibrated probability. Printing
        // "65.0%" next to it would dress a guess up as a measurement, so the band is shown
        // verbatim and only the trained detector gets a percentage.
        const strength = detection.trackId
          // A tracked thing is identified by which thing it is, not by how sure the model
          // was about one frame of it.
          ? `#${detection.trackId}${detection.coasted ? ', predicted' : ''}`
          : detection.certainty
            ? `${detection.certainty} certainty`
            : `${(detection.confidence * 100).toFixed(1)}%`;
        item.appendChild(document.createTextNode(`${detection.label} - ${strength}`));
        if (detection.note) {
          const note = document.createElement('span');
          note.className = 'detections__note';
          note.textContent = detection.note;
          item.appendChild(note);
        }
        list.appendChild(item);
      }
      body.appendChild(list);
    }

    // Findings the model described but could not place a box on. Dropping them would hide
    // real damage behind a clean-looking image, so they are listed without a box.
    if (result.unlocated?.length) {
      const heading = document.createElement('p');
      heading.className = 'card__meta';
      heading.textContent = 'Reported without a location:';
      body.appendChild(heading);

      const list = document.createElement('ul');
      list.className = 'detections detections--unlocated';
      for (const finding of result.unlocated) {
        const item = document.createElement('li');
        item.textContent = `${finding.label} - ${finding.certainty} certainty`
          + (finding.note ? ` - ${finding.note}` : '');
        list.appendChild(item);
      }
      body.appendChild(list);
    }

    // The people footnote, on every card whatever the subject. Written as a labelled number
    // rather than a sentence: phrased as prose, a zero reads as a claim about the scene,
    // where "People detected: 0" is a statement about what the detector marked. The
    // repository's own scanner rejects the prose form, and it is right to.
    if (result.people !== null && result.people !== undefined) {
      const people = document.createElement('p');
      people.className = 'card__people';
      people.textContent = `People detected: ${result.people}`;
      body.appendChild(people);
    }

    // A zero-detection result is the one most likely to be read as "this is fine", and it
    // is the one the model is least entitled to assert. Say what it looked for and what it
    // cannot see, rather than letting a green badge stand alone.
    const note = result.detections.length ? result.notes : (result.zeroDetectionNote ?? result.notes);
    if (note) {
      const element = document.createElement('p');
      element.className = 'card__note';
      element.textContent = note;
      body.appendChild(element);
    }
  } else {
    const message = document.createElement('p');
    message.className = `card__message card__message--${result.status}`;
    message.textContent = result.message;
    body.appendChild(message);

    if (result.gate) {
      const scores = document.createElement('p');
      scores.className = 'card__meta';
      scores.textContent = Object.entries(result.gate.scores)
        .map(([label, value]) => `${label} ${(value * 100).toFixed(0)}%`)
        .join(' · ');
      body.appendChild(scores);
    }
  }

  card.appendChild(body);
  el.results.appendChild(card);
}

function renderSummary() {
  const summary = summarise(state.results);
  const rejected = state.results.filter((r) => r.status === 'rejected').length;
  const errored = state.results.filter((r) => r.status === 'error').length;

  el.summary.classList.remove('hidden');
  el.summary.innerHTML = '';

  const overall = document.createElement('p');
  overall.className = 'summary__overall';
  overall.textContent = summary.overall;
  el.summary.appendChild(overall);

  const tiles = [
    ['Severe', summary.counts.severe, 'severe'],
    ['Moderate', summary.counts.moderate, 'moderate'],
    ['Minor', summary.counts.minor, 'minor'],
    ['Clear', summary.counts.none, 'none'],
    ['Rejected', rejected, 'rejected'],
    ['Errors', errored, 'error'],
  ];

  const row = document.createElement('div');
  row.className = 'summary__tiles';
  for (const [label, value, key] of tiles) {
    const tile = document.createElement('div');
    tile.className = `tile tile--${key}`;
    tile.innerHTML = `<span class="tile__value">${value}</span><span class="tile__label">${label}</span>`;
    row.appendChild(tile);
  }
  el.summary.appendChild(row);

  // The batch footnote. Summed across images, so somebody photographed three times counts
  // three times - these are images, not a headcount of a site, and the note says so.
  const counted = state.results.filter((r) => typeof r.people === 'number');
  if (counted.length) {
    const total = counted.reduce((sum, r) => sum + r.people, 0);
    const note = document.createElement('p');
    note.className = 'summary__people';
    note.innerHTML = `<strong>People detected: ${total}</strong> across `
      + `${counted.length} image${counted.length === 1 ? '' : 's'}`;

    const caveat = document.createElement('span');
    caveat.className = 'summary__caveat';
    caveat.textContent =
      'Summed per image, so anyone appearing in several is counted several times. It is what '
      + 'the on-device detector found, which is fewer than the people there: anyone small, '
      + 'distant, overlapping someone else or turned away is missed, and more are missed the '
      + 'higher the camera was. A floor, not a measurement.';
    note.appendChild(caveat);
    el.summary.appendChild(note);
  }
}

/** Date, image count and model versions, for the printed report only. */
function stampReport() {
  const stamp = document.getElementById('report-stamp');
  if (!stamp) return;

  const analysed = state.results.filter((r) => r.status === 'analysed').length;
  const models = usingApi()
    ? `${PROVIDERS[state.engine.provider].label} ${state.engine.model}`
    : ['gate', ...DOMAINS]
      .filter((k) => state.available[k])
      .map((k) => `${k} ${state.manifest?.[k]?.file ?? '?'}`)
      .join(', ');

  stamp.textContent =
    `Generated ${new Date().toLocaleString(undefined, { dateStyle: 'long', timeStyle: 'short' })}`
    + ` · ${analysed} of ${state.results.length} image${state.results.length === 1 ? '' : 's'} analysed`
    + (models ? ` · models: ${models}` : '');
  stamp.hidden = false;
}

function setProgress(percent, label) {
  el['progress-bar'].style.width = `${percent}%`;
  el['progress-label'].textContent = label;
}

function showBanner(kind, message) {
  el['status-banner'].className = `banner banner--${kind}`;
  el['status-banner'].textContent = message;
  el['status-banner'].classList.remove('hidden');
}

function updateExportButtons() {
  const hasResults = state.results.length > 0;
  for (const id of ['export-json', 'export-images', 'print-report', 'clear', 'export-training']) {
    el[id].disabled = !hasResults;
  }
}

function reset() {
  state.results = [];
  el.results.innerHTML = '';
  el.summary.classList.add('hidden');
  el['empty-state'].classList.remove('hidden');
  updateExportButtons();
}

// ---------------------------------------------------------------------------------------
// Export
// ---------------------------------------------------------------------------------------

function download(blob, filename) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}


/**
 * Download everything needed to turn this session into training data.
 *
 * WHY THIS BUTTON EXISTS
 *     The plan for getting off the API and back to a model on the device is that every
 *     inspection leaves a labelled photograph behind, and those accumulate into a dataset
 *     made of the operator's own imagery rather than one bought from a website. That is
 *     what tools/vlm_to_yolo.py builds from.
 *
 *     Until now only tools/vlm_inspect.py wrote those files. Anyone working in the app -
 *     which is everyone on a tablet - produced no training data at all, so the plan quietly
 *     did not apply to the way the thing is actually used.
 *
 * WHAT COMES OUT
 *     A zip laid out exactly as tools/vlm_inspect.py writes an inspection, so it drops
 *     straight into inspections/ and needs no conversion:
 *
 *         <name>/images/<file>          the original, untouched
 *         <name>/labels/<file>.json     the findings, with the image's dimensions
 *         <name>/inspection_summary.json
 *
 *     "reviewed" is false in every sidecar, and vlm_to_yolo.py ignores unreviewed labels by
 *     default. That is the point: a vision model's output is a first draft, and training on
 *     it unchecked teaches a detector to repeat its mistakes with more confidence.
 */
async function exportTrainingData() {
  const analysed = state.results.filter((r) => r.status === 'analysed' && r.sourceFile);
  if (!analysed.length) {
    showBanner('warning', state.results.some((r) => r.status === 'analysed')
      ? 'Only uploaded photographs can go into training data. Frames pulled out of a video '
        + 'have no original file behind them - save the frame first, then analyse it.'
      : 'Nothing analysed yet.');
    return;
  }

  const stamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
  const folder = `inspection-${stamp}`;
  const encoder = new TextEncoder();
  const entries = [];

  for (const result of analysed) {
    const bytes = new Uint8Array(await result.sourceFile.arrayBuffer());
    entries.push({ name: `${folder}/images/${result.file}`, data: bytes });

    const stem = result.file.replace(/\.[^.]+$/, '');
    entries.push({
      name: `${folder}/labels/${stem}.json`,
      data: encoder.encode(JSON.stringify({
        image: result.file,
        source: result.file,
        // Boxes are in pixels of this image, so the dimensions have to travel with them.
        width: result.image?.width ?? null,
        height: result.image?.height ?? null,
        domain: result.domain,
        provider: result.engine?.provider ?? 'local',
        model: result.engine?.model ?? null,
        generated: new Date().toISOString(),
        reviewed: false,
        overall: result.notes ?? '',
        detections: (result.detections ?? []).map((d) => ({
          label: d.label,
          class_id: d.classId ?? 0,
          certainty: d.certainty ?? 'medium',
          confidence: d.confidence,
          note: d.note ?? '',
          box: d.box.map((v) => Number(v.toFixed(1))),
        })),
        unlocated: (result.unlocated ?? []).map((d) => ({
          label: d.label, certainty: d.certainty, note: d.note ?? '',
        })),
      }, null, 2)),
    });
  }

  entries.push({
    name: `${folder}/inspection_summary.json`,
    data: encoder.encode(JSON.stringify(buildSummary(), null, 2)),
  });

  download(zip(entries), `${folder}.zip`);
  showBanner(
    'info',
    `${analysed.length} image${analysed.length === 1 ? '' : 's'} exported. Unzip into `
    + 'inspections/, correct the boxes and labels, set "reviewed": true in each, then run '
    + 'tools/vlm_to_yolo.py to build a training set from them.',
  );
}

/**
 * The inspection record, in the schema report_generator.py consumes.
 *
 * Shared by the JSON download and the training-data zip so the two cannot describe the
 * same session differently.
 */
function buildSummary() {
  const summary = summarise(state.results);
  return {
    generated: new Date().toISOString(),
    engine: usingApi()
      ? { provider: state.engine.provider, model: state.engine.model }
      : { provider: 'local', backend: activeBackend() },
    summary: { overall: summary.overall, counts: summary.counts, images: summary.total },
    results: state.results.map((r) => ({
      image: r.file,
      status: r.status,
      source: r.source ?? null,
      domain: r.domain ?? null,
      severity_label: r.severity ? severityLabel(r.severity, r.domain) : null,
      severity_band: r.severity?.key ?? null,
      severity_score: r.score != null ? Number(r.score.toFixed(3)) : null,
      people_detected: typeof r.people === 'number' ? r.people : null,
      message: r.message ?? null,
      gate: r.gate ? { verdict: r.gate.verdict, scores: r.gate.scores } : null,
      engine: r.engine ?? { provider: 'local', model: state.manifest?.[r.domain]?.file ?? null },
      notes: r.notes ?? null,
      detections: (r.detections ?? []).map((d) => ({
        class: d.label,
        confidence: Number(d.confidence.toFixed(4)),
        certainty: d.certainty ?? null,
        note: d.note ?? null,
        bbox: d.box.map((v) => Number(v.toFixed(1))),
      })),
      unlocated: (r.unlocated ?? []).map((d) => ({
        class: d.label, certainty: d.certainty ?? null, note: d.note ?? null,
      })),
    })),
  };
}

function exportJson() {
  download(
    new Blob([JSON.stringify(buildSummary(), null, 2)], { type: 'application/json' }),
    'inspection_summary.json',
  );
}

async function exportImages() {
  for (const result of state.results) {
    if (!result.canvas) continue;
    const blob = await toBlob(result.canvas);
    download(blob, `annotated_${result.file.replace(/\.[^.]+$/, '')}.png`);
  }
}

init();
