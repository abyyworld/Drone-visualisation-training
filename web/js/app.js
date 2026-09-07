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
import { assess, summarise } from './severity.js';
import { drawDetections, toBlob, colorFor } from './render.js';
import { PROVIDERS, ENGINE_LOCAL, inspect, listModels } from './vlm.js';
import { isVideo, extractFrames, VIDEO_DEFAULTS } from './video.js';

const MODELS_BASE = 'models/';
const MAX_FILES = 100;

const state = {
  manifest: null,
  available: {},
  results: [],
  busy: false,
  // The engine the next batch will run on. `apiKey` lives here and nowhere else - not in
  // localStorage, not in the URL, not in an exported file - so closing the tab discards it.
  engine: { provider: ENGINE_LOCAL, model: null, apiKey: '' },
};

const el = {};

// ---------------------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------------------

async function init() {
  for (const id of [
    'drop', 'file-input', 'results', 'summary', 'status-banner', 'backend',
    'domain-override', 'override-row', 'export-json', 'export-images', 'print-report',
    'empty-state', 'progress', 'progress-bar', 'progress-label', 'clear',
    'engine-provider', 'engine-model', 'engine-model-field', 'engine-model-hint',
    'engine-refresh', 'engine-key', 'engine-key-field', 'engine-key-label',
    'engine-key-hint', 'engine-key-toggle', 'engine-warning', 'privacy-pill',
    'video-frames',
  ]) {
    el[id] = document.getElementById(id);
  }

  el.backend.textContent = (await probeBackend()) === 'webgpu' ? 'WebGPU' : 'WASM (CPU)';

  await loadManifest();
  wireEngine();
  wireEvents();
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
  } catch (error) {
    showBanner(
      'error',
      'Could not load models/manifest.json. The site is deployed but not configured.',
    );
    return;
  }

  // A manifest entry is a promise, not a fact - check each file is actually there. A HEAD
  // request is enough and costs nothing next to downloading the weights.
  const keys = ['gate', 'turbine', 'solar'];
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
  const detectors = ['turbine', 'solar'].filter((k) => state.available[k]);
  const missing = ['gate', 'turbine', 'solar'].filter((k) => !state.available[k]);

  if (!detectors.length) {
    showBanner(
      'warning',
      'No detection models are deployed yet. Train them with the notebooks in training/, '
      + 'export with tools/export_onnx.py into web/models/, and this page will pick them up '
      + 'automatically - no code changes needed.',
    );
    el.drop.setAttribute('aria-disabled', 'true');
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

  // Only offer domains that actually have a model behind them.
  for (const option of el['domain-override'].options) {
    if (option.value !== 'auto' && !state.available[option.value]) option.disabled = true;
  }
  if (detectors.length === 1) el['domain-override'].value = detectors[0];
}


// ---------------------------------------------------------------------------------------
// Engine selection
// ---------------------------------------------------------------------------------------

function usingApi() {
  return state.engine.provider !== ENGINE_LOCAL;
}

function wireEngine() {
  el['engine-provider'].addEventListener('change', () => {
    state.engine.provider = el['engine-provider'].value;
    renderEngine();
  });

  el['engine-model'].addEventListener('change', () => {
    state.engine.model = el['engine-model'].value;
  });

  el['engine-key'].addEventListener('input', () => {
    // Keys are routinely pasted with a trailing newline or a stray space out of a password
    // manager, and every provider then returns a flat 401 that reads like a wrong key.
    state.engine.apiKey = el['engine-key'].value.trim();
    el['engine-refresh'].disabled = !state.engine.apiKey;
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
    reportModelStatus();
    return;
  }

  el['engine-key-label'].textContent = provider.keyLabel;
  el['engine-key-hint'].textContent = provider.keyHint;
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
  el['print-report'].addEventListener('click', () => window.print());

  // Stamp the report as it goes to paper rather than at page load, so a tab left open
  // overnight cannot print yesterday's date onto today's inspection.
  window.addEventListener('beforeprint', stampReport);
}

// ---------------------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------------------

async function handleFiles(files) {
  const images = files.filter((f) => f.type.startsWith('image/'));
  const videos = files.filter((f) => !f.type.startsWith('image/') && isVideo(f));
  const skipped = files.length - images.length - videos.length;

  if (skipped) {
    showBanner('warning', skipped === 1
      ? 'Skipped 1 file that is neither an image nor a video.'
      : `Skipped ${skipped} files that are neither images nor videos.`);
  }
  if ((!images.length && !videos.length) || state.busy) return;

  if (usingApi()) {
    if (!state.engine.apiKey) {
      showBanner('warning', 'Enter an API key for the selected engine before uploading.');
      return;
    }
    if (!state.engine.model) {
      showBanner('warning', 'Choose a model for the selected engine before uploading.');
      return;
    }
  } else if (!['turbine', 'solar'].some((k) => state.available[k])) {
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

  for (const [index, item] of work.entries()) {
    setProgress(
      (index / work.length) * 100,
      `Analysing ${index + 1} of ${work.length} with the ${engineName} - ${item.name}`,
    );
    const result = item.kind === 'file'
      ? await analyse(item.file)
      : await analyseImage(item.frame.bitmap, {
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
  return analyseImage(image, base);
}

/**
 * Analyse a decoded image.
 *
 * Split out from analyse() because a video frame arrives as an ImageBitmap with no File
 * behind it. Both paths converge here, so a frame is graded by exactly the same code as an
 * uploaded photograph rather than by a parallel implementation that can drift.
 */
async function analyseImage(image, base) {
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
    return { rejected: true, message: rejectionMessage(gate.verdict, gate.confidence), gate };
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
    badge.textContent = `${result.severity.label} · score ${result.score.toFixed(2)}`;
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
        const strength = detection.certainty
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
}

/** Date, image count and model versions, for the printed report only. */
function stampReport() {
  const stamp = document.getElementById('report-stamp');
  if (!stamp) return;

  const analysed = state.results.filter((r) => r.status === 'analysed').length;
  const models = usingApi()
    ? `${PROVIDERS[state.engine.provider].label} ${state.engine.model}`
    : ['gate', 'turbine', 'solar']
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
  for (const id of ['export-json', 'export-images', 'print-report', 'clear']) {
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

function exportJson() {
  const summary = summarise(state.results);
  const payload = {
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
      severity_label: r.severity?.label ?? null,
      severity_score: r.score != null ? Number(r.score.toFixed(3)) : null,
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

  download(
    new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' }),
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
