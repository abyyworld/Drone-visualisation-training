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

const MODELS_BASE = 'models/';
const MAX_FILES = 100;

const state = {
  manifest: null,
  available: {},
  results: [],
  busy: false,
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
  ]) {
    el[id] = document.getElementById(id);
  }

  el.backend.textContent = (await probeBackend()) === 'webgpu' ? 'WebGPU' : 'WASM (CPU)';

  await loadManifest();
  wireEvents();
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

  // A manifest entry is a promise, not a fact — check each file is actually there. A HEAD
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
      + 'automatically — no code changes needed.',
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
  const skipped = files.length - images.length;

  if (skipped) {
    showBanner('warning', `Skipped ${skipped} non-image file${skipped === 1 ? '' : 's'}.`);
  }
  if (!images.length || state.busy) return;
  if (!['turbine', 'solar'].some((k) => state.available[k])) return;

  const batch = images.slice(0, MAX_FILES);
  if (images.length > MAX_FILES) {
    showBanner('warning', `Processing the first ${MAX_FILES} of ${images.length} images.`);
  }

  state.busy = true;
  el['empty-state'].classList.add('hidden');
  el.progress.classList.remove('hidden');

  for (const [index, file] of batch.entries()) {
    setProgress((index / batch.length) * 100, `Analysing ${index + 1} of ${batch.length} — ${file.name}`);
    const result = await analyse(file);
    state.results.push(result);
    appendResultCard(result);
    renderSummary();
  }

  setProgress(100, 'Complete');
  el.progress.classList.add('hidden');
  el.backend.textContent = activeBackend() === 'webgpu' ? 'WebGPU' : 'WASM (CPU)';
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
        `Downloading ${domain.key} model — ${(loaded / 1e6).toFixed(1)} of ${(total / 1e6).toFixed(1)} MB`,
      ),
    );

    const { score, severity } = assess(detections, spec.severityWeights ?? {});
    return {
      ...base, image, status: 'analysed',
      domain: domain.key, displayName: spec.displayName ?? domain.key,
      notes: spec.notes, gate: domain.gate, detections, score, severity,
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
    }${result.gate ? ` · gate ${(result.gate.confidence * 100).toFixed(0)}%` : ''}`;
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
        item.appendChild(
          document.createTextNode(
            `${detection.label} — ${(detection.confidence * 100).toFixed(1)}%`,
          ),
        );
        list.appendChild(item);
      }
      body.appendChild(list);
    }

    if (result.notes) {
      const note = document.createElement('p');
      note.className = 'card__note';
      note.textContent = result.notes;
      body.appendChild(note);
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
  const models = ['gate', 'turbine', 'solar']
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
    summary: { overall: summary.overall, counts: summary.counts, images: summary.total },
    results: state.results.map((r) => ({
      image: r.file,
      status: r.status,
      domain: r.domain ?? null,
      severity_label: r.severity?.label ?? null,
      severity_score: r.score != null ? Number(r.score.toFixed(3)) : null,
      message: r.message ?? null,
      gate: r.gate ? { verdict: r.gate.verdict, scores: r.gate.scores } : null,
      detections: (r.detections ?? []).map((d) => ({
        class: d.label,
        confidence: Number(d.confidence.toFixed(4)),
        bbox: d.box.map((v) => Number(v.toFixed(1))),
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
