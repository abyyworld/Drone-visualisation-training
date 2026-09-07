/**
 * Vision-language-model inspection: an alternative to the on-device ONNX detector.
 *
 * WHY THIS EXISTS
 *     The trained detector only knows the defect classes that were in its dataset, and it
 *     only recognises them at the framing distance the dataset was shot at. It returned
 *     zero detections on a turbine with a blade snapped in half, because "blade severed"
 *     was not a class and no training image was a wide landscape shot. A general vision
 *     model has no such boundary - it describes what it sees. That is the trade this
 *     module makes: broader recall and no dataset to curate, in exchange for boxes that
 *     are approximate rather than pixel-tight, a per-image cost, and images leaving the
 *     device. All three are stated in the UI rather than buried here.
 *
 * WHAT IT RETURNS
 *     The same detection shape the ONNX path produces - {label, confidence, box} with box
 *     in pixels [x0, y0, x1, y1] - so severity scoring, the renderer, the JSON export and
 *     report_generator.py all work unchanged. One pipeline, two engines behind it.
 *
 * CONFIDENCE IS NOT A PROBABILITY
 *     A language model asked for a float will invent one. We ask for a certainty band
 *     instead and map it to a number, so the severity arithmetic still works without
 *     pretending the value is calibrated. Do not read 0.9 here as "90% likely".
 *
 * THE KEY
 *     Held in memory for the tab's lifetime and sent only to the provider you picked. It
 *     is never written to localStorage, never logged, and never sent anywhere else. A
 *     browser key is still a key visible to anyone using that machine - for anything
 *     beyond your own use, run tools/vlm_inspect.py server-side instead.
 */


export const ENGINE_LOCAL = 'local';

/**
 * Provider catalogue.
 *
 * `models` is a fallback list, not the source of truth - new model IDs ship faster than
 * this file is edited, so the UI can refresh it live from each provider's own models
 * endpoint once a key is entered. The fallback is what you get before that.
 */
export const PROVIDERS = {
  anthropic: {
    label: 'Anthropic Claude',
    keyLabel: 'Anthropic API key',
    keyHint: 'Starts with sk-ant-. console.anthropic.com -> API keys',
    models: [
      { id: 'claude-opus-5', label: 'Claude Opus 5 (most capable)' },
      { id: 'claude-sonnet-5', label: 'Claude Sonnet 5 (balanced)' },
      { id: 'claude-haiku-4-5-20251001', label: 'Claude Haiku 4.5 (fastest, cheapest)' },
    ],
  },
  gemini: {
    label: 'Google Gemini',
    keyLabel: 'Google AI Studio API key',
    keyHint: 'aistudio.google.com -> Get API key',
    models: [
      { id: 'gemini-2.5-pro', label: 'Gemini 2.5 Pro (most capable)' },
      { id: 'gemini-2.5-flash', label: 'Gemini 2.5 Flash (balanced)' },
      { id: 'gemini-2.5-flash-lite', label: 'Gemini 2.5 Flash Lite (fastest, cheapest)' },
    ],
  },
  openai: {
    label: 'OpenAI',
    keyLabel: 'OpenAI API key',
    keyHint: 'Starts with sk-. platform.openai.com -> API keys',
    models: [
      { id: 'gpt-5', label: 'GPT-5 (most capable)' },
      { id: 'gpt-5-mini', label: 'GPT-5 mini (balanced)' },
      { id: 'gpt-4.1', label: 'GPT-4.1' },
      { id: 'gpt-4o', label: 'GPT-4o' },
    ],
  },
};

// ---------------------------------------------------------------------------------------
// The prompt
// ---------------------------------------------------------------------------------------

/**
 * The prompt lives in web/prompts/inspection.json, not in this file.
 *
 * tools/vlm_inspect.py reads the same file. A prompt copied into two languages drifts, and
 * then the web app and the batch tool grade the same photograph differently - which is the
 * one thing an inspection record must not do. One fetch, cached for the page's lifetime.
 */
const PROMPT_URL = 'prompts/inspection.json';
let promptDoc = null;

async function loadPrompts() {
  if (promptDoc) return promptDoc;
  const response = await fetch(PROMPT_URL);
  if (!response.ok) {
    throw new Error(`Could not load ${PROMPT_URL} (${response.status}). The site is deployed but incomplete.`);
  }
  promptDoc = await response.json();
  return promptDoc;
}

async function promptFor(domain) {
  const doc = await loadPrompts();
  const brief = doc.domains[domain] ?? doc.domains.auto;
  return `${brief}\n\n${doc.schema}`;
}

// ---------------------------------------------------------------------------------------
// Image encoding
// ---------------------------------------------------------------------------------------

/**
 * Re-encode to a JPEG under `maxEdge` on the long side.
 *
 * Every provider downsamples above roughly this size before the model ever sees the
 * pixels, so uploading a 48-megapixel original buys no accuracy and costs upload time and
 * tokens. Re-encoding also strips EXIF, including GPS - the coordinates of an asset are
 * not something to hand to a third party by accident.
 */
async function encode(image, maxEdge) {
  const scale = Math.min(1, maxEdge / Math.max(image.width, image.height));
  const w = Math.max(1, Math.round(image.width * scale));
  const h = Math.max(1, Math.round(image.height * scale));

  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  canvas.getContext('2d').drawImage(image, 0, 0, w, h);

  const dataUrl = canvas.toDataURL('image/jpeg', 0.9);
  return { mediaType: 'image/jpeg', base64: dataUrl.slice(dataUrl.indexOf(',') + 1), dataUrl };
}

// ---------------------------------------------------------------------------------------
// Provider adapters
// ---------------------------------------------------------------------------------------

const ADAPTERS = {
  anthropic: {
    async infer({ model, apiKey, prompt, image }) {
      const response = await fetch('https://api.anthropic.com/v1/messages', {
        method: 'POST',
        headers: {
          'content-type': 'application/json',
          'x-api-key': apiKey,
          'anthropic-version': '2023-06-01',
          // Anthropic blocks browser-origin calls unless this is set, because a key in a
          // page is exposed to whoever opens the page. Opting in is deliberate here.
          'anthropic-dangerous-direct-browser-access': 'true',
        },
        body: JSON.stringify({
          model,
          max_tokens: 2000,
          messages: [{
            role: 'user',
            content: [
              { type: 'image', source: { type: 'base64', media_type: image.mediaType, data: image.base64 } },
              { type: 'text', text: prompt },
            ],
          }],
        }),
      });
      const body = await readJson(response, 'Anthropic');
      return (body.content ?? []).filter((b) => b.type === 'text').map((b) => b.text).join('');
    },

    async listModels(apiKey) {
      const response = await fetch('https://api.anthropic.com/v1/models?limit=100', {
        headers: {
          'x-api-key': apiKey,
          'anthropic-version': '2023-06-01',
          'anthropic-dangerous-direct-browser-access': 'true',
        },
      });
      const body = await readJson(response, 'Anthropic');
      return (body.data ?? []).map((m) => ({ id: m.id, label: m.display_name ?? m.id }));
    },
  },

  gemini: {
    async infer({ model, apiKey, prompt, image }) {
      const url = `https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(model)}:generateContent`;
      const response = await fetch(url, {
        method: 'POST',
        headers: { 'content-type': 'application/json', 'x-goog-api-key': apiKey },
        body: JSON.stringify({
          contents: [{
            role: 'user',
            parts: [
              { inline_data: { mime_type: image.mediaType, data: image.base64 } },
              { text: prompt },
            ],
          }],
          generationConfig: { responseMimeType: 'application/json', maxOutputTokens: 2000 },
        }),
      });
      const body = await readJson(response, 'Gemini');
      const parts = body.candidates?.[0]?.content?.parts ?? [];
      return parts.map((p) => p.text ?? '').join('');
    },

    async listModels(apiKey) {
      const response = await fetch(
        'https://generativelanguage.googleapis.com/v1beta/models?pageSize=200',
        { headers: { 'x-goog-api-key': apiKey } },
      );
      const body = await readJson(response, 'Gemini');
      return (body.models ?? [])
        .filter((m) => (m.supportedGenerationMethods ?? []).includes('generateContent'))
        .map((m) => ({
          id: m.name.replace(/^models\//, ''),
          label: m.displayName ?? m.name.replace(/^models\//, ''),
        }));
    },
  },

  openai: {
    async infer({ model, apiKey, prompt, image }) {
      const response = await fetch('https://api.openai.com/v1/chat/completions', {
        method: 'POST',
        headers: { 'content-type': 'application/json', authorization: `Bearer ${apiKey}` },
        body: JSON.stringify({
          model,
          response_format: { type: 'json_object' },
          messages: [{
            role: 'user',
            content: [
              { type: 'text', text: prompt },
              { type: 'image_url', image_url: { url: image.dataUrl, detail: 'high' } },
            ],
          }],
        }),
      });
      const body = await readJson(response, 'OpenAI');
      return body.choices?.[0]?.message?.content ?? '';
    },

    async listModels(apiKey) {
      const response = await fetch('https://api.openai.com/v1/models', {
        headers: { authorization: `Bearer ${apiKey}` },
      });
      const body = await readJson(response, 'OpenAI');
      return (body.data ?? [])
        .map((m) => ({ id: m.id, label: m.id }))
        .filter((m) => /^(gpt-|o[0-9]|chatgpt)/.test(m.id) && !/audio|realtime|tts|whisper|embedding|moderation|image/.test(m.id));
    },
  },
};

/**
 * Turn a provider response into JSON, or into an error a user can act on.
 *
 * Providers signal the same underlying problems with different status codes and different
 * body shapes, and the generic "request failed" that comes out of a naive fetch wrapper
 * sends people looking in the wrong place. A wrong key and an exhausted quota need
 * different fixes, so they get different messages.
 */
async function readJson(response, providerName) {
  const text = await response.text();
  let body = null;
  try { body = JSON.parse(text); } catch { /* keep the raw text for the message */ }

  if (response.ok) {
    if (body === null) throw new Error(`${providerName} returned a response that is not JSON.`);
    return body;
  }

  const detail = body?.error?.message ?? body?.message ?? text.slice(0, 300);

  if (response.status === 401 || response.status === 403) {
    throw new Error(`${providerName} rejected the API key (${response.status}). Check you pasted the whole key, that it is for ${providerName} and not another provider, and that it has not been revoked. ${detail}`);
  }
  if (response.status === 404) {
    throw new Error(`${providerName} does not recognise that model ID (404). Pick another from the list, or press Refresh to load the models your key can actually reach. ${detail}`);
  }
  if (response.status === 429) {
    throw new Error(`${providerName} rate-limited or out of credit (429). Wait and retry, or top up the account. ${detail}`);
  }
  if (response.status >= 500) {
    throw new Error(`${providerName} server error (${response.status}). This is their side, not yours - retry shortly. ${detail}`);
  }
  throw new Error(`${providerName} request failed (${response.status}). ${detail}`);
}

// ---------------------------------------------------------------------------------------
// Response parsing
// ---------------------------------------------------------------------------------------

/** Pull the JSON object out of a reply that may be fenced or padded with prose. */
export function parseReply(text) {
  const trimmed = (text ?? '').trim();
  if (!trimmed) throw new Error('The model returned an empty response.');

  const fenced = trimmed.match(/```(?:json)?\s*([\s\S]*?)```/);
  const candidate = fenced ? fenced[1] : trimmed;

  try {
    return JSON.parse(candidate);
  } catch { /* fall through to a brace scan */ }

  const start = candidate.indexOf('{');
  const end = candidate.lastIndexOf('}');
  if (start !== -1 && end > start) {
    try { return JSON.parse(candidate.slice(start, end + 1)); } catch { /* give up below */ }
  }
  throw new Error(`The model did not return usable JSON. It said: ${trimmed.slice(0, 200)}`);
}

/** Normalised fractions -> pixel box, clamped to the frame, with degenerate boxes dropped. */
export function toPixels(box, width, height) {
  if (!Array.isArray(box) || box.length !== 4 || box.some((v) => typeof v !== 'number' || !Number.isFinite(v))) {
    return null;
  }
  // Some models answer in 0-1000 or in pixels despite the instruction. Anything clearly
  // outside 0..1 is rescaled rather than thrown away - the box is still the useful part.
  let [x0, y0, x1, y1] = box;
  const max = Math.max(...box.map(Math.abs));
  if (max > 1.5 && max <= 1000) {
    [x0, y0, x1, y1] = box.map((v) => v / 1000);
  } else if (max > 1000) {
    x0 /= width; x1 /= width; y0 /= height; y1 /= height;
  }

  const clamp = (v) => Math.min(1, Math.max(0, v));
  const px0 = clamp(Math.min(x0, x1)) * width;
  const px1 = clamp(Math.max(x0, x1)) * width;
  const py0 = clamp(Math.min(y0, y1)) * height;
  const py1 = clamp(Math.max(y0, y1)) * height;

  if (px1 - px0 < 2 || py1 - py0 < 2) return null;
  return [px0, py0, px1, py1];
}

/**
 * Stable colour index for a free-form label.
 *
 * The ONNX path numbers its classes from the manifest; a vision model invents its label
 * set per image, so there is no index to use. Hashing the label keeps "crack" the same
 * colour in every image of a batch, which is what the palette is for.
 */
function colourIndex(label) {
  let hash = 0;
  for (let i = 0; i < label.length; i += 1) hash = (hash * 31 + label.charCodeAt(i)) % 4096;
  return hash;
}

export function tidyLabel(raw) {
  return String(raw ?? 'defect')
    .toLowerCase().trim()
    .replace(/[^a-z0-9]+/g, '_')
    .replace(/^_+|_+$/g, '')
    .slice(0, 40) || 'defect';
}

// ---------------------------------------------------------------------------------------
// Entry points
// ---------------------------------------------------------------------------------------

/**
 * Inspect one image.
 *
 * @returns {{asset: string, assetReason: string, overall: string, detections: Array,
 *            unlocated: Array, model: string, provider: string}}
 *   `unlocated` holds defects the model described but could not place a usable box on.
 *   They are kept and shown rather than dropped: "there is a crack here somewhere" is
 *   still worth an engineer's time, and silently discarding it would hide a real finding.
 */
export async function inspect({ provider, model, apiKey, image, domain = 'auto' }) {
  const adapter = ADAPTERS[provider];
  if (!adapter) throw new Error(`Unknown provider: ${provider}`);
  if (!apiKey) throw new Error('Enter an API key first.');
  if (!model) throw new Error('Choose a model first.');

  const doc = await loadPrompts();
  const encoded = await encode(image, doc.maxEdge);
  const raw = await adapter.infer({ model, apiKey, prompt: await promptFor(domain), image: encoded });
  const parsed = parseReply(raw);

  const detections = [];
  const unlocated = [];
  for (const item of parsed.defects ?? []) {
    const label = tidyLabel(item.label);
    const entry = {
      label,
      classId: colourIndex(label),
      confidence: doc.certainty[String(item.certainty ?? '').toLowerCase()] ?? doc.certainty.medium,
      certainty: String(item.certainty ?? 'medium').toLowerCase(),
      note: typeof item.note === 'string' ? item.note : '',
    };
    const box = toPixels(item.box, image.width, image.height);
    if (box) detections.push({ ...entry, box });
    else unlocated.push(entry);
  }

  return {
    asset: ['turbine', 'solar', 'neither'].includes(parsed.asset) ? parsed.asset : 'neither',
    assetReason: typeof parsed.asset_reason === 'string' ? parsed.asset_reason : '',
    overall: typeof parsed.overall === 'string' ? parsed.overall : '',
    detections,
    unlocated,
    provider,
    model,
  };
}

/** Ask the provider which models this key can actually reach. Falls back to the catalogue. */
export async function listModels(provider, apiKey) {
  const adapter = ADAPTERS[provider];
  if (!adapter) throw new Error(`Unknown provider: ${provider}`);
  const models = await adapter.listModels(apiKey);
  if (!models.length) throw new Error('The provider returned no models for this key.');
  return models.sort((a, b) => a.id.localeCompare(b.id));
}
