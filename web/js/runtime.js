/**
 * ONNX Runtime Web session management.
 *
 * Everything runs client-side: images never leave the visitor's device, there is no server
 * to pay for or keep warm, and GitHub Pages can host the whole thing as static files.
 *
 * WebGPU is tried first and falls back to WASM automatically. The fallback is not optional
 * politeness — Safari and older Chrome still ship without usable WebGPU, and a hard failure
 * there would take the whole site down.
 */

// Pinned deliberately. Bump it on purpose after testing, not incidentally: a runtime change
// can alter numerics and break an exported graph.
export const ORT_VERSION = '1.23.0';

const CDN_BASE = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;

// Where to load the runtime and its WASM assets from. Override via `runtime.ortBase` in
// models/manifest.json to self-host — worth doing for a production deployment, so the site
// does not depend on a third-party CDN staying up (and so the tests can run offline).
let ortBase = CDN_BASE;

let ortPromise = null;
let backend = null;
const sessions = new Map();

/** Point the runtime at a different copy of onnxruntime-web. Call before any inference. */
export function configureRuntime({ ortBase: base } = {}) {
  if (!base || ortPromise) return;
  ortBase = base.endsWith('/') ? base : `${base}/`;
}

/** Load onnxruntime-web once and point it at its own WASM assets. */
function loadOrt() {
  if (ortPromise) return ortPromise;

  ortPromise = import(/* @vite-ignore */ `${ortBase}ort.webgpu.min.mjs`)
    .catch(() => import(/* @vite-ignore */ `${ortBase}ort.min.mjs`))
    .then((module) => {
      const ort = module.default ?? module;
      ort.env.wasm.wasmPaths = ortBase;
      // Threads need cross-origin isolation, which GitHub Pages does not provide. Asking
      // for more than one would make the runtime fail to start rather than run slower.
      ort.env.wasm.numThreads = 1;
      ort.env.logLevel = 'error';
      return ort;
    });

  return ortPromise;
}

/** True when the browser exposes a usable WebGPU adapter. */
async function hasWebGPU() {
  if (!('gpu' in navigator)) return false;
  try {
    return Boolean(await navigator.gpu.requestAdapter());
  } catch {
    return false;
  }
}

/**
 * Create (or reuse) an inference session for a model file.
 *
 * @param {string} key   Model key from the manifest, used as the cache key.
 * @param {string} url   URL of the .onnx file.
 * @param {(loaded:number, total:number)=>void} [onProgress]
 * @returns {Promise<{session: object, ort: object}>}
 */
export async function getSession(key, url, onProgress) {
  if (sessions.has(key)) return sessions.get(key);

  const promise = (async () => {
    const ort = await loadOrt();

    // Fetch the weights ourselves so the UI can show real download progress — model files
    // are the slowest part of first load and a silent wait reads as a broken page.
    const buffer = await fetchWithProgress(url, onProgress);

    const providers = (await hasWebGPU()) ? ['webgpu', 'wasm'] : ['wasm'];
    let session;
    try {
      session = await ort.InferenceSession.create(buffer, {
        executionProviders: providers,
        graphOptimizationLevel: 'all',
      });
      backend = providers[0];
    } catch (error) {
      if (providers[0] === 'wasm') throw error;
      // A WebGPU adapter can exist and still fail on this particular graph.
      session = await ort.InferenceSession.create(buffer, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all',
      });
      backend = 'wasm';
    }

    return { session, ort };
  })();

  sessions.set(key, promise);
  promise.catch(() => sessions.delete(key)); // let a failed load be retried
  return promise;
}

async function fetchWithProgress(url, onProgress) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}`);
  }

  const total = Number(response.headers.get('content-length')) || 0;
  if (!response.body || !total || !onProgress) {
    return new Uint8Array(await response.arrayBuffer());
  }

  const reader = response.body.getReader();
  const chunks = [];
  let loaded = 0;

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    onProgress(loaded, total);
  }

  const bytes = new Uint8Array(loaded);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.length;
  }
  return bytes;
}

/** Which execution provider actually got used, once something has been loaded. */
export function activeBackend() {
  return backend;
}

/** Best-effort backend name for display before any model has loaded. */
export async function probeBackend() {
  return (await hasWebGPU()) ? 'webgpu' : 'wasm';
}
