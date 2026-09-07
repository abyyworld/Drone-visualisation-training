/**
 * On-device detection of people and vehicles, with no API key and no connection.
 *
 * WHY THIS EXISTS
 *     Two problems at once. The app could not analyse anything at all without a provider
 *     key, and a provider cannot do real time - one round trip is seconds, so the live
 *     screen samples frames and says how old its boxes are. Neither is acceptable as the
 *     only option.
 *
 *     This is a COCO-trained detector running in the browser through MediaPipe. It needs no
 *     key, works offline once loaded, costs nothing per image, and is fast enough that
 *     boxes follow a video rather than describing a frame from four seconds ago. Paired
 *     with track.js it produces identities that persist, which is the thing an API cannot
 *     give at any price.
 *
 * WHAT IT CAN AND CANNOT SEE
 *     COCO. So: person, car, truck, bus, motorcycle, bicycle, boat - and none of fire,
 *     smoke, cracks, corrosion, erosion or soiling, because those are not COCO classes and
 *     no threshold conjures them.
 *
 *     That is one half of the job. It is the half that matters most in the two live
 *     domains, where a person at a fire and people in a crowd are the highest-weighted
 *     findings in the manifest. Defects and hazards stay with the provider engines until a
 *     model is trained on the accumulated inspections. web/models/DETECTOR.md has the
 *     model's provenance and its licence, which is the reason it is this model and not a
 *     YOLO checkpoint.
 *
 * THE LIMIT WORTH SAYING OUT LOUD
 *     A person forty metres below a drone is a handful of pixels. This misses them, and
 *     misses more of them the higher you fly. It is a screening aid whose recall falls with
 *     altitude - a fact about optics and input resolution, not a threshold to tune.
 */

// Loaded from a CDN by default, exactly as the ONNX runtime is, and overridable through
// `runtime.mediapipeBase` in the model manifest so the APK can serve its own copy and work
// with no connection at all.
const CDN_BASE = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1';

let base = CDN_BASE;
let delegate = 'CPU';
let model = 'models/detector.tflite';
let spec = {};
let detectorPromise = null;

export function configureOnDevice(runtime = {}, modelSpec = {}) {
  if (typeof runtime.mediapipeBase === 'string' && runtime.mediapipeBase.length) {
    base = runtime.mediapipeBase.replace(/\/$/, '');
  }
  if (runtime.delegate === 'GPU' || runtime.delegate === 'CPU') {
    delegate = runtime.delegate;
  }
  spec = modelSpec ?? {};
  if (typeof spec.file === 'string' && spec.file.length) {
    model = `models/${spec.file}`;
  }
}

/**
 * Roughly how long one frame takes, measured rather than assumed.
 *
 * The live view uses it to pick a sensible cadence instead of running flat out and starving
 * the page of everything else.
 */
let lastInferenceMs = 0;

export function inferenceMillis() {
  return lastInferenceMs;
}

/**
 * The COCO classes worth reporting from a drone.
 *
 * The other 73 are furniture, food and household objects. Reporting a potted plant during a
 * crowd inspection is noise that costs an operator attention, so they are dropped here
 * rather than left for someone to scroll past.
 */
const KEEP = new Set([
  'person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck', 'boat', 'train', 'airplane',
]);

/** Which subjects this engine is any use for, and what it is looking for in each. */
export const SUBJECTS = {
  crowd: 'people and vehicles',
  wildfire: 'people and vehicles',
};

async function loadDetector(onProgress) {
  if (detectorPromise) return detectorPromise;

  detectorPromise = (async () => {
    onProgress?.('Loading the on-device detector');
    const { FilesetResolver, ObjectDetector } = await import(
      /* @vite-ignore */ `${base}/vision_bundle.mjs`
    );
    const vision = await FilesetResolver.forVisionTasks(`${base}/wasm`);

    return ObjectDetector.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: model,
        // CPU, deliberately, and this is the single most important line in the file.
        //
        // With delegate: 'GPU' this detector returns an empty list. Not an error, not a
        // warning - zero detections on a photograph with a person filling half of it,
        // where CPU scores that same person at 0.98. It fails silently, and a silent empty
        // result is the one failure this application must never have: it is
        // indistinguishable from a frame with nothing in it, which is exactly the
        // conclusion an operator would draw.
        //
        // Whether that is the model, the delegate or the machine is not worth finding out,
        // because there is no way to detect it at runtime - a wrong answer and a correct
        // empty answer look the same. So the option that is always right wins, and anyone
        // who has verified GPU on their own hardware can set `runtime.delegate` in the
        // manifest.
        delegate,
      },
      scoreThreshold: spec.scoreThreshold ?? 0.35,
      maxResults: 60,
      runningMode: 'IMAGE',
    });
  })().catch((error) => {
    // Never leave a rejected promise cached, or one bad load poisons every later attempt.
    detectorPromise = null;
    throw new Error(
      `The on-device detector could not be loaded: ${error.message}. `
      + 'It needs one connection to fetch its runtime, after which it works offline.',
    );
  });

  return detectorPromise;
}

/** Is it worth offering this engine for a given subject? */
export function handles(domain) {
  return domain === 'auto' || domain in SUBJECTS;
}

/**
 * Detect in one image.
 *
 * @returns {Promise<Array<{label:string, confidence:number, box:number[], classId:number}>>}
 *   Boxes in pixels of the image passed in, matching the shape the ONNX path returns so the
 *   renderer, the severity scoring and the exports all work unchanged.
 */
export async function detectOnDevice(image, onProgress) {
  const detector = await loadDetector(onProgress);
  const started = performance.now();
  const result = detector.detect(image);
  lastInferenceMs = performance.now() - started;

  const findings = [];
  for (const detection of result.detections ?? []) {
    const category = detection.categories?.[0];
    if (!category) continue;

    const label = String(category.categoryName ?? '').toLowerCase();
    if (!KEEP.has(label)) continue;

    const { originX, originY, width, height } = detection.boundingBox;
    findings.push({
      label,
      confidence: category.score,
      classId: colourIndex(label),
      box: [originX, originY, originX + width, originY + height],
    });
  }
  return findings;
}

/** Same hash as colourIndex() in vlm.js, so a label keeps one colour across both engines. */
function colourIndex(label) {
  let hash = 0;
  for (let i = 0; i < label.length; i += 1) hash = (hash * 31 + label.charCodeAt(i)) % 4096;
  return hash;
}
