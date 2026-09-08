/**
 * Detection, flame and smoke, and appearance signatures - all off the main thread.
 *
 * WHY A WORKER, AND WHY IT WAS NEVER GOING TO BE SMOOTH WITHOUT ONE
 *     MediaPipe's detectForVideo is synchronous. On a laptop it takes about 360 ms, and
 *     JavaScript has one thread: for those 360 ms nothing else in the page runs. Not the
 *     animation frame that draws the boxes, not the compositing of the video underneath
 *     them, not a click on the stop button. Splitting detection onto its own timer, which
 *     is what the code did before, changes when the freeze happens and not that it happens.
 *
 *     A worker is a second thread. The page keeps drawing at sixty frames a second on top
 *     of boxes the tracker coasts, and detection lands whenever it lands.
 *
 * WHAT CROSSES BETWEEN THEM
 *     One ImageBitmap per detection, transferred rather than copied, so the frame costs
 *     nothing to hand over and is owned by exactly one side at a time. Back comes a small
 *     object: boxes, flame and smoke regions, and a colour signature per person. Everything
 *     expensive stays here.
 */

import { FireScan } from './firescan.js';
import { describe } from './reid.js';
import { TILE_COUNT, merge, tileRegion, toFrame } from './tiles.js';

const SIGNATURE_WIDTH = 320;

/**
 * TILING, AND WHY A CROWD NEEDS IT
 *
 * The detector's input is 448 pixels square. A 1920-wide frame is squeezed into that, so a
 * person forty pixels tall in the original arrives nine pixels tall at the model. Nine
 * pixels is below what any detector can find, which is why a crowd shot returns five people
 * and not fifty: they are not being missed by a threshold, they are not being shown to the
 * model at all.
 *
 * So the frame is also cut into a grid and each piece is detected on its own, at its own
 * resolution. A sixth of a 1920-wide frame is 640 across, which the model squeezes by 1.4
 * rather than 4.3, and that same person now arrives at twenty-eight pixels. Same model,
 * same weights, three times the size on the thing being looked for.
 *
 * One tile per pass, cycling, rather than all six at once. Six detections in a row would be
 * six times the latency, and the full-frame pass that runs every time is what keeps every
 * track alive between them - the tiles only add the small distant people the full frame
 * cannot resolve. Over a couple of seconds every part of the frame has been looked at
 * closely, and the boxes never stop moving in the meantime.
 */
let tileIndex = 0;
let tileCanvas = null;
let tileContext = null;

let detector = null;
let fire = new FireScan();
let base = '';
let modelPath = '';
let delegate = 'CPU';
let scoreThreshold = 0.35;
let lastInferenceMs = 0;

const KEEP = new Set([
  'person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck', 'boat', 'train', 'airplane',
]);

/** A canvas for the signatures, reused across frames. */
let signatureCanvas = null;
let signatureContext = null;

self.onmessage = async (event) => {
  const message = event.data;
  try {
    if (message.type === 'configure') {
      base = message.base;
      modelPath = message.model;
      delegate = message.delegate ?? 'CPU';
      scoreThreshold = message.scoreThreshold ?? 0.35;
      self.postMessage({ type: 'configured' });
      return;
    }

    if (message.type === 'warm') {
      await load();
      self.postMessage({ type: 'ready' });
      return;
    }

    if (message.type === 'reset') {
      fire = new FireScan();
      return;
    }

    if (message.type === 'frame') {
      await handleFrame(message);
      return;
    }
  } catch (error) {
    // The frame is closed here rather than leaked: an ImageBitmap holds real memory and
    // the sender has already given up ownership of it.
    message.bitmap?.close?.();
    self.postMessage({ type: 'error', message: String(error?.message ?? error) });
  }
};

async function load() {
  if (detector) return detector;
  const { FilesetResolver, ObjectDetector } = await import(
    /* @vite-ignore */ `${base}/vision_bundle.mjs`
  );
  const vision = await FilesetResolver.forVisionTasks(`${base}/wasm`);
  detector = await ObjectDetector.createFromOptions(vision, {
    // CPU, for the reason set out at length in ondevice.js: the GPU delegate returns an
    // empty list on some machines, silently, and an empty list is indistinguishable from a
    // frame with nothing in it.
    baseOptions: { modelAssetPath: modelPath, delegate },
    scoreThreshold,
    // Per pass, and there are two passes a frame. Sixty was a number for a scene with a
    // few things in it; a crowd is not that, and a cap is a count that silently stops
    // climbing at a round number.
    maxResults: 300,
    // IMAGE, not VIDEO. VIDEO mode carries state from one call to the next, which is
    // exactly wrong when consecutive calls are different crops of the same frame: it would
    // read a tile as the whole scene having jumped. IMAGE is stateless and needs no
    // monotonic timestamp, which this loop cannot promise anyway.
    runningMode: 'IMAGE',
  });
  return detector;
}

async function handleFrame({ bitmap, scanFire, wantSignatures, tiled }) {
  await load();

  const started = performance.now();

  // The whole frame every time: it is what keeps every track alive, and it finds anyone
  // large enough to survive the downscale.
  let found = collect(detector.detect(bitmap));

  // Then one tile, at its own resolution, for the people the full frame cannot resolve.
  if (tiled) {
    const tile = nextTile(bitmap);
    if (tile) {
      const fromTile = collect(detector.detect(tile.bitmap))
        .map((finding) => ({ ...finding, box: toFrame(finding.box, tile.region, tile.scaleX, tile.scaleY) }));
      found = merge(found, fromTile);
    }
  }
  lastInferenceMs = performance.now() - started;

  const pixels = readPixels(bitmap, wantSignatures || scanFire);
  if (wantSignatures && pixels) attachSignatures(found, pixels, bitmap);

  let regions = [];
  if (scanFire) {
    const width = bitmap.width;
    const height = bitmap.height;
    const occluders = found.map((d) => [
      d.box[0] / width, d.box[1] / height, d.box[2] / width, d.box[3] / height,
    ]);
    try {
      regions = fire.scan(bitmap, width, height, occluders);
    } catch {
      regions = [];
    }
  }

  bitmap.close();
  self.postMessage({
    type: 'result',
    found,
    regions,
    inferenceMs: lastInferenceMs,
  });
}

/** Turn one detector result into findings, in whatever pixels it was given. */
function collect(result) {
  const found = [];
  for (const detection of result?.detections ?? []) {
    const category = detection.categories?.[0];
    if (!category) continue;
    const label = String(category.categoryName ?? '').toLowerCase();
    if (!KEEP.has(label)) continue;
    const { originX, originY, width, height } = detection.boundingBox;
    found.push({
      label,
      confidence: category.score,
      classId: colourIndex(label),
      box: [originX, originY, originX + width, originY + height],
    });
  }
  return found;
}

/**
 * Crop the next tile in the cycle, at its own resolution.
 *
 * Round robin rather than all at once, so one pass costs one extra detection rather than
 * six. Over a couple of seconds every part of the frame has had a close look, and the
 * full-frame pass that runs every time keeps the tracks alive in between.
 */
function nextTile(bitmap) {
  const region = tileRegion(tileIndex, bitmap.width, bitmap.height);
  tileIndex = (tileIndex + 1) % TILE_COUNT;
  if (region.width < 32 || region.height < 32) return null;

  // Drawn at its own size rather than blown up: the model downscales to its own input
  // whatever it is given, and handing it more pixels than the crop has adds nothing.
  const drawWidth = Math.max(64, Math.round(region.width));
  const drawHeight = Math.max(64, Math.round(region.height));
  if (!tileCanvas || tileCanvas.width !== drawWidth || tileCanvas.height !== drawHeight) {
    tileCanvas = new OffscreenCanvas(drawWidth, drawHeight);
    tileContext = tileCanvas.getContext('2d', { willReadFrequently: false });
  }
  tileContext.drawImage(
    bitmap, region.x, region.y, region.width, region.height, 0, 0, drawWidth, drawHeight,
  );

  return {
    bitmap: tileCanvas,
    region,
    scaleX: region.width / drawWidth,
    scaleY: region.height / drawHeight,
  };
}

/** One small copy of the frame, shared by every signature in it. */
function readPixels(bitmap, wanted) {
  if (!wanted) return null;
  const width = Math.min(SIGNATURE_WIDTH, bitmap.width);
  const height = Math.max(1, Math.round((bitmap.height / bitmap.width) * width));

  if (!signatureCanvas || signatureCanvas.width !== width || signatureCanvas.height !== height) {
    signatureCanvas = new OffscreenCanvas(width, height);
    signatureContext = signatureCanvas.getContext('2d', { willReadFrequently: true });
  }
  signatureContext.drawImage(bitmap, 0, 0, width, height);
  return signatureContext.getImageData(0, 0, width, height);
}

function attachSignatures(found, pixels, bitmap) {
  const scaleX = pixels.width / bitmap.width;
  const scaleY = pixels.height / bitmap.height;
  for (const detection of found) {
    if (detection.label !== 'person') continue;
    detection.signature = describe(pixels.data, pixels.width, pixels.height, [
      detection.box[0] * scaleX, detection.box[1] * scaleY,
      detection.box[2] * scaleX, detection.box[3] * scaleY,
    ]);
  }
}

/** Same hash as everywhere else, so a label keeps one colour across every engine. */
function colourIndex(label) {
  let hash = 0;
  for (let i = 0; i < label.length; i += 1) hash = (hash * 31 + label.charCodeAt(i)) % 4096;
  return hash;
}
