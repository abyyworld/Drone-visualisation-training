/**
 * YOLO inference and post-processing.
 *
 * Handles both output layouts you can get from an Ultralytics export, because which one you
 * get depends on the model family and the export flags — and a mismatch here produces
 * confident nonsense rather than an error:
 *
 *   [1, 4+nc, N]  classic YOLOv8/YOLO11 head. Needs sigmoid-free class scores, argmax and NMS.
 *   [1, N, 6]     end-to-end / NMS-free export (YOLO26, or `nms=True`). Already decoded:
 *                 x1, y1, x2, y2, confidence, class.
 */

import { getSession } from './runtime.js';
import { letterbox } from './preprocess.js';

/**
 * Run detection on one image.
 *
 * @param {string} key        Manifest key, used as the session cache key.
 * @param {object} spec       Manifest entry: { file, imgsz, labels, confThreshold, iouThreshold }.
 * @param {string} modelUrl   Resolved URL of the .onnx.
 * @param {ImageBitmap} image
 * @param {(l:number,t:number)=>void} [onProgress] Download progress for first load.
 * @returns {Promise<Array<{label:string, classId:number, confidence:number, box:[number,number,number,number]}>>}
 *   Boxes are `[x1, y1, x2, y2]` in the original image's pixel coordinates.
 */
export async function detect(key, spec, modelUrl, image, onProgress) {
  const { session, ort } = await getSession(key, modelUrl, onProgress);

  const size = spec.imgsz ?? 640;
  const { data, scale, padX, padY } = letterbox(image, size);

  const input = new ort.Tensor('float32', data, [1, 3, size, size]);
  const outputs = await session.run({ [session.inputNames[0]]: input });
  const output = outputs[session.outputNames[0]];

  const raw = decode(output, spec);
  const mapped = raw
    .map((d) => ({
      ...d,
      box: unletterbox(d.box, scale, padX, padY, image.width, image.height),
    }))
    // A box can survive NMS and still be degenerate after clamping to the image edge.
    .filter((d) => d.box[2] - d.box[0] > 1 && d.box[3] - d.box[1] > 1);

  return mapped.sort((a, b) => b.confidence - a.confidence);
}

function decode(output, spec) {
  const dims = output.dims;
  const labels = spec.labels ?? [];
  const confThreshold = spec.confThreshold ?? 0.25;

  // End-to-end export: already NMS'd, one row per detection.
  if (dims.length === 3 && dims[2] === 6) {
    const [, count] = dims;
    const values = output.data;
    const detections = [];
    for (let i = 0; i < count; i += 1) {
      const offset = i * 6;
      const confidence = values[offset + 4];
      if (confidence < confThreshold) continue;
      const classId = Math.round(values[offset + 5]);
      detections.push({
        classId,
        label: labels[classId] ?? `class_${classId}`,
        confidence,
        box: [values[offset], values[offset + 1], values[offset + 2], values[offset + 3]],
      });
    }
    return detections;
  }

  // Classic head: [1, 4+nc, anchors], channel-major.
  if (dims.length !== 3) {
    throw new Error(`unsupported model output shape [${dims}]`);
  }
  const [, channels, anchors] = dims;
  const numClasses = channels - 4;
  if (numClasses < 1) {
    throw new Error(`model output [${dims}] has no class channels`);
  }
  if (labels.length && labels.length !== numClasses) {
    throw new Error(
      `manifest lists ${labels.length} labels but the model predicts ${numClasses} classes`,
    );
  }

  const values = output.data;
  const detections = [];

  for (let i = 0; i < anchors; i += 1) {
    let bestScore = 0;
    let bestClass = -1;
    for (let c = 0; c < numClasses; c += 1) {
      const score = values[(4 + c) * anchors + i];
      if (score > bestScore) {
        bestScore = score;
        bestClass = c;
      }
    }
    if (bestClass < 0 || bestScore < confThreshold) continue;

    const cx = values[i];
    const cy = values[anchors + i];
    const w = values[anchors * 2 + i];
    const h = values[anchors * 3 + i];

    detections.push({
      classId: bestClass,
      label: labels[bestClass] ?? `class_${bestClass}`,
      confidence: bestScore,
      box: [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2],
    });
  }

  return nms(detections, spec.iouThreshold ?? 0.45);
}

/** Greedy per-class non-maximum suppression. */
function nms(detections, iouThreshold) {
  const kept = [];

  for (const classId of new Set(detections.map((d) => d.classId))) {
    const candidates = detections
      .filter((d) => d.classId === classId)
      .sort((a, b) => b.confidence - a.confidence);

    while (candidates.length) {
      const best = candidates.shift();
      kept.push(best);
      for (let i = candidates.length - 1; i >= 0; i -= 1) {
        if (iou(best.box, candidates[i].box) > iouThreshold) candidates.splice(i, 1);
      }
    }
  }

  return kept;
}

function iou(a, b) {
  const x0 = Math.max(a[0], b[0]);
  const y0 = Math.max(a[1], b[1]);
  const x1 = Math.min(a[2], b[2]);
  const y1 = Math.min(a[3], b[3]);

  const width = x1 - x0;
  const height = y1 - y0;
  if (width <= 0 || height <= 0) return 0;

  const intersection = width * height;
  const areaA = (a[2] - a[0]) * (a[3] - a[1]);
  const areaB = (b[2] - b[0]) * (b[3] - b[1]);
  const union = areaA + areaB - intersection;
  return union > 0 ? intersection / union : 0;
}

/** Undo the letterbox transform, back to original-image pixels, clamped to bounds. */
function unletterbox(box, scale, padX, padY, width, height) {
  return [
    Math.max(0, (box[0] - padX) / scale),
    Math.max(0, (box[1] - padY) / scale),
    Math.min(width, (box[2] - padX) / scale),
    Math.min(height, (box[3] - padY) / scale),
  ];
}
