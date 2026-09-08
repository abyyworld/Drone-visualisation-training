/**
 * Decoding a YOLO head, in one place, so the browser and the tablet cannot disagree.
 *
 * WHY THIS IS ITS OWN FILE
 *     The same arithmetic has to run in JavaScript for the web app and in Java for the
 *     Android app, and getting it wrong does not produce an error - it produces confident
 *     nonsense, or silence. Boxes an anchor-stride out land beside people instead of on
 *     them; a transposed layout finds nothing at all and looks exactly like an empty scene.
 *     That is the failure this project keeps having to design against, so the arithmetic
 *     lives here, is tested here, and the Java is compared against it on the same numbers.
 *
 * THE TWO LAYOUTS
 *     Which one an Ultralytics export gives depends on the model family and the flags:
 *
 *       [1, 4+nc, N]  the classic head. Channel-major: every anchor's x sits together,
 *                     then every anchor's y, and so on. Needs argmax over classes,
 *                     centre-form boxes converting to corners, and non-max suppression.
 *       [1, N, 6]     an end-to-end export, already suppressed: x1, y1, x2, y2, score,
 *                     class, one row per detection.
 *
 *     Both are handled, decided by the shape rather than by configuration, because a
 *     configuration that disagrees with the file is a silent wrong answer.
 */

/** Greedy per-class non-maximum suppression. */
export function nms(detections, iouThreshold = 0.45) {
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

export function iou(a, b) {
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

/**
 * Decode a classic YOLO head.
 *
 * @param {ArrayLike<number>} values  the raw output, channel-major
 * @param {number} channels  4 + number of classes
 * @param {number} anchors   how many boxes the head predicts
 * @param {object} options   { confThreshold, iouThreshold, keepClasses }
 * @returns {Array<{classId:number, confidence:number, box:number[]}>}
 *   Boxes as corners, in whatever units the model works in.
 */
export function decodeHead(values, channels, anchors, options = {}) {
  const confThreshold = options.confThreshold ?? 0.25;
  const numClasses = channels - 4;
  if (numClasses < 1) throw new Error(`a head of ${channels} channels has no classes`);

  // A set of the classes worth keeping, or null for all of them. Filtering here rather
  // than afterwards means the suppression below never has to consider a bicycle at all.
  const keep = options.keepClasses ? new Set(options.keepClasses) : null;

  const detections = [];
  for (let i = 0; i < anchors; i += 1) {
    let bestScore = 0;
    let bestClass = -1;
    for (let c = 0; c < numClasses; c += 1) {
      if (keep && !keep.has(c)) continue;
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
      confidence: bestScore,
      box: [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2],
    });
  }

  return nms(detections, options.iouThreshold ?? 0.45);
}

/**
 * Decode an end-to-end export, which has already been suppressed.
 *
 * @param {ArrayLike<number>} values  rows of x1, y1, x2, y2, score, class
 */
export function decodeRows(values, count, options = {}) {
  const confThreshold = options.confThreshold ?? 0.25;
  const keep = options.keepClasses ? new Set(options.keepClasses) : null;

  const detections = [];
  for (let i = 0; i < count; i += 1) {
    const at = i * 6;
    const confidence = values[at + 4];
    if (confidence < confThreshold) continue;
    const classId = Math.round(values[at + 5]);
    if (keep && !keep.has(classId)) continue;
    detections.push({
      classId,
      confidence,
      box: [values[at], values[at + 1], values[at + 2], values[at + 3]],
    });
  }
  return detections;
}

/** Undo a letterbox, back to the original image's pixels, clamped to its bounds. */
export function unletterbox(box, scale, padX, padY, width, height) {
  return [
    Math.max(0, (box[0] - padX) / scale),
    Math.max(0, (box[1] - padY) / scale),
    Math.min(width, (box[2] - padX) / scale),
    Math.min(height, (box[3] - padY) / scale),
  ];
}
