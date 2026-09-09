/**
 * YOLO inference and post-processing.
 *
 * Handles both output layouts you can get from an Ultralytics export, because which one you
 * get depends on the model family and the export flags - and a mismatch here produces
 * confident nonsense rather than an error:
 *
 *   [1, 4+nc, N]  classic YOLOv8/YOLO11 head. Needs sigmoid-free class scores, argmax and NMS.
 *   [1, N, 6]     end-to-end / NMS-free export (YOLO26, or `nms=True`). Already decoded:
 *                 x1, y1, x2, y2, confidence, class.
 */

import { getSession } from './runtime.js';
import { letterbox } from './preprocess.js';
import { decodeHead, decodeRows, unletterbox } from './yolo.js';

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

/**
 * Turn one model output into detections, whichever way it was exported.
 *
 * The arithmetic lives in yolo.js because the Android app has to do exactly the same thing
 * to exactly the same numbers, and a decode that disagrees between the two does not raise
 * an error - it draws boxes beside people on one of them, or finds nobody at all, which
 * looks identical to an empty scene.
 */
function decode(output, spec) {
  const dims = output.dims;
  const labels = spec.labels ?? [];
  const options = {
    confThreshold: spec.confThreshold ?? 0.25,
    iouThreshold: spec.iouThreshold ?? 0.45,
    // Dropped before non-max suppression rather than after. A model trained on aerial
    // imagery knows vehicles as well as people, and a van suppressed against a person
    // standing next to it would lose the person - which is the one thing that must not
    // happen. Filtering first means suppression only ever compares people with people.
    keepClasses: spec.keepClasses,
  };

  if (dims.length !== 3) {
    throw new Error(`unsupported model output shape [${dims}]`);
  }

  // A segmentation export puts mask coefficients after the class scores, so its head is
  // 4 + classes + coefficients wide. Those coefficients are not scores and read as scores
  // they are worse than useless: they are unbounded, so on real footage with nothing in it
  // every frame produces a box at over 1.0 "confidence". The model is still a perfectly
  // good detector with them ignored, which is what dropping them here does.
  //
  // Declared in the manifest rather than guessed from the shape. A head that is wider than
  // its label list is exactly as likely to be the wrong model file, and that has to keep
  // throwing.
  const maskCoefficients = spec.maskCoefficients ?? 0;
  let classChannels = dims[1];
  if (dims[2] !== 6 && labels.length) {
    const extra = dims[1] - 4 - labels.length;
    // Either the head is exactly the labels, or it is the labels plus the coefficients the
    // manifest says to expect. Any other width is the wrong file, and stays an error.
    if (extra !== 0 && extra !== maskCoefficients) {
      throw new Error(
        `manifest lists ${labels.length} labels but the model predicts ${dims[1] - 4} `
        + `classes${maskCoefficients ? ` plus ${maskCoefficients} mask coefficients` : ''}`,
      );
    }
    classChannels = 4 + labels.length;
  }

  // End-to-end export: already suppressed, one row per detection.
  const raw = dims[2] === 6
    ? decodeRows(output.data, dims[1], options)
    : decodeHead(output.data, classChannels, dims[2], options);
  return raw.map((d) => ({ ...d, label: labels[d.classId] ?? `class_${d.classId}` }));
}


