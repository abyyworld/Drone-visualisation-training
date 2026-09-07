/**
 * Domain gate - decides whether an upload is a turbine blade, a solar panel, or neither.
 *
 * This exists because a detector has no way to say "that's a cat". Shown an out-of-domain
 * image it will still emit boxes, often confident ones, and reporting those as inspection
 * findings would be worse than useless. A dedicated 3-class classifier in front of the
 * detectors is the cheap, honest way to refuse.
 *
 * Rejection is deliberately conservative: an image is only routed to a detector when the
 * gate is confident *and* the winning class is a real domain. Everything else is refused
 * with a message, including the ambiguous middle ground.
 */

import { getSession } from './runtime.js';
import { centerCrop } from './preprocess.js';

export const VERDICT = {
  TURBINE: 'turbine',
  SOLAR: 'solar',
  INVALID: 'invalid',
  UNCERTAIN: 'uncertain',
};

/**
 * Classify one image into a domain.
 *
 * @returns {Promise<{verdict:string, confidence:number, scores:Record<string,number>}>}
 */
export async function classify(spec, modelUrl, image, onProgress) {
  const { session, ort } = await getSession('gate', modelUrl, onProgress);

  const size = spec.imgsz ?? 224;
  const { data } = centerCrop(image, size);

  const input = new ort.Tensor('float32', data, [1, 3, size, size]);
  const outputs = await session.run({ [session.inputNames[0]]: input });
  const logits = Array.from(outputs[session.outputNames[0]].data);

  const labels = spec.labels ?? [];
  if (labels.length !== logits.length) {
    throw new Error(
      `gate manifest lists ${labels.length} labels but the model outputs ${logits.length}`,
    );
  }

  const probabilities = softmax(logits);
  const scores = Object.fromEntries(labels.map((label, i) => [label, probabilities[i]]));

  let bestIndex = 0;
  for (let i = 1; i < probabilities.length; i += 1) {
    if (probabilities[i] > probabilities[bestIndex]) bestIndex = i;
  }

  const confidence = probabilities[bestIndex];
  const winner = labels[bestIndex];
  const threshold = spec.minConfidence ?? 0.6;

  let verdict;
  if (winner === VERDICT.INVALID) {
    verdict = VERDICT.INVALID;
  } else if (confidence < threshold) {
    // Confidently nothing and unconfidently something are different failures, and the user
    // deserves to be told which one happened.
    verdict = VERDICT.UNCERTAIN;
  } else {
    verdict = winner;
  }

  return { verdict, confidence, scores };
}

/** Numerically stable softmax - the max subtraction stops large logits overflowing. */
function softmax(logits) {
  const max = Math.max(...logits);
  const exps = logits.map((v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / sum);
}

/** Human-readable explanation for a refused image. */
export function rejectionMessage(verdict, confidence) {
  if (verdict === VERDICT.UNCERTAIN) {
    return `Not confident this is a turbine blade or a solar panel (best guess ${(confidence * 100).toFixed(0)}%). Please upload a clearer inspection image.`;
  }
  return 'Please upload a valid turbine blade or solar panel image.';
}
