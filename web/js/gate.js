/**
 * Domain gate - decides which subject an upload is, or refuses it.
 *
 * This exists because a detector has no way to say "that's a cat". Shown an out-of-domain
 * image it will still emit boxes, often confident ones, and reporting those as findings
 * would be worse than useless. A classifier in front of the detectors is the cheap, honest
 * way to refuse.
 *
 * It is not hardcoded to a subject list. The classes come from the manifest and are checked
 * against the model's own output width, so adding a subject is a manifest change and a
 * retrained gate, not an edit here.
 *
 * Rejection is deliberately conservative: an image is only routed to a detector when the
 * gate is confident *and* the winning class is a real domain. Everything else is refused
 * with a message, including the ambiguous middle ground.
 */

import { getSession } from './runtime.js';
import { centerCrop } from './preprocess.js';

export const VERDICT = {
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

/**
 * Why an image was refused, naming the subjects that are actually available.
 *
 * Built from the deployed list rather than written into a sentence: this message told
 * people to upload a turbine or a solar panel for a while after crowd and wildfire were
 * added, which is the app describing a version of itself that no longer exists.
 */
export function rejectionMessage(verdict, confidence, subjects = []) {
  const named = list(subjects);
  if (verdict === VERDICT.UNCERTAIN) {
    return `Not confident this is ${named} (best guess ${(confidence * 100).toFixed(0)}%). `
      + 'Upload a clearer image, or choose the subject yourself above.';
  }
  return `This does not look like ${named}.`;
}

/** "a, b or c" - so the message reads as a sentence rather than a config dump. */
function list(subjects) {
  if (!subjects.length) return 'anything this can analyse';
  if (subjects.length === 1) return subjects[0];
  return `${subjects.slice(0, -1).join(', ')} or ${subjects[subjects.length - 1]}`;
}
