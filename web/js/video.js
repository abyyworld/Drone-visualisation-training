/**
 * Video -> a short list of frames worth analysing.
 *
 * WHY NOT EVERY FRAME
 *     A drone records video, not stills, so accepting a clip is the natural way to use
 *     this. But five minutes at 25fps is 7,500 frames, and consecutive frames of a slow
 *     orbit are the same photograph taken 7,500 times. Sending them all to a provider
 *     would cost hundreds of times what the inspection is worth and would produce hundreds
 *     of copies of the same finding.
 *
 *     So this samples the clip, throws away what is blurred or near-identical to a frame
 *     already kept, and hands back a small set of distinct, sharp views. That is what
 *     "relevant frames" has to mean if the word is to mean anything measurable.
 *
 * HOW A FRAME IS JUDGED
 *     Sharpness: variance of a Laplacian over a greyscale downscale. Motion blur and a
 *     missed focus both flatten local contrast, and the variance falls with it. Compared
 *     within the clip rather than against a fixed number, because what counts as sharp
 *     depends on the camera and the light.
 *
 *     Distinctness: a 64-bit difference hash, the same one training/common/imagehash.py
 *     uses on the datasets. Frames within HASH_DISTANCE bits of one already kept are the
 *     same view; the sharpest of each cluster survives.
 *
 * WHAT THIS CANNOT DO
 *     It does not know which frames contain damage - nothing does until they are analysed.
 *     It selects for *coverage*, not for interest. A defect visible in exactly one blurred
 *     frame of a clip can be dropped by this, and that is a real limit worth stating: a
 *     clip is a screening input, and a still photograph of anything suspicious is better.
 */

import { classify, decodeAdvice } from './formats.js';

const SAMPLE_WIDTH = 64;         // greyscale working size; enough for both measures
const SAMPLE_HEIGHT = 64;
const HASH_DISTANCE = 8;         // bits; below this two frames are the same view
const MIN_SHARPNESS_RATIO = 0.4; // relative to the clip's median, not an absolute number

export const VIDEO_DEFAULTS = { maxFrames: 24, minGapSeconds: 0.5, candidatesPerKeep: 4 };

/** Is this something we should try to decode as video? */
export function isVideo(file) {
  return classify(file) === 'video';
}

function loadVideoElement(file) {
  return new Promise((resolve, reject) => {
    const video = document.createElement('video');
    video.preload = 'auto';
    video.muted = true;
    video.playsInline = true;
    video.src = URL.createObjectURL(file);

    const fail = () => {
      URL.revokeObjectURL(video.src);
      // A browser refuses a codec it cannot decode with an empty error rather than an
      // exception, and the pipeline would otherwise report "0 frames" as though the clip
      // were empty. Name the actual cause, and the fix.
      reject(new Error(decodeAdvice(file)));
    };

    video.addEventListener('error', fail, { once: true });
    video.addEventListener('loadedmetadata', () => {
      if (!Number.isFinite(video.duration) || video.duration <= 0 || !video.videoWidth) {
        fail();
        return;
      }
      resolve(video);
    }, { once: true });
  });
}

function seek(video, time) {
  return new Promise((resolve, reject) => {
    const done = () => { cleanup(); resolve(); };
    const failed = () => { cleanup(); reject(new Error(`Could not seek to ${time.toFixed(2)}s.`)); };
    const cleanup = () => {
      video.removeEventListener('seeked', done);
      video.removeEventListener('error', failed);
      clearTimeout(timer);
    };
    // A seek that never fires 'seeked' hangs the whole batch. Time it out and move on:
    // one skipped frame is not worth a stuck page.
    const timer = setTimeout(failed, 8000);
    video.addEventListener('seeked', done, { once: true });
    video.addEventListener('error', failed, { once: true });
    video.currentTime = Math.min(time, Math.max(0, video.duration - 0.05));
  });
}

/** Greyscale luminance at SAMPLE_WIDTH x SAMPLE_HEIGHT, the input to both measures. */
function greyscale(video, ctx) {
  ctx.drawImage(video, 0, 0, SAMPLE_WIDTH, SAMPLE_HEIGHT);
  const { data } = ctx.getImageData(0, 0, SAMPLE_WIDTH, SAMPLE_HEIGHT);
  const grey = new Float32Array(SAMPLE_WIDTH * SAMPLE_HEIGHT);
  for (let i = 0; i < grey.length; i += 1) {
    const p = i * 4;
    grey[i] = 0.299 * data[p] + 0.587 * data[p + 1] + 0.114 * data[p + 2];
  }
  return grey;
}

/** Variance of the 4-neighbour Laplacian. Higher is sharper. */
function sharpness(grey) {
  const values = [];
  for (let y = 1; y < SAMPLE_HEIGHT - 1; y += 1) {
    for (let x = 1; x < SAMPLE_WIDTH - 1; x += 1) {
      const i = y * SAMPLE_WIDTH + x;
      values.push(
        4 * grey[i] - grey[i - 1] - grey[i + 1] - grey[i - SAMPLE_WIDTH] - grey[i + SAMPLE_WIDTH],
      );
    }
  }
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  return values.reduce((total, v) => total + (v - mean) ** 2, 0) / values.length;
}

/**
 * 64-bit difference hash, as two 32-bit halves.
 *
 * Same construction as dhash() in training/common/imagehash.py: compare each pixel of a
 * 9x8 greyscale to its right-hand neighbour. BigInt is avoided so this stays fast enough
 * to run on every candidate frame.
 */
function dhash(grey) {
  const cell = SAMPLE_WIDTH / 9;
  const row = SAMPLE_HEIGHT / 8;
  const at = (col, line) => grey[
    Math.min(SAMPLE_HEIGHT - 1, Math.floor(line * row + row / 2)) * SAMPLE_WIDTH
    + Math.min(SAMPLE_WIDTH - 1, Math.floor(col * cell + cell / 2))
  ];

  let high = 0;
  let low = 0;
  for (let line = 0; line < 8; line += 1) {
    for (let col = 0; col < 8; col += 1) {
      const bit = at(col, line) > at(col + 1, line) ? 1 : 0;
      if (line < 4) high = (high << 1) | bit;
      else low = (low << 1) | bit;
    }
  }
  return [high >>> 0, low >>> 0];
}

function hamming(a, b) {
  let count = 0;
  for (let half = 0; half < 2; half += 1) {
    let x = (a[half] ^ b[half]) >>> 0;
    while (x) { x &= x - 1; count += 1; }
  }
  return count;
}

/**
 * Sample a clip and return the frames worth analysing.
 *
 * @param {File} file
 * @param {object} options  maxFrames, minGapSeconds, candidatesPerKeep
 * @param {(done:number, total:number, label:string) => void} [onProgress]
 * @returns {Promise<Array<{bitmap: ImageBitmap, time: number, name: string, sharpness: number}>>}
 */
export async function extractFrames(file, options = {}, onProgress = () => {}) {
  const { maxFrames, minGapSeconds, candidatesPerKeep } = { ...VIDEO_DEFAULTS, ...options };
  const video = await loadVideoElement(file);

  try {
    // Look at more positions than we intend to keep, so there is something to choose
    // between: with one candidate per slot, "pick the sharp one" has no meaning.
    const wanted = Math.max(1, Math.floor(video.duration / minGapSeconds));
    const candidateCount = Math.max(1, Math.min(wanted, maxFrames * candidatesPerKeep));
    const step = video.duration / (candidateCount + 1);

    const probe = document.createElement('canvas');
    probe.width = SAMPLE_WIDTH;
    probe.height = SAMPLE_HEIGHT;
    const probeCtx = probe.getContext('2d', { willReadFrequently: true });

    const candidates = [];
    for (let index = 0; index < candidateCount; index += 1) {
      const time = step * (index + 1);
      onProgress(index, candidateCount, `Scanning ${file.name} at ${time.toFixed(1)}s`);
      try {
        await seek(video, time);
      } catch {
        continue; // a frame that will not seek is not worth failing the clip over
      }
      const grey = greyscale(video, probeCtx);
      candidates.push({ time, sharpness: sharpness(grey), hash: dhash(grey) });
    }

    if (!candidates.length) {
      throw new Error(`No frames could be read from ${file.name}.`);
    }

    // Sharpness floor, relative to this clip. An absolute threshold would reject a whole
    // overcast inspection or accept a whole blurred one.
    const ordered = [...candidates].sort((a, b) => a.sharpness - b.sharpness);
    const median = ordered[Math.floor(ordered.length / 2)].sharpness;
    const floor = median * MIN_SHARPNESS_RATIO;

    // Sharpest first, so the frame that represents each cluster is the best of it.
    const kept = [];
    for (const candidate of [...candidates].sort((a, b) => b.sharpness - a.sharpness)) {
      if (kept.length >= maxFrames) break;
      if (candidate.sharpness < floor) continue;
      if (kept.some((k) => hamming(k.hash, candidate.hash) < HASH_DISTANCE)) continue;
      kept.push(candidate);
    }

    // If everything looked alike - a static shot, a very short clip - keep the sharpest
    // rather than returning nothing and reporting the clip as empty.
    if (!kept.length) kept.push(candidates.sort((a, b) => b.sharpness - a.sharpness)[0]);

    kept.sort((a, b) => a.time - b.time);

    const stem = file.name.replace(/\.[^.]+$/, '');
    const frames = [];
    for (const [index, candidate] of kept.entries()) {
      onProgress(index, kept.length, `Extracting frame ${index + 1} of ${kept.length}`);
      await seek(video, candidate.time);

      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      canvas.getContext('2d').drawImage(video, 0, 0);

      frames.push({
        bitmap: await createImageBitmap(canvas),
        time: candidate.time,
        sharpness: candidate.sharpness,
        // Timestamp in the name so a finding can be traced back to a point in the clip.
        name: `${stem}@${candidate.time.toFixed(1)}s.jpg`,
      });
    }

    return { frames, scanned: candidates.length, duration: video.duration };
  } finally {
    URL.revokeObjectURL(video.src);
    video.removeAttribute('src');
    video.load();
  }
}
