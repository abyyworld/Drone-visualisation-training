/**
 * Recognising someone who has already been counted.
 *
 * THE PROBLEM THIS EXISTS FOR
 *     A tracker follows a thing while it can see it. The moment it cannot - someone walks
 *     behind a van, the drone pans away and back, the model misses them for two seconds -
 *     the track closes. When they reappear they are a new track with a new number, and the
 *     running total goes up by one for a person who was already in it. Fly a route over a
 *     crowd and the count is not a count, it is a tally of reappearances.
 *
 * WHAT IS ACTUALLY COMPARED
 *     What someone is wearing. The box is split into three horizontal bands, roughly head,
 *     torso and legs, and each band becomes a coarse colour histogram. Clothing is the one
 *     thing about a person that a drone camera can see reliably from a distance and that
 *     does not change while they are in the air: faces are a few pixels across at altitude
 *     and gone the moment someone turns around, and a gait needs a clean side view.
 *
 *     The torso band carries the most weight, because that is the largest area of a single
 *     colour on almost anyone.
 *
 * WHAT IT WILL GET WRONG, AND WHERE THAT LEAVES THE NUMBER
 *     Two people in the same dark jacket, in the same light, are the same person to this.
 *     So are one person before and after they take that jacket off, in reverse. It is a
 *     colour signature, not an identity, and at a football match where half the crowd is in
 *     one strip it will merge people who are not the same.
 *
 *     That direction of error is the deliberate one. The count was already a floor - it
 *     counts what the detector marked, and the detector misses anyone small, distant or
 *     overlapping. Merging two people keeps it a floor. Splitting one person into two would
 *     break that, and is the error this file exists to prevent.
 */

/**
 * Head, torso, legs, and the whole box.
 *
 * The fourth band is the drone's. The three spatial bands assume a person is seen roughly
 * side-on, so head is above torso is above legs. From a drone that assumption weakens with
 * every metre of altitude: from directly overhead the bands are shoulders, shoulders and
 * feet, and the same person seen from the front, the side and above puts different clothing
 * in different bands.
 *
 * The whole-box band has no spatial assumption in it at all. It says which colours this
 * person is wearing and in what proportion, which is the part that survives being flown
 * around. It carries the largest single weight for that reason, and the spatial bands are
 * kept because when the view is side-on they are what separates a red coat from red
 * trousers, and the whole-box band alone cannot.
 */
const BANDS = 4;
const WHOLE_BAND = 3;

/** Colour levels per channel: 4 x 4 x 4 gives 64 bins a band, 192 in total. */
const LEVELS = 4;

const BIN_COUNT = LEVELS * LEVELS * LEVELS;
export const SIZE = BANDS * BIN_COUNT;

/**
 * Whole box first, torso second.
 *
 * The whole box is what survives a change of viewing angle; the torso is the largest area
 * of one colour on almost anyone when the angle happens to be side-on.
 */
const BAND_WEIGHTS = [0.12, 0.28, 0.15, 0.45];

/**
 * Below this a box is too few pixels to describe.
 *
 * A person twelve pixels tall is four pixels a band, and a histogram of four pixels matches
 * almost anything. Refusing to describe them is the right answer: they are then tracked but
 * never re-identified, which counts them again if they leave and come back. That is a worse
 * count and an honest one, where a confident match on four pixels is neither.
 */
const MIN_BOX_WIDTH = 6;
const MIN_BOX_HEIGHT = 16;

/**
 * Build a colour signature for one box.
 *
 * @param {Uint8ClampedArray} pixels  RGBA of the frame the box is in
 * @param {number} width
 * @param {number} height
 * @param {number[]} box  [x0, y0, x1, y1] in that frame's pixels
 * @returns {Float32Array|null}  null when the box is too small to say anything about
 */
export function describe(pixels, width, height, box) {
  const x0 = Math.max(0, Math.round(box[0]));
  const y0 = Math.max(0, Math.round(box[1]));
  const x1 = Math.min(width, Math.round(box[2]));
  const y1 = Math.min(height, Math.round(box[3]));
  const boxWidth = x1 - x0;
  const boxHeight = y1 - y0;
  if (boxWidth < MIN_BOX_WIDTH || boxHeight < MIN_BOX_HEIGHT) return null;

  const signature = new Float32Array(SIZE);
  const counts = new Float32Array(BANDS);
  // Three spatial bands over the box, and the fourth spans all of it.
  const bandHeight = boxHeight / WHOLE_BAND;

  for (let y = y0; y < y1; y += 1) {
    const band = Math.min(WHOLE_BAND - 1, Math.floor((y - y0) / bandHeight));
    for (let x = x0; x < x1; x += 1) {
      const at = (y * width + x) * 4;
      // Integer division into LEVELS buckets per channel. Coarse on purpose: a finer
      // histogram is more precise about lighting and less about the person, and the
      // lighting changes as a drone moves while the person does not.
      const r = (pixels[at] * LEVELS) >> 8;
      const g = (pixels[at + 1] * LEVELS) >> 8;
      const b = (pixels[at + 2] * LEVELS) >> 8;
      const bin = (r * LEVELS + g) * LEVELS + b;
      signature[band * BIN_COUNT + bin] += 1;
      counts[band] += 1;
      signature[WHOLE_BAND * BIN_COUNT + bin] += 1;
      counts[WHOLE_BAND] += 1;
    }
  }

  for (let band = 0; band < BANDS; band += 1) {
    const total = counts[band];
    if (total <= 0) continue;
    for (let bin = 0; bin < BIN_COUNT; bin += 1) {
      signature[band * BIN_COUNT + bin] /= total;
    }
  }
  return signature;
}

/**
 * How alike two signatures are, from 0 to 1.
 *
 * Histogram intersection: the fraction of the two distributions that overlaps. It is the
 * standard measure for this and it has the property that matters here - a band that is
 * partly occluded in one view loses its share and no more, rather than poisoning the whole
 * comparison the way a squared distance would.
 */
export function similarity(a, b) {
  if (!a || !b || a.length !== b.length) return 0;

  let total = 0;
  for (let band = 0; band < BANDS; band += 1) {
    let shared = 0;
    const start = band * BIN_COUNT;
    for (let bin = 0; bin < BIN_COUNT; bin += 1) {
      const left = a[start + bin];
      const right = b[start + bin];
      shared += left < right ? left : right;
    }
    total += shared * BAND_WEIGHTS[band];
  }
  return total;
}
