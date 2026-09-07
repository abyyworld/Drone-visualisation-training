/**
 * Turns per-frame detections into tracks that keep an identity across frames.
 *
 * WHY A TRACKER AND NOT JUST BOXES
 *     A detector answers "what is in this frame" and forgets. Run it on video and the same
 *     person is a new box sixty times a second, boxes flicker out whenever they are missed
 *     for one frame, and nothing can be said about movement - which is most of what matters
 *     when watching a crowd or a fire. A tracker gives each thing a number that follows it,
 *     so it can be counted once, its path drawn, and its disappearance noticed.
 *
 * HOW
 *     Greedy IoU association, nearest first, plus a short memory. Each frame:
 *       - every detection is matched to the open track it overlaps most, above MIN_IOU
 *       - unmatched detections start a new track
 *       - unmatched tracks coast on their last box for MAX_MISSES frames before closing
 *
 *     Coasting is what stops a box vanishing because the model blinked, and it is why a
 *     track carries `missed`: a coasted box is a prediction, not an observation, and the
 *     renderer draws it differently.
 *
 * WHY NOT SORT OR BYTETRACK
 *     Both are better. Both are also a Kalman filter and a Hungarian solver for a problem
 *     that here involves at most a few dozen boxes a frame from a camera that mostly hovers.
 *     Greedy IoU is a hundred lines with no dependency and no licence to inherit - the
 *     original SORT is GPL-3.0, which is not a thing to pull into a product by accident.
 *     If the drone starts moving fast enough for identities to swap, the upgrade path is to
 *     replace `associate` and nothing else.
 */

const MIN_IOU = 0.25;
const MAX_MISSES = 12;
const CONFIRM_AFTER = 2;

/** Intersection over union of two [x0, y0, x1, y1] boxes. */
export function iou(a, b) {
  const x0 = Math.max(a[0], b[0]);
  const y0 = Math.max(a[1], b[1]);
  const x1 = Math.min(a[2], b[2]);
  const y1 = Math.min(a[3], b[3]);
  if (x1 <= x0 || y1 <= y0) return 0;

  const overlap = (x1 - x0) * (y1 - y0);
  const areaA = Math.max(0, a[2] - a[0]) * Math.max(0, a[3] - a[1]);
  const areaB = Math.max(0, b[2] - b[0]) * Math.max(0, b[3] - b[1]);
  const union = areaA + areaB - overlap;
  return union <= 0 ? 0 : overlap / union;
}

function centre(box) {
  return [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2];
}

export class Tracker {
  constructor({ minIou = MIN_IOU, maxMisses = MAX_MISSES, confirmAfter = CONFIRM_AFTER } = {}) {
    this.minIou = minIou;
    this.maxMisses = maxMisses;
    this.confirmAfter = confirmAfter;
    this.tracks = [];
    this.nextId = 1;
    this.frame = 0;
  }

  /**
   * Advance one frame.
   *
   * @param {Array<{label:string, confidence:number, box:number[]}>} detections
   * @returns {Array} the open tracks, coasted ones included
   */
  update(detections = []) {
    this.frame += 1;
    const pairs = [];

    // Every plausible pairing, best overlap first. Only a track and a detection of the same
    // class may pair: a car becoming a person between two frames is not a thing that
    // happens, and allowing it lets an identity jump across the frame.
    for (const track of this.tracks) {
      for (const [index, detection] of detections.entries()) {
        if (track.label !== detection.label) continue;
        const score = iou(track.box, detection.box);
        if (score >= this.minIou) pairs.push({ track, index, score });
      }
    }
    pairs.sort((a, b) => b.score - a.score);

    const usedTracks = new Set();
    const usedDetections = new Set();
    for (const pair of pairs) {
      if (usedTracks.has(pair.track) || usedDetections.has(pair.index)) continue;
      usedTracks.add(pair.track);
      usedDetections.add(pair.index);

      const detection = detections[pair.index];
      const previous = centre(pair.track.box);
      const now = centre(detection.box);

      pair.track.box = detection.box;
      pair.track.confidence = detection.confidence;
      pair.track.certainty = detection.certainty ?? pair.track.certainty;
      pair.track.missed = 0;
      pair.track.seen += 1;
      pair.track.lastFrame = this.frame;
      pair.track.velocity = [now[0] - previous[0], now[1] - previous[1]];
      pair.track.path.push(now);
      if (pair.track.path.length > 60) pair.track.path.shift();
    }

    for (const [index, detection] of detections.entries()) {
      if (usedDetections.has(index)) continue;
      this.tracks.push({
        id: this.nextId++,
        label: detection.label,
        confidence: detection.confidence,
        certainty: detection.certainty ?? null,
        box: detection.box,
        classId: detection.classId ?? 0,
        seen: 1,
        missed: 0,
        firstFrame: this.frame,
        lastFrame: this.frame,
        velocity: [0, 0],
        path: [centre(detection.box)],
      });
    }

    for (const track of this.tracks) {
      if (usedTracks.has(track)) continue;
      track.missed += 1;
      // Coast on the last observed motion. A hovering drone makes this nearly a no-op; a
      // panning one makes it the difference between holding a box and losing it.
      track.box = [
        track.box[0] + track.velocity[0], track.box[1] + track.velocity[1],
        track.box[2] + track.velocity[0], track.box[3] + track.velocity[1],
      ];
    }

    this.tracks = this.tracks.filter((t) => t.missed <= this.maxMisses);
    return this.open();
  }

  /**
   * Tracks worth drawing.
   *
   * A track seen once is as likely to be a flicker as a thing, so it is held back until it
   * has been seen `confirmAfter` times. That is the difference between a steady overlay and
   * one that strobes.
   */
  open() {
    return this.tracks.filter((t) => t.seen >= this.confirmAfter);
  }

  /** How many distinct things of a class have been seen since the start. */
  countOf(label) {
    return this.tracks.filter((t) => t.label === label && t.seen >= this.confirmAfter).length;
  }

  reset() {
    this.tracks = [];
    this.nextId = 1;
    this.frame = 0;
  }
}
