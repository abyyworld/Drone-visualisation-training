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

import { similarity } from './reid.js';

// Deliberately low. Detection runs a few times a second, not sixty, so a person walking
// normally can move most of their own width between two looks - and at IoU 0.25 that was
// enough to lose them and issue a new number, which is what "person #1, then #2, then #3
// while standing there moving" turns out to mean.
const MIN_IOU = 0.08;

// Centre-distance fallback, as a multiple of the box's own size. Two boxes that do not
// overlap at all are still obviously the same person if the second is half a body-width
// from the first and the same size. IoU alone cannot see that; this can.
const MAX_CENTRE_DRIFT = 1.6;
const MAX_SIZE_RATIO = 2.2;

// About four seconds at a few detections a second. Long enough to walk behind something.
const MAX_MISSES = 20;
/**
 * Sightings before a track is given a number and added to the total.
 *
 * Raised from two, on evidence. Measured against VisDrone's own labels, about three
 * boxes in ten do not land on a labelled person: street furniture, mostly, which from
 * above is a small dark blob like everything else. At two sightings any of those that
 * survived a second look was issued a number and added to the total, so the total
 * climbed on things that were not people and the numbers on screen churned.
 *
 * Four is a second and a bit of agreeing with itself. It does not fix the false boxes,
 * which is a limit of the model rather than of the tracking, but it stops them being
 * counted as people, and the cost is that somebody who crosses the frame very fast is
 * drawn a moment later.
 */
const CONFIRM_AFTER = 4;

/**
 * How long a track may coast, in milliseconds, before it is let go.
 *
 * Frames were the wrong unit and this is the bug it caused. Twenty missed frames is about
 * two and a half seconds at the eight detections a second a laptop manages, which is a
 * reasonable time to hold a box for someone who stepped behind a pillar. On a tablet
 * managing 1.4 a second it is fourteen seconds, and a spurious detection sits on screen,
 * dashed and drifting, for a quarter of a minute after everyone has agreed it was nothing.
 *
 * A second and a half is a person walking behind something and out the other side. Anything
 * longer is a box describing the past.
 */
const MAX_COAST_MS = 4000;
const IN_VIEW_MS = 1200;

/**
 * How alike two colour signatures must be to be the same person coming back.
 *
 * Histogram intersection weighted across three bands, so this is on a scale where the same
 * person in the same clothes under changing light lands around 0.7 to 0.9, and two
 * different people in different clothes land around 0.2 to 0.5. Sixty-two is inside that
 * gap and nearer the lower half of it, which is the deliberate direction: a missed match
 * counts someone twice and inflates the total, and a wrong match merges two people and
 * deflates it. The count is already a floor, so deflating it keeps it honest and inflating
 * it does not.
 */
const REID_SIMILARITY = 0.62;

/**
 * How long someone stays recognisable after leaving the frame.
 *
 * Five minutes is a route leg: long enough to fly past a group, turn, and come back over
 * the same people without counting them again. Beyond that a match on clothing colour
 * alone is not worth much - the light has moved, and so have they.
 */
const REID_WINDOW_MS = 5 * 60 * 1000;

/**
 * How many people can be remembered at once.
 *
 * This is not a detail: it is a ceiling on how many distinct people a flight can count.
 * Once it is full the oldest are forgotten, and a forgotten person who walks back into
 * frame is counted a second time. Two hundred and forty was far too low for a crowd.
 *
 * Two thousand signatures is about one and a half megabytes, which is affordable even on a
 * two-gigabyte controller, and the search is a scan of an array of floats.
 */
const REID_MAX_REMEMBERED = 2000;

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

/**
 * Fold a fresh signature into the one held, weighted towards what is held.
 *
 * A running average rather than a replacement. One frame of someone half behind a railing
 * is a bad description of them, and replacing outright would let that frame become their
 * identity; averaging lets it contribute and be outvoted.
 */
function blend(held, fresh) {
  const merged = new Float32Array(held.length);
  for (let i = 0; i < held.length; i += 1) merged[i] = held[i] * 0.8 + fresh[i] * 0.2;
  return merged;
}

function centre(box) {
  return [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2];
}

function sizeOf(box) {
  return [Math.abs(box[2] - box[0]), Math.abs(box[3] - box[1])];
}

/**
 * How strongly a detection belongs to a track. Zero means it does not.
 *
 * Overlap first, because when boxes overlap that is the better evidence. When they do not,
 * fall back to how far the centre has moved relative to the size of the thing - a person
 * who has stepped one body-width sideways between two looks is still that person, and a
 * detector running five times a second sees exactly that. The fallback also checks the
 * boxes are a similar size, so a distant person is not adopted by a nearby one's track.
 *
 * The track's own velocity is used to guess where it should be, so something moving
 * steadily is matched against its predicted position rather than its last one.
 */
function affinity(track, box, minIou) {
  const predicted = [
    track.box[0] + track.velocity[0], track.box[1] + track.velocity[1],
    track.box[2] + track.velocity[0], track.box[3] + track.velocity[1],
  ];

  const overlap = Math.max(iou(track.box, box), iou(predicted, box));
  if (overlap >= minIou) return 1 + overlap;   // always beats any distance-only match

  const [tw, th] = sizeOf(track.box);
  const [dw, dh] = sizeOf(box);
  if (tw <= 0 || th <= 0 || dw <= 0 || dh <= 0) return 0;

  const ratio = Math.max(tw / dw, dw / tw, th / dh, dh / th);
  if (ratio > MAX_SIZE_RATIO) return 0;

  const [px, py] = centre(predicted);
  const [bx, by] = centre(box);
  const drift = Math.hypot(px - bx, py - by) / Math.max(1, Math.hypot(tw, th) / 2);
  if (drift > MAX_CENTRE_DRIFT) return 0;

  // Closer is better, and never reaches the overlap band above.
  return 1 - drift / MAX_CENTRE_DRIFT;
}

export class Tracker {
  constructor({
    minIou = MIN_IOU,
    maxMisses = MAX_MISSES,
    confirmAfter = CONFIRM_AFTER,
    maxCoastMs = MAX_COAST_MS,
    reidSimilarity = REID_SIMILARITY,
    reidWindowMs = REID_WINDOW_MS,
  } = {}) {
    this.minIou = minIou;
    this.maxMisses = maxMisses;
    this.confirmAfter = confirmAfter;
    this.maxCoastMs = maxCoastMs;
    this.reidSimilarity = reidSimilarity;
    this.reidWindowMs = reidWindowMs;
    /**
     * People this tracker has seen and lost, so it knows them when they come back.
     * Each entry is {id, label, signature, lastSeen, counted}.
     */
    this.remembered = [];
    this.tracks = [];
    this.nextId = 1;
    this.frame = 0;
    this.everSeen = new Map();
  }

  /**
   * Advance one frame.
   *
   * @param {Array<{label:string, confidence:number, box:number[]}>} detections
   * @returns {Array} the open tracks, coasted ones included
   */
  update(detections = [], now = Date.now()) {
    this.frame += 1;
    this.now = now;
    const pairs = [];

    // Every plausible pairing, best first. Only a track and a detection of the same class
    // may pair: a car becoming a person between two frames is not a thing that happens, and
    // allowing it lets an identity jump across the frame.
    for (const track of this.tracks) {
      for (const [index, detection] of detections.entries()) {
        if (track.label !== detection.label) continue;
        const score = affinity(track, detection.box, this.minIou);
        if (score > 0) pairs.push({ track, index, score });
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
      const observedCentre = centre(detection.box);

      pair.track.box = detection.box;
      pair.track.confidence = detection.confidence;
      pair.track.certainty = detection.certainty ?? pair.track.certainty;
      pair.track.missed = 0;
      pair.track.seen += 1;
      pair.track.lastFrame = this.frame;
      pair.track.lastSeenAt = now;
      if (detection.signature) {
        // Kept current rather than frozen at first sight: someone turning around, or the
        // light changing as the drone moves, should update what they look like now.
        pair.track.signature = pair.track.signature
          ? blend(pair.track.signature, detection.signature)
          : detection.signature;
      }
      // Smoothed, so one noisy frame does not send the coasting prediction sideways.
      const observed = [
        observedCentre[0] - previous[0],
        observedCentre[1] - previous[1],
      ];
      pair.track.velocity = [
        pair.track.velocity[0] * 0.6 + observed[0] * 0.4,
        pair.track.velocity[1] * 0.6 + observed[1] * 0.4,
      ];
      pair.track.path.push(observedCentre);
      if (pair.track.path.length > 60) pair.track.path.shift();
    }

    for (const [index, detection] of detections.entries()) {
      if (usedDetections.has(index)) continue;

      // Before issuing a new number, ask whether this is someone already known. A track
      // that closed because its subject walked behind something is not a different person
      // when they walk out the other side, and giving them a second number is what turned
      // a count of people into a count of reappearances.
      const known = this.recognise(detection, now);
      this.tracks.push({
        id: known ? known.id : this.nextId++,
        label: detection.label,
        confidence: detection.confidence,
        certainty: detection.certainty ?? null,
        box: detection.box,
        classId: detection.classId ?? 0,
        signature: detection.signature ?? known?.signature ?? null,
        seen: 1,
        missed: 0,
        // Carried across, and this is the whole point: someone already counted is not
        // counted again when they come back.
        counted: known ? known.counted : false,
        returned: Boolean(known),
        firstFrame: this.frame,
        lastFrame: this.frame,
        lastSeenAt: now,
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

    // Counted once, at the moment a track becomes confirmed - not while it is a one-frame
    // flicker, and not again on every frame after.
    for (const track of this.tracks) {
      if (track.seen === this.confirmAfter && !track.counted) {
        track.counted = true;
        this.everSeen.set(track.label, (this.everSeen.get(track.label) ?? 0) + 1);
      }
    }

    // Both bounds, and the time one is the one that matters. See MAX_COAST_MS.
    const keeping = [];
    for (const track of this.tracks) {
      if (track.missed <= this.maxMisses && now - track.lastSeenAt <= this.maxCoastMs) {
        keeping.push(track);
      } else {
        this.remember(track, now);
      }
    }
    this.tracks = keeping;
    return this.open();
  }

  /**
   * Is this detection someone the tracker has already seen and let go?
   *
   * Only ever consulted for a brand new track. A detection that matched an open track is
   * that track, and no amount of colour similarity should be able to overrule a box that is
   * where the last one was.
   */
  recognise(detection, now) {
    if (!detection.signature) return null;

    let best = null;
    let bestScore = this.reidSimilarity;
    for (const entry of this.remembered) {
      if (entry.label !== detection.label) continue;
      if (now - entry.lastSeen > this.reidWindowMs) continue;
      const score = similarity(entry.signature, detection.signature);
      if (score >= bestScore) {
        bestScore = score;
        best = entry;
      }
    }
    if (best) {
      // Taken out of the gallery: they are being tracked again, and leaving a copy behind
      // would let a second person match the same identity while the first still holds it.
      this.remembered = this.remembered.filter((entry) => entry !== best);
    }
    return best;
  }

  /** Put a closing track into the gallery, so it can be recognised later. */
  remember(track, now) {
    if (!track.signature || !track.counted) return;

    this.remembered.push({
      id: track.id,
      label: track.label,
      signature: track.signature,
      lastSeen: now,
      counted: true,
    });
    // Oldest out first. A gallery that grows without limit turns every new detection into
    // a linear scan of the whole flight.
    if (this.remembered.length > REID_MAX_REMEMBERED) {
      this.remembered.splice(0, this.remembered.length - REID_MAX_REMEMBERED);
    }
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

  /**
   * How many of a class are in view right now.
   *
   * Being held and being visible are different questions, and one number used to answer
   * both. A track coasts for seconds because a close look at any one part of the frame comes
   * round only every few passes, and holding the identity across that gap is the whole point.
   * Counting those as present would report a crowd that has already walked off.
   */
  countOf(label, now = this.now ?? Date.now()) {
    return this.tracks.filter((t) => t.label === label
      && t.seen >= this.confirmAfter
      && now - t.lastSeenAt <= IN_VIEW_MS).length;
  }

  /**
   * How many distinct things of a class have been seen since the start.
   *
   * A running total that never goes down, counted from the identities issued rather than
   * from the boxes on screen - so somebody who walks through and leaves is counted once,
   * and somebody who stands still is not counted again every frame. It is the number worth
   * reporting after a pass over a site, and it is not the same as how many are in view.
   */
  countSeen(label) {
    return this.everSeen.get(label) ?? 0;
  }

  reset() {
    this.tracks = [];
    this.remembered = [];
    this.nextId = 1;
    this.frame = 0;
    this.everSeen = new Map();
  }
}
