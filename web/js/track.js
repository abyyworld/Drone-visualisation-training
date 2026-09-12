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

// Working out how far the whole picture moved between two looks. See estimateDrift.
// The bin is coarse because the answer only has to be good enough to bring a box back
// inside the gate above, and a coarse bin is what makes the vote decisive.
const DRIFT_BIN = 8;
// Counted in distinct tracks agreeing, not in pairs. Two tracks that both shifted by the
// same amount are the camera moving; one track contributing several near-identical offsets
// because the crowd around it is dense is not evidence of anything.
const DRIFT_MIN_VOTES = 2;
const DRIFT_MIN_SHARE = 0.2;
// A ceiling on one cycle's worth of movement. Past this the frames have nothing to do with
// each other and pairing them up would be invention rather than tracking.
const DRIFT_MAX = 400;
// The frame is split this many ways along each axis when working out how it moved. See
// estimateDrift: a turn or a climb does not move the whole picture by one amount, but any
// smooth warp looks like a translation over a small enough piece of it.
const DRIFT_CELLS = 4;

// Pairing something up when it is the only candidate. See unambiguousPairs.
// The runner-up has to be this much further away before a pairing counts as obvious, and
// the score is below every real affinity so these are only ever used on what is left over.
// 1.4, from the geometry rather than from taste. Two people a gap apart, the camera moving
// m per frame: from a track's old position its own new box is m away and its neighbour's is
// gap minus m, so the ratio is (gap - m) / m. It only becomes genuinely ambiguous when the
// camera has moved half the gap, and a threshold of R resolves everything up to gap/(1 + R).
// 1.4 covers camera motion up to 42% of the gap between two people, which is most of the way
// to the point where no rule could be right.
const LONELY_RATIO = 1.4;

/**
 * How sure the detector has to be before a box may start a new identity.
 *
 * WHY STARTING AND CONTINUING ARE DIFFERENT QUESTIONS
 *     A weak box in a place nothing is being tracked is probably a bin. The same weak box
 *     landing where somebody already is, is almost certainly that person, seen badly for a
 *     moment. One number should not answer both.
 *
 *     So a detection below this can keep an existing track alive but can never create one.
 *     That lets the detector be run at a low threshold without letting faint rubbish into
 *     the count: measured against VisDrone's labels, boxes that land on a labelled person
 *     average 0.465 and the rest average 0.330, which overlaps far too much to threshold in
 *     one frame and separates cleanly once a track has to keep earning it.
 *
 *     This is ByteTrack's association, which does the same thing for the same reason: match
 *     the confident boxes first, then offer what is left to the tracks that are still
 *     looking, and require more of a box that wants to be somebody new.
 */
const NEW_TRACK_CONFIDENCE = 0.25;
const LONELY_SCORE = 1e-4;

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
 * Three, with the weak boxes above now able to keep a track alive between good looks.
 * Measured over nine synthetic flights across a labelled frame, 1388 people between them,
 * with the real detector run on every rendered frame so its misses and false positives are
 * all present:
 *
 *     new-track 0.25, three sightings    707 people counted, 69% of boxes on a person
 *     new-track 0.25, four sightings     691 people counted, 69%
 *     new-track 0.30, four sightings     533 people counted, 72%
 *
 * against 513 at 72% for what shipped before any of this. The extra sighting was buying
 * almost nothing once a weak box could carry a track through a bad frame, and it was
 * costing people who are only ever seen briefly.
 *
 * AND BACK TO FOUR, UNDER THE CADENCE THAT REPLACED THAT ONE
 *     The measurement above was taken when the detector looked at one sixth of the frame per
 *     cycle. Three sightings then meant waiting three rounds of six tiles, four and a half
 *     seconds, so a fourth sighting really did cost people who are only ever seen briefly.
 *
 *     The whole frame is now looked at every cycle, so a sighting is a cycle and four of
 *     them is one second. Measured over four crowded frames at the real cadence, charging
 *     every number to the person it spent the most frames on:
 *
 *         three sightings   49% of the people reached, 1.50 numbers each, 150 on nobody
 *         four              49% reached, 1.46 each, 149 on nobody
 *         five              48% reached, 1.44 each, 148 on nobody
 *
 *     Four costs no reach at all and takes a number off roughly one person in twenty five.
 *     Five starts costing people. The trade the old comment describes has not changed sign;
 *     the cadence moved underneath it.
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
 *
 * WHY A SHORT COAST IS NOW THE BETTER POLICY
 *     This was 4000, then 1200, and both were reasoned from "how long might somebody be out of
 *     sight". That is the wrong question once re-identification works.
 *
 *     Measured on five real VisDrone MOT sequences with colour signatures switched on - which
 *     is the first time they have been measured with the tracker's gallery actually running:
 *
 *         coast   people reached   numbers each   numbers on nobody
 *         4000         74%            1.88              232
 *         2000         72%            1.71              215
 *         1200         68%            1.63              176
 *          800         68%            1.57              155
 *
 *     Holding a box longer buys people and costs numbering, which is the trade the old values
 *     were picked on. But the gallery changes what a release costs: a track let go is
 *     REMEMBERED, and somebody who walks back into view is recognised and gets their own number
 *     back rather than a new one. So letting go early is nearly free, while coasting is not -
 *     a stale box drifts, and a drifting box wins the detection belonging to whoever is
 *     standing where it drifted to. That person then misses, coasts, and is renumbered.
 *
 *     Three missed looks at the target period. The safety net is the gallery, not the coast.
 *
 * WHY IT IS 1200 AND WAS 4000
 *     4000 was not chosen, it was forced. Under the old shape the detector looked at one
 *     sixth of the frame per cycle, so a person outside the current tile was not observed
 *     for six cycles running and a track that let go after a second and a half would drop
 *     them and renumber them on the next look. The window was stretched until that stopped
 *     happening, and the prose right above it went on saying a second and a half, because
 *     that is what it should be.
 *
 *     Every person is now looked at every cycle, so five missed looks in a row is five
 *     failures to detect somebody being looked straight at - which is a person who left,
 *     not a person waiting their turn. Measured over four crowded frames at the real
 *     cadence, dropping 4000 to 1200 moves the share of drawn boxes that are actually on a
 *     person from 74.7% to 80.8%, for two extra numbers issued across three flights. A
 *     coasting box IS the box sitting in the old place, so this is the same complaint the
 *     tile change answers, met from the other side.
 */
const MAX_COAST_MS = 800;

/**
 * How recently a track must have been seen to count as being in view now.
 *
 * Holding an identity and being visible are two different questions, and one number was
 * answering both. This has to stay BELOW the coast window or the two questions collapse back
 * into one: a track would stop being reported as present at the same instant it is let go,
 * and the window where somebody is held without being counted as present - the window that
 * lets them keep their number through a couple of missed looks - would be empty.
 *
 * 500 ms is two cycles at the target period. Somebody not detected for two looks running is
 * not in view, whatever is still being held for them. It came down with the coast window: at
 * 750 against a coast of 800 the gap was smaller than a single cycle, which is the collapse
 * this note warns about, arriving by the back door.
 */
const IN_VIEW_MS = 500;

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
// Measured, at last. 0.62 was reasoned and never measured: with no signatures in any
// harness the comparison never ran, so every value scored identically. On five real
// VisDrone MOT sequences, with the gallery actually running:
//
//     similarity   people reached   numbers each   numbers on nobody
//     0.80              70%             1.70              215
//     0.70              69%             1.67              184
//     0.62              68%             1.63              176
//     0.55              67%             1.62              169
//     0.45              67%             1.60              165
//
// Alone it is marginal. With the shorter coast it is not: together they give 1.55 numbers
// per person and 144 on nobody, against 1.63 and 176, at the same people reached. A shorter
// coast releases tracks sooner, which asks the gallery more questions, which makes how
// readily it says yes matter more than it used to.
//
// Lower is not free. 0.45 keeps improving the count because it starts merging people, and a
// crowd counted as fewer than are in it is a worse answer than one counted twice.
const REID_SIMILARITY = 0.55;

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

function shift(box, by) {
  return [box[0] + by[0], box[1] + by[1], box[2] + by[0], box[3] + by[1]];
}

/**
 * How far the whole picture moved, worked out before anything is matched.
 *
 * WHY THIS IS NOT OPTIONAL
 *     The camera is on a drone. When it moves, every box in the frame moves at once, by the
 *     same amount, and none of the people did anything. From altitude a person is about ten
 *     pixels across, so the centre-distance gate below is worth roughly seventeen pixels,
 *     and a drone in forward flight covers far more than that between two looks a quarter of
 *     a second apart.
 *
 *     So every track fails its gate on the same frame, every one of them is let go, and
 *     everybody in the crowd is issued a new number. Measured on a labelled frame panned
 *     under the tracker, with the dataset's own boxes so the detector cannot be blamed: a
 *     still camera numbers 140 people as 140, and the same crowd under a moving camera comes
 *     out as 320. That is the numbering complaint, entirely.
 *
 *     It cannot be recovered from a track's own velocity either, and that is the trap. A
 *     track needs to survive a frame to learn how fast it is moving, and at these speeds
 *     none of them survive one, so none of them ever learn. The estimate has to come from
 *     the boxes themselves, before any of them are paired up.
 *
 * HOW
 *     Every track against every detection of its class, each pair offering the offset that
 *     would join them, and the offsets voted into coarse bins. A rigid translation puts one
 *     vote per person into the same bin; wrong pairings scatter across all the others. The
 *     winning bin is the drone's own motion, and the median inside it is the value.
 *
 *     A vote per pair is arithmetic on a few thousand numbers even for a full crowd, which
 *     is nothing beside the inference that produced the boxes.
 */
function estimateDrift(tracks, detections) {
  if (tracks.length < 2 || detections.length < 2) return [0, 0];

  // Keyed by binned offset. Each bucket keeps the offsets that landed in it and the set of
  // tracks that put them there, and it is the second of those that decides the winner.
  const votes = new Map();
  let best = null;
  for (const track of tracks) {
    const [tw, th] = sizeOf(track.box);
    if (tw <= 0 || th <= 0) continue;
    const [tx, ty] = centre(track.box);

    for (const detection of detections) {
      if (track.label !== detection.label) continue;
      const [dw, dh] = sizeOf(detection.box);
      if (dw <= 0 || dh <= 0) continue;
      // Same reasoning as the size gate in affinity: a box twice the size is a different
      // person, and its offset is not evidence about where the picture went.
      if (Math.max(tw / dw, dw / tw, th / dh, dh / th) > MAX_SIZE_RATIO) continue;

      const [cx, cy] = centre(detection.box);
      const dx = cx - tx;
      const dy = cy - ty;
      if (Math.abs(dx) > DRIFT_MAX || Math.abs(dy) > DRIFT_MAX) continue;

      const key = `${Math.round(dx / DRIFT_BIN)},${Math.round(dy / DRIFT_BIN)}`;
      let bucket = votes.get(key);
      if (!bucket) votes.set(key, bucket = { offsets: [], tracks: new Set() });
      bucket.offsets.push([dx, dy]);
      bucket.tracks.add(track);
      if (!best || bucket.tracks.size > best.tracks.size) best = bucket;
    }
  }

  // Enough of the frame has to agree, or this is not a translation, it is coincidence.
  if (!best || best.tracks.size < DRIFT_MIN_VOTES
    || best.tracks.size < tracks.length * DRIFT_MIN_SHARE) return [0, 0];

  const median = (values) => {
    const sorted = values.slice().sort((a, b) => a - b);
    return sorted[Math.floor(sorted.length / 2)];
  };
  return [median(best.offsets.map((o) => o[0])), median(best.offsets.map((o) => o[1]))];
}

/**
 * Pairings that are obvious because there is nothing else they could be.
 *
 * WHY THE GATES ABOVE ARE NOT ENOUGH
 *     The centre-distance gate exists to stop an identity jumping to a *competing*
 *     candidate. With one person in the frame there is no competitor, so it is guarding
 *     against a risk that is not there and refusing the only sensible answer.
 *
 *     And the drift estimate cannot help, because it works by vote: a crowd fills a bin and
 *     one person casts one vote, which is refused as coincidence. So the sparse case had
 *     nothing at all. Measured, one person about ten pixels across with the camera moving
 *     twenty pixels a frame: every frame started a fresh track, none survived long enough to
 *     be confirmed, and the screen showed a number that changed constantly or no box at all.
 *
 * WHAT MAKES A PAIRING OBVIOUS
 *     They are each other's nearest, and the runner-up on both sides is far enough behind to
 *     leave no real doubt. That is the ratio test used for matching image features, and it
 *     says exactly the right thing here: distance alone is a poor reason to refuse a match,
 *     but distance *relative to the next best candidate* is a good one.
 *
 *     In a crowd the runner-up is close, the ratio fails, and this does nothing - the drift
 *     vote already has that case. In an empty scene there is no runner-up and this does all
 *     of the work. The size check and the DRIFT_MAX ceiling still apply, so this can never
 *     join two things of different sizes or on opposite sides of the frame.
 */
function unambiguousPairs(tracks, detections) {
  const nearest = (from, candidates, sizeOfFrom) => {
    const [fw, fh] = sizeOfFrom;
    if (fw <= 0 || fh <= 0) return null;
    const [fx, fy] = centre(from.box);
    let best = null;
    let second = Infinity;
    for (const [index, other] of candidates.entries()) {
      if (other.label !== from.label) continue;
      const [ow, oh] = sizeOf(other.box);
      if (ow <= 0 || oh <= 0) continue;
      if (Math.max(fw / ow, ow / fw, fh / oh, oh / fh) > MAX_SIZE_RATIO) continue;
      const [ox, oy] = centre(other.box);
      const away = Math.hypot(ox - fx, oy - fy);
      if (away > DRIFT_MAX) continue;
      if (!best || away < best.away) {
        second = best ? best.away : second;
        best = { index, away };
      } else if (away < second) {
        second = away;
      }
    }
    // No runner-up at all is the clearest case there is.
    if (!best || second < best.away * LONELY_RATIO) return null;
    return best.index;
  };

  const pairs = [];
  for (const [index, track] of tracks.entries()) {
    const pick = nearest(track, detections, sizeOf(track.box));
    if (pick === null) continue;
    // And the same answer looking the other way, so two tracks cannot both claim one
    // detection just because it is the only thing near either of them.
    if (nearest(detections[pick], tracks, sizeOf(detections[pick].box)) !== index) continue;
    pairs.push({ track, index: pick, score: LONELY_SCORE });
  }
  return pairs;
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
 * Three guesses at where the track should be, and the best of them wins: where it was, where
 * its own velocity says it went, and where the whole picture went. Best-of rather than a sum,
 * because a track that has been alive a while has already absorbed the drone's motion into
 * its velocity, and adding the drift on top of that would carry it twice as far.
 */
function affinity(track, box, minIou, drift) {
  const moved = shift(track.box, track.velocity);
  const drifted = shift(track.box, drift);

  const overlap = Math.max(iou(track.box, box), iou(moved, box), iou(drifted, box));
  if (overlap >= minIou) return 1 + overlap;   // always beats any distance-only match

  const [tw, th] = sizeOf(track.box);
  const [dw, dh] = sizeOf(box);
  if (tw <= 0 || th <= 0 || dw <= 0 || dh <= 0) return 0;

  const ratio = Math.max(tw / dw, dw / tw, th / dh, dh / th);
  if (ratio > MAX_SIZE_RATIO) return 0;

  const [bx, by] = centre(box);
  const reach = Math.max(1, Math.hypot(tw, th) / 2);
  let closest = Infinity;
  for (const guess of [track.box, moved, drifted]) {
    const [px, py] = centre(guess);
    closest = Math.min(closest, Math.hypot(px - bx, py - by) / reach);
  }
  if (closest > MAX_CENTRE_DRIFT) return 0;

  // Closer is better, and never reaches the overlap band above.
  return 1 - closest / MAX_CENTRE_DRIFT;
}

export class Tracker {
  constructor({
    minIou = MIN_IOU,
    maxMisses = MAX_MISSES,
    confirmAfter = CONFIRM_AFTER,
    newTrackConfidence = NEW_TRACK_CONFIDENCE,
    maxCoastMs = MAX_COAST_MS,
    reidSimilarity = REID_SIMILARITY,
    reidWindowMs = REID_WINDOW_MS,
    inViewMs = IN_VIEW_MS,
  } = {}) {
    this.minIou = minIou;
    this.maxMisses = maxMisses;
    this.confirmAfter = confirmAfter;
    this.newTrackConfidence = newTrackConfidence;
    this.maxCoastMs = maxCoastMs;
    this.reidSimilarity = reidSimilarity;
    this.reidWindowMs = reidWindowMs;
    this.inViewMs = inViewMs;
    /**
     * People this tracker has seen and lost, so it knows them when they come back.
     * Each entry is {id, label, signature, lastSeen, counted}.
     */
    this.remembered = [];
    this.tracks = [];
    /** The last measured movement of the whole picture, for the status line and for tests. */
    this.drift = [0, 0];
    this.nextId = 1;
    // The number an operator reads off a box, which is NOT the internal id. See issue().
    this.nextNumber = 1;
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

    // Before any pairing: how far the whole picture moved. See estimateDrift.
    const drift = estimateDrift(this.tracks, detections);
    this.drift = drift;

    const pairs = [];

    // Every plausible pairing, best first. Only a track and a detection of the same class
    // may pair: a car becoming a person between two frames is not a thing that happens, and
    // allowing it lets an identity jump across the frame.
    for (const track of this.tracks) {
      for (const [index, detection] of detections.entries()) {
        if (track.label !== detection.label) continue;
        const score = affinity(track, detection.box, this.minIou, drift);
        if (score > 0) pairs.push({ track, index, score });
      }
    }
    // Appended rather than merged: they score below every real affinity, so the greedy pass
    // below only reaches them for a track and a detection nothing else wanted.
    pairs.push(...unambiguousPairs(this.tracks, detections));
    pairs.sort((a, b) => b.score - a.score);

    const usedTracks = new Set();
    const usedDetections = new Set();
    for (const pair of pairs) {
      if (usedTracks.has(pair.track) || usedDetections.has(pair.index)) continue;
      usedTracks.add(pair.track);
      usedDetections.add(pair.index);

      const detection = detections[pair.index];
      const observedCentre = centre(detection.box);
      // How many cycles since this track was last actually SEEN, rather than predicted.
      const coasted = Math.max(1, pair.track.missed + 1);

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
      //
      // Measured from where the track was last SEEN, and divided by how many cycles ago
      // that was. It used to be measured against track.box, which by then had been coasted
      // forward by the velocity on every missed cycle - so the difference was the
      // prediction's leftover error, not how far the person had gone, and feeding that back
      // in as velocity is a loop that fights itself. Under one look per six cycles, which is
      // what the tile rotation gave every track, it oscillates instead of settling: the box
      // overshoots, gets pulled back, overshoots the other way. That is a box that sits in
      // the wrong place and jumps.
      const last = pair.track.lastObserved ?? observedCentre;
      const observed = [
        (observedCentre[0] - last[0]) / coasted,
        (observedCentre[1] - last[1]) / coasted,
      ];
      // A pairing made only because there was nothing else it could be is not evidence
      // about motion. Letting it set velocity is what flings a coasting box across the
      // frame. See unambiguousPairs.
      if (pair.score > LONELY_SCORE) {
        pair.track.velocity = [
          pair.track.velocity[0] * 0.6 + observed[0] * 0.4,
          pair.track.velocity[1] * 0.6 + observed[1] * 0.4,
        ];
      }
      pair.track.lastObserved = observedCentre;
      pair.track.path.push(observedCentre);
      if (pair.track.path.length > 60) pair.track.path.shift();
    }

    for (const [index, detection] of detections.entries()) {
      if (usedDetections.has(index)) continue;

      // A box too weak to be somebody new. It was offered to every track above and none of
      // them wanted it, so it stops here: it may keep a person alive through a bad moment,
      // and it may not invent one. See NEW_TRACK_CONFIDENCE.
      //
      // DRAWING THESE ANYWAY WAS TRIED, AND MEASURED, AND TAKEN BACK OUT
      //     The ask was to box everyone without waiting for verification. Half of that is
      //     right and is what visible() does. The other half - drawing a blob too faint to
      //     start an identity - was measured across five scene densities, per frame, as an
      //     operator sees it:
      //
      //         scene            boxes on people      boxes on nobody
      //         light  (6-15)      4.4 -> 5.6           2.9 -> 4.5
      //         busy   (16-40)    10.2 -> 12.7          4.5 -> 9.9
      //         dense  (100+)     52.4 -> 64.6         17.7 -> 26.9
      //
      //     In a crowd it roughly breaks even. In the scenes this actually flies - 16 to 40
      //     people is 261 of the 548 validation frames, and 100+ is twelve of them - it puts
      //     more boxes on empty ground than on people. An operator cannot use a screen where
      //     two boxes in five are on nothing.
      //
      //     It is also redundant. The sensitivity slider sets this threshold, so anybody who
      //     wants those faint blobs boxed can have them by lowering it, and can see what it
      //     costs while they do it. See docs/metrics-crowd.txt.
      if ((detection.confidence ?? 1) < this.newTrackConfidence) continue;

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
        // Carried across with the count for the same reason: somebody who walks behind a
        // van and out the other side is not a new person and must not get a new number.
        number: known ? known.number : 0,
        returned: Boolean(known),
        firstFrame: this.frame,
        lastFrame: this.frame,
        lastSeenAt: now,
        velocity: [0, 0],
        // Where this track was last actually seen. Velocity is measured from here, never
        // from the coasted box. See the velocity update in update().
        lastObserved: centre(detection.box),
        path: [centre(detection.box)],
      });
      usedTracks.add(this.tracks[this.tracks.length - 1]);
    }

    // Newly created tracks count as used, because they were: the detection that made each
    // of them was seen on THIS frame. Without this they fall into the loop below and are
    // marked as having missed the very frame they were born on, which is wrong three ways.
    // Their miss count is permanently one too high, so the budget that decides when to let
    // them go is one short; `coasted()` is true from the first frame, so a brand-new box is
    // drawn dimmed and dashed as though it were a guess; and velocity is divided by a
    // coast that never happened.
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
      if (track.seen >= this.confirmAfter && !track.counted) {
        track.counted = true;
        // The number is issued HERE, when somebody is confirmed to be somebody, and not
        // when a box first appears.
        //
        // WHY THIS IS NOT track.id
        //     id is spent the moment any box arrives that no existing track wanted, which
        //     includes every flicker of gravel, roof vent and shadow that never survives to
        //     a second look. It is bookkeeping and it is meant to be thrown away. Printing
        //     it on screen meant that one real person standing in a scene with sixty-six
        //     discarded flickers was labelled "person 67", and the operator reasonably read
        //     that as the tracker having counted sixty-seven people.
        //
        //     This counter only ever moves when somebody is confirmed, so the highest number
        //     on screen is the number of people counted, which is what countSeen returns and
        //     what the number was always meant to mean.
        if (!track.number) track.number = this.nextNumber++;
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
      // The number they were given, so they come back as themselves and not as a new one.
      number: track.number,
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
  /**
   * Set the coast and in-view windows from the cadence the device is ACHIEVING.
   *
   * MAX_COAST_MS is not a fact about people. It is three looks at a 250 ms cycle, and
   * 250 ms is what a desktop CPU managed while these were tuned rather than what a
   * controller does. On a tablet running at 600 ms a cycle, 800 ms is barely one look:
   * everybody would be let go between consecutive looks at them and renumbered on the next,
   * which is the exact failure the value was tuned to avoid.
   *
   * Looks is the right unit on both sides. Holding through a brief miss is worth it because
   * a miss is a LOOK that failed; coasting costs because a stale box gets one chance per
   * LOOK to steal the detection belonging to whoever stands where it drifted to.
   *
   * Kept in step with Tracker.setCadence in the Java.
   *
   * @param cycleMs measured time from the start of one detect cycle to the next
   */
  setCadence(cycleMs) {
    const cycle = Math.max(80, Math.min(2000, cycleMs));
    this.maxCoastMs = Math.max(400, Math.min(6000, cycle * 3));
    this.inViewMs = Math.max(250, Math.min(4000, cycle * 2));
  }

  open() {
    return this.tracks.filter((t) => t.seen >= this.confirmAfter);
  }

  /**
   * Everything worth drawing a box around, which is more than everything worth numbering.
   *
   * WHY THESE ARE TWO QUESTIONS
   *     They used to be one, and it forced a choice nobody should have to make. Drawing only
   *     confirmed tracks meant a person who had just walked into frame had no box at all for
   *     four cycles, about a second, which reads as the detector not seeing them. Drawing
   *     everything the instant it appeared would have put a number on every flicker, and a
   *     number that appears and vanishes is worse than no number.
   *
   *     So: a box as soon as anything is detected, and a number only once it has agreed with
   *     itself. An unconfirmed track comes back with number 0, and the overlay draws it
   *     without a label. Nothing is hidden from the operator and nothing unproven is counted.
   */
  visible() {
    return this.tracks.filter((t) => t.missed === 0 || t.seen >= this.confirmAfter);
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
      && now - t.lastSeenAt <= this.inViewMs).length;
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
    // The number an operator reads off a box, which is NOT the internal id. See issue().
    this.nextNumber = 1;
    this.frame = 0;
    this.everSeen = new Map();
  }
}
