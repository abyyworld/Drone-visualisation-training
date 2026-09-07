/**
 * Unit tests for app/js/sync.js -- the three-tier overlay synchroniser and,
 * above all, the staleness rule.
 *
 * This is the piece of the tablet app most worth testing and the hardest to
 * test in a browser: it decides, once per drawn frame, whether the boxes on
 * screen belong to the frame on screen. Get it wrong in the lenient direction
 * and the operator gets boxes computed from a scene that has already moved
 * on, over live video, looking authoritative -- which docs/CONTRACT.md names
 * as this system's single most dangerous failure mode.
 *
 * sync.js is pure by construction (no DOM, no timers, no network, injectable
 * clock), so all of it runs under plain node. Payloads are built through
 * wire.js's own parser rather than by hand, so a test can never assert against
 * a shape the real parser would have rejected.
 *
 * Run directly:  node tests/test_sync.js
 * Under pytest:  tests/test_sync_js.py shells out to exactly that.
 */

import { strict as assert } from 'node:assert';

import {
  OverlaySync,
  SYNC_TIER,
  TIER_LABEL,
  median,
  rtpDelta,
  selectNearest,
  RTP_CLOCK_RATE,
  RTP_MODULO,
} from '../app/js/sync.js';
import { WIRE_VERSION, parseDetections, parseStatus } from '../app/js/wire.js';

// -------------------------------------------------------------------------
// a very small test harness
// -------------------------------------------------------------------------

const results = { passed: 0, failed: 0 };
const failures = [];
let currentGroup = '';

function group(name) {
  currentGroup = name;
}

function test(name, fn) {
  const label = currentGroup ? `${currentGroup} :: ${name}` : name;
  try {
    fn();
    results.passed += 1;
  } catch (err) {
    results.failed += 1;
    failures.push({ label, err });
  }
}

// -------------------------------------------------------------------------
// builders -- everything goes through the real wire parser
// -------------------------------------------------------------------------

const BOX = [0.4, 0.35, 0.55, 0.5];

/**
 * Build a parsed detections payload.
 *
 * @param {number} pts Media pts in seconds.
 * @param {{rtp?: number|null, boxes?: number, frameId?: number}} [opts] Extras.
 * @returns {object} The payload as wire.js would hand it to the synchroniser.
 */
function payload(pts, opts = {}) {
  const detections = [];
  for (let i = 0; i < (opts.boxes === undefined ? 1 : opts.boxes); i += 1) {
    detections.push({ cls: 'fire', conf: 0.8, box: BOX, persisted: 3, track: i + 1 });
  }
  const raw = {
    v: WIRE_VERSION,
    type: 'detections',
    frame_id: opts.frameId === undefined ? Math.round(pts * 10) : opts.frameId,
    pts,
    wall_time: '2026-09-06T14:22:31.412Z',
    detections,
  };
  if (opts.rtp !== undefined && opts.rtp !== null) raw.rtp_ts = opts.rtp;
  return parseDetections(raw);
}

/** Build a parsed status message. */
function status(fields = {}) {
  return parseStatus({
    v: WIRE_VERSION,
    type: 'status',
    state: 'running',
    wall_time: '2026-09-06T14:22:31.500Z',
    dropped_frames: 0,
    ...fields,
  });
}

/** A synchroniser with a controllable wall clock. */
function makeSync(options = {}) {
  const clock = { ms: 1_000_000 };
  const sync = new OverlaySync({ now: () => clock.ms, ...options });
  return { sync, clock };
}

/** 90 kHz RTP timestamp for a media time, from an arbitrary origin. */
const RTP_ORIGIN = 4_294_000_000; // deliberately close to the 32-bit wrap
function rtpFor(mediaTime) {
  return (RTP_ORIGIN + Math.round(mediaTime * RTP_CLOCK_RATE)) % RTP_MODULO;
}

// -------------------------------------------------------------------------
// pure helpers
// -------------------------------------------------------------------------

group('rtpDelta');

test('plain difference', () => {
  assert.equal(rtpDelta(90000, 0), 90000);
  assert.equal(rtpDelta(0, 90000), -90000);
});

test('wraps correctly across the 32-bit boundary', () => {
  // A long incident genuinely reaches the wrap (~13.25 h at 90 kHz). An
  // unwrapped subtraction there would look like a half-day jump and park the
  // overlay in the wrong tier -- or match a box to a frame hours away.
  const before = RTP_MODULO - 1000;
  const after = 2000; // 3000 ticks later, having wrapped
  assert.equal(rtpDelta(after, before), 3000);
  assert.equal(rtpDelta(before, after), -3000);
});

test('is zero for equal timestamps', () => {
  assert.equal(rtpDelta(12345, 12345), 0);
});

group('median');

test('odd and even lengths', () => {
  assert.equal(median([3, 1, 2]), 2);
  assert.equal(median([4, 1, 3, 2]), 2.5);
});

test('empty list is null, not zero', () => {
  assert.equal(median([]), null);
});

test('does not mutate its input', () => {
  const values = [3, 1, 2];
  median(values);
  assert.deepEqual(values, [3, 1, 2]);
});

test('resists a single wild outlier where a mean would not', () => {
  // The whole reason the contract specifies a median: one very late payload
  // must not drag every box sideways.
  const samples = [1.0, 1.01, 0.99, 1.02, 0.98, 1.0, 30.0];
  const mean = samples.reduce((a, b) => a + b, 0) / samples.length;
  assert.ok(Math.abs(median(samples) - 1.0) < 0.05);
  assert.ok(mean > 5);
});

group('selectNearest');

test('prefers the most recent item at or before the target', () => {
  const items = [{ t: 0 }, { t: 1 }, { t: 2 }, { t: 3 }];
  assert.equal(selectNearest(items, 2.4).t, 2);
  assert.equal(selectNearest(items, 3).t, 3);
});

test('never returns an item ahead of the target when one behind exists', () => {
  // A payload from a frame the operator has not been shown yet would put
  // boxes ahead of the scene.
  const items = [{ t: 0 }, { t: 5 }];
  assert.equal(selectNearest(items, 4.9).t, 0);
});

test('falls back to the earliest item when everything is ahead', () => {
  const items = [{ t: 10 }, { t: 11 }];
  assert.equal(selectNearest(items, 3).t, 10);
});

test('empty list is null', () => {
  assert.equal(selectNearest([], 1), null);
});

// -------------------------------------------------------------------------
// parameters
// -------------------------------------------------------------------------

group('setParameters');

test('adopts the station values', () => {
  const { sync } = makeSync();
  sync.setParameters({ max_overlay_age_s: 2, overlay_buffer_s: 5 });
  assert.equal(sync.maxOverlayAgeS, 2);
  assert.equal(sync.overlayBufferS, 5);
});

test('refuses values that would disable the staleness rule', () => {
  // These arrive over the network. A station (or something pretending to be
  // one) must not be able to switch the safety rule off.
  const { sync } = makeSync();
  const before = sync.maxOverlayAgeS;
  for (const bad of [0, -1, 1e9, NaN, null, 'soon', undefined]) {
    sync.setParameters({ max_overlay_age_s: bad });
    assert.equal(sync.maxOverlayAgeS, before, `accepted max_overlay_age_s=${bad}`);
  }
});

test('a buffer shorter than the age limit is raised to it', () => {
  // Mirrors config.validate(): a shorter buffer would discard payloads the
  // tablet is still allowed to draw.
  const { sync } = makeSync();
  sync.setParameters({ max_overlay_age_s: 2, overlay_buffer_s: 1 });
  assert.equal(sync.overlayBufferS, 2);
});

test('null parameters are ignored rather than throwing', () => {
  const { sync } = makeSync();
  sync.setParameters(null);
  assert.equal(sync.maxOverlayAgeS, 1.0);
});

// -------------------------------------------------------------------------
// tier 3
// -------------------------------------------------------------------------

group('tier 3 (unsynchronised)');

test('an empty buffer draws nothing and says so', () => {
  const { sync } = makeSync();
  const result = sync.select(10);
  assert.equal(result.tier, SYNC_TIER.UNSYNCHRONISED);
  assert.deepEqual(result.detections, []);
  assert.equal(result.payload, null);
  assert.equal(result.stale, false);
  assert.equal(result.reason, 'no payloads buffered');
});

test('with no media clock at all, the latest payload is drawn and labelled', () => {
  const { sync } = makeSync();
  sync.addPayload(payload(1.0), null);
  const result = sync.select(null);
  assert.equal(result.tier, SYNC_TIER.UNSYNCHRONISED);
  assert.equal(result.tierLabel, TIER_LABEL[SYNC_TIER.UNSYNCHRONISED]);
  assert.equal(result.detections.length, 1);
});

test('tier 3 still ages out: approximate is allowed, old is not', () => {
  const { sync, clock } = makeSync();
  sync.addPayload(payload(1.0), null);
  clock.ms += 3000;
  const result = sync.select(null);
  assert.equal(result.stale, true);
  assert.deepEqual(result.detections, []);
});

// -------------------------------------------------------------------------
// tier 2
// -------------------------------------------------------------------------

group('tier 2 (pts offset)');

test('seeds the offset from stream_start_pts on the first status', () => {
  // Those first seconds are when an operator forms an opinion about whether
  // the boxes line up at all.
  const { sync } = makeSync();
  sync.addStatus(status({ stream_start_pts: 100 }), 0);
  const estimate = sync.offsetEstimate();
  assert.equal(estimate.value, 100);
  assert.match(estimate.source, /stream_start_pts/);
});

test('measured samples take over from the seed', () => {
  const { sync } = makeSync({ minOffsetSamples: 3 });
  sync.addStatus(status({ stream_start_pts: 100 }), 0);
  for (let i = 0; i < 3; i += 1) sync.addPayload(payload(100 + i * 0.1), i * 0.1);
  const estimate = sync.offsetEstimate();
  assert.ok(Math.abs(estimate.value - 100) < 1e-9);
  assert.match(estimate.source, /median of 3/);
});

test('a single very late payload does not drag the estimate', () => {
  const { sync } = makeSync({ minOffsetSamples: 3 });
  for (let i = 0; i < 9; i += 1) sync.addPayload(payload(10 + i * 0.1), 10 + i * 0.1 - 0.5);
  sync.addPayload(payload(10.9), 5.0); // one payload with a wild arrival time
  const estimate = sync.offsetEstimate();
  assert.ok(Math.abs(estimate.value - 0.5) < 0.05, `offset drifted to ${estimate.value}`);
});

test('the offset window is bounded', () => {
  const { sync } = makeSync({ offsetWindow: 5, overlayBufferS: 1000 });
  for (let i = 0; i < 50; i += 1) sync.addPayload(payload(i * 0.1), i * 0.1);
  assert.equal(sync._offsets.length, 5);
});

test('matches the nearest payload at or before the target pts', () => {
  const { sync } = makeSync({ minOffsetSamples: 1 });
  // offset 0: pts and currentTime share a timeline.
  for (const pts of [10.0, 10.1, 10.2, 10.3]) sync.addPayload(payload(pts), pts);
  const result = sync.select(10.25);
  assert.equal(result.tier, SYNC_TIER.PTS_OFFSET);
  assert.ok(Math.abs(result.matchedPts - 10.2) < 1e-9, `matched ${result.matchedPts}`);
  assert.ok(Math.abs(result.ageS - 0.05) < 1e-9);
  assert.equal(result.detections.length, 1);
});

test('a constant offset between the timelines is removed', () => {
  const { sync } = makeSync({ minOffsetSamples: 3 });
  // Station pts runs 40 s ahead of the tablet's currentTime (a tablet that
  // joined 40 s into the incident).
  for (let i = 0; i < 10; i += 1) sync.addPayload(payload(40 + i * 0.1), i * 0.1);
  const result = sync.select(0.85);
  assert.ok(Math.abs(result.offsetS - 40) < 1e-6, `offset ${result.offsetS}`);
  assert.ok(Math.abs(result.matchedPts - 40.8) < 1e-6, `matched ${result.matchedPts}`);
  assert.equal(result.stale, false);
});

// -------------------------------------------------------------------------
// tier 1
// -------------------------------------------------------------------------

group('tier 1 (rtp match)');

test('is used when both the browser and the station supply rtp', () => {
  const { sync } = makeSync();
  for (let i = 0; i < 5; i += 1) {
    const t = 10 + i * 0.1;
    sync.addVideoFrame({ mediaTime: t, rtpTimestamp: rtpFor(t) });
  }
  for (let i = 0; i < 5; i += 1) {
    const t = 10 + i * 0.1;
    sync.addPayload(payload(t, { rtp: rtpFor(t) }), t);
  }
  const result = sync.select(10.4);
  assert.equal(result.tier, SYNC_TIER.RTP);
  assert.ok(Math.abs(result.matchedPts - 10.4) < 1e-9, `matched ${result.matchedPts}`);
  assert.ok(Math.abs(result.ageS) < 1e-6);
  assert.equal(result.stale, false);
});

test('matches the frame it was computed from, not the newest payload', () => {
  // The point of tier 1. The newest payload here is 0.3 s ahead of the frame
  // actually on the glass.
  const { sync } = makeSync();
  for (let i = 0; i < 4; i += 1) {
    const t = 5 + i * 0.1;
    sync.addVideoFrame({ mediaTime: t, rtpTimestamp: rtpFor(t) });
  }
  for (let i = 0; i < 7; i += 1) {
    const t = 5 + i * 0.1;
    sync.addPayload(payload(t, { rtp: rtpFor(t) }), t);
  }
  const result = sync.select(5.3);
  assert.equal(result.tier, SYNC_TIER.RTP);
  assert.ok(Math.abs(result.matchedPts - 5.3) < 1e-9, `matched ${result.matchedPts}`);
});

test('falls back to tier 2 when the browser reports no rtpTimestamp', () => {
  // A MediaMTX relay, or a browser without the optional field. Correct
  // behaviour, and labelled -- not an error.
  const { sync } = makeSync({ minOffsetSamples: 1 });
  for (let i = 0; i < 4; i += 1) sync.addVideoFrame({ mediaTime: 5 + i * 0.1 });
  for (let i = 0; i < 4; i += 1) {
    const t = 5 + i * 0.1;
    sync.addPayload(payload(t, { rtp: rtpFor(t) }), t);
  }
  assert.equal(sync.select(5.3).tier, SYNC_TIER.PTS_OFFSET);
});

test('falls back to tier 2 when the station sends no rtp_ts', () => {
  const { sync } = makeSync({ minOffsetSamples: 1 });
  for (let i = 0; i < 4; i += 1) {
    const t = 5 + i * 0.1;
    sync.addVideoFrame({ mediaTime: t, rtpTimestamp: rtpFor(t) });
  }
  for (let i = 0; i < 4; i += 1) sync.addPayload(payload(5 + i * 0.1), 5 + i * 0.1);
  assert.equal(sync.select(5.3).tier, SYNC_TIER.PTS_OFFSET);
});

test('refuses an rtp mapping built from stale video frames', () => {
  // rVFC stops firing in a hidden tab or on a stalled decoder, and a mapping
  // from that far back may sit on the far side of an RTP discontinuity. The
  // labelled drop to tier 2 is the safe answer.
  const { sync } = makeSync({ maxRtpMapAgeS: 1.5, minOffsetSamples: 1 });
  sync.addVideoFrame({ mediaTime: 5.0, rtpTimestamp: rtpFor(5.0) });
  sync.addVideoFrame({ mediaTime: 5.1, rtpTimestamp: rtpFor(5.1) });
  sync.addPayload(payload(9.0, { rtp: rtpFor(9.0) }), 9.0);
  assert.equal(sync.select(9.0).tier, SYNC_TIER.PTS_OFFSET);
});

test('mapRtpToMedia interpolates inside the sample window', () => {
  const { sync } = makeSync();
  sync.addVideoFrame({ mediaTime: 1.0, rtpTimestamp: rtpFor(1.0) });
  sync.addVideoFrame({ mediaTime: 2.0, rtpTimestamp: rtpFor(2.0) });
  assert.ok(Math.abs(sync.mapRtpToMedia(rtpFor(1.5)) - 1.5) < 1e-6);
});

test('mapRtpToMedia extrapolates at unit slope outside it', () => {
  const { sync } = makeSync();
  sync.addVideoFrame({ mediaTime: 1.0, rtpTimestamp: rtpFor(1.0) });
  sync.addVideoFrame({ mediaTime: 2.0, rtpTimestamp: rtpFor(2.0) });
  assert.ok(Math.abs(sync.mapRtpToMedia(rtpFor(2.5)) - 2.5) < 1e-6);
  assert.ok(Math.abs(sync.mapRtpToMedia(rtpFor(0.5)) - 0.5) < 1e-6);
});

test('mapRtpToMedia is null with no samples', () => {
  const { sync } = makeSync();
  assert.equal(sync.mapRtpToMedia(12345), null);
});

test('a repeated frame does not tilt the mapping', () => {
  const { sync } = makeSync();
  sync.addVideoFrame({ mediaTime: 1.0, rtpTimestamp: rtpFor(1.0) });
  sync.addVideoFrame({ mediaTime: 1.2, rtpTimestamp: rtpFor(1.0) }); // same source frame
  assert.equal(sync.stats.rtpSamples, 1);
});

// -------------------------------------------------------------------------
// THE STALENESS RULE
// -------------------------------------------------------------------------

group('staleness rule');

test('tier 2: past max_overlay_age_s the boxes stop', () => {
  // The rule that outranks all three tiers. Live video under boxes computed
  // from a scene that has already moved on is this system's most dangerous
  // single failure mode; plain video is the safe state.
  const { sync } = makeSync({ minOffsetSamples: 1, maxOverlayAgeS: 1.0 });
  sync.addPayload(payload(10.0), 10.0);
  const fresh = sync.select(10.5);
  assert.equal(fresh.stale, false);
  assert.equal(fresh.detections.length, 1);

  const stale = sync.select(11.5); // 1.5 s of media time later
  assert.equal(stale.stale, true);
  assert.equal(stale.staleKind, 'age');
  assert.equal(stale.payload, null, 'a stale payload must not be handed to the renderer');
  assert.deepEqual(stale.detections, [], 'boxes are dropped, not dimmed');
  assert.match(stale.staleReason, /old/);
});

test('the boundary is exact and inclusive up to the limit', () => {
  const { sync } = makeSync({ minOffsetSamples: 1, maxOverlayAgeS: 1.0 });
  sync.addPayload(payload(10.0), 10.0);
  assert.equal(sync.select(11.0).stale, false, 'exactly at the limit is still drawable');
  assert.equal(sync.select(11.0001).stale, true, 'a hair past the limit must stop drawing');
});

test('the station-configured limit is the one enforced', () => {
  const { sync } = makeSync({ minOffsetSamples: 1 });
  sync.setParameters({ max_overlay_age_s: 0.4, overlay_buffer_s: 3 });
  sync.addPayload(payload(10.0), 10.0);
  assert.equal(sync.select(10.3).stale, false);
  assert.equal(sync.select(10.5).stale, true);
});

test('tier 1: an old rtp match is stale too', () => {
  const { sync } = makeSync({ maxOverlayAgeS: 0.5, maxRtpMapAgeS: 100 });
  for (let i = 0; i < 4; i += 1) {
    const t = 20 + i * 0.1;
    sync.addVideoFrame({ mediaTime: t, rtpTimestamp: rtpFor(t) });
  }
  sync.addPayload(payload(18.0, { rtp: rtpFor(18.0) }), 20.3);
  const result = sync.select(20.3);
  assert.equal(result.tier, SYNC_TIER.RTP);
  assert.equal(result.stale, true);
  assert.deepEqual(result.detections, []);
});

test('tier 3: an old latest-payload is stale too', () => {
  const { sync } = makeSync({ maxOverlayAgeS: 0.5 });
  sync.addPayload(payload(1.0), 1.0);   // arrived at media time 1.0
  const result = sync.select(3.0);      // ...and the video is now at 3.0
  assert.equal(result.tier, SYNC_TIER.UNSYNCHRONISED);
  assert.equal(result.stale, true);
  assert.deepEqual(result.detections, []);
});

test('a payload far ahead of the frame on screen also stops the boxes', () => {
  // Not staleness, but the same disease: boxes computed from a scene the
  // operator has not been shown yet. The contract only names the old
  // direction because a healthy station cannot produce the other one.
  const { sync } = makeSync({ minOffsetSamples: 1, maxOverlayAgeS: 1.0 });
  sync.addPayload(payload(50.0), 10.0); // a wildly wrong offset sample
  const result = sync.select(10.0);
  assert.equal(result.stale, true);
  assert.equal(result.staleKind, 'ahead');
  assert.deepEqual(result.detections, []);
});

test('a small buffering lead is not treated as stale', () => {
  // A payload belonging to a frame about to be presented is normal.
  const { sync } = makeSync({ minOffsetSamples: 1, maxOverlayAgeS: 1.0 });
  sync.addPayload(payload(10.0), 10.0);
  sync.addPayload(payload(10.3), 10.3);
  const result = sync.select(10.1);
  assert.equal(result.stale, false);
  assert.ok(result.detections.length > 0);
});

test('a stalled pipeline goes stale rather than replaying the ring buffer', () => {
  // The buffer holds 3 s of payloads. When the station stops sending, the
  // overlay must go dark quickly rather than redrawing what it still holds.
  const { sync } = makeSync({ minOffsetSamples: 1, maxOverlayAgeS: 1.0, overlayBufferS: 3.0 });
  for (let i = 0; i < 30; i += 1) sync.addPayload(payload(10 + i * 0.1), 10 + i * 0.1);
  assert.equal(sync.select(12.9).stale, false);
  // ...and now nothing arrives for two seconds while the video runs on.
  const result = sync.select(14.9);
  assert.equal(result.stale, true);
  assert.deepEqual(result.detections, []);
  assert.equal(sync.stale, true, 'the flag on the object must agree with the result');
});

test('the stale flag on the instance tracks the last select()', () => {
  const { sync } = makeSync({ minOffsetSamples: 1, maxOverlayAgeS: 1.0 });
  sync.addPayload(payload(10.0), 10.0);
  sync.select(10.1);
  assert.equal(sync.stale, false);
  sync.select(12.0);
  assert.equal(sync.stale, true);
});

test('an empty payload is not staleness and renders as nothing', () => {
  // The safety invariant, at the point where drawing is decided: an empty
  // detections list is a normal, frequent, unremarkable result. It is not an
  // error, it is not staleness, and it renders as nothing at all.
  const { sync } = makeSync({ minOffsetSamples: 1 });
  sync.addPayload(payload(10.0, { boxes: 0 }), 10.0);
  const result = sync.select(10.1);
  assert.equal(result.stale, false);
  assert.notEqual(result.payload, null);
  assert.deepEqual(result.detections, []);
});

// -------------------------------------------------------------------------
// the ring buffer
// -------------------------------------------------------------------------

group('ring buffer');

test('evicts payloads older than the buffer window', () => {
  const { sync } = makeSync({ overlayBufferS: 1.0 });
  for (let i = 0; i < 50; i += 1) sync.addPayload(payload(i * 0.1), i * 0.1);
  const result = sync.select(4.9);
  assert.ok(result.bufferedCount <= 12, `buffered ${result.bufferedCount}`);
  assert.ok(sync.stats.payloadsEvicted > 0);
});

test('the window is measured from the newest pts, not the wall clock', () => {
  // A pipeline that pauses and resumes must not lose the payloads it is
  // about to need.
  const { sync, clock } = makeSync({ overlayBufferS: 3.0 });
  for (let i = 0; i < 10; i += 1) sync.addPayload(payload(i * 0.1), i * 0.1);
  clock.ms += 60_000;
  assert.equal(sync.select(0.9).bufferedCount, 10);
});

test('never evicts the last payload', () => {
  const { sync } = makeSync({ overlayBufferS: 0.01 });
  sync.addPayload(payload(1.0), 1.0);
  sync.addPayload(payload(99.0), 99.0);
  sync.addPayload(payload(99.001), 99.001);
  assert.ok(sync._payloads.length >= 1);
});

test('a hard cap bounds the buffer whatever pts the station sends', () => {
  const { sync } = makeSync({ maxPayloads: 8, overlayBufferS: 1e9 });
  for (let i = 0; i < 200; i += 1) sync.addPayload(payload(i * 0.001), i * 0.001);
  sync.select(0.2);
  assert.ok(sync._payloads.length <= 8, `buffer grew to ${sync._payloads.length}`);
});

test('out-of-order arrivals are inserted in pts order', () => {
  const { sync } = makeSync();
  sync.addPayload(payload(1.0), 1.0);
  sync.addPayload(payload(1.2), 1.2);
  sync.addPayload(payload(1.1), 1.1);
  assert.deepEqual(sync._payloads.map((e) => e.pts), [1.0, 1.1, 1.2]);
});

test('a payload older than the whole buffer is discarded', () => {
  const { sync } = makeSync({ overlayBufferS: 1.0 });
  for (let i = 0; i < 10; i += 1) sync.addPayload(payload(10 + i * 0.1), 10 + i * 0.1);
  const before = sync._payloads.length;
  sync.addPayload(payload(1.0), 1.0);
  assert.equal(sync._payloads.length, before);
  assert.equal(sync.stats.payloadsTooOld, 1);
});

test('a payload with no usable pts is ignored', () => {
  const { sync } = makeSync();
  sync.addPayload({ pts: NaN, detections: [] }, 1.0);
  sync.addPayload(null, 1.0);
  assert.equal(sync._payloads.length, 0);
});

// -------------------------------------------------------------------------
// timeline jumps and reset
// -------------------------------------------------------------------------

group('timeline jumps');

test('one wild pts does not evict the buffer', () => {
  // The buffer window is measured from the newest pts, so admitting a corrupt
  // pts straight away would throw away every good payload behind it.
  const { sync } = makeSync({ overlayBufferS: 3.0, timelineJumpPayloads: 3 });
  for (let i = 0; i < 10; i += 1) sync.addPayload(payload(10 + i * 0.1), 10 + i * 0.1);
  sync.addPayload(payload(9999), 11.0);
  assert.equal(sync._payloads.length, 10);
  assert.equal(sync._payloads[sync._payloads.length - 1].pts, 10.9);
});

test('a persistent jump is adopted as a new timeline', () => {
  const { sync } = makeSync({ overlayBufferS: 3.0, timelineJumpPayloads: 3 });
  for (let i = 0; i < 10; i += 1) sync.addPayload(payload(10 + i * 0.1), 10 + i * 0.1);
  for (let i = 0; i < 3; i += 1) sync.addPayload(payload(500 + i * 0.1), 11 + i * 0.1);
  assert.equal(sync.stats.timelineJumps, 1);
  assert.ok(sync._payloads.every((e) => e.pts >= 500));
});

test('a changed stream_start_pts resets everything', () => {
  // The station restarted its source: everything measured against the old
  // timeline is worse than useless.
  const { sync } = makeSync();
  sync.addStatus(status({ stream_start_pts: 0 }), 0);
  for (let i = 0; i < 5; i += 1) sync.addPayload(payload(i * 0.1), i * 0.1);
  sync.addStatus(status({ stream_start_pts: 900 }), 10);
  assert.equal(sync._payloads.length, 0);
  assert.equal(sync.stats.resets, 1);
});

test('an unchanged stream_start_pts does not reset', () => {
  const { sync } = makeSync();
  sync.addStatus(status({ stream_start_pts: 0 }), 0);
  sync.addPayload(payload(0.5), 0.5);
  sync.addStatus(status({ stream_start_pts: 0 }), 1);
  assert.equal(sync._payloads.length, 1);
  assert.equal(sync.stats.resets, 0);
});

test('reset clears every timeline-derived thing', () => {
  const { sync } = makeSync();
  sync.addStatus(status({ stream_start_pts: 5 }), 0);
  sync.addVideoFrame({ mediaTime: 1.0, rtpTimestamp: rtpFor(1.0) });
  sync.addPayload(payload(1.0, { rtp: rtpFor(1.0) }), 1.0);
  sync.reset();
  assert.equal(sync._payloads.length, 0);
  assert.equal(sync._rtpSamples.length, 0);
  assert.equal(sync._offsets.length, 0);
  assert.equal(sync.offsetEstimate(), null);
  assert.equal(sync.tier, SYNC_TIER.UNSYNCHRONISED);
  assert.equal(sync.stale, false);
  assert.equal(sync.select(1.0).detections.length, 0);
});

// -------------------------------------------------------------------------
// the frozen-decoder case
// -------------------------------------------------------------------------

group('frozen decoder');

test('currentTime wins when rVFC has stopped firing', () => {
  // A frozen `presented` would freeze the reference clock and let the
  // staleness rule sleep through exactly the failure it exists to catch.
  const { sync } = makeSync({ minOffsetSamples: 1, maxOverlayAgeS: 1.0, presentedSkewToleranceS: 0.5 });
  sync.addVideoFrame({ mediaTime: 10.0 });
  sync.addPayload(payload(10.0), 10.0);
  assert.equal(sync.select(10.2).stale, false);
  // rVFC stops; currentTime runs on.
  const result = sync.select(12.0);
  assert.equal(result.presentedMediaTime, 12.0, 'the frozen rVFC time must not win');
  assert.equal(result.stale, true);
});

test('a small skew keeps the presented frame as the reference', () => {
  const { sync } = makeSync({ minOffsetSamples: 1, presentedSkewToleranceS: 0.5 });
  sync.addVideoFrame({ mediaTime: 10.0 });
  sync.addPayload(payload(10.0), 10.0);
  assert.equal(sync.select(10.2).presentedMediaTime, 10.0);
});

test('a stalled video clock is reported but does not itself blank the overlay', () => {
  // A frozen picture with boxes matched to it is self-consistent. The
  // operator is told; the boxes stay.
  const { sync, clock } = makeSync({ minOffsetSamples: 1 });
  sync.addVideoFrame({ mediaTime: 10.0, rtpTimestamp: rtpFor(10.0) });
  sync.addPayload(payload(10.0, { rtp: rtpFor(10.0) }), 10.0);
  clock.ms += 4000;
  sync.addVideoFrame({ mediaTime: 10.0, rtpTimestamp: rtpFor(10.0) });
  const result = sync.select(10.0);
  assert.ok(result.videoStalledS >= 4, `videoStalledS ${result.videoStalledS}`);
  assert.equal(result.stale, false);
  assert.ok(result.detections.length > 0);
});

// -------------------------------------------------------------------------
// invariant 1, at the point of drawing
// -------------------------------------------------------------------------

group('safety invariant');

test('nothing the synchroniser returns asserts an absence of fire', () => {
  const { sync } = makeSync({ minOffsetSamples: 1 });
  sync.addPayload(payload(10.0, { boxes: 0 }), 10.0);
  const results = [sync.select(10.1), sync.select(30.0), sync.select(null)];
  const banned = [
    /\ball[\s\-_]*clear\b/i,
    /\bno\s+(fire|smoke|detections?)\s+(detected|found|present)\b/i,
    /\b0\s+(fires?|detections?)\s+(detected|found)\b/i,
    /\bnothing\s+(detected|found)\b/i,
    /\b(area|zone|scene)\s+(is\s+)?(clear|safe)\b/i,
  ];
  const text = JSON.stringify(results.map((r) => [r.reason, r.staleReason, r.tierLabel, r.tierDescription]));
  for (const pattern of banned) {
    assert.ok(!pattern.test(text), `synchroniser text matched ${pattern}: ${text}`);
  }
});

test('a stale overlay returns no payload at all, so nothing can be drawn from it', () => {
  const { sync } = makeSync({ minOffsetSamples: 1, maxOverlayAgeS: 0.5 });
  sync.addPayload(payload(10.0), 10.0);
  const result = sync.select(20.0);
  assert.equal(result.payload, null);
  assert.equal(result.detections.length, 0);
});

// -------------------------------------------------------------------------

if (failures.length) {
  console.error(`\n${failures.length} failing test(s):\n`);
  for (const { label, err } of failures) {
    console.error(`  FAIL ${label}`);
    console.error(`       ${String(err && err.message).split('\n').join('\n       ')}`);
  }
}
console.log(`\n${results.passed} passed, ${results.failed} failed`);
process.exit(results.failed ? 1 : 0);
