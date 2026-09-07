/**
 * Overlay synchronisation: the three tiers of docs/CONTRACT.md, plus the
 * staleness rule that outranks all three.
 *
 * The problem this module exists for: video and detections travel as separate
 * streams, so a box can be drawn over a frame it was not computed from. On a
 * moving aerial scene at 15 m/s, 300 ms of drift puts the box ~4.5 m off the
 * ground truth while looking entirely authoritative. So the newest payload is
 * never rendered on arrival. It is buffered, and then matched to a frame.
 *
 * Tier 1 -- rtp_ts matched through requestVideoFrameCallback's
 *           `rtpTimestamp` / `mediaTime` pair. Exact; no estimation.
 * Tier 2 -- median-of-last-60 pts-offset estimate, seeded from
 *           `stream_start_pts`. The default fallback.
 * Tier 3 -- unsynchronised: draw the latest payload and say so.
 *
 * Staleness outranks all three: past `maxOverlayAgeS` of *media* time, stop
 * drawing boxes. Note the asymmetry and take it seriously -- tier 3 keeps
 * drawing because it is labelled approximate; staleness stops drawing because
 * the boxes are wrong, not merely imprecise. Dropping the overlay degrades the
 * tool to plain video, which is a safe state: the operator is watching the
 * video anyway. That is the whole premise of the design.
 *
 * This module is deliberately pure: no DOM, no timers, no network. Every input
 * is pushed in, the clock is injectable, and `select()` is a function of state.
 * It runs under node, and it is the piece of this app most worth testing.
 */

/** The three tiers, best first. `OverlaySync.tier` is always one of these. */
export const SYNC_TIER = Object.freeze({
  RTP: 1,
  PTS_OFFSET: 2,
  UNSYNCHRONISED: 3,
});

/** Operator-facing tier names. Short: they live in a status chip. */
export const TIER_LABEL = Object.freeze({
  1: 'RTP-MATCHED',
  2: 'PTS-ESTIMATED',
  3: 'UNSYNCHRONISED',
});

/** Longer explanations, for the detail line under the chip. */
export const TIER_DESCRIPTION = Object.freeze({
  1: 'boxes matched to the exact frame they were computed from',
  2: 'boxes aligned by a rolling median offset estimate',
  3: 'boxes are the latest available and may not match the frame on screen',
});

/** RTP video clock, fixed by RFC 3551 and the unit rtpTimestamp is reported in. */
/**
 * Offset samples required before the tier-2 estimate is trusted at all, no
 * matter how low `minOffsetSamples` is configured.
 *
 * A median of one is not a measurement, it is that one number -- and it is
 * believed completely. One absurd sample (a payload arriving 40 s from the
 * frame on screen) then *defines* the offset, the correction hides the error,
 * and `ageS` comes out at 0: the overlay reports itself perfectly synchronised
 * while every box sits over the wrong ground. That is worse than admitting the
 * timelines are not lined up, because tier 3 says so on screen and the
 * ahead/stale guards still see the raw discrepancy.
 *
 * Two samples cannot corroborate much, but they can disagree, and the median
 * of a pair moves when the second one contradicts the first.
 */
export const MIN_CORROBORATING_SAMPLES = 2;

export const RTP_CLOCK_RATE = 90000;

/** RTP timestamps are 32-bit and wrap; at 90 kHz that is every ~13.25 hours. */
export const RTP_MODULO = 4294967296;

/**
 * Signed difference between two 32-bit RTP timestamps, wrap-correct.
 *
 * A long incident genuinely reaches the wrap, and an unwrapped subtraction
 * there yields a ~13 hour jump that would park the overlay in tier 3 (or,
 * worse, match a box to a frame half a day away) for the rest of the shift.
 *
 * @param {number} a Later timestamp.
 * @param {number} b Earlier timestamp.
 * @returns {number} `a - b` in ticks, in the range [-2^31, 2^31).
 */
export function rtpDelta(a, b) {
  let d = (a - b) % RTP_MODULO;
  if (d < 0) d += RTP_MODULO;
  return d >= RTP_MODULO / 2 ? d - RTP_MODULO : d;
}

/**
 * Median of a list of numbers.
 *
 * Median and not mean, and that is the point of the estimator: arrival-time
 * jitter and the odd very late payload are routine on a WiFi link, and a mean
 * would let a single outlier drag every box sideways.
 *
 * @param {number[]} values Sample values; not mutated.
 * @returns {number|null} The median, or null for an empty list.
 */
export function median(values) {
  if (!values.length) return null;
  const sorted = Array.from(values).sort((a, b) => a - b);
  const mid = sorted.length >> 1;
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

/**
 * Pick the payload to draw from a list sorted ascending by `t`.
 *
 * Prefers the most recent payload at or before the target: a payload from a
 * frame the operator has not been shown yet would put boxes ahead of the
 * scene. If every payload is ahead of the target (start-up, or a badly
 * mis-estimated offset) the nearest one is used instead, which tier 2's
 * estimate then converges to within a second or so.
 *
 * @param {Array<{t: number}>} items Candidates, ascending by `t`.
 * @param {number} target Media-timeline seconds to match.
 * @returns {{t: number}|null} The chosen candidate, or null when empty.
 */
export function selectNearest(items, target) {
  if (!items.length) return null;
  let best = null;
  for (const item of items) {
    if (item.t <= target) best = item;
    else break;
  }
  if (best) return best;
  return items[0];
}

/**
 * Three-tier overlay synchroniser plus the staleness rule.
 *
 * Feed it: `addStatus` (heartbeats), `addPayload` (detection messages) and
 * `addVideoFrame` (one per presented video frame, from
 * requestVideoFrameCallback). Ask it `select()` once per frame you are about
 * to draw.
 */
export class OverlaySync {
  /**
   * @param {object} [options] Tuning; the defaults match StreamConfig.
   * @param {number} [options.maxOverlayAgeS=1.0] Media-seconds after which the
   *   overlay stops drawing boxes entirely. The single most safety-relevant
   *   number in the tablet app.
   * @param {number} [options.overlayBufferS=3.0] Seconds of payloads held for
   *   matching. Long enough to absorb jitter, short enough that a stalled
   *   pipeline hits the staleness rule instead of replaying a ring buffer.
   * @param {number} [options.offsetWindow=60] Samples in the tier-2 median.
   * @param {number} [options.minOffsetSamples=3] Samples before the measured
   *   median is trusted over the `stream_start_pts` seed.
   * @param {number} [options.rtpWindow=48] rtpTimestamp/mediaTime pairs kept.
   * @param {number} [options.maxRtpMapAgeS=1.5] A mapping older than this (in
   *   media time) is not used: rVFC stops firing when the tab is hidden or the
   *   decoder stalls, and a mapping built from frames that far back may predate
   *   an RTP timestamp discontinuity nobody told the tablet about.
   * @param {number} [options.presentedSkewToleranceS=0.5] How far
   *   `video.currentTime` may run ahead of the last frame requestVideoFrameCallback
   *   reported before that report is treated as stopped rather than current.
   * @param {number} [options.maxPayloads=2000] Hard cap on buffered payloads,
   *   so a station sending nonsense pts cannot grow the buffer without bound.
   * @param {() => number} [options.now] Millisecond clock, injectable for tests.
   */
  constructor(options = {}) {
    this.maxOverlayAgeS = numberOr(options.maxOverlayAgeS, 1.0);
    this.overlayBufferS = numberOr(options.overlayBufferS, 3.0);
    this.offsetWindow = Math.max(1, Math.trunc(numberOr(options.offsetWindow, 60)));
    this.minOffsetSamples = Math.max(1, Math.trunc(numberOr(options.minOffsetSamples, 3)));
    this.rtpWindow = Math.max(2, Math.trunc(numberOr(options.rtpWindow, 48)));
    this.maxRtpMapAgeS = numberOr(options.maxRtpMapAgeS, 1.5);
    this.presentedSkewToleranceS = numberOr(options.presentedSkewToleranceS, 0.5);
    this.maxPayloads = Math.max(8, Math.trunc(numberOr(options.maxPayloads, 2000)));
    this.timelineJumpPayloads = Math.max(2, Math.trunc(numberOr(options.timelineJumpPayloads, 3)));
    this.now = typeof options.now === 'function' ? options.now : () => Date.now();

    /** Active tier. Read it; the operator is entitled to see it. @type {number} */
    this.tier = SYNC_TIER.UNSYNCHRONISED;
    /** True when the staleness rule is suppressing boxes. @type {boolean} */
    this.stale = false;

    /** @type {Array<{t: number, pts: number, rtp: number|null, frame: object, arrivalMediaTime: number|null, arrivalWallMs: number}>} */
    this._payloads = [];
    /** Payloads whose pts jumped clear of the buffer, pending corroboration. */
    this._future = [];
    /** @type {Array<{rtp: number, mediaTime: number}>} */
    this._rtpSamples = [];
    /** @type {number[]} */
    this._offsets = [];
    this._seedOffset = null;
    this._seedStreamStartPts = null;

    /** @type {object|null} Most recent heartbeat, for the status bar. */
    this.lastStatus = null;
    /** @type {{mediaTime: number, rtpTimestamp: number|null}|null} */
    this._lastPresented = null;
    this._lastPresentedWallMs = null;
    this._videoAdvancedWallMs = null;
    this._lastPayloadWallMs = null;

    this.stats = {
      payloads: 0,
      payloadsEvicted: 0,
      statuses: 0,
      videoFrames: 0,
      rtpSamples: 0,
      resets: 0,
      timelineJumps: 0,
      payloadsTooOld: 0,
    };
  }

  /**
   * Apply the overlay parameters the station sent with the answer.
   *
   * They are configured on the station (StreamConfig) and enforced here, so a
   * station and a tablet cannot disagree about when boxes stop being drawn.
   * Values that would disable the safety rule are refused rather than trusted:
   * this arrives over the network.
   *
   * @param {{max_overlay_age_s?: number, overlay_buffer_s?: number}} params
   *   The `client_parameters()` block from /webrtc/offer or /webrtc/config.
   * @returns {void}
   */
  setParameters(params) {
    if (!params) return;
    const age = Number(params.max_overlay_age_s);
    if (Number.isFinite(age) && age > 0 && age <= 10) this.maxOverlayAgeS = age;
    const buf = Number(params.overlay_buffer_s);
    if (Number.isFinite(buf) && buf > 0 && buf <= 30) this.overlayBufferS = buf;
    // Mirrors config.validate(): a buffer shorter than the age would discard
    // payloads the tablet is still allowed to draw.
    if (this.overlayBufferS < this.maxOverlayAgeS) this.overlayBufferS = this.maxOverlayAgeS;
  }

  /**
   * Drop all timeline state. Call on reconnect: a new peer connection is a new
   * media timeline, and pts from the old one would match nothing (or, worse,
   * something).
   *
   * @returns {void}
   */
  reset() {
    this._payloads.length = 0;
    this._future.length = 0;
    this._rtpSamples.length = 0;
    this._offsets.length = 0;
    this._seedOffset = null;
    this._seedStreamStartPts = null;
    this._lastPresented = null;
    this._lastPresentedWallMs = null;
    this._videoAdvancedWallMs = null;
    this._lastPayloadWallMs = null;
    this.tier = SYNC_TIER.UNSYNCHRONISED;
    this.stale = false;
    this.stats.resets += 1;
  }

  /**
   * Record a heartbeat and seed the tier-2 estimate from `stream_start_pts`.
   *
   * Seeding matters for the first seconds only, but those are the seconds
   * during which an operator forms an opinion about whether the boxes line up.
   *
   * @param {object} status A parsed `status` message (wire.parseStatus).
   * @param {number|null} [mediaTimeNow] `video.currentTime` when it arrived.
   * @returns {void}
   */
  addStatus(status, mediaTimeNow = null) {
    if (!status) return;
    this.lastStatus = status;
    this.stats.statuses += 1;
    const start = status.stream_start_pts;
    if (start === null || start === undefined || !Number.isFinite(start)) return;
    if (this._seedStreamStartPts !== null && start === this._seedStreamStartPts) return;
    // A changed stream_start_pts means the station restarted its source: the
    // media timeline is new, so everything measured against the old one is
    // worse than useless.
    if (this._seedStreamStartPts !== null) this.reset();
    this._seedStreamStartPts = start;
    const media = Number.isFinite(mediaTimeNow) ? /** @type {number} */ (mediaTimeNow) : 0;
    this._seedOffset = start - media;
  }

  /**
   * Buffer one detections payload and sample the pts offset.
   *
   * @param {object} frame A parsed `detections` message (wire.parseDetections).
   * @param {number|null} [mediaTimeNow] `video.currentTime` at arrival. This is
   *   the tier-2 offset sample; pass null when the video has no timeline yet.
   * @returns {void}
   */
  addPayload(frame, mediaTimeNow = null) {
    if (!frame || !Number.isFinite(frame.pts)) return;
    const wallMs = this.now();
    this._lastPayloadWallMs = wallMs;
    this.stats.payloads += 1;

    const entry = {
      t: frame.pts,
      pts: frame.pts,
      rtp: Number.isFinite(frame.rtp_ts) ? frame.rtp_ts : null,
      frame,
      arrivalMediaTime: Number.isFinite(mediaTimeNow) ? /** @type {number} */ (mediaTimeNow) : null,
      arrivalWallMs: wallMs,
    };

    const newest = this._payloads.length ? this._payloads[this._payloads.length - 1].pts : null;
    if (newest !== null && entry.pts > newest + this.overlayBufferS) {
      // A pts far beyond everything else is either corruption or a genuine
      // timeline jump (the station reconnected its source and restarted pts
      // without a heartbeat in between). Both must be survived, and they are
      // told apart by whether the jump *persists*. Admitting the first such
      // payload straight into the buffer would evict every good payload behind
      // it -- the ring buffer is windowed on the newest pts -- so it is held
      // aside until it is corroborated.
      this._future.push(entry);
      if (this._future.length >= this.timelineJumpPayloads) {
        this._payloads = this._future;
        this._future = [];
        // Offsets measured against the old timeline are now wrong, and one
        // stale sample would sit in a 60-wide median for a full minute.
        this._offsets = this._payloads
          .filter((e) => e.arrivalMediaTime !== null)
          .map((e) => e.pts - /** @type {number} */ (e.arrivalMediaTime));
        this._seedOffset = null;
        this.stats.timelineJumps += 1;
      }
      return;
    }
    this._future.length = 0;

    if (newest !== null && entry.pts < newest - this.overlayBufferS) {
      // Older than the whole buffer: it could never be selected, and its
      // offset sample belongs to a timeline the video has long left behind.
      this.stats.payloadsTooOld += 1;
      return;
    }

    if (Number.isFinite(mediaTimeNow)) {
      this._offsets.push(frame.pts - /** @type {number} */ (mediaTimeNow));
      if (this._offsets.length > this.offsetWindow) {
        this._offsets.splice(0, this._offsets.length - this.offsetWindow);
      }
    }

    // Usually an append; the backward scan handles the out-of-order arrivals
    // that a reliable-ordered channel still produces when the station requeues.
    let i = this._payloads.length;
    while (i > 0 && this._payloads[i - 1].pts > entry.pts) i -= 1;
    this._payloads.splice(i, 0, entry);
    this._prune();
  }

  /**
   * Record one presented video frame, from requestVideoFrameCallback.
   *
   * `rtpTimestamp` is what makes tier 1 possible. It is optional in the spec
   * and absent on some builds and on relayed streams; when it never appears,
   * the tier-1 sample set stays empty and the synchroniser sits in tier 2,
   * which is the correct outcome and not an error.
   *
   * @param {{mediaTime: number, rtpTimestamp?: number|null}} metadata The
   *   VideoFrameCallbackMetadata (or the fields of it that matter).
   * @returns {void}
   */
  addVideoFrame(metadata) {
    if (!metadata || !Number.isFinite(metadata.mediaTime)) return;
    const wallMs = this.now();
    const mediaTime = metadata.mediaTime;
    if (!this._lastPresented || mediaTime !== this._lastPresented.mediaTime) {
      this._videoAdvancedWallMs = wallMs;
    }
    const rtp = Number.isFinite(metadata.rtpTimestamp) ? Math.trunc(/** @type {number} */ (metadata.rtpTimestamp)) : null;
    this._lastPresented = { mediaTime, rtpTimestamp: rtp };
    this._lastPresentedWallMs = wallMs;
    this.stats.videoFrames += 1;

    if (rtp === null) return;
    const last = this._rtpSamples[this._rtpSamples.length - 1];
    if (last && last.rtp === rtp) {
      // Same source frame presented twice (a repeat on a stalled decoder).
      // Keeping the newer mediaTime would tilt the mapping; drop it.
      return;
    }
    this._rtpSamples.push({ rtp, mediaTime });
    this.stats.rtpSamples += 1;
    if (this._rtpSamples.length > this.rtpWindow) {
      this._rtpSamples.splice(0, this._rtpSamples.length - this.rtpWindow);
    }
  }

  /**
   * Map an RTP timestamp onto the media timeline.
   *
   * Interpolates between the two bracketing samples when the timestamp falls
   * inside the observed window, and extrapolates from the nearest sample
   * otherwise. Extrapolation is exact rather than a guess: both clocks run at
   * one second per second, so the slope is 1 by construction and only the
   * intercept has to be observed. The interpolation still earns its place --
   * it averages out the per-sample noise in what the browser reports.
   *
   * @param {number} rtp A 32-bit RTP timestamp from a detections payload.
   * @returns {number|null} Media-timeline seconds, or null with no samples.
   */
  mapRtpToMedia(rtp) {
    const samples = this._rtpSamples;
    if (!samples.length) return null;
    const ref = samples[samples.length - 1];
    const xq = rtpDelta(rtp, ref.rtp) / RTP_CLOCK_RATE;
    let before = null;
    let after = null;
    for (const s of samples) {
      const x = rtpDelta(s.rtp, ref.rtp) / RTP_CLOCK_RATE;
      if (x <= xq && (before === null || x > before.x)) before = { x, y: s.mediaTime };
      if (x >= xq && (after === null || x < after.x)) after = { x, y: s.mediaTime };
    }
    if (before && after && after.x > before.x) {
      const w = (xq - before.x) / (after.x - before.x);
      return before.y + w * (after.y - before.y);
    }
    const anchor = before || after;
    if (!anchor) return null;
    return anchor.y + (xq - anchor.x);
  }

  /**
   * The tier-2 offset currently in force.
   *
   * @returns {{value: number, source: string}|null} The offset in seconds and
   *   where it came from, or null when neither samples nor a seed exist.
   */
  offsetEstimate() {
    if (this._offsets.length >= this.minOffsetSamples) {
      return { value: /** @type {number} */ (median(this._offsets)), source: `median of ${this._offsets.length}` };
    }
    if (this._offsets.length >= MIN_CORROBORATING_SAMPLES) {
      // Fewer samples than the floor: still better than nothing, but say so.
      return { value: /** @type {number} */ (median(this._offsets)), source: `median of ${this._offsets.length} (warming up)` };
    }
    if (this._seedOffset !== null) return { value: this._seedOffset, source: 'seeded from stream_start_pts' };
    return null;
  }

  /**
   * Choose the payload to draw over the frame currently on screen.
   *
   * This is the whole algorithm in one call, and the order of the checks is
   * the contract's order: best tier available, then the staleness rule on top
   * of whatever that tier produced.
   *
   * @param {number|null} [mediaTimeNow] `video.currentTime` at draw time. The
   *   presented frame's own `mediaTime` is preferred when rVFC is running,
   *   because that is the frame the operator is actually looking at.
   * @returns {{tier: number, tierLabel: string, tierDescription: string,
   *   stale: boolean, staleKind: string|null, staleReason: string|null, payload: object|null,
   *   detections: Array<object>, ageS: number|null, matchedPts: number|null,
   *   targetPts: number|null, offsetS: number|null, offsetSource: string|null,
   *   bufferedCount: number, presentedMediaTime: number|null,
   *   videoStalledS: number|null, lastPayloadAgeWallS: number|null,
   *   reason: string}} What to draw, and everything the status bar needs to
   *   explain why.
   */
  select(mediaTimeNow = null) {
    // Which media time is "now". The presented frame's own mediaTime is the
    // truth when rVFC is running, because that is the frame on the glass. But
    // rVFC stops firing in a hidden tab and on a stalled decoder while
    // `currentTime` keeps advancing, and a frozen `presented` would let the
    // staleness rule sleep through exactly the failure it exists to catch --
    // so `currentTime` running clear ahead of it wins.
    let presented = this._lastPresented ? this._lastPresented.mediaTime : null;
    if (Number.isFinite(mediaTimeNow)) {
      const clockNow = /** @type {number} */ (mediaTimeNow);
      if (presented === null || clockNow - presented > this.presentedSkewToleranceS) presented = clockNow;
    }
    const clock = Number.isFinite(mediaTimeNow) ? /** @type {number} */ (mediaTimeNow) : presented;
    const nowMs = this.now();
    this._prune();

    const result = {
      tier: SYNC_TIER.UNSYNCHRONISED,
      tierLabel: TIER_LABEL[SYNC_TIER.UNSYNCHRONISED],
      tierDescription: TIER_DESCRIPTION[SYNC_TIER.UNSYNCHRONISED],
      stale: false,
      staleKind: null,
      staleReason: null,
      payload: null,
      detections: [],
      ageS: null,
      matchedPts: null,
      targetPts: null,
      offsetS: null,
      offsetSource: null,
      bufferedCount: this._payloads.length,
      presentedMediaTime: presented,
      videoStalledS: this._videoStalledS(nowMs),
      lastPayloadAgeWallS: this._lastPayloadWallMs === null ? null : (nowMs - this._lastPayloadWallMs) / 1000,
      reason: 'no payloads buffered',
    };

    if (!this._payloads.length) {
      this.tier = result.tier;
      this.stale = false;
      return result;
    }

    const tier1 = this._selectTier1(presented);
    const chosen = tier1 || this._selectTier2(clock) || this._selectTier3(presented, nowMs);
    Object.assign(result, chosen);
    result.tierLabel = TIER_LABEL[result.tier];
    result.tierDescription = TIER_DESCRIPTION[result.tier];

    // The staleness rule, applied last because it outranks all three tiers.
    // A small negative age is a buffering lead -- the payload belongs to a
    // frame about to be presented -- and is fine.
    if (result.ageS !== null && result.ageS > this.maxOverlayAgeS) {
      result.stale = true;
      result.staleKind = 'age';
      result.staleReason =
        `best match is ${result.ageS.toFixed(2)} s of media time old ` +
        `(limit ${this.maxOverlayAgeS.toFixed(2)} s)`;
    } else if (result.ageS !== null && result.ageS < -this.maxOverlayAgeS) {
      // Not staleness, but the same disease: a payload a second or more ahead
      // of the frame on screen was computed from a scene the operator has not
      // been shown. The contract does not name this case because a healthy
      // station cannot produce it; a mis-estimated offset or a timeline jump
      // can, and the boxes would be just as wrong.
      result.stale = true;
      result.staleKind = 'ahead';
      result.staleReason =
        `best match is ${(-result.ageS).toFixed(2)} s ahead of the frame on screen; ` +
        'the detection and video timelines do not line up';
    }
    if (result.stale) {
      // Boxes are dropped here and not merely dimmed. Live video under boxes
      // computed from a scene that has already moved on is this system's most
      // dangerous single failure mode; plain video is the safe state.
      result.payload = null;
      result.detections = [];
    } else if (result.payload) {
      result.detections = result.payload.detections;
    }

    this.tier = result.tier;
    this.stale = result.stale;
    return result;
  }

  // ------------------------------------------------------------- internals

  /**
   * Tier 1: match on rtp_ts through the rtpTimestamp/mediaTime mapping.
   *
   * @param {number|null} presented Media time of the frame on screen.
   * @returns {object|null} A partial result, or null when tier 1 is unavailable.
   */
  _selectTier1(presented) {
    if (presented === null || !this._rtpSamples.length) return null;
    const newest = this._rtpSamples[this._rtpSamples.length - 1];
    // rVFC stops firing when the tab is hidden or the decoder stalls. The
    // mapping's slope is exactly 1 so it does not drift with age -- but its
    // *intercept* only holds until the next RTP timestamp discontinuity, and
    // frames this far back may already be on the wrong side of one. Refusing
    // it drops the overlay to tier 2, which is labelled; trusting it would
    // misplace every box while looking exact.
    if (Math.abs(presented - newest.mediaTime) > this.maxRtpMapAgeS) return null;

    const candidates = [];
    for (const entry of this._payloads) {
      if (entry.rtp === null) continue;
      const t = this.mapRtpToMedia(entry.rtp);
      if (t === null) continue;
      candidates.push({ t, entry });
    }
    if (!candidates.length) return null;
    candidates.sort((a, b) => a.t - b.t);
    const best = /** @type {{t: number, entry: object}} */ (selectNearest(candidates, presented));
    return {
      tier: SYNC_TIER.RTP,
      payload: best.entry.frame,
      detections: best.entry.frame.detections,
      ageS: presented - best.t,
      matchedPts: best.entry.pts,
      targetPts: null,
      offsetS: null,
      offsetSource: null,
      reason: `rtp_ts matched to media time ${best.t.toFixed(3)} s from ${this._rtpSamples.length} frame samples`,
    };
  }

  /**
   * Tier 2: the median pts-offset estimate.
   *
   * @param {number|null} clock `video.currentTime` at draw time.
   * @returns {object|null} A partial result, or null when no offset exists yet.
   */
  _selectTier2(clock) {
    if (clock === null) return null;
    const offset = this.offsetEstimate();
    if (!offset) return null;
    const target = clock + offset.value;
    const best = /** @type {{pts: number, frame: object}} */ (selectNearest(this._payloads, target));
    if (!best) return null;
    return {
      tier: SYNC_TIER.PTS_OFFSET,
      payload: best.frame,
      detections: best.frame.detections,
      ageS: target - best.pts,
      matchedPts: best.pts,
      targetPts: target,
      offsetS: offset.value,
      offsetSource: offset.source,
      reason: `pts offset ${offset.value.toFixed(3)} s (${offset.source})`,
    };
  }

  /**
   * Tier 3: unsynchronised. Draw the latest payload, labelled.
   *
   * The age here is measured from how much media time (or, with no media
   * clock at all, wall time) has passed since the payload arrived. It is
   * cruder than tiers 1 and 2, which is exactly why the staleness rule still
   * has to be able to act on it: an unsynchronised overlay may be approximate,
   * but it may not be old.
   *
   * @param {number|null} presented Media time of the frame on screen.
   * @param {number} nowMs Wall clock, milliseconds.
   * @returns {object} A partial result. Tier 3 always produces one.
   */
  _selectTier3(presented, nowMs) {
    const best = this._payloads[this._payloads.length - 1];
    let ageS = null;
    let basis = 'wall clock';
    if (presented !== null && best.arrivalMediaTime !== null) {
      ageS = presented - best.arrivalMediaTime;
      basis = 'media time since arrival';
    } else {
      ageS = (nowMs - best.arrivalWallMs) / 1000;
    }
    return {
      tier: SYNC_TIER.UNSYNCHRONISED,
      payload: best.frame,
      detections: best.frame.detections,
      ageS,
      matchedPts: best.pts,
      targetPts: null,
      offsetS: null,
      offsetSource: null,
      reason: `latest payload, unmatched; age from ${basis}`,
    };
  }

  /**
   * How long the video clock has been standing still, in seconds.
   *
   * Reported, not acted on. A frozen picture with boxes matched to it is
   * self-consistent, so this is not staleness in the contract's sense -- but
   * the operator still needs to know the picture is frozen, and the status bar
   * shows it.
   *
   * @param {number} nowMs Wall clock, milliseconds.
   * @returns {number|null} Seconds since the media time last advanced.
   */
  _videoStalledS(nowMs) {
    if (this._videoAdvancedWallMs === null) return null;
    return (nowMs - this._videoAdvancedWallMs) / 1000;
  }

  /**
   * Drop payloads outside the ring buffer window.
   *
   * The window is measured against the newest pts, not against the wall
   * clock, so a pipeline that pauses and resumes does not lose the payloads
   * it is about to need.
   *
   * @returns {void}
   */
  _prune() {
    if (!this._payloads.length) return;
    const newest = this._payloads[this._payloads.length - 1].pts;
    const cutoff = newest - this.overlayBufferS;
    let drop = 0;
    while (drop < this._payloads.length - 1 && this._payloads[drop].pts < cutoff) drop += 1;
    if (this._payloads.length - drop > this.maxPayloads) {
      drop = this._payloads.length - this.maxPayloads;
    }
    if (drop > 0) {
      this._payloads.splice(0, drop);
      this.stats.payloadsEvicted += drop;
    }
  }
}

/**
 * Numeric option with a default, tolerant of undefined but not of nonsense.
 *
 * @param {unknown} value Candidate value.
 * @param {number} fallback Default.
 * @returns {number} `value` when finite, otherwise `fallback`.
 */
function numberOr(value, fallback) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}
