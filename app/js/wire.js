/**
 * The wire contract, tablet side. Mirrors station/core/types.py exactly.
 *
 * Field names here are the Python attribute names (snake_case, `track_id` for
 * the wire's `track`) so that a reader can diff this file against types.py line
 * by line. That matters more than JS naming convention: these two files are one
 * protocol, and drift between them is invisible until it is drawn on a screen
 * a crew is acting on.
 *
 * The whole module is pure -- no DOM, no network -- so it can be unit-tested
 * under node.
 *
 * Two rules that are not style preferences (docs/SAFETY.md):
 *
 *  1. An unknown `v` is a hard parse error. A build that silently dropped the
 *     fields it did not understand would render a partial overlay, and a
 *     partial overlay looks exactly like a quiet scene.
 *  2. An empty `detections` array is normal, frequent, and carries no
 *     information about the world. It parses successfully, it renders as
 *     nothing at all, and nothing in this module invites any other reading.
 */

/**
 * Bumped on any incompatible change; must equal types.py WIRE_VERSION.
 *
 * v2 added the `person` class. The message shape did not change, so a v1 build
 * would have parsed it and drawn the box in the unknown-class fallback style --
 * a thin white dashed rectangle labelled '?'. For a box around a human being,
 * being mislabelled is worse than the tablet refusing to connect, which is why
 * this is a version bump and not an additive change.
 */
export const WIRE_VERSION = 2;

export const MSG_DETECTIONS = 'detections';
export const MSG_STATUS = 'status';

export const CLASS_FIRE = 'fire';
export const CLASS_SMOKE = 'smoke';
export const CLASS_PERSON = 'person';
/** Ordered, and the order is load-bearing: it is the model's class index order. */
export const CLASSES = Object.freeze([CLASS_FIRE, CLASS_SMOKE, CLASS_PERSON]);

/**
 * Classes that describe a human being.
 *
 * Kept as a set rather than a string comparison because several rules key on
 * it: the overlay refuses to let one fall through to the unknown-class style,
 * and nothing may summarise these away as a count.
 */
export const LIFE_SAFETY_CLASSES = Object.freeze(new Set([CLASS_PERSON]));

/** Liveness of the station pipeline. Never a statement about the scene. */
export const PipelineState = Object.freeze({
  STARTING: 'starting',
  RUNNING: 'running',
  DEGRADED: 'degraded',
  STALLED: 'stalled',
  STOPPED: 'stopped',
  ALL: Object.freeze(['starting', 'running', 'degraded', 'stalled', 'stopped']),
});

/** Data channel label. Normative -- see docs/CONTRACT.md. */
export const DETECTIONS_CHANNEL = 'detections';

/**
 * A message that could not be parsed against this build's contract.
 *
 * `fatal` marks the subset the app must not try to recover from by waiting for
 * the next message: a version mismatch means every subsequent message is also
 * unreadable, and the operator has to be told the overlay is off rather than
 * left watching a screen that has quietly stopped drawing.
 */
export class WireError extends Error {
  /**
   * @param {string} message Human-readable failure, shown to the operator.
   * @param {{fatal?: boolean, version?: unknown}} [options]
   */
  constructor(message, options = {}) {
    super(message);
    this.name = 'WireError';
    this.fatal = options.fatal === true;
    this.version = options.version;
  }
}

/**
 * Coerce to a finite number or throw. Mirrors `_finite` in types.py.
 *
 * @param {string} name Field name, for the error message.
 * @param {unknown} value Raw value off the wire.
 * @returns {number} The value as a finite float.
 */
function finite(name, value) {
  const out = typeof value === 'number' ? value : Number(value);
  if (value === null || value === '' || !Number.isFinite(out)) {
    throw new WireError(`${name} must be a finite number, got ${JSON.stringify(value)}`);
  }
  return out;
}

/**
 * Optional finite number: absent/null stays null, anything else is validated.
 *
 * @param {string} name Field name, for the error message.
 * @param {unknown} value Raw value off the wire.
 * @returns {number|null} The value, or null when the field was not sent.
 */
function optionalFinite(name, value) {
  if (value === undefined || value === null) return null;
  return finite(name, value);
}

/**
 * Parse `[x1, y1, x2, y2]`, normalised 0..1, origin top-left.
 *
 * Inverted corners throw, exactly as BBox.__post_init__ does. Out-of-range
 * values are kept, not clipped: the station already clamps, and a box that
 * runs off the frame edge is the case we least want to silently discard.
 *
 * @param {unknown} raw The wire array.
 * @returns {{x1: number, y1: number, x2: number, y2: number}} Normalised box.
 */
export function parseBox(raw) {
  if (!Array.isArray(raw) || raw.length !== 4) {
    throw new WireError(`box must have 4 elements, got ${JSON.stringify(raw)}`);
  }
  const [x1, y1, x2, y2] = raw.map((v, i) => finite(`box[${i}]`, v));
  if (x2 < x1 || y2 < y1) {
    throw new WireError(`box corners are inverted: ${JSON.stringify(raw)}`);
  }
  return { x1, y1, x2, y2 };
}

/**
 * Parse one detection object.
 *
 * @param {Record<string, unknown>} raw One entry of the `detections` array.
 * @returns {{cls: string, conf: number, box: object, track_id: number|null,
 *            persisted: number, first_seen_pts: number|null}} The detection.
 */
export function parseDetection(raw) {
  if (raw === null || typeof raw !== 'object') {
    throw new WireError(`detection must be an object, got ${JSON.stringify(raw)}`);
  }
  const cls = typeof raw.cls === 'string' ? raw.cls : '';
  if (!cls) throw new WireError('detection class must be non-empty');
  const conf = finite('conf', raw.conf);
  if (conf < 0 || conf > 1) throw new WireError(`conf must be in 0..1, got ${conf}`);
  const persisted = raw.persisted === undefined ? 1 : Math.trunc(finite('persisted', raw.persisted));
  if (persisted < 1) throw new WireError(`persisted must be >= 1, got ${persisted}`);
  const track = raw.track === undefined || raw.track === null ? null : Math.trunc(finite('track', raw.track));
  return {
    cls,
    conf,
    box: parseBox(raw.box),
    track_id: track,
    persisted,
    first_seen_pts: optionalFinite('first_seen_pts', raw.first_seen_pts),
    /** True for a class this build has no drawing rule for; still rendered. */
    unknown_class: !CLASSES.includes(cls),
  };
}

/**
 * Parse the model block. Absent is allowed; the fields inside are tolerant,
 * matching ModelInfo.from_wire, because provenance is for the log and the
 * status bar, and a missing `imgsz` must not cost the operator an overlay.
 *
 * @param {unknown} raw The `model` object, or null/undefined.
 * @returns {object|null} Model info, or null when not sent.
 */
export function parseModel(raw) {
  if (raw === undefined || raw === null) return null;
  if (typeof raw !== 'object') throw new WireError('model must be an object');
  const r = /** @type {Record<string, unknown>} */ (raw);
  return {
    name: r.name === undefined ? 'unknown' : String(r.name),
    version: r.version === undefined ? 'unknown' : String(r.version),
    classes: Array.isArray(r.classes) ? r.classes.map(String) : Array.from(CLASSES),
    weights_sha: r.weights_sha === undefined || r.weights_sha === null ? null : String(r.weights_sha),
    imgsz: optionalFinite('model.imgsz', r.imgsz),
    conf_threshold: optionalFinite('model.conf_threshold', r.conf_threshold),
  };
}

/**
 * Parse a `detections` message: everything the model concluded about one frame.
 *
 * @param {Record<string, unknown>} raw The decoded JSON object.
 * @returns {object} A FrameDetections-shaped object.
 */
export function parseDetections(raw) {
  const list = raw.detections === undefined ? [] : raw.detections;
  if (!Array.isArray(list)) throw new WireError('detections must be an array');
  return {
    type: MSG_DETECTIONS,
    frame_id: Math.trunc(finite('frame_id', raw.frame_id)),
    pts: finite('pts', raw.pts),
    wall_time: raw.wall_time === undefined ? '' : String(raw.wall_time),
    detections: list.map(parseDetection),
    model: parseModel(raw.model),
    inference_ms: optionalFinite('inference_ms', raw.inference_ms),
    // Null here is not a defect: a MediaMTX relay cannot supply it, and its
    // absence is precisely what drops the tablet from sync tier 1 to tier 2.
    rtp_ts: raw.rtp_ts === undefined || raw.rtp_ts === null ? null : Math.trunc(finite('rtp_ts', raw.rtp_ts)),
    source_id: raw.source_id === undefined || raw.source_id === null ? null : String(raw.source_id),
  };
}

/**
 * Parse a `status` heartbeat. Describes the pipeline, never the scene.
 *
 * @param {Record<string, unknown>} raw The decoded JSON object.
 * @returns {object} A PipelineStatus-shaped object.
 */
export function parseStatus(raw) {
  const state = String(raw.state ?? '');
  if (!PipelineState.ALL.includes(state)) {
    throw new WireError(`unknown pipeline state ${JSON.stringify(raw.state)}; expected one of ${PipelineState.ALL.join(', ')}`);
  }
  return {
    type: MSG_STATUS,
    state,
    wall_time: raw.wall_time === undefined ? '' : String(raw.wall_time),
    source: raw.source === undefined || raw.source === null ? null : String(raw.source),
    model: parseModel(raw.model),
    source_fps: optionalFinite('source_fps', raw.source_fps),
    inference_fps: optionalFinite('inference_fps', raw.inference_fps),
    last_inference_pts: optionalFinite('last_inference_pts', raw.last_inference_pts),
    last_inference_wall_time:
      raw.last_inference_wall_time === undefined || raw.last_inference_wall_time === null
        ? null
        : String(raw.last_inference_wall_time),
    dropped_frames: raw.dropped_frames === undefined ? 0 : Math.trunc(finite('dropped_frames', raw.dropped_frames)),
    uptime_s: optionalFinite('uptime_s', raw.uptime_s),
    stream_start_pts: optionalFinite('stream_start_pts', raw.stream_start_pts),
    note: raw.note === undefined || raw.note === null ? null : String(raw.note),
  };
}

/**
 * Parse one data-channel message, rejecting versions this build cannot read.
 *
 * Mirrors types.parse_message. Refusing an unknown `v` is deliberate: an old
 * tablet against a new station would otherwise drop the fields it cannot see,
 * and a silently partial overlay is indistinguishable from a quiet scene.
 *
 * @param {string|ArrayBuffer|Uint8Array|object} raw One data-channel message.
 * @returns {object} A parsed `detections` or `status` object; check `.type`.
 * @throws {WireError} On malformed JSON, an unknown version, or an unknown type.
 */
export function parseMessage(raw) {
  let payload = raw;
  if (raw instanceof ArrayBuffer || ArrayBuffer.isView(raw)) {
    const bytes = raw instanceof ArrayBuffer ? new Uint8Array(raw) : new Uint8Array(
      /** @type {ArrayBufferView} */ (raw).buffer,
      /** @type {ArrayBufferView} */ (raw).byteOffset,
      /** @type {ArrayBufferView} */ (raw).byteLength,
    );
    payload = new TextDecoder('utf-8', { fatal: true }).decode(bytes);
  }
  if (typeof payload === 'string') {
    try {
      payload = JSON.parse(payload);
    } catch (err) {
      throw new WireError(`message is not valid JSON: ${err.message}`);
    }
  }
  if (payload === null || typeof payload !== 'object' || Array.isArray(payload)) {
    throw new WireError('message must be a JSON object');
  }
  const obj = /** @type {Record<string, unknown>} */ (payload);
  const version = obj.v;
  if (version !== WIRE_VERSION) {
    throw new WireError(
      `unsupported wire version ${JSON.stringify(version)}; this tablet build speaks v${WIRE_VERSION}. ` +
        'Update the tablet app from the station before relying on the overlay.',
      { fatal: true, version },
    );
  }
  const kind = obj.type;
  if (kind === MSG_DETECTIONS) return parseDetections(obj);
  if (kind === MSG_STATUS) return parseStatus(obj);
  throw new WireError(`unknown message type ${JSON.stringify(kind)}`);
}

/**
 * Highest confidence on a frame, or null when the frame carried none.
 *
 * Deliberately null and not 0: a zero is a number, and a number invites being
 * drawn as a gauge reading "nothing here", which is the one thing this
 * interface may never say. Mirrors FrameDetections.max_conf.
 *
 * @param {{detections: Array<{conf: number}>}} frame A parsed detections message.
 * @returns {number|null} The maximum confidence, or null.
 */
export function maxConf(frame) {
  if (!frame || !frame.detections.length) return null;
  return frame.detections.reduce((acc, d) => (d.conf > acc ? d.conf : acc), 0);
}

/**
 * The detections of one class on a frame.
 *
 * @param {{detections: Array<{cls: string}>}} frame A parsed detections message.
 * @param {string} cls Class name, e.g. CLASS_FIRE.
 * @returns {Array<object>} The matching detections, in wire order.
 */
export function ofClass(frame, cls) {
  if (!frame) return [];
  return frame.detections.filter((d) => d.cls === cls);
}
