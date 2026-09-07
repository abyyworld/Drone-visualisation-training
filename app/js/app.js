/**
 * Application wiring: connection -> synchroniser -> overlay, plus the status
 * bar, the screen wake lock and the offline shell.
 *
 * The status bar is not decoration. Three of its four chips exist so that a
 * pipeline which has quietly stopped looking is visible as a stopped pipeline
 * instead of as a calm picture: the link state, the station's own liveness,
 * and the timestamp of the last completed inference. An operator can trust
 * boxes that appear. What they cannot do -- ever -- is read the absence of
 * boxes as information, and the only defence against that is showing them,
 * continuously, whether the machine is still looking.
 */

import { Overlay } from './overlay.js';
import { OverlaySync, SYNC_TIER } from './sync.js';
import { StationConnection, CONN, DEFAULT_PREFIX } from './connection.js';
import { PipelineState } from './wire.js';

const el = (id) => document.getElementById(id);

const video = /** @type {HTMLVideoElement} */ (el('video'));
const canvas = /** @type {HTMLCanvasElement} */ (el('overlay'));
const banner = el('banner');

/**
 * Provenance strip, inserted next to the banner rather than declared in the
 * markup so this stays a pure addition. Styled inline: it must be legible even
 * if a stylesheet fails to load, because what it says is exactly what must not
 * be missed.
 */
const provenance = (() => {
  if (!banner || !banner.parentNode) return null;
  const node = document.createElement('div');
  node.id = 'provenance';
  node.setAttribute('role', 'status');
  node.hidden = true;
  node.style.cssText = [
    'background:#3b1d00', 'color:#ffd9a0', 'border:2px solid #ff9e1e',
    'font-weight:700', 'letter-spacing:0.04em', 'padding:0.5rem 0.75rem',
    'text-align:center', 'font-size:clamp(0.8rem,2.2vw,1rem)',
  ].join(';');
  banner.parentNode.insertBefore(node, banner);
  return node;
})();
const playButton = el('playButton');

const overlay = new Overlay(canvas, { fit: 'contain' });
const sync = new OverlaySync();

const prefix = document.querySelector('meta[name="signaling-prefix"]')?.getAttribute('content') || DEFAULT_PREFIX;
const conn = new StationConnection({ prefix });

/** Everything the status bar needs that is not in the sync result. */
const state = {
  link: { state: CONN.IDLE, error: null, nextRetryS: null, attempt: 0 },
  status: null,
  statusSeenAt: null,
  /** Local monotonic stamp of when `last_inference_wall_time` last changed. */
  lastInferenceSeenAt: null,
  lastInferenceWall: null,
  /** Station wall clock minus tablet wall clock, seconds. */
  clockSkewS: null,
  wake: 'off',
  wakeWanted: true,
  cache: 'not registered',
  videoPlaying: false,
  parameters: null,
};

// --------------------------------------------------------------- rendering

/**
 * Draw one overlay frame and refresh the status bar.
 *
 * Called from requestVideoFrameCallback when the browser presents a frame, and
 * from a 10 Hz safety-net timer regardless. The timer is the important half:
 * when the decoder stalls, rVFC stops firing, and without an independent tick
 * the last boxes drawn would simply stay on the screen -- which is precisely
 * the failure the staleness rule exists to prevent.
 *
 * @returns {void}
 */
function render() {
  const mediaTime = Number.isFinite(video.currentTime) ? video.currentTime : null;
  const result = sync.select(mediaTime);
  overlay.draw(video, result, { protocolError: conn.protocolError });
  updateStatus(result);
}

let framePending = false;

/**
 * Feed one presented video frame to the synchroniser and redraw on it.
 *
 * @param {number} _now DOMHighResTimeStamp, unused.
 * @param {VideoFrameCallbackMetadata} metadata Frame metadata; `rtpTimestamp`
 *   is what makes sync tier 1 possible and is absent on some builds.
 * @returns {void}
 */
function onVideoFrame(_now, metadata) {
  framePending = false;
  sync.addVideoFrame({ mediaTime: metadata.mediaTime, rtpTimestamp: metadata.rtpTimestamp });
  render();
  armFrameCallback();
}

/**
 * (Re)arm requestVideoFrameCallback.
 *
 * The chain has to be re-armed rather than assumed: a reconnect replaces the
 * element's source, and a callback registered against the old stream is never
 * called again -- which would leave the overlay running on the 10 Hz timer
 * alone and silently out of reach of sync tier 1.
 *
 * @returns {void}
 */
function armFrameCallback() {
  if (framePending || !video.requestVideoFrameCallback) return;
  framePending = true;
  video.requestVideoFrameCallback(onVideoFrame);
}

if (video.requestVideoFrameCallback) {
  armFrameCallback();
} else {
  // No rVFC: tier 1 is unreachable on this browser and the synchroniser will
  // sit in tier 2, which the overlay labels. Nothing else changes.
  console.info('requestVideoFrameCallback is unavailable; overlay sync tier 1 is out of reach here');
}
setInterval(render, 100);

// -------------------------------------------------------------- status bar

/**
 * Set one status chip.
 *
 * @param {'link'|'pipeline'|'overlay'|'inference'} name Chip name; the markup
 *   ids are `chip<Name>`, `<name>Value` and `<name>Note`.
 * @param {'good'|'warn'|'bad'|'idle'} level Severity, drives the edge bar.
 * @param {string} value The big word.
 * @param {string} note The small line under it.
 * @returns {void}
 */
function chip(name, level, value, note) {
  const node = el(`chip${name[0].toUpperCase()}${name.slice(1)}`);
  if (node.dataset.state !== level) node.dataset.state = level;
  text(`${name}Value`, value);
  text(`${name}Note`, note);
}

/**
 * Write text into a node only when it changed.
 *
 * The status bar is refreshed on every presented frame -- up to 60 times a
 * second on a tablet that is also decoding video -- and unconditional
 * textContent writes there cost a layout pass each time for nothing.
 *
 * @param {string} id Element id.
 * @param {string} value New text.
 * @returns {void}
 */
function text(id, value) {
  const node = el(id);
  if (node.textContent !== value) node.textContent = value;
}

/**
 * Seconds since a local performance stamp.
 *
 * @param {number|null} stamp A `performance.now()` value, or null.
 * @returns {number|null} Elapsed seconds, or null.
 */
function since(stamp) {
  return stamp === null ? null : (performance.now() - stamp) / 1000;
}

/**
 * Format an RFC 3339 stamp as a wall clock time.
 *
 * @param {string|null} iso The timestamp, or null.
 * @returns {string} `HH:MM:SS`, or an em dash.
 */
function clockOf(iso) {
  if (!iso) return '—';
  const ms = Date.parse(iso);
  if (!Number.isFinite(ms)) return iso;
  const d = new Date(ms);
  return [d.getHours(), d.getMinutes(), d.getSeconds()]
    .map((n) => String(n).padStart(2, '0'))
    .join(':');
}

/**
 * Refresh every chip and the details panel.
 *
 * @param {object} result The current `OverlaySync.select()` result.
 * @returns {void}
 */
function updateStatus(result) {
  // --- link
  const link = state.link;
  if (conn.protocolError) {
    chip('link', 'bad', 'INCOMPATIBLE', 'tablet and station speak different protocol versions');
  } else if (link.state === CONN.CONNECTED) {
    chip('link', 'good', 'LIVE', `peer ${conn.peerId ? conn.peerId.slice(0, 8) : '—'}`);
  } else if (link.state === CONN.RETRYING) {
    const wait = link.nextRetryS === null ? '' : ` — retry in ${link.nextRetryS.toFixed(0)} s`;
    chip('link', 'bad', 'LINK DOWN', `${link.error || 'connection lost'}${wait}`);
  } else if (link.state === CONN.CONNECTING) {
    chip('link', 'warn', 'CONNECTING', `attempt ${link.attempt}`);
  } else if (link.state === CONN.CLOSED) {
    chip('link', 'idle', 'CLOSED', 'reconnect to resume');
  } else {
    chip('link', 'idle', 'STARTING', 'opening peer connection');
  }

  // --- pipeline (the station's liveness, never the scene)
  const st = state.status;
  const beatAgeS = state.statusSeenAt === null || state.statusSeenAt === undefined
    ? null
    : (performance.now() - state.statusSeenAt) / 1000;
  // Three missed heartbeats. The station sends one per status_interval_s, so a
  // gap this long means the station is gone, not merely slow -- and a stale
  // "RUNNING" is a claim the tablet has no basis to keep making.
  const beatDeadlineS = (state.parameters?.status_interval_s ?? 1.0) * 3;
  if (!st) {
    chip('pipeline', 'warn', 'UNKNOWN', 'awaiting heartbeat from the station');
  } else if (beatAgeS !== null && beatAgeS > beatDeadlineS) {
    chip('pipeline', 'bad', 'NO HEARTBEAT',
      `nothing from the station for ${beatAgeS.toFixed(0)} s; last state was ${st.state}`);
  } else {
    const fps = st.inference_fps === null ? '—' : `${st.inference_fps.toFixed(1)}/s`;
    const src = st.source_fps === null ? '—' : `${st.source_fps.toFixed(1)}/s`;
    const level =
      st.state === PipelineState.RUNNING ? 'good'
        : st.state === PipelineState.DEGRADED ? 'warn'
          : st.state === PipelineState.STARTING ? 'warn' : 'bad';
    chip('pipeline', level, st.state.toUpperCase(), st.note || `inference ${fps} · source ${src}`);
  }

  // --- overlay
  if (conn.protocolError) {
    chip('overlay', 'bad', 'OFF', conn.protocolError);
  } else if (result.stale) {
    chip('overlay', 'bad', 'STALE — BOXES OFF', result.staleReason || 'overlay older than the limit');
  } else if (!result.payload) {
    chip('overlay', 'idle', 'STANDBY', 'awaiting detection payloads');
  } else if (result.tier === SYNC_TIER.UNSYNCHRONISED) {
    chip('overlay', 'warn', result.tierLabel, result.tierDescription);
  } else {
    const age = result.ageS === null ? '' : ` · match ${(result.ageS * 1000).toFixed(0)} ms`;
    chip('overlay', 'good', result.tierLabel, `${result.tierDescription}${age}`);
  }

  // --- last inference
  // Measured against the tablet's own monotonic clock, from the moment the
  // value changed. The station's wall time is shown as it sent it, but never
  // subtracted from the tablet's clock: the two machines are on a LAN with no
  // time source, and a 40 s skew would otherwise read as a 40 s stall.
  const age = since(state.lastInferenceSeenAt);
  if (!st || !state.lastInferenceWall) {
    chip('inference', 'warn', '—', 'the station has not reported an inference yet');
  } else {
    const stall = state.parameters?.stall_after_s ?? 3.0;
    const level = age === null ? 'warn' : age > stall * 2 ? 'bad' : age > stall ? 'warn' : 'good';
    const suffix = age === null ? '' : ` · ${age.toFixed(1)} s ago`;
    chip('inference', level, clockOf(state.lastInferenceWall), `station clock${suffix}`);
  }

  updateBanner(result);
  updateProvenance();
  if (!el('details').hidden) updateDetails(result);
}


/**
 * The provenance strip: says when the boxes are not a live model on a live feed.
 *
 * Deliberately separate from the banner, which shows one message in priority
 * order -- provenance would be masked the moment the link dropped, and "these
 * boxes are fabricated" must not be the message that gets outranked. It is
 * persistent for the same reason: someone walking up to the tablet mid-session
 * has to be able to tell a demonstration from an incident.
 *
 * Two cases, both of which produce boxes indistinguishable from the real thing:
 * the station running its stub runner (no model at all, generated detections),
 * and replay of a recorded incident (real detections, but from the past).
 *
 * @returns {void}
 */
function updateProvenance() {
  if (!provenance) return;
  const replay = state.parameters?.replay || null;
  const modelName = state.status?.model?.name || null;
  const stub = modelName === 'stub';

  let text = null;
  if (replay) {
    const which = replay.incident ? ` — ${replay.incident}` : '';
    text = `REPLAY OF A RECORDING${which} — NOT A LIVE FEED`;
  } else if (stub) {
    text = 'GENERATED BOXES — NO MODEL IS RUNNING';
  }

  if (text) {
    provenance.textContent = text;
    provenance.hidden = false;
  } else {
    provenance.hidden = true;
  }
}

/**
 * The one banner over the picture, in priority order.
 *
 * @param {object} result The current sync result.
 * @returns {void}
 */
function updateBanner(result) {
  let text = null;
  let kind = 'warn';
  if (conn.protocolError) {
    text = `OVERLAY DISABLED — ${conn.protocolError}`;
    kind = 'fatal';
  } else if (state.link.state === CONN.RETRYING) {
    text = `LINK DOWN — ${state.link.error || 'reconnecting'}. The picture below is frozen or blank.`;
    kind = 'fatal';
  } else if (state.status && state.status.state === PipelineState.STALLED) {
    text = 'STATION PIPELINE STALLED — the model is not processing frames.';
    kind = 'fatal';
  } else if (result.videoStalledS !== null && result.videoStalledS > 2 && state.videoPlaying) {
    text = `VIDEO FROZEN — no new frame presented for ${result.videoStalledS.toFixed(0)} s.`;
    kind = 'warn';
  }
  if (text) {
    banner.textContent = text;
    banner.dataset.kind = kind;
    banner.hidden = false;
  } else {
    banner.hidden = true;
  }
}

/**
 * Refresh the details panel, when it is open.
 *
 * @param {object} result The current sync result.
 * @returns {void}
 */
function updateDetails(result) {
  const st = state.status;
  const p = state.parameters || {};
  const model = st?.model || null;
  const set = text;
  set('dStation', conn.peerId ? `connected · peer ${conn.peerId}` : 'not connected');
  set('dSource', st?.source || '—');
  set('dModel', model ? `${model.name} ${model.version}` : '—');
  set('dWeights', model?.weights_sha ? `${model.weights_sha} · imgsz ${model.imgsz ?? '—'} · conf ≥ ${model.conf_threshold ?? '—'}` : '—');
  set('dTier', `${result.tier} — ${result.tierLabel}`);
  set('dSyncDetail', result.reason + (result.offsetS === null ? '' : ` · offset ${result.offsetS.toFixed(3)} s`));
  set('dMaxAge', `${sync.maxOverlayAgeS.toFixed(2)} s of media time · buffer ${sync.overlayBufferS.toFixed(1)} s`);
  set('dBuffer', `${result.bufferedCount} payloads`);
  set('dFps', `${st?.source_fps?.toFixed(1) ?? '—'} / ${st?.inference_fps?.toFixed(1) ?? '—'} fps`);
  set('dDropped', String(st?.dropped_frames ?? '—'));
  set('dVideo', video.videoWidth
    ? `${video.videoWidth}×${video.videoHeight} · t=${video.currentTime.toFixed(2)} s · ${state.videoPlaying ? 'playing' : 'not playing'}`
    : 'no frames yet');
  set('dPeer', `${state.link.state} · attempt ${state.link.attempt} · drops ${conn.stats.drops}`);
  set('dMessages', `${conn.stats.messages} / ${conn.stats.parseErrors}`);
  set('dWake', state.wake);
  set('dCache', state.cache);
  set('dSkew', state.clockSkewS === null ? '—' : `${state.clockSkewS >= 0 ? '+' : ''}${state.clockSkewS.toFixed(1)} s (station ahead of tablet)`);
}

// ------------------------------------------------------------- connection

conn.on('state', (payload) => {
  state.link = { state: payload.state, error: payload.error, nextRetryS: payload.nextRetryS ?? null, attempt: payload.attempt };
  if (payload.state === CONN.CONNECTED) sync.reset(); // new peer, new media timeline
  render();
});

conn.on('parameters', (params) => {
  state.parameters = params;
  // The staleness limit is configured on the station and enforced here.
  sync.setParameters(params);
});

conn.on('track', (stream) => {
  video.srcObject = stream;
  void startPlayback();
});

conn.on('detections', (frame) => {
  sync.addPayload(frame, Number.isFinite(video.currentTime) ? video.currentTime : null);
});

conn.on('status', (status) => {
  state.status = status;
  // When the heartbeat arrived, in real time. Without this the chip below would
  // keep asserting the last known state forever -- painting a confident
  // "RUNNING" for a station that died minutes ago.
  state.statusSeenAt = performance.now();
  sync.addStatus(status, Number.isFinite(video.currentTime) ? video.currentTime : null);
  if (status.last_inference_wall_time && status.last_inference_wall_time !== state.lastInferenceWall) {
    state.lastInferenceWall = status.last_inference_wall_time;
    state.lastInferenceSeenAt = performance.now();
  }
  const stationMs = Date.parse(status.wall_time);
  if (Number.isFinite(stationMs)) state.clockSkewS = (stationMs - Date.now()) / 1000;
});

conn.on('protocol-error', () => render());

// ---------------------------------------------------------------- playback

/**
 * Start (or restart) playback, surfacing the tap-to-start button when the
 * browser refuses to autoplay.
 *
 * @returns {Promise<void>} Resolves once play has been attempted.
 */
async function startPlayback() {
  try {
    await video.play();
    playButton.hidden = true;
  } catch (err) {
    // iOS in particular will refuse until a gesture. Say so plainly rather
    // than leaving a black rectangle that looks like a working camera feed
    // pointed at a dark scene.
    playButton.hidden = false;
    console.info('autoplay refused; waiting for a tap', err);
  }
}

video.addEventListener('playing', () => {
  state.videoPlaying = true;
  playButton.hidden = true;
  armFrameCallback();
});
video.addEventListener('loadedmetadata', () => { armFrameCallback(); overlay.resize(); });
video.addEventListener('pause', () => { state.videoPlaying = false; });
video.addEventListener('emptied', () => { state.videoPlaying = false; });
playButton.addEventListener('click', () => {
  void startPlayback();
  void acquireWakeLock();
});

// --------------------------------------------------------------- wake lock

/** @type {WakeLockSentinel|null} */
let wakeSentinel = null;

/**
 * Take the screen wake lock, if it is wanted and available.
 *
 * A tablet that sleeps mid-incident is useless, and the operator will not
 * notice it happened until they need it. The lock is dropped by the browser on
 * every visibility change, so this is called again on every return.
 *
 * @returns {Promise<void>} Resolves when the attempt has settled.
 */
async function acquireWakeLock() {
  if (!('wakeLock' in navigator)) {
    state.wake = 'unsupported on this browser — set the screen timeout manually';
    return;
  }
  if (!state.wakeWanted || document.visibilityState !== 'visible') return;
  if (wakeSentinel && !wakeSentinel.released) return;
  try {
    wakeSentinel = await navigator.wakeLock.request('screen');
    state.wake = 'held';
    wakeSentinel.addEventListener('release', () => {
      state.wake = 'released by the system';
    });
  } catch (err) {
    state.wake = `refused: ${err.message}`;
  }
}

/**
 * Release the wake lock.
 *
 * @returns {Promise<void>} Resolves when released.
 */
async function releaseWakeLock() {
  try {
    if (wakeSentinel && !wakeSentinel.released) await wakeSentinel.release();
  } catch (err) {
    console.debug('wake lock release failed', err);
  }
  wakeSentinel = null;
  if (state.wake === 'held') state.wake = 'off';
}

el('wakeButton').addEventListener('click', () => {
  state.wakeWanted = !state.wakeWanted;
  el('wakeButton').setAttribute('aria-pressed', String(state.wakeWanted));
  if (state.wakeWanted) void acquireWakeLock();
  else void releaseWakeLock();
});
el('wakeButton').setAttribute('aria-pressed', 'true');

document.addEventListener('visibilitychange', () => {
  if (document.visibilityState !== 'visible') return;
  void acquireWakeLock();
  // Timers are throttled in a backgrounded tab, so a link that dropped while
  // the tablet was in someone's pocket may still be sitting on a long backoff.
  if (conn.state === CONN.RETRYING) conn.reconnectNow();
  overlay.resize();
  render();
});

// ----------------------------------------------------------------- chrome

el('reconnectButton').addEventListener('click', () => conn.reconnectNow());

el('fullscreenButton').addEventListener('click', async () => {
  // Never `video.webkitEnterFullscreen()`: the platform player draws over the
  // canvas, so the boxes vanish while the picture stays -- an overlay that has
  // silently switched itself off. Where element fullscreen is unavailable
  // (iOS Safari), collapse the chrome instead and keep the canvas.
  const root = document.documentElement;
  try {
    if (document.fullscreenElement) await document.exitFullscreen();
    else if (root.requestFullscreen) await root.requestFullscreen({ navigationUI: 'hide' });
    else document.body.classList.toggle('immersive');
  } catch (err) {
    document.body.classList.toggle('immersive');
  }
  setTimeout(() => { overlay.resize(); render(); }, 120);
});

el('detailsButton').addEventListener('click', () => {
  const panel = el('details');
  panel.hidden = !panel.hidden;
  el('detailsButton').setAttribute('aria-expanded', String(!panel.hidden));
  overlay.resize();
});

if (window.ResizeObserver) {
  new ResizeObserver(() => { overlay.resize(); render(); }).observe(el('stage'));
} else {
  window.addEventListener('resize', () => { overlay.resize(); render(); });
}
window.addEventListener('orientationchange', () => setTimeout(() => { overlay.resize(); render(); }, 200));

window.addEventListener('pagehide', () => {
  // Frees the encoder the station is holding for this tablet. Best effort:
  // the station's watchdog collects the rest.
  conn.sendClose();
});

// ------------------------------------------------------------------- start

/**
 * Register the offline shell. Deliberately not awaited and never fatal.
 *
 * The station serves this app over a self-signed LAN certificate, and whether
 * a given tablet trusts it enough to allow a service worker is not something
 * this app can control. So the worker is an optimisation for the next cold
 * start, never a precondition: the app runs identically as a plain tab.
 *
 * @returns {void}
 */
function registerServiceWorker() {
  if (!('serviceWorker' in navigator)) {
    state.cache = 'unsupported on this browser';
    return;
  }
  if (!window.isSecureContext) {
    state.cache = 'unavailable — page is not a secure context';
    return;
  }
  navigator.serviceWorker.register('sw.js', { scope: './' }).then(
    () => { state.cache = 'app shell cached for offline start'; },
    (err) => { state.cache = `registration refused: ${err.message}`; },
  );
}

/**
 * Boot: parameters first (so the staleness limit is right from the first
 * payload), then the peer connection.
 *
 * @returns {Promise<void>} Resolves once the first connection attempt settles.
 */
async function main() {
  registerServiceWorker();
  void acquireWakeLock();
  overlay.resize();
  render();
  await conn.fetchParameters();
  await conn.connect();
}

void main();
