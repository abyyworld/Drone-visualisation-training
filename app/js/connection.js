/**
 * WebRTC client: one peer connection to the ground station, carrying the video
 * track and the `detections` data channel.
 *
 * Signalling is one POST, matching station/stream/signaling.py: the tablet
 * offers, the station answers, and because aiortc gathers its host candidates
 * before answering there is no candidate exchange to run afterwards. So this
 * module waits for its *own* ICE gathering to finish before posting, and then
 * has nothing left to negotiate.
 *
 * Reconnection is not a nicety here. Tablets on a fire ground walk behind
 * vehicles, out of range, and into their own screen-lock; the link drops
 * constantly and nobody is going to be reading a reconnect button. So the
 * connection re-offers on its own with a backoff, and the connection state is
 * pushed to the UI on every transition rather than being inspected on demand.
 */

import { parseMessage, WireError, WIRE_VERSION, DETECTIONS_CHANNEL } from './wire.js';

/** Connection states surfaced to the UI. `fatal` needs an operator to act. */
export const CONN = Object.freeze({
  IDLE: 'idle',
  CONNECTING: 'connecting',
  CONNECTED: 'connected',
  RETRYING: 'retrying',
  FATAL: 'fatal',
  CLOSED: 'closed',
});

/** Default signalling mount, matching signaling.DEFAULT_PREFIX. */
export const DEFAULT_PREFIX = '/webrtc';

/**
 * A tiny event emitter, so the app can subscribe without a framework.
 */
class Emitter {
  constructor() {
    /** @type {Map<string, Set<Function>>} */
    this._handlers = new Map();
  }

  /**
   * Subscribe to an event.
   *
   * @param {string} event Event name.
   * @param {Function} handler Called with the event payload.
   * @returns {() => void} Unsubscribe function.
   */
  on(event, handler) {
    if (!this._handlers.has(event)) this._handlers.set(event, new Set());
    this._handlers.get(event).add(handler);
    return () => this._handlers.get(event)?.delete(handler);
  }

  /**
   * Emit an event. A throwing handler is logged and does not stop the others:
   * a rendering bug must not be able to take the connection down with it.
   *
   * @param {string} event Event name.
   * @param {*} [payload] Payload passed to handlers.
   * @returns {void}
   */
  emit(event, payload) {
    for (const handler of this._handlers.get(event) || []) {
      try {
        handler(payload);
      } catch (err) {
        console.error(`handler for "${event}" threw`, err);
      }
    }
  }
}

/**
 * One connection to the station, with automatic reconnection.
 *
 * Events: `state` ({state, attempt, error, nextRetryS}), `track` (MediaStream),
 * `detections` (parsed payload), `status` (parsed heartbeat), `parameters`
 * (the station's client_parameters block), `protocol-error` (fatal, sticky).
 */
export class StationConnection extends Emitter {
  /**
   * @param {object} [options] Connection tuning.
   * @param {string} [options.prefix='/webrtc'] Signalling mount point.
   * @param {number} [options.offerTimeoutMs=12000] Give up on one POST.
   * @param {number} [options.iceGatherTimeoutMs=2500] Cap on waiting for ICE
   *   gathering. On a LAN with host candidates this completes in milliseconds;
   *   the cap exists for the browsers that never fire the completion event.
   * @param {number} [options.minBackoffMs=500] First retry delay.
   * @param {number} [options.maxBackoffMs=15000] Longest retry delay.
   * @param {number} [options.silenceTimeoutMs=8000] Reconnect when the data
   *   channel has been silent this long. The station heartbeats at ~1 Hz, so
   *   silence is a dead link -- and a dead link that still looks "connected"
   *   is a tablet showing a frozen picture with no way to tell.
   */
  constructor(options = {}) {
    super();
    this.prefix = (options.prefix || DEFAULT_PREFIX).replace(/\/+$/, '');
    this.offerTimeoutMs = options.offerTimeoutMs ?? 12000;
    this.iceGatherTimeoutMs = options.iceGatherTimeoutMs ?? 2500;
    this.minBackoffMs = options.minBackoffMs ?? 500;
    this.maxBackoffMs = options.maxBackoffMs ?? 15000;
    this.silenceTimeoutMs = options.silenceTimeoutMs ?? 8000;

    this.state = CONN.IDLE;
    this.attempt = 0;
    this.lastError = null;
    this.peerId = null;
    /** Station `client_parameters()` from the last successful answer. */
    this.parameters = null;
    /** Sticky: a version mismatch does not heal by waiting for more messages. */
    this.protocolError = null;
    this.stats = { messages: 0, parseErrors: 0, connects: 0, drops: 0 };

    this._pc = null;
    this._channel = null;
    this._retryTimer = null;
    this._silenceTimer = null;
    this._closedByUs = false;
    this._connecting = false;
  }

  /**
   * Start connecting, and keep connecting until `close()`.
   *
   * @returns {Promise<void>} Resolves once the first attempt settles; failures
   *   are reported through the `state` event rather than thrown, because after
   *   the first attempt there is no caller left to catch them.
   */
  async connect() {
    this._closedByUs = false;
    await this._attempt();
  }

  /**
   * Ask for an immediate reconnection: the operator pressed the button, or the
   * tablet just came back to the foreground.
   *
   * @returns {void}
   */
  reconnectNow() {
    if (this._closedByUs) return;
    this._clearRetry();
    this.attempt = 0;
    this._teardownPeer();
    void this._attempt();
  }

  /**
   * Close for good and tell the station, so it can free the encoder it is
   * holding for this tablet rather than waiting for its watchdog.
   *
   * @returns {void}
   */
  close() {
    this._closedByUs = true;
    this._clearRetry();
    this._clearSilence();
    this.sendClose();
    this._teardownPeer();
    this._setState(CONN.CLOSED);
  }

  /**
   * Best-effort "I am leaving" to the station. Safe to call from `pagehide`,
   * where a normal fetch would be cancelled with the page.
   *
   * @returns {void}
   */
  sendClose() {
    if (!this.peerId) return;
    const body = JSON.stringify({ peer_id: this.peerId });
    try {
      if (navigator.sendBeacon) {
        navigator.sendBeacon(`${this.prefix}/close`, new Blob([body], { type: 'application/json' }));
        return;
      }
      void fetch(`${this.prefix}/close`, { method: 'POST', body, keepalive: true, headers: { 'Content-Type': 'application/json' } });
    } catch (err) {
      // The station's own watchdog collects peers that never say goodbye,
      // which in the field is most of them. Nothing to recover here.
      console.debug('close beacon failed', err);
    }
  }

  /**
   * Fetch the overlay parameters without opening a peer.
   *
   * Used at start-up so the staleness limit is in force before the first
   * payload arrives, rather than a hard-coded default being in force.
   *
   * @returns {Promise<object|null>} The parameters, or null when unreachable.
   */
  async fetchParameters() {
    try {
      const response = await fetch(`${this.prefix}/config`, { cache: 'no-store' });
      if (!response.ok) return null;
      const params = await response.json();
      this.parameters = params;
      this.emit('parameters', params);
      return params;
    } catch (err) {
      return null;
    }
  }

  // ------------------------------------------------------------- internals

  /**
   * One offer/answer attempt, scheduling a retry if it fails.
   *
   * @returns {Promise<void>} Resolves when the attempt has settled.
   */
  async _attempt() {
    if (this._connecting || this._closedByUs) return;
    this._connecting = true;
    this.attempt += 1;
    this._setState(CONN.CONNECTING);
    try {
      await this._negotiate();
      this.stats.connects += 1;
      this.attempt = 0;
      this.lastError = null;
      this._setState(CONN.CONNECTED);
      this._armSilence();
    } catch (err) {
      this.lastError = err instanceof Error ? err.message : String(err);
      this._teardownPeer();
      this._scheduleRetry();
    } finally {
      this._connecting = false;
    }
  }

  /**
   * Build the peer connection, exchange SDP, and wire up the streams.
   *
   * @returns {Promise<void>} Resolves once the answer has been applied.
   * @throws {Error} On any signalling or negotiation failure.
   */
  async _negotiate() {
    const iceServers = (this.parameters?.ice_servers || []).map((url) => ({ urls: url }));
    const pc = new RTCPeerConnection({ iceServers, bundlePolicy: 'max-bundle' });
    this._pc = pc;

    // Receive-only: inference happens once, on the station. The tablet never
    // sends media and never runs a model.
    pc.addTransceiver('video', { direction: 'recvonly' });

    // The tablet opens the channel; the station adopts it on `datachannel`.
    // Reliable and ordered: a dropped payload is a missing overlay frame, and
    // an out-of-order one would be matched to the wrong video frame.
    const channel = pc.createDataChannel(DETECTIONS_CHANNEL, { ordered: true });
    this._channel = channel;
    channel.onmessage = (event) => this._onMessage(event.data);
    channel.onclose = () => this._onDrop('data channel closed');

    pc.ontrack = (event) => {
      const stream = event.streams && event.streams[0] ? event.streams[0] : new MediaStream([event.track]);
      this.emit('track', stream);
    };
    pc.onconnectionstatechange = () => {
      const state = pc.connectionState;
      if (state === 'failed' || state === 'closed') this._onDrop(`peer connection ${state}`);
      // `disconnected` is deliberately not acted on here: it recovers on its
      // own often enough that dropping the peer immediately would cost a
      // reconnect on every brief WiFi hiccup. The silence watchdog catches it
      // if it does not recover.
    };

    await pc.setLocalDescription(await pc.createOffer());
    await this._waitForIceGathering(pc);

    const answer = await this._postOffer(pc.localDescription);
    this.peerId = answer.peer_id || null;
    this.parameters = answer;
    this.emit('parameters', answer);

    if (answer.wire_version !== undefined && answer.wire_version !== WIRE_VERSION) {
      // Loud, sticky, and it does not stop the video: the picture is the
      // ground truth and the operator keeps it. What stops is the overlay,
      // because boxes drawn from a protocol this build cannot fully read
      // would be a partial overlay, and a partial overlay looks exactly like
      // a quiet scene.
      this.protocolError =
        `station speaks wire v${answer.wire_version}, this tablet speaks v${WIRE_VERSION}. ` +
        'Overlay disabled. Reinstall the app from the station.';
      this.emit('protocol-error', this.protocolError);
    }

    await pc.setRemoteDescription({ type: answer.type || 'answer', sdp: answer.sdp });
  }

  /**
   * POST the offer to the station and return the parsed answer.
   *
   * @param {RTCSessionDescription|null} description The local description.
   * @returns {Promise<object>} The answer body.
   * @throws {Error} With the station's own message when it sends one -- a 501
   *   naming a missing package is far more useful than "connection failed".
   */
  async _postOffer(description) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.offerTimeoutMs);
    try {
      const response = await fetch(`${this.prefix}/offer`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sdp: description?.sdp, type: description?.type || 'offer' }),
        cache: 'no-store',
        signal: controller.signal,
      });
      const text = await response.text();
      let body = {};
      try {
        body = text ? JSON.parse(text) : {};
      } catch (err) {
        throw new Error(`station sent a non-JSON answer (HTTP ${response.status})`);
      }
      if (!response.ok) throw new Error(body.error || `station refused the offer (HTTP ${response.status})`);
      if (!body.sdp) throw new Error('station answer carried no SDP');
      return body;
    } catch (err) {
      if (err.name === 'AbortError') throw new Error(`station did not answer within ${this.offerTimeoutMs} ms`);
      throw err;
    } finally {
      clearTimeout(timer);
    }
  }

  /**
   * Wait until ICE gathering completes, or the cap expires.
   *
   * The station's signalling is non-trickle, so the offer has to be complete
   * when it is posted -- there is no second endpoint to send candidates to.
   *
   * @param {RTCPeerConnection} pc The peer connection.
   * @returns {Promise<void>} Resolves on completion or timeout.
   */
  _waitForIceGathering(pc) {
    if (pc.iceGatheringState === 'complete') return Promise.resolve();
    return new Promise((resolve) => {
      const done = () => {
        clearTimeout(timer);
        pc.removeEventListener('icegatheringstatechange', onChange);
        resolve();
      };
      const onChange = () => {
        if (pc.iceGatheringState === 'complete') done();
      };
      const timer = setTimeout(done, this.iceGatherTimeoutMs);
      pc.addEventListener('icegatheringstatechange', onChange);
    });
  }

  /**
   * Handle one data-channel message.
   *
   * @param {string|ArrayBuffer} data The raw message.
   * @returns {void}
   */
  _onMessage(data) {
    this.stats.messages += 1;
    this._armSilence();
    let message;
    try {
      message = parseMessage(data);
    } catch (err) {
      this.stats.parseErrors += 1;
      if (err instanceof WireError && err.fatal) {
        if (!this.protocolError) {
          this.protocolError = `${err.message}`;
          this.emit('protocol-error', this.protocolError);
        }
        return;
      }
      // A single malformed message is not fatal, but it is not shrugged off
      // either: a payload that failed to parse is detections the operator did
      // not get, and the count is on screen.
      console.warn('discarded an unparseable message', err);
      this.emit('parse-error', err);
      return;
    }
    if (this.protocolError) return; // overlay is off; do not half-render it
    if (message.type === 'detections') this.emit('detections', message);
    else this.emit('status', message);
  }

  /**
   * React to a link that has gone away.
   *
   * @param {string} reason What noticed.
   * @returns {void}
   */
  _onDrop(reason) {
    if (this._closedByUs || this._connecting) return;
    if (this.state === CONN.RETRYING) return;
    this.stats.drops += 1;
    this.lastError = reason;
    this._teardownPeer();
    this._scheduleRetry();
  }

  /**
   * Schedule the next attempt with exponential backoff and jitter.
   *
   * Jitter matters with several tablets on one station: without it they all
   * come back at the same instant after a WiFi outage and the laptop has to
   * negotiate every peer at once, which is when it drops them again.
   *
   * @returns {void}
   */
  _scheduleRetry() {
    if (this._closedByUs) return;
    this._clearRetry();
    this._clearSilence();
    const exponent = Math.min(this.attempt, 6);
    const base = Math.min(this.maxBackoffMs, this.minBackoffMs * 2 ** Math.max(0, exponent - 1));
    const delay = Math.round(base * (0.7 + Math.random() * 0.6));
    this._setState(CONN.RETRYING, { nextRetryS: delay / 1000 });
    this._retryTimer = setTimeout(() => {
      this._retryTimer = null;
      void this._attempt();
    }, delay);
  }

  /**
   * (Re)arm the data-channel silence watchdog.
   *
   * @returns {void}
   */
  _armSilence() {
    this._clearSilence();
    this._silenceTimer = setTimeout(() => {
      this._silenceTimer = null;
      this._onDrop(`no data channel message for ${(this.silenceTimeoutMs / 1000).toFixed(0)} s`);
    }, this.silenceTimeoutMs);
  }

  /** @returns {void} */
  _clearSilence() {
    if (this._silenceTimer !== null) clearTimeout(this._silenceTimer);
    this._silenceTimer = null;
  }

  /** @returns {void} */
  _clearRetry() {
    if (this._retryTimer !== null) clearTimeout(this._retryTimer);
    this._retryTimer = null;
  }

  /**
   * Drop the peer connection and its channel without touching retry state.
   *
   * @returns {void}
   */
  _teardownPeer() {
    this._clearSilence();
    const channel = this._channel;
    const pc = this._pc;
    this._channel = null;
    this._pc = null;
    try {
      if (channel) {
        channel.onmessage = null;
        channel.onclose = null;
        channel.close();
      }
    } catch (err) {
      console.debug('channel close failed', err);
    }
    try {
      if (pc) {
        pc.ontrack = null;
        pc.onconnectionstatechange = null;
        pc.close();
      }
    } catch (err) {
      console.debug('peer close failed', err);
    }
  }

  /**
   * Publish a state transition.
   *
   * @param {string} state One of CONN.
   * @param {object} [extra] Extra fields for the UI (e.g. `nextRetryS`).
   * @returns {void}
   */
  _setState(state, extra = {}) {
    this.state = state;
    this.emit('state', {
      state,
      attempt: this.attempt,
      error: this.lastError,
      peerId: this.peerId,
      protocolError: this.protocolError,
      ...extra,
    });
  }
}
