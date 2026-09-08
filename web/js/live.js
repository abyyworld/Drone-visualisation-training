/**
 * Live tracking in the browser: a camera in, tracked boxes out, at frame rate.
 *
 * WHY THIS IS THE ENGINE THAT MATTERS
 *     A provider API cannot do this and never will. One round trip is seconds, so the best
 *     it can offer is a frame sampled every few seconds with the overlay confessing how old
 *     its boxes are - and nothing to associate between frames, so no identity, no counting,
 *     no tracking. Detection on the device is what turns that into something that follows
 *     what it is looking at.
 *
 * THE LOOP
 *     Detection and drawing are deliberately decoupled. Drawing runs on every animation
 *     frame so the boxes move with the video; detection runs as often as the device can
 *     manage and no faster, and the tracker coasts the boxes in between. On a machine
 *     managing eight detections a second at sixty frames of video, that is the difference
 *     between boxes that follow a person and boxes that jump.
 *
 *     The loop never queues work behind itself. If a detection is still running when the
 *     next frame arrives, that frame is drawn from the existing tracks and no second
 *     detection is started - which is what stops a slow device spiralling into a backlog it
 *     can never clear.
 *
 * WHAT IT CANNOT DO
 *     A browser cannot open an RTSP stream, so this sees the tablet's own camera and any
 *     screen or capture device the operating system presents as one. The drone's own feed
 *     arrives over RTSP and is handled natively in the Android app.
 */

import { detectFrame, warmUp, inferenceMillis } from './ondevice.js';
import { Tracker } from './track.js';
import { colorFor } from './render.js';

const PALETTE_ALPHA_COASTED = 0.45;

// The floor stops a fast machine spending every millisecond in the detector; the ceiling
// stops a slow one leaving the boxes stale for longer than the tracker can sensibly coast.
const MIN_GAP_MS = 60;
const MAX_GAP_MS = 400;

export class LiveView {
  /**
   * @param {HTMLVideoElement} video
   * @param {HTMLCanvasElement} canvas  drawn over the video, at the video's own resolution
   */
  constructor(video, canvas, { onStatus = () => {} } = {}) {
    this.video = video;
    this.canvas = canvas;
    this.onStatus = onStatus;
    this.tracker = new Tracker();
    this.stream = null;
    this.running = false;
    this.detecting = false;
    this.frameHandle = 0;
    this.detectHandle = 0;
    this.lastTimestamp = -1;
    // Trails are off by default. They read as scribbles over a moving picture, and the
    // number on the box already says which thing is which.
    this.showTrails = false;
    this.countPeople = false;
    this.detections = 0;
    this.startedAt = 0;
    this.recorder = null;
    this.recorded = [];
  }

  /** Cameras the browser will admit to having, for the picker. */
  static async cameras() {
    if (!navigator.mediaDevices?.enumerateDevices) return [];
    const devices = await navigator.mediaDevices.enumerateDevices();
    return devices
      .filter((d) => d.kind === 'videoinput')
      .map((d, index) => ({ id: d.deviceId, label: d.label || `Camera ${index + 1}` }));
  }

  async start(deviceId) {
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error(
        'This browser will not open a camera. That needs a secure page, so http:// will '
        + 'not do it - use the deployed https:// site, or the Android app.',
      );
    }

    this.onStatus('Loading the detector');
    // Warmed before the camera opens, so the first frames are not the slow ones and the
    // first thing on screen is not a stutter.
    await warmUp('VIDEO', (label) => this.onStatus(label));

    this.onStatus('Opening the camera');
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        video: deviceId
          ? { deviceId: { exact: deviceId } }
          // Rear camera by default: a handheld pointed at something is using the back one.
          : { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      });
    } catch (error) {
      throw new Error(describeCameraFailure(error));
    }

    this.video.srcObject = this.stream;
    this.video.muted = true;
    this.video.playsInline = true;
    await this.video.play();

    this.tracker.reset();
    this.detections = 0;
    this.startedAt = performance.now();
    this.running = true;
    this.loop();
    this.detectLoop();
  }

  stop() {
    this.running = false;
    cancelAnimationFrame(this.frameHandle);
    clearTimeout(this.detectHandle);
    this.stopRecording();
    for (const track of this.stream?.getTracks() ?? []) track.stop();
    this.stream = null;
    this.video.srcObject = null;
  }

  /**
   * Drawing. Runs on every animation frame and does nothing expensive.
   *
   * This used to also run the detector, on the belief that a `detecting` flag kept them
   * apart. It did not: detectForVideo is synchronous, so the flag was set and cleared
   * inside one tick and detection ran on every single frame, blocking the thread for a
   * tenth of a second each time. The video stuttered, and detections landed so far apart
   * that a walking person outran the tracker and kept being issued a new number.
   *
   * So the two are genuinely separate now. This draws; detectLoop below detects on its own
   * cadence and deliberately leaves the browser room to render between calls.
   */
  loop() {
    if (!this.running) return;
    this.frameHandle = requestAnimationFrame(() => this.loop());

    const { videoWidth: width, videoHeight: height } = this.video;
    if (!width || !height) return;

    if (this.canvas.width !== width || this.canvas.height !== height) {
      this.canvas.width = width;
      this.canvas.height = height;
    }

    this.draw();
    this.report();
  }

  /**
   * Detection, on a timer rather than a frame callback.
   *
   * The gap after each detection is the length of that detection, floored at MIN_GAP_MS.
   * A machine taking 150 ms per frame therefore spends about half its time detecting and
   * half of it free to decode and paint video, which is what keeps the picture smooth
   * instead of running the detector flat out and starving everything else.
   */
  detectLoop() {
    if (!this.running) return;

    const started = performance.now();
    if (this.video.readyState >= 2 && this.video.videoWidth) {
      // MediaPipe rejects a timestamp that does not advance, and at high frame rates two
      // calls can land in the same millisecond.
      const timestamp = Math.max(this.lastTimestamp + 1, Math.round(performance.now()));
      this.lastTimestamp = timestamp;
      try {
        const found = detectFrame(this.video, timestamp);
        if (found) {
          this.tracker.update(found);
          this.detections += 1;
        }
      } catch (error) {
        this.onStatus(`Detection stopped: ${error.message}`);
        return;
      }
    }

    const took = performance.now() - started;
    const gap = Math.max(MIN_GAP_MS, Math.min(took, MAX_GAP_MS));
    this.detectHandle = setTimeout(() => this.detectLoop(), gap);
  }

  draw() {
    const ctx = this.canvas.getContext('2d');
    ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

    const longEdge = Math.max(this.canvas.width, this.canvas.height);
    const lineWidth = Math.max(2, Math.round(longEdge / 300));
    const fontSize = Math.max(12, Math.round(longEdge / 45));
    ctx.lineWidth = lineWidth;
    ctx.font = `600 ${fontSize}px system-ui, sans-serif`;
    ctx.textBaseline = 'top';

    for (const track of this.tracker.open()) {
      const colour = colorFor(track.classId ?? 0);
      // A coasted box is a prediction rather than an observation, and is drawn as one. An
      // operator should be able to see at a glance which boxes the model is still looking at.
      ctx.globalAlpha = track.missed > 0 ? PALETTE_ALPHA_COASTED : 1;
      ctx.strokeStyle = colour;
      ctx.setLineDash(track.missed > 0 ? [lineWidth * 3, lineWidth * 2] : []);

      const [x0, y0, x1, y1] = track.box;
      ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
      ctx.setLineDash([]);

      const label = `${track.label} #${track.id}`;
      const padding = lineWidth * 2;
      const textWidth = ctx.measureText(label).width;
      const top = Math.max(0, y0 - fontSize - padding * 2);

      ctx.fillStyle = colour;
      ctx.fillRect(x0, top, textWidth + padding * 2, fontSize + padding * 2);
      ctx.fillStyle = '#ffffff';
      ctx.fillText(label, x0 + padding, top + padding);

      // The trail says where something came from, which is useful when you are studying a
      // flow and is a scribble over the picture when you are not. Off unless asked for.
      if (this.showTrails && track.path.length > 2) {
        ctx.globalAlpha = 0.5;
        ctx.beginPath();
        ctx.moveTo(track.path[0][0], track.path[0][1]);
        for (const [x, y] of track.path.slice(1)) ctx.lineTo(x, y);
        ctx.stroke();
      }
      ctx.globalAlpha = 1;
    }
  }

  report() {
    const seconds = (performance.now() - this.startedAt) / 1000;
    const open = this.tracker.open();
    const counts = new Map();
    for (const track of open) counts.set(track.label, (counts.get(track.label) ?? 0) + 1);

    this.onStatus(null, {
      fps: seconds > 0 ? this.detections / seconds : 0,
      inferenceMs: inferenceMillis(),
      onScreen: [...counts].map(([label, n]) => `${n} ${label}${n === 1 ? '' : 's'}`),
      // Two different numbers, and confusing them is the classic mistake. `people` is how
      // many are in view right now; `peopleTotal` is how many distinct ones have been seen
      // since the camera opened, which keeps climbing as people walk through.
      people: counts.get('person') ?? 0,
      peopleTotal: this.tracker.countSeen('person'),
      seenTotal: this.tracker.nextId - 1,
      recording: Boolean(this.recorder),
    });
  }

  // -------------------------------------------------------------------------------------
  // Recording
  // -------------------------------------------------------------------------------------

  /**
   * Record what is on screen, boxes included.
   *
   * The canvas is composited with the video into a second canvas rather than recorded on
   * its own, because the overlay canvas is transparent - recording it alone would produce
   * boxes floating on black.
   */
  startRecording() {
    if (this.recorder || !this.stream) return;

    const composite = document.createElement('canvas');
    composite.width = this.canvas.width;
    composite.height = this.canvas.height;
    const ctx = composite.getContext('2d');

    const paint = () => {
      if (!this.recorder) return;
      ctx.drawImage(this.video, 0, 0, composite.width, composite.height);
      ctx.drawImage(this.canvas, 0, 0);
      requestAnimationFrame(paint);
    };

    const type = ['video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm']
      .find((t) => MediaRecorder.isTypeSupported(t));
    if (!type) throw new Error('This browser cannot record video.');

    this.recorded = [];
    this.recorder = new MediaRecorder(composite.captureStream(30), { mimeType: type });
    this.recorder.ondataavailable = (event) => {
      if (event.data.size) this.recorded.push(event.data);
    };
    this.recorder.start(1000);
    paint();
  }

  /** @returns {Blob|null} the annotated recording */
  stopRecording() {
    if (!this.recorder) return null;
    const recorder = this.recorder;
    this.recorder = null;
    if (recorder.state !== 'inactive') recorder.stop();
    return new Blob(this.recorded, { type: recorder.mimeType });
  }
}

/** Camera failures are all "NotAllowedError" until someone explains which one happened. */
function describeCameraFailure(error) {
  const name = error?.name ?? '';
  if (name === 'NotAllowedError' || name === 'SecurityError') {
    return 'The camera was refused. Allow camera access for this site in the browser\'s '
      + 'address bar, then try again.';
  }
  if (name === 'NotFoundError' || name === 'OverconstrainedError') {
    return 'No camera the browser can use. If one is plugged in, pick it from the list.';
  }
  if (name === 'NotReadableError') {
    return 'The camera is open in another application. Close that and try again.';
  }
  return `The camera could not be opened: ${error?.message ?? name}`;
}
