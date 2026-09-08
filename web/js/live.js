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

import { detectFrame, warmUp, inferenceMillis, onDeviceConfig } from './ondevice.js';
import { Tracker } from './track.js';
import { describe } from './reid.js';
import { FireScan } from './firescan.js';
import { colorFor } from './render.js';

const PALETTE_ALPHA_COASTED = 0.45;

// The floor stops a fast machine spending every millisecond in the detector; the ceiling
// stops a slow one leaving the boxes stale for longer than the tracker can sensibly coast.
/**
 * The width the appearance signatures are computed at.
 *
 * Larger does not help. What is being measured is which colours someone is wearing and in
 * what proportion, and that is settled long before this resolution; past it the histogram
 * starts describing the lighting and the sensor rather than the person.
 */
const SIGNATURE_WIDTH = 320;

/** The recording's own frame rate. The display draws faster; the file does not need to. */
const RECORD_INTERVAL_MS = 1000 / 30;

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
    // Flame and smoke run beside the detector on the same frames. They cost about a
    // millisecond, because they work on a 160-wide copy, so this is not a trade against
    // detection rate - see firescan.js.
    this.fire = new FireScan();
    this.fireRegions = [];
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
    // Which subject the camera is pointed at, and therefore which engines run. See
    // setSubject(): running the flame scan indoors is how a white wall became smoke.
    this.subject = 'crowd';
    // A small copy of the frame, kept for the appearance signatures. Small on purpose: a
    // colour histogram of someone's clothing does not get better with more pixels, and this
    // runs on every detection.
    this.signatureCanvas = null;
    // Detection's own thread. Null when the browser would not give us one, in which case
    // everything below falls back to running it here, slowly and visibly.
    this.worker = null;
    this.workerBusy = false;
    this.workerInference = 0;
    // Off for a turbine or a panel, where the subject fills the frame and there is nothing
    // small to find. On for a crowd or a fire, where there is.
    this.tiled = true;
    this.detections = 0;
    this.startedAt = 0;
    this.recorder = null;
    this.recorded = [];
    this.composite = null;
    this.compositeContext = null;
    this.recordTrack = null;
    this.lastRecordedAt = 0;
  }

  /**
   * What this camera is looking at.
   *
   * Not cosmetic, and this is the lesson from a screenshot of an office: the flame and
   * smoke scan was running on a webcam pointed at a person in front of a white wall, and
   * marked the wall as smoke at 82 percent. It was not wrong by its own rules - a flat,
   * desaturated, mid-brightness region that loses its texture when someone moves across it
   * is exactly what the smoke rule describes. It was being asked a question that made no
   * sense indoors.
   *
   * So the scan runs when the operator says they are looking for fire, and not otherwise.
   * A subject is a statement about what is in front of the camera, and it is the cheapest
   * and most reliable false-positive filter available: context no algorithm can infer.
   */
  setSubject(subject) {
    this.subject = subject;
    this.tiled = subject === 'crowd' || subject === 'wildfire';
    this.fire.reset();
    this.fireRegions = [];
    this.worker?.postMessage({ type: 'reset' });
  }

  /** Does the flame and smoke scan apply to what we are looking at? */
  scansForFire() {
    return this.subject === 'wildfire';
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
    // A thread of its own, if this browser has one to give. Everything expensive goes
    // there: the model, the flame scan and the appearance signatures. See detect-worker.js
    // for why nothing short of this makes the picture smooth.
    await this.openWorker();
    if (!this.worker) {
      // Warmed before the camera opens, so the first frames are not the slow ones and the
      // first thing on screen is not a stutter.
      await warmUp('VIDEO', (label) => this.onStatus(label));
    }

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
    this.fire.reset();
    this.fireRegions = [];
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
    if (this.worker) {
      this.worker.terminate();
      this.worker = null;
      this.workerBusy = false;
    }
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
    this.paintComposite(performance.now());
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

    if (this.worker) {
      this.detectOnWorker();
      return;
    }

    const started = performance.now();
    if (this.video.readyState >= 2 && this.video.videoWidth) {
      // MediaPipe rejects a timestamp that does not advance, and at high frame rates two
      // calls can land in the same millisecond.
      const timestamp = Math.max(this.lastTimestamp + 1, Math.round(performance.now()));
      this.lastTimestamp = timestamp;
      try {
        const found = detectFrame(this.video, timestamp);
        if (found) {
          this.signPeople(found);
          // The clock goes in explicitly. The tracker lets a box go once it is older than a
          // second and a half, and it can only know that if it is told what time it is.
          this.tracker.update(found, performance.now());
          this.detections += 1;
        }
      } catch (error) {
        this.onStatus(`Detection stopped: ${error.message}`);
        return;
      }

      // Deliberately outside that try. If the model fails the run is over; if the flame
      // scan fails it is one frame of one of two engines, and taking the whole live view
      // down over it would be the wrong trade.
      if (this.scansForFire()) {
        try {
          // The tracked boxes go in with the frame. A smoke region mostly covered by
          // something the detector is already following has its missing texture explained
          // already, and does not need a second explanation invented for it.
          const width = this.video.videoWidth;
          const height = this.video.videoHeight;
          const occluders = this.tracker.open().map((track) => [
            track.box[0] / width, track.box[1] / height,
            track.box[2] / width, track.box[3] / height,
          ]);
          this.fireRegions = this.fire.scan(this.video, width, height, occluders);
        } catch {
          this.fireRegions = [];
        }
      }
    }

    const took = performance.now() - started;
    const gap = Math.max(MIN_GAP_MS, Math.min(took, MAX_GAP_MS));
    this.detectHandle = setTimeout(() => this.detectLoop(), gap);
  }

  /**
   * Start the detection thread, or decide to do without one.
   *
   * Never fatal. A browser that refuses module workers, or a page served from a context
   * that forbids them, falls back to detecting on this thread: slower and visibly juddery,
   * but working. Refusing to open the camera because a worker would not start would be a
   * far worse trade.
   */
  async openWorker() {
    if (this.worker) return;
    if (typeof Worker !== 'function' || typeof createImageBitmap !== 'function') return;

    try {
      const worker = new Worker(new URL('./detect-worker.js', import.meta.url), {
        type: 'module',
      });
      const config = onDeviceConfig();

      await new Promise((resolve, reject) => {
        const timer = setTimeout(() => reject(new Error('the detector did not start')), 30000);
        worker.onmessage = (event) => {
          if (event.data?.type === 'ready') {
            clearTimeout(timer);
            resolve();
          } else if (event.data?.type === 'error') {
            clearTimeout(timer);
            reject(new Error(event.data.message));
          }
        };
        worker.onerror = (error) => {
          clearTimeout(timer);
          reject(new Error(error.message || 'the detector thread failed to start'));
        };
        worker.postMessage({ type: 'configure', ...config });
        worker.postMessage({ type: 'warm' });
      });

      worker.onmessage = (event) => this.onWorkerMessage(event.data);
      worker.onerror = () => { this.worker = null; };
      this.worker = worker;
    } catch (error) {
      this.worker = null;
      this.onStatus(`Detecting on the main thread: ${error.message}`);
    }
  }

  /**
   * Hand the worker a frame, if it has finished with the last one.
   *
   * Never more than one frame in flight. Queueing them would build a backlog the device can
   * never clear, and every box would describe somewhere the camera used to be pointed.
   */
  detectOnWorker() {
    const send = async () => {
      if (this.workerBusy || !this.running) return;
      if (this.video.readyState < 2 || !this.video.videoWidth) return;
      this.workerBusy = true;
      try {
        // Transferred, not copied. The frame is GPU-side and costs nothing to hand over.
        const bitmap = await createImageBitmap(this.video);
        this.worker.postMessage({
          type: 'frame',
          bitmap,
          scanFire: this.scansForFire(),
          wantSignatures: true,
          // Tiling is what finds the people who are only a few pixels tall, which is most
          // of a crowd from any altitude. It costs one extra detection per pass.
          tiled: this.tiled,
        }, [bitmap]);
      } catch {
        this.workerBusy = false;
      }
    };
    send();
    // A short, fixed tick. The worker's own speed is what actually paces this: a frame is
    // only sent when the last one has come back.
    this.detectHandle = setTimeout(() => this.detectLoop(), MIN_GAP_MS);
  }

  onWorkerMessage(message) {
    if (message?.type === 'error') {
      this.workerBusy = false;
      this.onStatus(`Detection stopped: ${message.message}`);
      return;
    }
    if (message?.type !== 'result') return;

    this.workerBusy = false;
    this.workerInference = message.inferenceMs ?? 0;
    this.tracker.update(message.found ?? [], performance.now());
    this.fireRegions = message.regions ?? [];
    this.detections += 1;
  }

  /**
   * Attach a colour signature to every person found, so the tracker can recognise them
   * again after they have gone out of view.
   *
   * One small copy of the frame serves every box in it. Never allowed to throw: without a
   * signature someone is still tracked and still counted, they just get counted a second
   * time if they leave and come back, and that is a worse number rather than no number.
   */
  signPeople(found) {
    const people = found.filter((d) => d.label === 'person');
    if (!people.length) return;

    try {
      const width = this.video.videoWidth;
      const height = this.video.videoHeight;
      if (!width || !height) return;

      const small = Math.min(SIGNATURE_WIDTH, width);
      const tall = Math.max(1, Math.round((height / width) * small));
      if (!this.signatureCanvas
        || this.signatureCanvas.width !== small || this.signatureCanvas.height !== tall) {
        this.signatureCanvas = typeof OffscreenCanvas === 'function'
          ? new OffscreenCanvas(small, tall)
          : Object.assign(document.createElement('canvas'), { width: small, height: tall });
        this.signatureContext = this.signatureCanvas.getContext(
          '2d', { willReadFrequently: true },
        );
      }
      this.signatureContext.drawImage(this.video, 0, 0, small, tall);
      const frame = this.signatureContext.getImageData(0, 0, small, tall);

      const scaleX = small / width;
      const scaleY = tall / height;
      for (const person of people) {
        person.signature = describe(frame.data, small, tall, [
          person.box[0] * scaleX, person.box[1] * scaleY,
          person.box[2] * scaleX, person.box[3] * scaleY,
        ]);
      }
    } catch {
      // Nothing to do: the tracker treats a missing signature as "cannot be recognised".
    }
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

    this.drawFire(ctx, lineWidth, fontSize);

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

  /**
   * Flame and smoke regions, drawn underneath the tracked boxes and drawn differently.
   *
   * A dashed outline with no identity number, because these are not things being followed -
   * they are areas that look and behave like burning. Drawn first so a person standing in
   * front of a fire still gets a solid box over the top of it.
   */
  drawFire(ctx, lineWidth, fontSize) {
    for (const region of this.fireRegions) {
      const colour = region.label === 'smoke' ? '#9aa3ad' : '#ff5a1f';
      const [x0, y0, x1, y1] = region.box;

      ctx.save();
      ctx.globalAlpha = 0.16;
      ctx.fillStyle = colour;
      ctx.fillRect(x0, y0, x1 - x0, y1 - y0);
      ctx.globalAlpha = 1;
      ctx.strokeStyle = colour;
      ctx.setLineDash([lineWidth * 4, lineWidth * 2]);
      ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
      ctx.setLineDash([]);

      const label = `${region.label} ${Math.round(region.confidence * 100)}%`;
      const padding = lineWidth * 2;
      const width = ctx.measureText(label).width;
      const top = Math.min(this.canvas.height - fontSize - padding * 2, y1 + padding);
      ctx.fillStyle = colour;
      ctx.fillRect(x0, top, width + padding * 2, fontSize + padding * 2);
      ctx.fillStyle = '#000000';
      ctx.fillText(label, x0 + padding, top + padding);
      ctx.restore();
    }
  }

  report() {
    const seconds = (performance.now() - this.startedAt) / 1000;
    const open = this.tracker.open();
    const counts = new Map();
    for (const track of open) counts.set(track.label, (counts.get(track.label) ?? 0) + 1);

    this.onStatus(null, {
      fps: seconds > 0 ? this.detections / seconds : 0,
      inferenceMs: this.worker ? this.workerInference : inferenceMillis(),
      offThread: Boolean(this.worker),
      onScreen: [...counts].map(([label, n]) => `${n} ${label}${n === 1 ? '' : 's'}`),
      // Two different numbers, and confusing them is the classic mistake. `people` is how
      // many are in view right now; `peopleTotal` is how many distinct ones have been seen
      // since the camera opened, which keeps climbing as people walk through.
      people: counts.get('person') ?? 0,
      peopleTotal: this.tracker.countSeen('person'),
      seenTotal: this.tracker.nextId - 1,
      // Regions rather than a count. Two boxes on one fire is two regions and one fire, and
      // reporting it as "2 fires" would be inventing a number the method cannot support.
      flame: this.fireRegions.filter((r) => r.label === 'flame').length,
      smoke: this.fireRegions.filter((r) => r.label === 'smoke').length,
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

    this.composite = document.createElement('canvas');
    this.composite.width = this.canvas.width;
    this.composite.height = this.canvas.height;
    this.compositeContext = this.composite.getContext('2d', { alpha: false });

    const type = ['video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm']
      .find((t) => MediaRecorder.isTypeSupported(t));
    if (!type) throw new Error('This browser cannot record video.');

    // A stream this code feeds by hand, rather than one that samples the canvas on a clock
    // of its own. captureStream(30) polls whatever is on the canvas thirty times a second
    // whether or not it has changed, so a frame that arrives late is missed and a frame
    // that has not changed is encoded twice. requestFrame() puts exactly one frame in, at
    // the moment it was drawn.
    const stream = this.composite.captureStream(0);
    this.recordTrack = stream.getVideoTracks()[0];
    if (typeof this.recordTrack?.requestFrame !== 'function') {
      // An older browser without manual frames. Fall back to the polled stream rather than
      // refusing to record.
      this.recordTrack = null;
      this.recorded = [];
      this.recorder = new MediaRecorder(this.composite.captureStream(30), { mimeType: type });
      this.recorder.ondataavailable = (event) => {
        if (event.data.size) this.recorded.push(event.data);
      };
      this.recorder.start();
      return;
    }

    this.recorded = [];
    this.lastRecordedAt = 0;
    this.recorder = new MediaRecorder(stream, {
      mimeType: type,
      // Named rather than left to the browser: the default for a captured canvas is low
      // enough that a crowd turns to mush, which is the one thing a review recording of a
      // crowd must not do.
      videoBitsPerSecond: 6_000_000,
    });
    this.recorder.ondataavailable = (event) => {
      if (event.data.size) this.recorded.push(event.data);
    };
    // No timeslice, and this is the fix for a stutter once a second. start(1000) asks the
    // encoder to close off and hand over a chunk every second, and that flush lands on the
    // main thread in the middle of drawing. One blob at the end costs nothing while
    // recording.
    this.recorder.start();
  }

  /**
   * Compose one frame for the recording, from the same animation frame that draws the
   * screen.
   *
   * Called from loop(), rather than from a second requestAnimationFrame of its own. Two
   * loops both waking on every frame is twice the scheduling and two chances to be late,
   * for one picture.
   */
  paintComposite(now) {
    if (!this.recorder || !this.composite) return;
    if (this.composite.width !== this.canvas.width
      || this.composite.height !== this.canvas.height) {
      this.composite.width = this.canvas.width;
      this.composite.height = this.canvas.height;
    }
    // Held to the recording's frame rate rather than the display's: encoding sixty frames
    // a second of a thirty frame camera is work for no picture.
    if (now - this.lastRecordedAt < RECORD_INTERVAL_MS) return;
    this.lastRecordedAt = now;

    this.compositeContext.drawImage(
      this.video, 0, 0, this.composite.width, this.composite.height,
    );
    this.compositeContext.drawImage(this.canvas, 0, 0);
    this.recordTrack?.requestFrame();
  }

  /**
   * Finish the recording and hand back the file.
   *
   * A promise, and it has to be. Without a timeslice the encoder holds everything until it
   * is stopped, and the data arrives in a `dataavailable` event *after* stop() returns.
   * Building the blob synchronously, as this used to, would have handed back an empty file
   * the moment the per-second flush was removed.
   *
   * @returns {Promise<Blob|null>} the annotated recording
   */
  stopRecording() {
    if (!this.recorder) return Promise.resolve(null);

    const recorder = this.recorder;
    this.recorder = null;
    this.recordTrack = null;
    this.composite = null;
    this.compositeContext = null;

    if (recorder.state === 'inactive') {
      return Promise.resolve(new Blob(this.recorded, { type: recorder.mimeType }));
    }
    return new Promise((resolve) => {
      // A ceiling, so a browser that never fires the event cannot leave the button stuck
      // and the flight's recording unreachable.
      const timer = setTimeout(
        () => resolve(new Blob(this.recorded, { type: recorder.mimeType })), 4000,
      );
      recorder.onstop = () => {
        clearTimeout(timer);
        resolve(new Blob(this.recorded, { type: recorder.mimeType }));
      };
      recorder.stop();
    });
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
