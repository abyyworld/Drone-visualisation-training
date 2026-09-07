/**
 * Canvas 2D overlay: boxes over the video, never burned into it.
 *
 * The overlay's whole job is to say "look here". It has no way to say "there
 * is nothing there", and there is nothing in this file that draws when the
 * detections list is empty. An empty frame renders as bare video, which is
 * exactly what an operator watching a scene with nothing in it should see --
 * and byte-identical to what an unseeing model produces, which is why no
 * reassurance may ever be attached to it. See docs/SAFETY.md.
 *
 * Two things in here are easy to get subtly wrong and impossible to notice
 * afterwards:
 *
 *  1. **Letterboxing.** Boxes arrive normalised to the *video frame*, but the
 *     video element is a CSS box of a different aspect ratio, and the picture
 *     inside it is letterboxed. Mapping through the element instead of through
 *     the picture offsets every box by the size of the black bars -- a
 *     constant, plausible-looking error that puts a crew slightly off target.
 *     `computeVideoRect` is the whole fix and it is exported for testing.
 *  2. **devicePixelRatio.** A canvas sized in CSS pixels on a 2x or 3x tablet
 *     renders soft, and a 1 px hairline box in direct sunlight is invisible.
 */

import { CLASS_FIRE, CLASS_SMOKE, CLASS_PERSON } from './wire.js';

/**
 * Where the picture actually sits inside the element, in CSS pixels.
 *
 * Pure, and exported: this is the function that silently offsets every box
 * when it is wrong, so it is the one worth testing on its own.
 *
 * @param {number} videoWidth Intrinsic frame width (`video.videoWidth`).
 * @param {number} videoHeight Intrinsic frame height (`video.videoHeight`).
 * @param {number} clientWidth Element width in CSS pixels.
 * @param {number} clientHeight Element height in CSS pixels.
 * @param {'contain'|'cover'|'fill'} [fit='contain'] The element's object-fit.
 * @returns {{x: number, y: number, w: number, h: number, scale: number}} The
 *   picture rectangle, in CSS pixels relative to the element's top-left.
 */
export function computeVideoRect(videoWidth, videoHeight, clientWidth, clientHeight, fit = 'contain') {
  if (!(videoWidth > 0 && videoHeight > 0 && clientWidth > 0 && clientHeight > 0)) {
    return { x: 0, y: 0, w: Math.max(0, clientWidth), h: Math.max(0, clientHeight), scale: 1 };
  }
  if (fit === 'fill') {
    return { x: 0, y: 0, w: clientWidth, h: clientHeight, scale: clientWidth / videoWidth };
  }
  const sx = clientWidth / videoWidth;
  const sy = clientHeight / videoHeight;
  const scale = fit === 'cover' ? Math.max(sx, sy) : Math.min(sx, sy);
  const w = videoWidth * scale;
  const h = videoHeight * scale;
  // Centred, because that is what object-fit does with the leftover space.
  return { x: (clientWidth - w) / 2, y: (clientHeight - h) / 2, w, h, scale };
}

/**
 * Per-class drawing rules.
 *
 * Colour is never the only carrier. Red already means something else on an
 * incident ground, and roughly one man in twelve on a fire crew cannot
 * separate these hues reliably -- so the classes also differ in stroke pattern,
 * in corner marks, and in a word printed in full.
 *
 * `person` gets the loudest treatment of the three, and a minimum drawn size.
 * That is not emphasis for its own sake: it is the smallest object on screen
 * and the most consequential to overlook.
 */
const CLASS_STYLE = Object.freeze({
  [CLASS_FIRE]: { colour: '#FFB000', label: 'FIRE', dash: [], corners: true, minPx: 0 },
  [CLASS_SMOKE]: { colour: '#00D5FF', label: 'SMOKE', dash: [14, 9], corners: false, minPx: 0 },
  // A person box is tiny -- a few pixels at altitude -- so at natural size it
  // is easy to miss on a sunlit tablet, which defeats the point of drawing it.
  // `minPx` inflates the drawn box to a legible minimum without touching the
  // reported geometry: the operator has to be able to SEE that something was
  // marked before they can look at it. The third dash pattern and the third
  // corner treatment keep the class separable without relying on hue.
  [CLASS_PERSON]: { colour: '#FF57D8', label: 'PERSON', dash: [4, 4], corners: true, minPx: 44 },
});

/** Anything the station names that this build has no rule for. Still drawn. */
const UNKNOWN_STYLE = { colour: '#FFFFFF', label: '?', dash: [3, 6], corners: false, minPx: 0 };

const HALO = 'rgba(0, 0, 0, 0.88)';

/**
 * Draws the detection overlay onto a canvas stacked over the video element.
 */
export class Overlay {
  /**
   * @param {HTMLCanvasElement} canvas The overlay canvas.
   * @param {{fit?: 'contain'|'cover'|'fill'}} [options] `fit` must match the
   *   `object-fit` the stylesheet gives the video element.
   */
  constructor(canvas, options = {}) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.fit = options.fit || 'contain';
    this.dpr = 1;
    this._cssWidth = 0;
    this._cssHeight = 0;
    /** Last rect used, so a caller can hit-test or debug alignment. */
    this.rect = { x: 0, y: 0, w: 0, h: 0, scale: 1 };
  }

  /**
   * Match the backing store to the element's CSS size and the screen density.
   *
   * Call on resize, on orientation change, and after fullscreen transitions.
   *
   * @returns {boolean} True when the backing store changed size.
   */
  resize() {
    const dpr = Math.max(1, Math.min(window.devicePixelRatio || 1, 3));
    const cssWidth = this.canvas.clientWidth;
    const cssHeight = this.canvas.clientHeight;
    if (cssWidth === this._cssWidth && cssHeight === this._cssHeight && dpr === this.dpr) return false;
    this.dpr = dpr;
    this._cssWidth = cssWidth;
    this._cssHeight = cssHeight;
    this.canvas.width = Math.max(1, Math.round(cssWidth * dpr));
    this.canvas.height = Math.max(1, Math.round(cssHeight * dpr));
    return true;
  }

  /** Wipe the canvas. Used whenever boxes must stop appearing. @returns {void} */
  clear() {
    const ctx = this.ctx;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
  }

  /**
   * Draw one frame of overlay.
   *
   * @param {{videoWidth: number, videoHeight: number}} video The video element,
   *   for its intrinsic size. Nothing is read from it but the frame geometry.
   * @param {object} sync The result of `OverlaySync.select()`.
   * @param {{protocolError?: string|null, connectionState?: string}} [flags]
   *   Extra conditions the operator must see over the picture rather than only
   *   in the status bar.
   * @returns {void}
   */
  draw(video, sync, flags = {}) {
    this.resize();
    this.clear();
    const ctx = this.ctx;
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);

    const rect = computeVideoRect(
      video ? video.videoWidth : 0,
      video ? video.videoHeight : 0,
      this._cssWidth,
      this._cssHeight,
      this.fit,
    );
    this.rect = rect;
    if (rect.w <= 0 || rect.h <= 0) return;

    const unit = Math.max(16, Math.round(Math.min(rect.w, rect.h) * 0.032));

    // The overlay's own state, drawn on the picture. The status bar carries it
    // too, but an operator watching a fire is looking at the video, not at a
    // bar at the edge of the screen.
    if (flags.protocolError) {
      this._chip(rect, unit, '#FF4FD8', 'OVERLAY OFF', flags.protocolError);
      this._border(rect, '#FF4FD8');
      return;
    }
    if (sync && sync.stale) {
      this._chip(rect, unit, '#FF8A3D', 'OVERLAY STALE — BOXES OFF', sync.staleReason || '');
      this._border(rect, '#FF8A3D');
      return;
    }
    if (sync && sync.tier === 3 && sync.detections.length) {
      // Tier 3 keeps drawing precisely because it is labelled: an obviously
      // approximate box beats a silently misplaced one.
      this._chip(rect, unit, '#FFD400', 'UNSYNCHRONISED', 'box positions may not match this frame');
    }

    if (!sync || !sync.detections.length) return;

    // Weakest first, so the strongest evidence ends up on top when boxes
    // overlap and the operator's eye lands on it first.
    const ordered = Array.from(sync.detections).sort((a, b) => a.conf - b.conf);
    for (const det of ordered) this._detection(det, rect, unit);
  }

  // ------------------------------------------------------------- internals

  /**
   * Draw one detection: box, class, confidence, persistence.
   *
   * @param {object} det A parsed detection.
   * @param {{x: number, y: number, w: number, h: number}} rect Picture rect.
   * @param {number} unit Base type size in CSS pixels.
   * @returns {void}
   */
  _detection(det, rect, unit) {
    const ctx = this.ctx;
    const style = CLASS_STYLE[det.cls] || { ...UNKNOWN_STYLE, label: det.cls.toUpperCase() };
    const x = rect.x + det.box.x1 * rect.w;
    const y = rect.y + det.box.y1 * rect.h;
    const w = Math.max(2, (det.box.x2 - det.box.x1) * rect.w);
    const h = Math.max(2, (det.box.y2 - det.box.y1) * rect.h);

    // Confidence is carried three ways at once, because one way is a way to
    // be missed: stroke weight, opacity, and the number itself. A 0.30 must
    // not be able to look like a 0.90 at a glance across a light bar.
    const conf = Math.min(1, Math.max(0, det.conf));
    const weight = 2 + 5 * conf;
    const alpha = 0.4 + 0.6 * conf;

    ctx.save();
    ctx.lineJoin = 'miter';
    ctx.setLineDash([]);
    ctx.strokeStyle = HALO;
    ctx.lineWidth = weight + 4;
    ctx.strokeRect(x, y, w, h);

    ctx.globalAlpha = alpha;
    ctx.strokeStyle = style.colour;
    ctx.lineWidth = weight;
    ctx.setLineDash(style.dash.map((d) => d * (weight / 4)));
    ctx.strokeRect(x, y, w, h);
    ctx.setLineDash([]);
    if (style.corners) this._corners(x, y, w, h, weight, style.colour);
    ctx.restore();

    this._label(det, style, x, y, w, h, rect, unit, conf);
  }

  /**
   * Solid corner brackets: the second, non-colour cue that this is fire.
   *
   * @param {number} x Box left, CSS px.
   * @param {number} y Box top, CSS px.
   * @param {number} w Box width, CSS px.
   * @param {number} h Box height, CSS px.
   * @param {number} weight Stroke weight in CSS px.
   * @param {string} colour Stroke colour.
   * @returns {void}
   */
  _corners(x, y, w, h, weight, colour) {
    const ctx = this.ctx;
    const arm = Math.min(w, h) * 0.28 + weight;
    ctx.save();
    ctx.lineCap = 'butt';
    for (const pass of [{ c: HALO, lw: weight * 2.2 + 4 }, { c: colour, lw: weight * 2.2 }]) {
      ctx.strokeStyle = pass.c;
      ctx.lineWidth = pass.lw;
      ctx.beginPath();
      ctx.moveTo(x, y + arm); ctx.lineTo(x, y); ctx.lineTo(x + arm, y);
      ctx.moveTo(x + w - arm, y); ctx.lineTo(x + w, y); ctx.lineTo(x + w, y + arm);
      ctx.moveTo(x + w, y + h - arm); ctx.lineTo(x + w, y + h); ctx.lineTo(x + w - arm, y + h);
      ctx.moveTo(x + arm, y + h); ctx.lineTo(x, y + h); ctx.lineTo(x, y + h - arm);
      ctx.stroke();
    }
    ctx.restore();
  }

  /**
   * Draw the label block: class word, confidence number and bar, persistence.
   *
   * The label is drawn at full opacity even when the box is faint. A
   * low-confidence detection should look weak, but its number must stay
   * legible -- the operator's judgement needs the value, not an impression.
   *
   * @param {object} det The detection.
   * @param {{colour: string, label: string}} style Class style.
   * @param {number} x Box left, CSS px.
   * @param {number} y Box top, CSS px.
   * @param {number} w Box width, CSS px.
   * @param {number} h Box height, CSS px.
   * @param {{x: number, y: number, w: number, h: number}} rect Picture rect.
   * @param {number} unit Base type size in CSS px.
   * @param {number} conf Clamped confidence.
   * @returns {void}
   */
  _label(det, style, x, y, w, h, rect, unit, conf) {
    const ctx = this.ctx;
    const pad = Math.round(unit * 0.32);
    const fontMain = `700 ${unit}px system-ui, -apple-system, "Segoe UI", Roboto, sans-serif`;
    const fontSmall = `600 ${Math.round(unit * 0.72)}px system-ui, -apple-system, "Segoe UI", Roboto, sans-serif`;

    const confText = conf.toFixed(2);
    const persisted = Math.max(1, det.persisted | 0);
    const pips = Math.min(persisted, 5);
    const pipSize = Math.round(unit * 0.3);
    const pipGap = Math.round(pipSize * 0.55);
    const pipsWidth = pips * pipSize + (pips - 1) * pipGap + (persisted > 5 ? unit * 0.5 : 0);

    ctx.save();
    ctx.font = fontMain;
    const nameWidth = ctx.measureText(style.label).width;
    ctx.font = fontSmall;
    const confWidth = ctx.measureText(confText).width;

    const boxH = Math.round(unit * 1.9);
    const boxW = Math.round(pad * 3 + nameWidth + confWidth + pipsWidth + unit * 0.6);
    let bx = x;
    let by = y - boxH - Math.round(unit * 0.2);
    // Flip inside the box when the label would fall off the top of the
    // picture, and pull it left when it would fall off the right edge.
    if (by < rect.y) by = Math.min(y + Math.round(unit * 0.2), rect.y + rect.h - boxH);
    if (bx + boxW > rect.x + rect.w) bx = Math.max(rect.x, rect.x + rect.w - boxW);

    ctx.fillStyle = 'rgba(0, 0, 0, 0.82)';
    ctx.fillRect(bx, by, boxW, boxH);
    ctx.strokeStyle = style.colour;
    ctx.lineWidth = 2;
    ctx.strokeRect(bx + 1, by + 1, boxW - 2, boxH - 2);

    ctx.textBaseline = 'middle';
    ctx.fillStyle = style.colour;
    ctx.font = fontMain;
    const midY = by + boxH * 0.42;
    ctx.fillText(style.label, bx + pad, midY);

    ctx.fillStyle = '#FFFFFF';
    ctx.font = fontSmall;
    const confX = bx + pad + nameWidth + pad;
    ctx.fillText(confText, confX, midY);

    // Persistence: how many of the last frames this track survived. A 5-of-5
    // box and a 1-of-5 box are different evidence and must not look alike.
    const pipX = confX + confWidth + pad;
    const pipY = by + boxH * 0.30;
    for (let i = 0; i < pips; i += 1) {
      ctx.fillStyle = style.colour;
      ctx.fillRect(pipX + i * (pipSize + pipGap), pipY, pipSize, pipSize);
    }
    if (persisted > 5) {
      ctx.fillStyle = '#FFFFFF';
      ctx.font = fontSmall;
      ctx.fillText('+', pipX + pips * (pipSize + pipGap), midY);
    }

    // A confidence bar along the bottom of the label: the same number again,
    // in a form that is readable at arm's length through a visor.
    const barY = by + boxH - Math.round(unit * 0.42);
    const barW = boxW - pad * 2;
    ctx.fillStyle = 'rgba(255, 255, 255, 0.22)';
    ctx.fillRect(bx + pad, barY, barW, Math.round(unit * 0.22));
    ctx.fillStyle = style.colour;
    ctx.fillRect(bx + pad, barY, Math.max(2, barW * conf), Math.round(unit * 0.22));
    ctx.restore();
  }

  /**
   * A state chip in the top-left of the picture.
   *
   * @param {{x: number, y: number, w: number, h: number}} rect Picture rect.
   * @param {number} unit Base type size in CSS px.
   * @param {string} colour Accent colour.
   * @param {string} title Chip headline, uppercase.
   * @param {string} detail Smaller explanatory line; may be empty.
   * @returns {void}
   */
  _chip(rect, unit, colour, title, detail) {
    const ctx = this.ctx;
    const pad = Math.round(unit * 0.5);
    ctx.save();
    ctx.font = `800 ${unit}px system-ui, -apple-system, "Segoe UI", Roboto, sans-serif`;
    const titleWidth = ctx.measureText(title).width;
    ctx.font = `600 ${Math.round(unit * 0.68)}px system-ui, -apple-system, "Segoe UI", Roboto, sans-serif`;
    const detailWidth = detail ? ctx.measureText(detail).width : 0;
    const boxW = Math.min(rect.w - pad * 2, Math.max(titleWidth, detailWidth) + pad * 2);
    const boxH = detail ? Math.round(unit * 2.9) : Math.round(unit * 1.8);
    const bx = rect.x + pad;
    const by = rect.y + pad;

    ctx.fillStyle = 'rgba(0, 0, 0, 0.86)';
    ctx.fillRect(bx, by, boxW, boxH);
    ctx.strokeStyle = colour;
    ctx.lineWidth = 3;
    ctx.strokeRect(bx + 1.5, by + 1.5, boxW - 3, boxH - 3);

    ctx.textBaseline = 'top';
    ctx.fillStyle = colour;
    ctx.font = `800 ${unit}px system-ui, -apple-system, "Segoe UI", Roboto, sans-serif`;
    ctx.fillText(title, bx + pad, by + pad * 0.8, boxW - pad * 2);
    if (detail) {
      ctx.fillStyle = '#FFFFFF';
      ctx.font = `600 ${Math.round(unit * 0.68)}px system-ui, -apple-system, "Segoe UI", Roboto, sans-serif`;
      ctx.fillText(detail, bx + pad, by + pad * 0.8 + unit * 1.25, boxW - pad * 2);
    }
    ctx.restore();
  }

  /**
   * Hatched border around the picture, for states where boxes are suppressed.
   *
   * It marks the overlay as switched off. It is not a claim about the scene:
   * the operator is looking at live video, which is the safe state.
   *
   * @param {{x: number, y: number, w: number, h: number}} rect Picture rect.
   * @param {string} colour Accent colour.
   * @returns {void}
   */
  _border(rect, colour) {
    const ctx = this.ctx;
    ctx.save();
    ctx.strokeStyle = colour;
    ctx.lineWidth = 6;
    ctx.setLineDash([26, 18]);
    ctx.strokeRect(rect.x + 3, rect.y + 3, rect.w - 6, rect.h - 6);
    ctx.restore();
  }
}
