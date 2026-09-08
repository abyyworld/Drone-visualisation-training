/**
 * Flame and smoke on the device, with no model file and no network.
 *
 * WHY THIS EXISTS AT ALL
 *     The COCO detector that does people and vehicles has no fire class and never will -
 *     the dataset has no such category. So on-device wildfire work had exactly two routes:
 *     ship a trained model for flame, or compute it. Every pretrained one that is actually
 *     reachable and actually accurate is a YOLO derivative under AGPL-3.0, which is a
 *     licence this project cannot take. Training one here is not possible either: no GPU,
 *     no labelled flame imagery on disk.
 *
 *     So it is computed. Flame and smoke are two of the few things in vision that have a
 *     genuine physical signature rather than a learned shape, and the rules below are the
 *     published ones - Chen's RGB rule and Celik and Demirel's YCbCr rule for flame colour,
 *     with the piece that actually separates fire from a red jacket bolted on top:
 *     time. Fire flickers. It is the flicker, not the colour, that carries the evidence.
 *
 * WHAT IT ACTUALLY MEASURES
 *     Per cell of a coarse grid, over a rolling window of frames:
 *       flameRatio   how much of the cell is flame-coloured
 *       flicker      how much that fraction moves between frames
 *       smokeRatio   how much of the cell is desaturated in the smoke luminance band
 *       edgeDrop     how far the cell's texture has fallen from its own recent best,
 *                    because smoke veils whatever is behind it
 *       yMove        how much the cell's brightness drifts, because a plume moves
 *     A cell has to pass on colour AND on time. A red car parked in shot passes the first
 *     and fails the second, which is the entire point.
 *
 * WHAT IT IS NOT
 *     It is a candidate finder. It says "this region looks and behaves like flame" - it
 *     does not confirm a wildfire, and a single frame with no history behind it earns a
 *     deliberately capped confidence, because on one photograph the time evidence does not
 *     exist. Regions are for an operator to look at, and for the provider models to be
 *     pointed at. None of it reads in the other direction: a frame with nothing marked on
 *     it has not been cleared of anything, it has only failed to meet a threshold.
 */

export const FLAME = 'flame';
export const SMOKE = 'smoke';

// Working resolution. Everything is computed on a downscaled copy: the signal here lives in
// regions, not in pixels, and 160 across is enough to find a plume while costing about a
// millisecond a frame, which is what lets it run beside the detector without slowing it.
const WORK_WIDTH = 160;
const CELL = 8;

// A window of ~1.5 s at the cadence the live loop detects at. Long enough to measure
// flicker, short enough that a fire appearing is reported within a couple of seconds.
const HISTORY = 14;
const MIN_HISTORY = 4;

// How fast a cell forgets how much texture it used to have. Deliberately far slower than
// the flicker window: a plume that drifts in and then sits there would otherwise become its
// own baseline within a second and stop being reported, which is the exact moment an
// operator most needs the box. At 0.995 a frame the memory half-lives in about fifteen
// seconds, so a settled plume stays flagged and a genuinely flat grey wall fades out.
const EDGE_DECAY = 0.995;

// Flame colour, from the published rules.
const R_MIN = 110;          // Chen's R_T
const S_T = 55;             // Chen's saturation threshold at R_T
const CBCR_GAP = 40;        // Celik and Demirel's tau

// Smoke colour: near-grey, in a luminance band that excludes both black shadow and blown
// highlight. Both bounds matter - without the lower one, every dark corner is smoke.
const SMOKE_SPREAD = 26;
const SMOKE_Y_MIN = 55;
const SMOKE_Y_MAX = 245;

// Cell gates. Below any of these the cell is not offered, whatever the arithmetic says.
const FLAME_RATIO_FLOOR = 0.25;
const FLAME_FLICKER_FLOOR = 0.012;
const FLAME_OCCUPANCY_FLOOR = 0.50;
const FLAME_PRESENT = 0.15;
// How much a cell's flame fraction has to move between two frames to count as having
// changed at all, and how often it has to do that. Rate, not amplitude: see below.
const FLICKER_DELTA = 0.06;
const FLICKER_RATE_FLOOR = 0.45;
const FLAME_STILL_FLOOR = 0.15;
const SMOKE_RATIO_FLOOR = 0.35;
const SMOKE_EDGE_DROP_FLOOR = 0.20;
const SMOKE_MOVE_FLOOR = 0.35;
const SMOKE_STILL_FLOOR = 0.50;
const SMOKE_STILL_EDGE_MAX = 6;

// A confidence ceiling for a region found without any time evidence behind it.
const STILL_CEILING_FLAME = 0.60;
const STILL_CEILING_SMOKE = 0.45;

const MIN_CELLS_FLAME = 2;
const MIN_CELLS_SMOKE = 6;
const MIN_SPAN_SMOKE = 2;   // cells, in both directions: a plume is a mass, not a line

const clamp01 = (value) => (value < 0 ? 0 : value > 1 ? 1 : value);

export class FireScan {
  constructor({ workWidth = WORK_WIDTH, cell = CELL, history = HISTORY } = {}) {
    this.workWidth = workWidth;
    this.cell = cell;
    this.historyLength = history;
    this.reset();
  }

  /** Forget every frame seen so far. Called whenever the source changes. */
  reset() {
    this.cols = 0;
    this.rows = 0;
    this.history = [];   // one array of samples per cell, oldest first
    this.edgeBase = null;   // per cell, how much texture it used to have
    this.frames = 0;
  }

  /**
   * Scan one frame of raw RGBA.
   *
   * @param {Uint8ClampedArray|Uint8Array|Array} data  RGBA, 4 bytes per pixel
   * @param {number} width
   * @param {number} height
   * @returns {Array<{label:string, confidence:number, box:number[], cells:number,
   *                  evidence:object, temporal:boolean}>}
   *   Boxes normalised to 0..1, so the caller scales them to whatever it is drawing on.
   */
  scanPixels(data, width, height) {
    const small = downsample(data, width, height, this.workWidth);
    const grid = this.measure(small);
    this.remember(grid);
    return this.regions(grid);
  }

  /**
   * Scan whatever the DOM has: a video element, a canvas, an image, an ImageBitmap.
   * Boxes come back in pixels of `width` x `height`, matching every other engine here.
   */
  scan(source, width, height) {
    const ratio = height > 0 && width > 0 ? height / width : 0.75;
    const w = this.workWidth;
    const h = Math.max(1, Math.round(w * ratio));

    if (!this.canvas || this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas = makeCanvas(w, h);
      this.context = this.canvas.getContext('2d', { willReadFrequently: true });
    }
    this.context.drawImage(source, 0, 0, w, h);
    const frame = this.context.getImageData(0, 0, w, h);

    return this.scanPixels(frame.data, w, h).map((finding) => ({
      ...finding,
      box: [
        finding.box[0] * width, finding.box[1] * height,
        finding.box[2] * width, finding.box[3] * height,
      ],
    }));
  }

  /** Per-cell colour, texture and brightness for one frame. */
  measure({ rgb, width, height }) {
    const cell = this.cell;
    const cols = Math.max(1, Math.ceil(width / cell));
    const rows = Math.max(1, Math.ceil(height / cell));
    if (cols !== this.cols || rows !== this.rows) {
      this.cols = cols;
      this.rows = rows;
      this.history = Array.from({ length: cols * rows }, () => []);
      this.edgeBase = new Float32Array(cols * rows).fill(-1);
      this.frames = 0;
    }

    const pixels = width * height;
    const luma = new Float32Array(pixels);
    const cb = new Float32Array(pixels);
    const cr = new Float32Array(pixels);
    let sumY = 0; let sumCb = 0; let sumCr = 0;

    for (let i = 0; i < pixels; i += 1) {
      const r = rgb[i * 3];
      const g = rgb[i * 3 + 1];
      const b = rgb[i * 3 + 2];
      const y = 0.299 * r + 0.587 * g + 0.114 * b;
      luma[i] = y;
      cb[i] = -0.168736 * r - 0.331264 * g + 0.5 * b + 128;
      cr[i] = 0.5 * r - 0.418688 * g - 0.081312 * b + 128;
      sumY += y; sumCb += cb[i]; sumCr += cr[i];
    }
    // Celik and Demirel's region rules compare each pixel against the frame's own means,
    // which is what makes the test hold under a bright sky and under dusk alike.
    const meanY = sumY / pixels;
    const meanCb = sumCb / pixels;
    const meanCr = sumCr / pixels;

    const flame = new Float32Array(cols * rows);
    const smoke = new Float32Array(cols * rows);
    const bright = new Float32Array(cols * rows);
    const edge = new Float32Array(cols * rows);
    const counts = new Float32Array(cols * rows);

    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        const i = y * width + x;
        const r = rgb[i * 3];
        const g = rgb[i * 3 + 1];
        const b = rgb[i * 3 + 2];
        const c = Math.floor(y / cell) * cols + Math.floor(x / cell);

        counts[c] += 1;
        bright[c] += luma[i];

        const max = r > g ? (r > b ? r : b) : (g > b ? g : b);
        const min = r < g ? (r < b ? r : b) : (g < b ? g : b);
        const saturation = max > 0 ? ((max - min) / max) * 255 : 0;

        const isFlame = r > g && g > b
          && r >= R_MIN
          && saturation >= ((255 - r) * S_T) / R_MIN
          && luma[i] > cb[i] && cr[i] > cb[i]
          && Math.abs(cb[i] - cr[i]) >= CBCR_GAP
          && luma[i] >= meanY && cb[i] <= meanCb && cr[i] >= meanCr;

        if (isFlame) {
          flame[c] += 1;
        } else if (max - min <= SMOKE_SPREAD
          && luma[i] >= SMOKE_Y_MIN && luma[i] <= SMOKE_Y_MAX) {
          smoke[c] += 1;
        }

        // Texture, as the plain gradient of brightness. Smoke pulls this down because it
        // veils whatever detail was behind it, which is a more reliable smoke cue than the
        // colour is: plenty of things are grey, very few of them erase the background.
        const right = x + 1 < width ? luma[i + 1] : luma[i];
        const below = y + 1 < height ? luma[i + width] : luma[i];
        edge[c] += Math.abs(right - luma[i]) + Math.abs(below - luma[i]);
      }
    }

    for (let c = 0; c < counts.length; c += 1) {
      const n = counts[c] || 1;
      flame[c] /= n; smoke[c] /= n; bright[c] /= n; edge[c] /= n;
    }
    return { cols, rows, flame, smoke, bright, edge };
  }

  /** Push this frame's cell measurements onto each cell's rolling window. */
  remember(grid) {
    for (let c = 0; c < grid.flame.length; c += 1) {
      const window = this.history[c];
      window.push({
        flame: grid.flame[c], smoke: grid.smoke[c], bright: grid.bright[c], edge: grid.edge[c],
      });
      if (window.length > this.historyLength) window.shift();
    }
    this.frames += 1;
  }

  /** Score every cell, then join the ones that pass into regions. */
  regions(grid) {
    const { cols, rows } = grid;
    const flameScore = new Float32Array(cols * rows);
    const smokeScore = new Float32Array(cols * rows);
    const evidence = new Array(cols * rows);
    let temporal = false;

    for (let c = 0; c < cols * rows; c += 1) {
      const window = this.history[c];
      const seen = window.length;
      const hasTime = seen >= MIN_HISTORY;
      if (hasTime) temporal = true;

      const flameNow = grid.flame[c];
      const smokeNow = grid.smoke[c];
      const edgeNow = grid.edge[c];

      let flameMean = flameNow;
      let flicker = 0;
      let move = 0;
      let occupancy = flameNow >= FLAME_PRESENT ? 1 : 0;
      let flickerRate = 0;
      if (seen > 1) {
        let sum = 0;
        let held = 0;
        let changed = 0;
        for (let i = 0; i < seen; i += 1) {
          sum += window[i].flame;
          if (window[i].flame >= FLAME_PRESENT) held += 1;
          if (i > 0) {
            const delta = Math.abs(window[i].flame - window[i - 1].flame);
            flicker += delta;
            if (delta >= FLICKER_DELTA) changed += 1;
            move += Math.abs(window[i].bright - window[i - 1].bright);
          }
        }
        flameMean = sum / seen;
        occupancy = held / seen;
        flickerRate = changed / (seen - 1);
        flicker /= seen - 1;
        move /= seen - 1;
      }

      // Texture is compared against the cell's own decaying memory, not against the frames
      // still in the flicker window - see EDGE_DECAY. The comparison is made against the
      // memory as it stood before this frame, so a cell whose detail has just returned
      // reports no drop rather than a spurious one.
      const prior = this.edgeBase[c] < 0 ? edgeNow : this.edgeBase[c] * EDGE_DECAY;
      const edgeDrop = prior > 0.5 ? clamp01((prior - edgeNow) / prior) : 0;
      this.edgeBase[c] = Math.max(edgeNow, prior);

      evidence[c] = {
        flameMean, flicker, flickerRate, occupancy, smokeNow, edgeDrop, move, hasTime,
      };

      if (hasTime) {
        // Four conditions, and the last two are the ones that earn their keep.
        //
        // Amplitude of flicker is not enough, and this is where a naive version of this
        // method falls over. A red van driving through the shot makes a cell swing from no
        // flame colour to all of it and back - a bigger swing than a real fire ever
        // produces. But it does it twice, on the two frames the van's edge crosses that
        // cell, and holds perfectly steady for the ten frames in between. Fire never holds
        // steady: the tongues rise and fall, so the fraction moves on nearly every frame.
        //
        // So the test is the rate at which the cell changes, not how far it changes, and
        // occupancy catches whatever crosses too fast to be measured that way. Colour says
        // where to look; these two say whether it is burning.
        if (flameMean >= FLAME_RATIO_FLOOR
          && flicker >= FLAME_FLICKER_FLOOR
          && flickerRate >= FLICKER_RATE_FLOOR
          && occupancy >= FLAME_OCCUPANCY_FLOOR) {
          flameScore[c] = clamp01(
            0.40 * clamp01(flameMean / 0.30)
            + 0.25 * clamp01(flicker / 0.05)
            + 0.25 * flickerRate
            + 0.10 * clamp01(flameNow / 0.25),
          );
        }
        if (smokeNow >= SMOKE_RATIO_FLOOR
          && edgeDrop >= SMOKE_EDGE_DROP_FLOOR
          && move >= SMOKE_MOVE_FLOOR) {
          smokeScore[c] = clamp01(
            0.40 * clamp01(smokeNow / 0.60)
            + 0.35 * clamp01(edgeDrop / 0.50)
            + 0.25 * clamp01(move / 2),
          );
        }
      } else {
        // One frame, no history. Colour is all there is, so the ceiling is lower and the
        // floor is higher: a still is judged on stronger colour evidence than a stream,
        // and can never reach the confidence a flickering region earns.
        if (flameNow >= FLAME_STILL_FLOOR) {
          flameScore[c] = Math.min(STILL_CEILING_FLAME, 0.60 * clamp01(flameNow / 0.35));
        }
        if (smokeNow >= SMOKE_STILL_FLOOR && edgeNow <= SMOKE_STILL_EDGE_MAX) {
          smokeScore[c] = Math.min(STILL_CEILING_SMOKE, 0.45 * clamp01(smokeNow / 0.75));
        }
      }
    }

    const found = [
      ...join(flameScore, cols, rows, MIN_CELLS_FLAME, FLAME, this.cell, temporal, evidence),
      ...join(smokeScore, cols, rows, MIN_CELLS_SMOKE, SMOKE, this.cell, temporal, evidence),
    ];
    return dropSky(found, cols, rows);
  }
}

/**
 * Flood-fill the passing cells into rectangles.
 *
 * Four-connected on purpose. Eight-connected joins a flame region to an unrelated red thing
 * touching it only at a corner, and a box drawn around both is worse than two boxes.
 */
function join(score, cols, rows, minCells, label, cell, temporal, evidence) {
  const seen = new Uint8Array(cols * rows);
  const out = [];

  for (let start = 0; start < score.length; start += 1) {
    if (seen[start] || score[start] <= 0) continue;

    const stack = [start];
    const members = [];
    seen[start] = 1;

    while (stack.length) {
      const at = stack.pop();
      members.push(at);
      const x = at % cols;
      const y = Math.floor(at / cols);
      const neighbours = [
        x > 0 ? at - 1 : -1,
        x + 1 < cols ? at + 1 : -1,
        y > 0 ? at - cols : -1,
        y + 1 < rows ? at + cols : -1,
      ];
      for (const next of neighbours) {
        if (next >= 0 && !seen[next] && score[next] > 0) { seen[next] = 1; stack.push(next); }
      }
    }
    if (members.length < minCells) continue;

    let minX = cols; let minY = rows; let maxX = 0; let maxY = 0; let total = 0;
    let flicker = 0; let edgeDrop = 0;
    for (const at of members) {
      const x = at % cols;
      const y = Math.floor(at / cols);
      if (x < minX) minX = x;
      if (y < minY) minY = y;
      if (x > maxX) maxX = x;
      if (y > maxY) maxY = y;
      total += score[at];
      flicker += evidence[at].flickerRate;
      edgeDrop += evidence[at].edgeDrop;
    }

    out.push({
      label,
      confidence: total / members.length,
      classId: colourIndex(label),
      cells: members.length,
      temporal,
      evidence: {
        flicker: flicker / members.length,
        edgeDrop: edgeDrop / members.length,
      },
      box: [minX / cols, minY / rows, (maxX + 1) / cols, (maxY + 1) / rows],
    });
  }
  return out;
}

/**
 * Throw away the smoke region that is actually the sky.
 *
 * Overcast, haze and a flat grey sky all pass the smoke colour test and, on a moving
 * handheld camera, can pass the movement test too. What they cannot do is be small. A grey
 * region that spans the frame and hangs off the top edge is weather; a plume is a shape.
 */
function dropSky(found, cols, rows) {
  const flames = found.filter((f) => f.label === FLAME);
  return found.filter((finding) => {
    if (finding.label !== SMOKE) return true;
    const [x0, y0, x1, y1] = finding.box;
    const width = x1 - x0;
    const height = y1 - y0;

    // A flame in the same frame settles the argument: grey above a fire is smoke.
    if (flames.some((flame) => overlaps(flame.box, finding.box))) return true;

    // The horizon. Where flat overcast meets the ground, the boundary row wobbles with the
    // camera, and that one row of cells shows both the grey and the movement smoke shows.
    // It is a hairline across the whole frame, and a plume is never that shape.
    if (width >= 0.90 && height <= 0.20) return false;
    if (width < MIN_SPAN_SMOKE / cols || height < MIN_SPAN_SMOKE / rows) return false;

    // Hanging off the top edge and spanning the frame: sky, not plume.
    if (y0 <= 1 / rows && width >= 0.80) return false;
    return width * height <= 0.60;
  });
}

function overlaps(a, b) {
  return !(a[2] < b[0] || b[2] < a[0] || a[3] < b[1] || b[3] < a[1]);
}

/**
 * Area-average down to the working width.
 *
 * Averaging rather than nearest-neighbour, because nearest-neighbour throws away exactly
 * the thing being measured: a thin flame edge that happens to fall between sample points
 * simply disappears, and the ratio the whole method rests on comes out wrong.
 */
function downsample(data, width, height, targetWidth) {
  const w = Math.min(width, targetWidth);
  const h = Math.max(1, Math.round((height / width) * w));
  const rgb = new Float32Array(w * h * 3);

  for (let y = 0; y < h; y += 1) {
    const y0 = Math.floor((y * height) / h);
    const y1 = Math.max(y0 + 1, Math.floor(((y + 1) * height) / h));
    for (let x = 0; x < w; x += 1) {
      const x0 = Math.floor((x * width) / w);
      const x1 = Math.max(x0 + 1, Math.floor(((x + 1) * width) / w));
      let r = 0; let g = 0; let b = 0; let n = 0;
      for (let sy = y0; sy < y1 && sy < height; sy += 1) {
        for (let sx = x0; sx < x1 && sx < width; sx += 1) {
          const i = (sy * width + sx) * 4;
          r += data[i]; g += data[i + 1]; b += data[i + 2]; n += 1;
        }
      }
      const at = (y * w + x) * 3;
      rgb[at] = r / n; rgb[at + 1] = g / n; rgb[at + 2] = b / n;
    }
  }
  return { rgb, width: w, height: h };
}

function makeCanvas(width, height) {
  if (typeof OffscreenCanvas === 'function') return new OffscreenCanvas(width, height);
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  return canvas;
}

/** Same hash as ondevice.js and vlm.js, so a label keeps one colour across every engine. */
function colourIndex(label) {
  let hash = 0;
  for (let i = 0; i < label.length; i += 1) hash = (hash * 31 + label.charCodeAt(i)) % 4096;
  return hash;
}
