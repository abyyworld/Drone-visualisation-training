/** What the flame scanner does on real drone video, at the cadence the tablet scans it.
 *
 *     node tools/scan_fire_video.mjs <dir of .rgba frames> <width> <height>
 *
 * WHY THIS IS THE MEASUREMENT THAT WAS MISSING
 *     FireScan has never been scored against anything. The fire MODEL has: 311 labelled
 *     pictures, a threshold sweep, a miss list. The scanner beside it was reasoned about
 *     and shipped, and it is the half that runs on every wildfire flight whether a model
 *     loads or not.
 *
 *     Its recall cannot be measured here - that needs video of real fire, and FLAME and
 *     FLAME2 are behind an account no machine in this project can reach. But the half that
 *     decides whether it is usable CAN be: how often it marks flame or smoke on real drone
 *     footage with no fire in it. A candidate finder that marks every third frame of an
 *     empty field is not a candidate finder, it is noise, and an operator stops reading it.
 *
 *     Stills would not do. The whole method is that a red car passes on colour and fails on
 *     time, so a scanner shown single photographs is being asked a question it declines to
 *     answer. These are consecutive frames, sampled at the cycle the device achieves, which
 *     is the input it is written for.
 *
 * WHY RAW BYTES
 *     scanPixels takes RGBA, which is what the device hands it. Decoding JPEG belongs to
 *     the step that writes these files, so this stays the same code path the app runs.
 */
import { readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { FireScan, FLAME, SMOKE } from '../web/js/firescan.js';

const [dir, widthArg, heightArg] = process.argv.slice(2);
if (!dir || !widthArg || !heightArg) {
  console.error('usage: node tools/scan_fire_video.mjs <dir> <width> <height>');
  process.exit(2);
}
const width = Number(widthArg);
const height = Number(heightArg);
const frames = readdirSync(dir).filter((f) => f.endsWith('.rgba')).sort();
if (!frames.length) {
  console.error(`no .rgba frames in ${dir}`);
  process.exit(2);
}

const scan = new FireScan();
let marked = 0;
let flameFrames = 0;
let smokeFrames = 0;
let worst = 0;
for (const name of frames) {
  const bytes = readFileSync(join(dir, name));
  if (bytes.length !== width * height * 4) {
    console.error(`${name}: ${bytes.length} bytes, expected ${width * height * 4}`);
    process.exit(2);
  }
  const found = scan.scanPixels(new Uint8ClampedArray(bytes), width, height);
  if (found.length) marked += 1;
  if (found.some((r) => r.label === FLAME)) flameFrames += 1;
  if (found.some((r) => r.label === SMOKE)) smokeFrames += 1;
  for (const region of found) worst = Math.max(worst, region.confidence);
}
console.log(`  ${dir.split('/').pop().padEnd(22)}`
  + `${String(frames.length).padStart(8)} frames`
  + `${String(marked).padStart(9)} marked`
  + `${String(flameFrames).padStart(8)} flame`
  + `${String(smokeFrames).padStart(8)} smoke`
  + `${worst.toFixed(2).padStart(9)} worst`);
