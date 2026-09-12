/** What tracker settings do best on REAL footage.
 *
 *     node tools/sweep_tracker.mjs flight-*.json
 *
 * Every constant in the tracker was chosen against tools/fly.py, which pans across a still
 * frame. Real video is a different problem: people move independently, walk behind each
 * other, and are blurred exactly when the scene is hardest. Measured on VisDrone MOT, the
 * repeat numbering is about 1.85 numbers per person against the 1.35 the panned stills
 * predicted, so the settings were tuned for a problem easier than the real one.
 *
 * Judged on what the operator asked for, in this order:
 *   reached  - of the people in view, how many ever got a number. Missing somebody is the
 *              worst outcome, so this is not traded away for a tidier count.
 *   each     - numbers per person reached. Above 1.00 is the same person numbered twice,
 *              which is the "why does it say person 67" complaint.
 *   stray    - numbers on nobody.
 */
import { readFileSync } from 'node:fs';
import { Tracker } from '../web/js/track.js';

function iou(a, b) {
  const x0 = Math.max(a[0], b[0]), y0 = Math.max(a[1], b[1]);
  const x1 = Math.min(a[2], b[2]), y1 = Math.min(a[3], b[3]);
  if (x1 <= x0 || y1 <= y0) return 0;
  const i = (x1 - x0) * (y1 - y0);
  return i / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i);
}

const files = process.argv.slice(2);
if (!files.length) {
  console.error('usage: node tools/sweep_tracker.mjs flight-*.json');
  process.exit(2);
}

function score(options) {
  let people = 0, reached = 0, numbers = 0, stray = 0;
  for (const file of files) {
    const data = JSON.parse(readFileSync(file, 'utf8'));
    for (const run of data.runs) {
      const tracker = new Tracker(options);
      // Which real person each number spent the most frames on. `who` is the dataset's own
      // target id here, so this is a measured identity switch and not an estimate of one.
      const hits = new Map();
      let clock = 1000;
      for (const frame of run.frames) {
        for (const track of tracker.update(frame.detections, clock)) {
          let best = -1, bestScore = 0.3;
          frame.truth.forEach((p, i) => {
            const v = iou(track.box, p);
            if (v >= bestScore) { bestScore = v; best = i; }
          });
          if (!hits.has(track.id)) hits.set(track.id, new Map());
          if (best < 0) continue;
          const m = hits.get(track.id), who = frame.who[best];
          m.set(who, (m.get(who) ?? 0) + 1);
        }
        clock += data.period;
      }
      const owners = new Set();
      for (const [, m] of hits) {
        if (!m.size) { stray += 1; continue; }
        let who = -1, n = -1;
        for (const [k, v] of m) if (v > n) { n = v; who = k; }
        owners.add(who);
        numbers += 1;
      }
      people += run.peoplePresent;
      reached += owners.size;
    }
  }
  return { reached: reached / Math.max(1, people), each: numbers / Math.max(1, reached), stray };
}

const base = score({});
const show = (label, s) => {
  const dr = (s.reached - base.reached) * 100;
  const de = s.each - base.each;
  console.log(`  ${label.padEnd(34)}${(s.reached*100).toFixed(0).padStart(8)}%`
    + `${(dr >= 0 ? '+' : '') + dr.toFixed(0)}`.padStart(6)
    + `${s.each.toFixed(2).padStart(9)}${(de >= 0 ? '+' : '') + de.toFixed(2)}`.padStart(7)
    + `${String(s.stray).padStart(8)}`);
};

console.log(`\n  ${files.length} real sequences\n`);
console.log(`  ${'setting'.padEnd(34)}${'reached'.padStart(9)}${''.padStart(5)}${'each'.padStart(8)}${''.padStart(7)}${'stray'.padStart(8)}`);
show('as shipped', base);
console.log();
for (const v of [0.10, 0.15, 0.20, 0.30]) show(`minIou ${v}`, score({ minIou: v }));
for (const v of [800, 1200, 2000, 3000, 4000]) show(`maxCoastMs ${v}`, score({ maxCoastMs: v }));
for (const v of [2, 3, 4, 5, 6, 8]) show(`confirmAfter ${v}`, score({ confirmAfter: v }));
for (const v of [0.20, 0.25, 0.30, 0.35]) show(`newTrackConfidence ${v}`, score({ newTrackConfidence: v }));
for (const v of [3, 5, 10, 20]) show(`maxMisses ${v}`, score({ maxMisses: v }));
