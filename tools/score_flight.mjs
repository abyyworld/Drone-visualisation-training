/** Score a flight the way the operator reads the screen: who got a number, and how many
 *  numbers each of them got.
 *
 *  countSeen alone cannot separate "found more people" from "numbered the same person
 *  again", and those pull in opposite directions - a model that flickers scores higher on
 *  a raw count while being worse at the job. Truth identity is carried through from the
 *  harness, so every confirmed box is attributed to the person it sits on:
 *
 *    reached   of the people in view, how many ever got a number at all
 *    numbers   how many numbers those people got between them
 *    repeats   numbers beyond one per person, which is the failure the operator sees
 *    stray     confirmed boxes sitting on nobody
 */
import { readFileSync } from 'node:fs';
import { Tracker } from '../web/js/track.js';

const data = JSON.parse(readFileSync(process.argv[2], 'utf8'));
function iou(a, b) {
  const x0 = Math.max(a[0], b[0]), y0 = Math.max(a[1], b[1]);
  const x1 = Math.min(a[2], b[2]), y1 = Math.min(a[3], b[3]);
  if (x1 <= x0 || y1 <= y0) return 0;
  const i = (x1 - x0) * (y1 - y0);
  return i / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i);
}

let P = 0, R = 0, N = 0, X = 0, B = 0, ON = 0;
const rows = [];
for (const run of data.runs) {
  const tracker = new Tracker();
  // A number belongs to the person it spent the most frames on. In a crowd this matters:
  // 175 people in one frame means a box that has drifted a little sits on SOMEBODY at
  // IoU 0.3, so crediting a number to everyone it ever brushed counts drift as if it were
  // re-numbering and reports three times the real figure.
  const hits = new Map();     // track id -> Map(person -> frames)
  let boxes = 0, on = 0, clock = 1000;
  for (const frame of run.frames) {
    const open = tracker.update(frame.detections, clock);
    clock += data.period;
    for (const t of open) {
      boxes += 1;
      let best = -1, bestScore = 0.3;
      frame.truth.forEach((p, i) => { const v = iou(t.box, p); if (v >= bestScore) { bestScore = v; best = i; } });
      if (!hits.has(t.id)) hits.set(t.id, new Map());
      if (best < 0) continue;
      on += 1;
      const who = frame.who[best];
      const m = hits.get(t.id);
      m.set(who, (m.get(who) ?? 0) + 1);
    }
  }
  const owner = new Map();    // person -> how many numbers claimed them
  let stray = 0;
  for (const [, m] of hits) {
    if (!m.size) { stray += 1; continue; }
    let who = -1, n = -1;
    for (const [k, v] of m) if (v > n) { n = v; who = k; }
    owner.set(who, (owner.get(who) ?? 0) + 1);
  }
  const reached = owner.size;
  const numbers = [...owner.values()].reduce((a, v) => a + v, 0);
  rows.push([run.name, run.peoplePresent, reached, numbers, numbers - reached, stray, boxes ? on / boxes * 100 : 0]);
  P += run.peoplePresent; R += reached; N += numbers; X += stray; B += boxes; ON += on;
}
const c = data.config;
console.log(`${c.tiles_per_cycle} tile/cycle, ${c.grid}, ${data.period} ms, ${c.model.split('/').pop()}`);
console.log(`  ${'flight'.padEnd(14)}${'present'.padStart(8)}${'reached'.padStart(9)}${'numbers'.padStart(9)}${'repeats'.padStart(9)}${'stray'.padStart(7)}${'on a person'.padStart(13)}`);
for (const [n, p, r, nn, rep, st, pct] of rows)
  console.log(`  ${n.padEnd(14)}${String(p).padStart(8)}${String(r).padStart(9)}${String(nn).padStart(9)}${String(rep).padStart(9)}${String(st).padStart(7)}${(pct.toFixed(0)+'%').padStart(13)}`);
console.log(`  ${'TOTAL'.padEnd(14)}${String(P).padStart(8)}${String(R).padStart(9)}${String(N).padStart(9)}${String(X === 0 ? N-R : N-R).padStart(9)}${String(X).padStart(7)}${((ON/Math.max(1,B))*100).toFixed(1).padStart(12)}%`);
console.log(`  reached ${(R/P*100).toFixed(0)}% of the people, and gave them ${(N/Math.max(1,R)).toFixed(2)} numbers each`);
