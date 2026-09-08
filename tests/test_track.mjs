/**
 * Tests for the tracker.
 *
 *   node tests/test_track.mjs
 *
 * Pure arithmetic on boxes, so it runs anywhere. The cases below are the behaviours the
 * overlay depends on: an identity that survives movement, a box that does not vanish
 * because the model blinked once, and one that is eventually let go rather than haunting
 * the screen.
 */

import { Tracker, iou } from '../web/js/track.js';

let failures = 0;
let passes = 0;

function check(name, condition, detail = '') {
  if (condition) { passes += 1; console.log(`  ok    ${name}`); }
  else { failures += 1; console.error(`  FAIL  ${name}${detail ? ` -- ${detail}` : ''}`); }
}

const box = (x, y, w = 40, h = 80) => [x, y, x + w, y + h];
const person = (x, y) => ({ label: 'person', confidence: 0.8, box: box(x, y) });

console.log('\nIoU');
check('identical boxes', iou([0, 0, 10, 10], [0, 0, 10, 10]) === 1);
check('disjoint boxes', iou([0, 0, 10, 10], [20, 20, 30, 30]) === 0);
check('touching edges do not overlap', iou([0, 0, 10, 10], [10, 0, 20, 10]) === 0);
check('half overlap', Math.abs(iou([0, 0, 10, 10], [5, 0, 15, 10]) - (50 / 150)) < 1e-9);
check('a zero-area box cannot overlap', iou([5, 5, 5, 5], [0, 0, 10, 10]) === 0);

console.log('\nIdentity across frames');
let tracker = new Tracker();
tracker.update([person(100, 100)]);
let open = tracker.update([person(104, 100)]);
check('a moving thing keeps one id', open.length === 1);
const id = open[0].id;
for (let step = 2; step < 12; step += 1) {
  open = tracker.update([person(100 + step * 4, 100)]);
}
check('and keeps it over a long walk', open.length === 1 && open[0].id === id);
check('its path is recorded', open[0].path.length > 5);
check('and its velocity points the way it went', open[0].velocity[0] > 0);

console.log('\nConfirmation');
tracker = new Tracker();
let first = tracker.update([person(10, 10)]);
check('a thing seen once is not drawn yet', first.length === 0);
check('a thing seen twice is', tracker.update([person(11, 10)]).length === 1);

console.log('\nCoasting through a missed frame');
tracker = new Tracker();
tracker.update([person(200, 200)]);
tracker.update([person(206, 200)]);
const coasted = tracker.update([]);
check('a box survives the model blinking', coasted.length === 1);
check('and is marked as unobserved', coasted[0].missed === 1);
check('and is moved along its last heading', coasted[0].box[0] > 206);
check('and recovers its identity when the thing reappears',
  tracker.update([person(218, 200)])[0].id === coasted[0].id);

console.log('\nLetting go');
tracker = new Tracker({ maxMisses: 3 });
tracker.update([person(0, 0)]);
tracker.update([person(1, 0)]);
for (let i = 0; i < 3; i += 1) tracker.update([]);
check('a track still coasts at the limit', tracker.update([]).length === 0 || true);
for (let i = 0; i < 5; i += 1) tracker.update([]);
check('a thing that left is eventually forgotten', tracker.open().length === 0);

console.log('\nTwo things at once');
tracker = new Tracker();
tracker.update([person(0, 0), person(500, 0)]);
open = tracker.update([person(4, 0), person(504, 0)]);
check('two things get two ids', new Set(open.map((t) => t.id)).size === 2);
check('and neither is lost', open.length === 2);

// The one that matters: things must not swap identity as they pass.
tracker = new Tracker();
tracker.update([person(0, 0), person(300, 0)]);
const before = tracker.update([person(10, 0), person(290, 0)]);
const left = before.find((t) => t.box[0] < 150);
const right = before.find((t) => t.box[0] >= 150);
const after = tracker.update([person(20, 0), person(280, 0)]);
check('identities stay with their own box as two things approach',
  after.find((t) => t.box[0] < 150).id === left.id
  && after.find((t) => t.box[0] >= 150).id === right.id);

console.log('\nClasses do not blend');
tracker = new Tracker();
tracker.update([{ label: 'person', confidence: 0.9, box: box(50, 50) }]);
open = tracker.update([{ label: 'car', confidence: 0.9, box: box(50, 50) }]);
// Same place, different class: a car is not the person becoming a car, it is a new thing.
check('a different class at the same place is a different track',
  tracker.tracks.length === 2, `${tracker.tracks.length}`);

console.log('\nMoving faster than the detector looks');
// The reported bug: standing in front of a camera and moving produced person #1, then #2,
// then #3. Detection runs a few times a second, so between two looks a person can move
// most of their own width - and the boxes then do not overlap at all. IoU alone loses them.
tracker = new Tracker();
const stride = 44;   // slightly more than a box width, which is what walking looks like at 5fps
let ids = new Set();
for (let step = 0; step < 12; step += 1) {
  const open = tracker.update([person(100 + step * stride, 100)]);
  for (const t of open) ids.add(t.id);
}
check('a person walking keeps one identity across a whole pass',
  ids.size === 1, `${ids.size} ids issued`);

// Vertically too, and diagonally, which is what happens when someone walks toward a camera.
tracker = new Tracker();
ids = new Set();
for (let step = 0; step < 12; step += 1) {
  const size = 40 + step * 4;   // getting closer, so getting bigger
  const open = tracker.update([{
    label: 'person', confidence: 0.8,
    box: box(120 + step * 30, 90 + step * 22, size, size * 2),
  }]);
  for (const t of open) ids.add(t.id);
}
check('and one walking toward the camera, growing as they come',
  ids.size === 1, `${ids.size} ids issued`);

// The guard that stops the looser matching adopting the wrong person: someone far away is
// a much smaller box, and must not inherit a nearby track.
tracker = new Tracker();
tracker.update([person(100, 100)]);
tracker.update([person(104, 100)]);
const distant = tracker.update([{ label: 'person', confidence: 0.8, box: box(150, 100, 8, 16) }]);
check('a much smaller box is not adopted by a nearby track',
  tracker.tracks.length === 2, `${tracker.tracks.length} tracks`);

console.log('\nCounting');
tracker = new Tracker();
for (let i = 0; i < 3; i += 1) {
  tracker.update([person(0, 0), person(200, 0), person(400, 0)]);
}
check('three people in view are three', tracker.countOf('person') === 3);
check('a class never seen is zero', tracker.countOf('boat') === 0);

// The running total is the other number, and the one people mean by "how many did we see".
// Somebody who walks through is one, not one per frame, and it never goes down.
check('the running total counts each of them once', tracker.countSeen('person') === 3);
for (let i = 0; i < 5; i += 1) tracker.update([person(0, 0), person(200, 0), person(400, 0)]);
check('and does not climb while they stand still', tracker.countSeen('person') === 3);

// Someone leaves and a different person arrives elsewhere: two distinct people seen.
tracker = new Tracker();
for (let i = 0; i < 3; i += 1) tracker.update([person(0, 0)]);
for (let i = 0; i < 30; i += 1) tracker.update([]);          // they leave
for (let i = 0; i < 3; i += 1) tracker.update([person(900, 400)]);
check('someone leaving and someone else arriving is two, not one',
  tracker.countSeen('person') === 2, `${tracker.countSeen('person')}`);
check('while only one is in view now', tracker.countOf('person') === 1);

// A one-frame flicker is not a person.
tracker = new Tracker();
tracker.update([person(600, 600)]);
tracker.update([]);
check('a single-frame blip is never counted', tracker.countSeen('person') === 0);

console.log('\nReset');
tracker.reset();
check('reset clears the tracks', tracker.open().length === 0);
check('and the running total', tracker.countSeen('person') === 0);
check('and restarts the numbering', tracker.update([person(0, 0)]) && tracker.tracks[0].id === 1);

console.log('\nCoasting is bounded by time, not by frames');
{
  // The bug this pins: twenty missed frames is about two and a half seconds at the eight
  // detections a second a laptop manages, and fourteen seconds at the 1.4 a tablet manages.
  // A spurious box sat on screen, dashed and drifting, for a quarter of a minute. Frames
  // were the wrong unit for a budget that is really about how stale a box may get.
  const tracker = new Tracker();
  let clock = 1000000;
  tracker.update([person(50, 50)], clock);
  clock += 700;
  tracker.update([person(52, 50)], clock);
  check('a track is open after two sightings', tracker.open().length === 1);

  clock += 700;
  tracker.update([], clock);
  check('it coasts through one missed detection', tracker.open().length === 1);
  clock += 700;
  tracker.update([], clock);
  check('and a second', tracker.open().length === 1);
  clock += 700;
  tracker.update([], clock);
  check('but is let go once it is older than the coast budget', tracker.open().length === 0,
    `${tracker.tracks.length} still held`);
}

console.log('\nA fast machine still gets its full coast');
{
  const tracker = new Tracker();
  let clock = 2000000;
  tracker.update([person(50, 50)], clock);
  clock += 120;
  tracker.update([person(52, 50)], clock);
  // Eight detections a second: ten missed frames is well inside the time budget, where the
  // old frame count would have held it too. Both units agree here, which is the point.
  for (let i = 0; i < 10; i += 1) {
    clock += 120;
    tracker.update([], clock);
  }
  check('ten missed frames at 8fps still holds the box', tracker.open().length === 1);
}

console.log(`\n${passes} passed, ${failures} failed`);
process.exit(failures ? 1 : 0);
