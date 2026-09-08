/**
 * Tests for the flame and smoke scanner.
 *
 *   node tests/test_firescan.mjs
 *
 * Synthetic frames, painted by hand, because the thing worth testing is not "does it fire
 * on a picture of fire" - it is the separation. A red car, a sunset, a grey sky and an
 * overcast horizon all look like fire or smoke to a colour rule, and the whole method rests
 * on time evidence throwing them out. So every positive case here has a negative twin that
 * differs only in whether it moves.
 */

import { FireScan, FLAME, SMOKE } from '../web/js/firescan.js';

let failures = 0;
let passes = 0;

function check(name, condition, detail = '') {
  if (condition) { passes += 1; console.log(`  ok    ${name}`); }
  else { failures += 1; console.error(`  FAIL  ${name}${detail ? ` -- ${detail}` : ''}`); }
}

const W = 160;
const H = 120;

/** A blank RGBA frame, painted by a callback that returns [r,g,b] for each pixel. */
function paint(painter) {
  const data = new Uint8ClampedArray(W * H * 4);
  for (let y = 0; y < H; y += 1) {
    for (let x = 0; x < W; x += 1) {
      const [r, g, b] = painter(x, y);
      const i = (y * W + x) * 4;
      data[i] = r; data[i + 1] = g; data[i + 2] = b; data[i + 3] = 255;
    }
  }
  return data;
}

// A forest floor: dark, green, and textured, so there is detail for smoke to hide.
const ground = (x, y) => {
  const noise = ((x * 7 + y * 13) % 11) * 6;
  return [28 + noise, 58 + noise, 22 + (noise >> 1)];
};

const inside = (x, y, box) => x >= box[0] && x < box[2] && y >= box[1] && y < box[3];

const FIRE_BOX = [40, 60, 80, 100];

console.log('\nFlame, moving');
{
  const scan = new FireScan();
  let found = [];
  for (let frame = 0; frame < 20; frame += 1) {
    // Tongues, rising and falling. Each column of the fire has its own height, and that
    // height oscillates, so the flame fraction of a cell moves on nearly every frame. That
    // is what a fire does and what the red panel in the next case cannot do - and it is
    // measured as how often the fraction changes, not how far, because the panel wins on
    // how far every time.
    const data = paint((x, y) => {
      if (!inside(x, y, FIRE_BOX)) return ground(x, y);
      const height = 22 + 14 * Math.sin(frame * 1.1 + x * 0.35);
      if (y < FIRE_BOX[3] - height) return ground(x, y);
      const ember = ((x * 3 + y * 5 + frame * 7) % 9) === 0;
      return ember ? [120, 45, 12] : [255, 140, 30];
    });
    found = scan.scanPixels(data, W, H);
  }
  const flames = found.filter((f) => f.label === FLAME);
  check('a flickering flame region is found', flames.length > 0);
  check('and is confident about it', flames[0]?.confidence > 0.45,
    `confidence ${flames[0]?.confidence?.toFixed(2)}`);
  check('and says the evidence was temporal', flames[0]?.temporal === true);
  if (flames[0]) {
    // The painted fire is x 0.25..0.50, y 0.50..0.83. The box should sit inside that, give
    // or take the grid it is quantised to, and cover most of its width. It will not reach
    // the top of the region, and should not: the tongues only get there some of the time.
    const [x0, y0, x1, y1] = flames[0].box;
    const where = `box ${[x0, y0, x1, y1].map((v) => v.toFixed(2)).join(', ')}`;
    check('the box sits on the painted fire', x0 >= 0.20 && x1 <= 0.56, where);
    check('and covers most of its width', x1 - x0 >= 0.20, where);
    check('and reaches down to its base', y1 >= 0.75 && y0 >= 0.45, where);
  }
}

console.log('\nA red thing that is not on fire');
{
  const scan = new FireScan();
  let found = [];
  for (let frame = 0; frame < 20; frame += 1) {
    // Identical colour to the flame above, identical size, identical place. The one
    // difference is that it holds still, because it is a painted panel and not a fire.
    const data = paint((x, y) => (inside(x, y, FIRE_BOX) ? [255, 140, 30] : ground(x, y)));
    found = scan.scanPixels(data, W, H);
  }
  check('a static red panel is not called flame', found.every((f) => f.label !== FLAME),
    JSON.stringify(found.map((f) => [f.label, f.confidence.toFixed(2)])));
}

console.log('\nA red thing that merely moves');
{
  const scan = new FireScan();
  let found = [];
  for (let frame = 0; frame < 20; frame += 1) {
    // A red van driving across the shot. Cells at its leading and trailing edge do see the
    // flame fraction change, so a couple of edge cells can pass; what must not happen is a
    // confident region over the body of it, where the colour is solid and unmoving.
    const shift = frame * 4;
    const box = [10 + shift, 60, 50 + shift, 90];
    const data = paint((x, y) => (inside(x, y, box) ? [255, 140, 30] : ground(x, y)));
    found = scan.scanPixels(data, W, H);
  }
  const flames = found.filter((f) => f.label === FLAME);
  check('a moving red vehicle does not become a confident flame',
    flames.every((f) => f.confidence < 0.6),
    JSON.stringify(flames.map((f) => f.confidence.toFixed(2))));
}

console.log('\nSunset');
{
  const scan = new FireScan();
  let found = [];
  for (let frame = 0; frame < 20; frame += 1) {
    // The classic false positive for every colour rule ever written for flame: half the
    // sky the colour of flame, with the sensor noise and the slow drift of dusk on top of
    // it. It is a huge, bright, orange region and it is not burning.
    const data = paint((x, y) => {
      if (y >= 55) return ground(x, y);
      const drift = frame * 0.4;
      const noise = ((x + y * 3 + frame) % 5) - 2;
      return [252 - drift + noise, 138 - y + noise, 40 + y + noise];
    });
    found = scan.scanPixels(data, W, H);
  }
  check('a sunset sky is not called flame', found.every((f) => f.label !== FLAME),
    JSON.stringify(found.map((f) => [f.label, f.confidence.toFixed(2)])));
}

console.log('\nSmoke drifting over cover');
{
  const scan = new FireScan();
  let found = [];
  for (let frame = 0; frame < 24; frame += 1) {
    // Five clean frames first, so each cell learns how much detail it used to have. Then
    // the plume drifts in and erases it, which is the measurement.
    const arrived = Math.max(0, frame - 5);
    const top = Math.max(0, 100 - arrived * 6);
    const data = paint((x, y) => {
      const inPlume = frame > 5 && y >= top && y < 100
        && Math.abs(x - 80) < 30 + ((y + frame) % 7);
      if (!inPlume) return ground(x, y);
      const grey = 168 + ((x + y * 2 + frame * 9) % 5);
      return [grey, grey + 2, grey - 1];
    });
    found = scan.scanPixels(data, W, H);
  }
  const smoke = found.filter((f) => f.label === SMOKE);
  check('a drifting plume is found', smoke.length > 0,
    JSON.stringify(found.map((f) => [f.label, f.confidence.toFixed(2)])));
  check('and it is not the whole frame', smoke.every((f) => {
    const [x0, y0, x1, y1] = f.box;
    return (x1 - x0) * (y1 - y0) < 0.6;
  }));
}

console.log('\nAn office, which is the one that got out');
{
  // The screenshot that started this, reduced to its essentials: a flat pale wall, and a
  // person moving across it. The wall is desaturated and sits in the smoke luminance band,
  // and the moving person erases its texture the way a plume erases a hillside. The
  // shipped version marked that wall as smoke at 82 percent.
  //
  // It was not a threshold that was slightly too low. It was a measure being applied where
  // it means nothing: a painted wall has about one luma level of detail, so an ordinary
  // flicker of one level is a fifty percent drop. A cell with nothing to lose cannot lose
  // it, and now says so.
  const scan = new FireScan();
  let found = [];
  for (let frame = 0; frame < 24; frame += 1) {
    const shift = frame * 3;
    const data = paint((x, y) => {
      if (Math.abs(x - (30 + shift)) < 26 && y > 30) return [46, 40, 38];
      const wall = 190 + ((x + y) % 2);
      return [wall, wall + 1, wall - 1];
    });
    found = scan.scanPixels(data, W, H);
  }
  check('a painted wall behind a moving person is not smoke',
    found.every((f) => f.label !== SMOKE),
    JSON.stringify(found.map((f) => [f.label, f.box.map((v) => v.toFixed(2))])));
}

console.log('\nA tracked object explains away what it is standing in front of');
{
  // The second line of defence, for a textured background where the drop is real evidence
  // but has an ordinary explanation: something the detector is already following is
  // standing there. The caller passes the boxes it is tracking; a smoke region mostly
  // covered by one of them has its missing texture accounted for already.
  const scan = new FireScan();
  const withBoxes = new FireScan();
  let blind = [];
  let informed = [];
  for (let frame = 0; frame < 24; frame += 1) {
    const arrived = Math.max(0, frame - 5);
    const top = Math.max(0, 100 - arrived * 6);
    const data = paint((x, y) => {
      const inPlume = frame > 5 && y >= top && y < 100
        && Math.abs(x - 80) < 30 + ((y + frame) % 7);
      if (!inPlume) return ground(x, y);
      const grey = 168 + ((x + y * 2 + frame * 9) % 5);
      return [grey, grey + 2, grey - 1];
    });
    blind = scan.scanPixels(data, W, H);
    informed = withBoxes.scanPixels(data, W, H, [[0.2, 0, 0.8, 1]]);
  }
  check('without the boxes it is smoke', blind.some((f) => f.label === SMOKE));
  check('with them it is a tracked thing', informed.every((f) => f.label !== SMOKE),
    JSON.stringify(informed.map((f) => f.label)));
}

console.log('\nWeather, which is not smoke');
{
  const scan = new FireScan();
  let found = [];
  for (let frame = 0; frame < 24; frame += 1) {
    // Flat overcast across the top half, with the handheld wobble that would otherwise
    // pass for a plume moving. It hangs off the top edge and spans the frame, so it is sky.
    const wobble = (frame % 3) - 1;
    const data = paint((x, y) => {
      if (y < 60 + wobble) {
        const grey = 176 + ((x + frame) % 4);
        return [grey, grey, grey + 1];
      }
      return ground(x, y);
    });
    found = scan.scanPixels(data, W, H);
  }
  check('an overcast sky is not called smoke', found.every((f) => f.label !== SMOKE),
    JSON.stringify(found.map((f) => [f.label, f.box.map((v) => v.toFixed(2))])));
}

console.log('\nNothing at all');
{
  const scan = new FireScan();
  let found = [];
  for (let frame = 0; frame < 20; frame += 1) {
    const data = paint((x, y) => ground(x + frame, y));
    found = scan.scanPixels(data, W, H);
  }
  check('a moving forest floor produces no regions', found.length === 0,
    JSON.stringify(found.map((f) => [f.label, f.confidence.toFixed(2)])));
}

console.log('\nOne photograph, no history');
{
  const scan = new FireScan();
  const data = paint((x, y) => {
    if (!inside(x, y, FIRE_BOX)) return ground(x, y);
    const burning = ((x * 3 + y * 5) % 10) > 2;
    return burning ? [255, 140, 30] : [120, 40, 10];
  });
  const found = scan.scanPixels(data, W, H);
  const flames = found.filter((f) => f.label === FLAME);
  check('a still frame still finds the flame', flames.length > 0);
  check('but says so with capped confidence', flames.every((f) => f.confidence <= 0.6),
    JSON.stringify(flames.map((f) => f.confidence.toFixed(2))));
  check('and does not claim time evidence it does not have',
    flames.every((f) => f.temporal === false));
}

console.log('\nHousekeeping');
{
  const scan = new FireScan();
  const data = paint(ground);
  for (let frame = 0; frame < 6; frame += 1) scan.scanPixels(data, W, H);
  check('frames are counted', scan.frames === 6);
  scan.reset();
  check('reset forgets them', scan.frames === 0 && scan.history.length === 0);

  // A different aspect ratio must not carry the previous grid's history into the new one.
  const wide = new Uint8ClampedArray(320 * 100 * 4).fill(255);
  const out = scan.scanPixels(wide, 320, 100);
  check('a new frame size rebuilds the grid', Array.isArray(out));
}

console.log(`\n${passes} passed, ${failures} failed\n`);
process.exit(failures === 0 ? 0 : 1);
