/**
 * Tests for tiling and for putting the pieces back together.
 *
 *   node tests/test_tiles.mjs
 *
 * The reason this file exists: a crowd shot returned five people, not fifty, and it was not
 * a threshold. The detector's input is 448 pixels square, so a 1920-wide frame is squeezed
 * by more than four, and a person forty pixels tall arrives at nine. Nine pixels is below
 * what any detector can find. Cutting the frame up and looking at each piece at its own
 * resolution is what puts those people in front of the model at a size it can work with.
 *
 * What is tested here is the arithmetic around that, because it is where the mistakes are:
 * a tile mapped back one pixel wrong puts a box beside a person instead of on them, and a
 * merge that is too shy counts one person twice - which is the thing the whole system is
 * trying not to do.
 */

import {
  TILE_COUNT, TILE_COLUMNS, TILE_ROWS,
  tileRegion, overlap, merge, toFrame, coversFrame, containment,
} from '../web/js/tiles.js';

let failures = 0;
let passes = 0;

function check(name, condition, detail = '') {
  if (condition) { passes += 1; console.log(`  ok    ${name}`); }
  else { failures += 1; console.error(`  FAIL  ${name}${detail ? ` -- ${detail}` : ''}`); }
}

const W = 1920;
const H = 1080;

console.log('\nTiles cover the frame');
{
  check('there are as many tiles as the grid says', TILE_COUNT === TILE_COLUMNS * TILE_ROWS);

  const regions = coversFrame(W, H);
  check('every tile is inside the frame', regions.every(
    (r) => r.x >= 0 && r.y >= 0 && r.x + r.width <= W + 0.001 && r.y + r.height <= H + 0.001,
  ));

  // The real question: is any pixel of the frame in no tile at all? A gap is a strip of the
  // picture that never gets a close look, and on a wide crowd shot that is people missed.
  let uncovered = 0;
  for (let y = 4; y < H; y += 8) {
    for (let x = 4; x < W; x += 8) {
      const inside = regions.some(
        (r) => x >= r.x && x < r.x + r.width && y >= r.y && y < r.y + r.height,
      );
      if (!inside) uncovered += 1;
    }
  }
  check('no part of the frame is missed', uncovered === 0, `${uncovered} sample points uncovered`);

  // And they must overlap, or someone standing on a seam is half in two tiles and whole in
  // neither, which is exactly the person a crowd shot is full of.
  const first = regions[0];
  const second = regions[1];
  check('neighbouring tiles overlap', first.x + first.width > second.x,
    `${(first.x + first.width).toFixed(0)} vs ${second.x.toFixed(0)}`);
}

console.log('\nThe cycle visits every tile');
{
  const seen = new Set();
  for (let i = 0; i < TILE_COUNT; i += 1) {
    const region = tileRegion(i, W, H);
    seen.add(`${Math.round(region.x)},${Math.round(region.y)}`);
  }
  check('one full cycle is every tile, once', seen.size === TILE_COUNT);
  const wrapped = tileRegion(TILE_COUNT, W, H);
  const first = tileRegion(0, W, H);
  check('and then it starts again', wrapped.x === first.x && wrapped.y === first.y);
}

console.log('\nA box found in a tile lands where the person is');
{
  const region = tileRegion(4, W, H);
  // A tile drawn at half its source size, with a box in the middle of the drawn image.
  const scaleX = 2;
  const scaleY = 2;
  const box = toFrame([10, 20, 30, 60], region, scaleX, scaleY);
  check('the offset is applied', box[0] === region.x + 20 && box[1] === region.y + 40);
  // Compared with a tolerance: the tile's origin is a fraction of a frame width and the
  // arithmetic is floating point, so an exact equality here fails on the last bit and says
  // nothing about whether the mapping is right.
  check('and the scale',
    Math.abs((box[2] - box[0]) - 40) < 1e-6 && Math.abs((box[3] - box[1]) - 80) < 1e-6,
    `${(box[2] - box[0]).toFixed(6)} x ${(box[3] - box[1]).toFixed(6)}`);
}

console.log('\nThe same person seen twice is one person');
{
  const person = (x, y, confidence) => ({
    label: 'person', confidence, box: [x, y, x + 40, y + 90],
  });

  // The full frame found them roughly; the tile found them precisely and is more confident,
  // because it saw three times as many of their pixels.
  const fromFrame = [person(100, 100, 0.51)];
  const fromTile = [person(104, 98, 0.88)];
  const merged = merge(fromFrame, fromTile);

  check('they are one finding, not two', merged.length === 1, `${merged.length} findings`);
  check('and the closer look won', merged[0].confidence === 0.88);
  check('with its box', merged[0].box[0] === 104);

  // Two people standing near each other are still two people.
  const crowd = merge([person(100, 100, 0.7)], [person(160, 100, 0.7)]);
  check('two people side by side stay two', crowd.length === 2);

  // A car and a person in the same place are not the same thing.
  const mixed = merge(
    [person(100, 100, 0.7)],
    [{ label: 'car', confidence: 0.7, box: [100, 100, 140, 190] }],
  );
  check('different classes never merge', mixed.length === 2);
}

console.log('\nTwo people standing close together');
{
  // Reported from a real flight: the app drew one box around two people and counted them
  // as one. At full-frame scale two people side by side are one blob of pixels, so that
  // pass returns one wide box; the tile, looking three times larger, sees both. The old
  // merge asked only "do these overlap?" and threw both tile boxes away in favour of the
  // blob, which is the exact opposite of what tiling is for.
  const blob = { label: 'person', confidence: 0.62, box: [100, 100, 190, 200] };
  const left = { label: 'person', confidence: 0.81, box: [102, 100, 142, 198] };
  const right = { label: 'person', confidence: 0.78, box: [148, 102, 188, 200] };

  const merged = merge([blob], [left, right]);
  check('the closer look wins and there are two people', merged.length === 2,
    `${merged.length} findings: ${JSON.stringify(merged.map((f) => f.box))}`);
  check('the blur over both of them is gone',
    merged.every((f) => f.box[2] - f.box[0] < 60),
    JSON.stringify(merged.map((f) => f.box[2] - f.box[0])));

  // And the ordinary case still holds: one person seen by both passes is one person, not
  // one person plus a blur.
  const one = merge(
    [{ label: 'person', confidence: 0.5, box: [100, 100, 140, 190] }],
    [{ label: 'person', confidence: 0.9, box: [102, 99, 142, 189] }],
  );
  check('one person seen twice is still one', one.length === 1, `${one.length}`);
  check('and keeps the confident reading', one[0].confidence === 0.9);
}

console.log('\nContainment');
{
  check('fully inside is one', containment([2, 2, 8, 8], [0, 0, 10, 10]) === 1);
  check('fully outside is zero', containment([20, 20, 30, 30], [0, 0, 10, 10]) === 0);
  check('half in is a half', Math.abs(containment([5, 0, 15, 10], [0, 0, 10, 10]) - 0.5) < 1e-9);
}

console.log('\nMerging never loses anyone');
{
  // Fifty people spread across the frame, found by the tiles, none of them overlapping.
  const people = [];
  for (let i = 0; i < 50; i += 1) {
    const x = (i % 10) * 190;
    const y = Math.floor(i / 10) * 210;
    people.push({ label: 'person', confidence: 0.6, box: [x, y, x + 30, y + 70] });
  }
  const merged = merge([], people);
  check('fifty separate people stay fifty', merged.length === 50, `${merged.length}`);

  // And the merge does not mutate what it was given, which would corrupt the frame's own
  // findings for whatever looks at them next.
  const original = [{ label: 'person', confidence: 0.5, box: [0, 0, 10, 10] }];
  merge(original, [{ label: 'person', confidence: 0.9, box: [0, 0, 10, 10] }]);
  check('the input is left alone', original[0].confidence === 0.5);
}

console.log('\nOverlap');
{
  check('identical boxes fully overlap', overlap([0, 0, 10, 10], [0, 0, 10, 10]) === 1);
  check('disjoint boxes do not', overlap([0, 0, 10, 10], [20, 20, 30, 30]) === 0);
  check('a box inside another is a high overlap',
    overlap([0, 0, 10, 10], [2, 2, 8, 8]) > 0.35);
}

console.log(`\n${passes} passed, ${failures} failed\n`);
process.exit(failures === 0 ? 0 : 1);
