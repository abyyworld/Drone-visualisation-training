/**
 * Tests for the YOLO decode.
 *
 *   node tests/test_yolo.mjs
 *
 * This arithmetic runs twice - JavaScript in the browser, Java on the tablet - and getting
 * it wrong does not raise an error. A transposed layout finds nobody, which looks exactly
 * like an empty scene; an anchor stride out by one puts boxes beside people instead of on
 * them. Both are silent, and silence is the failure this project keeps designing against.
 * So the numbers are pinned here, and the Java is compared against these same cases.
 */

import { decodeHead, decodeRows, nms, iou, unletterbox } from '../web/js/yolo.js';

let failures = 0;
let passes = 0;

function check(name, condition, detail = '') {
  if (condition) { passes += 1; console.log(`  ok    ${name}`); }
  else { failures += 1; console.error(`  FAIL  ${name}${detail ? ` -- ${detail}` : ''}`); }
}

/**
 * Build a classic YOLO head by hand: channel-major, so every anchor's x sits together,
 * then every anchor's y, and so on. Laying it out here by hand is the point - if the
 * decode reads it any other way, these tests fail rather than the field does.
 */
function head(anchors, numClasses, boxes) {
  const channels = 4 + numClasses;
  const values = new Float32Array(channels * anchors);
  for (const { at, cx, cy, w, h, scores } of boxes) {
    values[at] = cx;
    values[anchors + at] = cy;
    values[anchors * 2 + at] = w;
    values[anchors * 3 + at] = h;
    scores.forEach((score, c) => { values[(4 + c) * anchors + at] = score; });
  }
  return { values, channels };
}

console.log('\nA classic head');
{
  const anchors = 8;
  const { values, channels } = head(anchors, 2, [
    { at: 1, cx: 100, cy: 200, w: 40, h: 90, scores: [0.9, 0.1] },
    { at: 5, cx: 300, cy: 220, w: 30, h: 80, scores: [0.2, 0.8] },
  ]);
  const found = decodeHead(values, channels, anchors, { confThreshold: 0.25 });

  check('finds both', found.length === 2, `${found.length}`);
  const person = found.find((d) => d.classId === 0);
  // Compared with a tolerance: the head is a Float32Array, so 0.9 comes back as
  // 0.8999999761581421. An exact comparison here fails on the last bit and says nothing
  // about whether the right class won.
  check('picks the winning class',
    person && Math.abs(person.confidence - 0.9) < 1e-6,
    `${person?.confidence}`);
  // Centre form in, corner form out. Getting this backwards is the classic silent bug:
  // every box ends up a quarter of its size, in the wrong place, and still looks plausible.
  check('converts centre form to corners',
    person && person.box[0] === 80 && person.box[1] === 155
      && person.box[2] === 120 && person.box[3] === 245,
    JSON.stringify(person?.box));
}

console.log('\nThresholds and class filtering');
{
  const anchors = 4;
  const { values, channels } = head(anchors, 2, [
    { at: 0, cx: 10, cy: 10, w: 4, h: 4, scores: [0.9, 0] },
    { at: 1, cx: 50, cy: 50, w: 4, h: 4, scores: [0.1, 0] },
    { at: 2, cx: 90, cy: 90, w: 4, h: 4, scores: [0, 0.95] },
  ]);

  check('drops anything under the threshold',
    decodeHead(values, channels, anchors, { confThreshold: 0.5 }).length === 2);

  // People only, which is what crowd mode wants. Filtering inside the decode rather than
  // afterwards means the suppression never considers a vehicle at all.
  const peopleOnly = decodeHead(values, channels, anchors,
    { confThreshold: 0.25, keepClasses: [0] });
  check('keeps only the classes asked for', peopleOnly.length === 1, `${peopleOnly.length}`);
  check('and it is the right one', peopleOnly[0]?.classId === 0);
}

console.log('\nSuppression');
{
  const overlapping = [
    { classId: 0, confidence: 0.9, box: [100, 100, 140, 190] },
    { classId: 0, confidence: 0.7, box: [104, 102, 144, 192] },
    { classId: 0, confidence: 0.8, box: [300, 100, 340, 190] },
  ];
  const kept = nms(overlapping, 0.45);
  check('one person seen twice becomes one', kept.length === 2, `${kept.length}`);
  check('the confident box survives', kept.some((d) => d.confidence === 0.9));
  check('the distant one is untouched', kept.some((d) => d.box[0] === 300));

  // Two people standing close together are two people, not one. This is the same failure
  // reported from a real flight, at a different layer, and suppression must not cause it.
  const shoulderToShoulder = nms([
    { classId: 0, confidence: 0.9, box: [100, 100, 140, 190] },
    { classId: 0, confidence: 0.8, box: [145, 100, 185, 190] },
  ], 0.45);
  check('people side by side both survive', shoulderToShoulder.length === 2);

  // Suppression is per class: a person standing where a car is does not delete either.
  const mixed = nms([
    { classId: 0, confidence: 0.9, box: [100, 100, 140, 190] },
    { classId: 2, confidence: 0.8, box: [100, 100, 140, 190] },
  ], 0.45);
  check('different classes never suppress each other', mixed.length === 2);
}

console.log('\nEnd-to-end exports, already suppressed');
{
  const rows = new Float32Array([
    80, 155, 120, 245, 0.91, 0,
    300, 100, 340, 190, 0.10, 0,
    500, 100, 540, 190, 0.77, 2,
  ]);
  const found = decodeRows(rows, 3, { confThreshold: 0.25 });
  check('reads corner form straight through',
    found.length === 2 && found[0].box[0] === 80 && found[0].box[2] === 120);
  check('and the class index', found[1].classId === 2);
  check('filtering applies here too',
    decodeRows(rows, 3, { confThreshold: 0.25, keepClasses: [0] }).length === 1);
}

console.log('\nOverlap and letterbox');
{
  check('identical boxes overlap fully', iou([0, 0, 10, 10], [0, 0, 10, 10]) === 1);
  check('disjoint boxes do not', iou([0, 0, 10, 10], [20, 20, 30, 30]) === 0);

  // A 640-wide picture letterboxed into 320: scaled by half, with 40 pixels of padding
  // above and below. A box at the top of the content must come back at the top of the
  // picture, not 80 pixels into it.
  const box = unletterbox([100, 40, 200, 140], 0.5, 0, 40, 640, 480);
  check('undoes the scale and the padding',
    box[0] === 200 && box[1] === 0 && box[2] === 400 && box[3] === 200,
    JSON.stringify(box));
  check('and clamps to the picture',
    unletterbox([-10, -10, 9999, 9999], 0.5, 0, 40, 640, 480)
      .every((v, i) => (i < 2 ? v === 0 : v === [640, 480][i - 2])));
}

console.log(`\n${passes} passed, ${failures} failed\n`);
process.exit(failures === 0 ? 0 : 1);
