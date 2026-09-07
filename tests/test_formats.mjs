/**
 * Tests for file classification and the advice given when something will not open.
 *
 *   node tests/test_formats.mjs
 *
 * This exists because of a real bug: a photograph copied off an iPhone arrives with an
 * empty File.type, `file.type.startsWith('image/')` said it was not an image, and it was
 * dropped before anything tried to open it. The message said it was neither an image nor a
 * video, about a photograph.
 *
 * So the cases below are mostly "no MIME type, judge it by the name", and the rule that the
 * table never refuses a file in advance - it only explains a failure that already happened.
 */

import { classify, extensionOf, decodeAdvice, ACCEPT_ATTRIBUTE } from '../web/js/formats.js';

let failures = 0;
let passes = 0;

function check(name, condition, detail = '') {
  if (condition) {
    passes += 1;
    console.log(`  ok    ${name}`);
  } else {
    failures += 1;
    console.error(`  FAIL  ${name}${detail ? ` -- ${detail}` : ''}`);
  }
}

const file = (name, type = '') => ({ name, type });

console.log('\nMIME type present');
check('a declared image is an image', classify(file('a.jpg', 'image/jpeg')) === 'image');
check('a declared video is a video', classify(file('a.mp4', 'video/mp4')) === 'video');
check('a declared HEIC is an image', classify(file('IMG_0001.HEIC', 'image/heic')) === 'image');
check('a text file is neither', classify(file('notes.txt', 'text/plain')) === 'other');

console.log('\nNo MIME type - the case that was broken');
for (const name of ['IMG_0001.HEIC', 'IMG_0001.heic', 'photo.HEIF', 'shot.avif', 'scan.tiff', 'raw.DNG']) {
  check(`${name} is an image`, classify(file(name)) === 'image');
}
for (const name of ['IMG_0002.MOV', 'clip.mov', 'flight.MP4', 'a.m4v', 'b.mkv', 'c.avi',
                    'camcorder.MTS', 'insta.insv', 'old.3gp']) {
  check(`${name} is a video`, classify(file(name)) === 'video');
}
check('a file with no extension is neither', classify(file('README')) === 'other');
check('an archive is neither', classify(file('photos.zip')) === 'other');

console.log('\nThe MIME type wins when there is one');
// A picker that mislabels a .mov as video/quicktime is still telling us it is video.
check('type beats extension', classify(file('weird.bin', 'image/png')) === 'image');

console.log('\nExtensions');
check('case is ignored', extensionOf('IMG.HEIC') === 'heic');
check('a dotted name takes the last part', extensionOf('my.holiday.photo.jpg') === 'jpg');
check('no extension is empty', extensionOf('README') === '');
check('a leading dot is not an extension', extensionOf('.gitignore') === 'gitignore');

console.log('\nAdvice after a failure');
const heic = decodeAdvice(file('IMG_0001.HEIC'));
check('HEIC advice names the format', heic.includes('HEIC'));
check('HEIC advice names Safari', heic.includes('Safari'));
check('HEIC advice names the iPhone setting that fixes it', heic.includes('Most Compatible'));
check('HEIC advice names the file', heic.includes('IMG_0001.HEIC'));

const mov = decodeAdvice(file('IMG_0002.MOV'));
check('MOV advice names the codec that causes it', mov.includes('HEVC'));
check('MOV advice gives a way out', mov.includes('Most Compatible') || mov.includes('convert'));

const raw = decodeAdvice(file('DSC_0001.NEF'));
check('a raw file is explained as a raw file', raw.includes('raw file'));

const unknown = decodeAdvice(file('mystery.qqq'));
check('an unknown format still names itself', unknown.includes('.qqq'));
check('and still says what to do', unknown.includes('Convert it'));

const noExtension = decodeAdvice(file('mystery'));
check('a nameless format does not print an empty extension', !noExtension.includes('.  '));
check('and still says what to do', noExtension.includes('Convert it'));

console.log('\nThe picker offers what the classifier accepts');
// A picker that greys out a HEIC is the same bug one step earlier, so every extension the
// classifier will accept has to be in the accept attribute.
for (const extension of ['heic', 'heif', 'mov', 'mp4', 'dng', 'mts', 'avif']) {
  check(`accept offers .${extension}`, ACCEPT_ATTRIBUTE.includes(`.${extension}`));
}
check('accept keeps the wildcards for everything else',
  ACCEPT_ATTRIBUTE.includes('image/*') && ACCEPT_ATTRIBUTE.includes('video/*'));

console.log(`\n${passes} passed, ${failures} failed`);
process.exit(failures ? 1 : 0);
