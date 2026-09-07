/**
 * What counts as an image, what counts as a video, and what to say when one will not open.
 *
 * WHY THIS EXISTS
 *     Files arriving from a phone routinely have no MIME type at all. A browser sets
 *     `File.type` from the operating system, and an operating system that has never heard
 *     of HEIC reports an empty string - so `file.type.startsWith('image/')` rejects a
 *     photograph for not being a photograph. That was a real bug: iPhone stills and clips
 *     were dropped before anything looked at them, with a message saying they were neither
 *     an image nor a video.
 *
 *     So classification falls back to the extension, and the table below is the only place
 *     that knows the list.
 *
 * THE TABLE DOES NOT DECIDE WHAT WORKS
 *     It is used to accept files and to explain failures, never to refuse one in advance.
 *     Whether a browser can decode a given file is the browser's business and it changes
 *     with every release; the honest way to find out is to try, and to have something
 *     useful to say when it does not work. A hardcoded "unsupported" list would still be
 *     refusing formats years after they started working.
 */

const IMAGE_EXTENSIONS = [
  'jpg', 'jpeg', 'jpe', 'png', 'webp', 'gif', 'bmp', 'avif',
  'heic', 'heif', 'tif', 'tiff', 'dng', 'cr2', 'cr3', 'nef', 'arw', 'raf', 'orf', 'rw2',
];

const VIDEO_EXTENSIONS = [
  'mp4', 'm4v', 'mov', 'qt', 'webm', 'mkv', 'avi', 'mts', 'm2ts', 'ts', '3gp', '3g2',
  'mpg', 'mpeg', 'wmv', 'flv', 'insv', 'lrv',
];

/** Everything the file picker should offer, so a phone's own formats are not hidden. */
export const ACCEPT_ATTRIBUTE = [
  'image/*',
  'video/*',
  ...IMAGE_EXTENSIONS.map((e) => `.${e}`),
  ...VIDEO_EXTENSIONS.map((e) => `.${e}`),
].join(',');

export function extensionOf(name) {
  const match = /\.([a-z0-9]+)$/i.exec(name ?? '');
  return match ? match[1].toLowerCase() : '';
}

/**
 * "image", "video" or "other".
 *
 * The MIME type wins when there is one, because it is what the file actually claims to be.
 * The extension is the fallback for the very common case of there being none.
 */
export function classify(file) {
  const type = file.type ?? '';
  if (type.startsWith('image/')) return 'image';
  if (type.startsWith('video/')) return 'video';

  const extension = extensionOf(file.name);
  if (IMAGE_EXTENSIONS.includes(extension)) return 'image';
  if (VIDEO_EXTENSIONS.includes(extension)) return 'video';
  return 'other';
}

// What to say after a decode has actually failed. Keyed by extension, because that is what
// the person is looking at in their file manager.
const ADVICE = {
  heic: 'HEIC is the format an iPhone uses by default, and only Safari can decode it. '
    + 'Either open this page in Safari, or set the phone to shoot JPEG - Settings, Camera, '
    + 'Formats, Most Compatible - or export the photograph as JPEG before uploading it.',
  heif: 'HEIF is decoded only by Safari. Open this page in Safari, or export as JPEG first.',
  mov: 'A .mov from a recent iPhone usually holds HEVC video, which most browsers will not '
    + 'decode. Safari can. Otherwise set the phone to Most Compatible - Settings, Camera, '
    + 'Formats - which records H.264, or convert the clip to MP4.',
  mkv: 'Browsers do not open Matroska. Convert it to MP4 first; the video inside usually '
    + 'needs no re-encoding, so it is quick and lossless.',
  avi: 'Browsers do not open AVI. Convert it to MP4 first.',
  wmv: 'Browsers do not open WMV. Convert it to MP4 first.',
  flv: 'Browsers do not open FLV. Convert it to MP4 first.',
  mts: 'AVCHD from a camcorder is not something a browser opens. Convert it to MP4 first.',
  m2ts: 'AVCHD from a camcorder is not something a browser opens. Convert it to MP4 first.',
  tif: 'TIFF is not a web format and no browser decodes it. Export as JPEG or PNG.',
  tiff: 'TIFF is not a web format and no browser decodes it. Export as JPEG or PNG.',
  dng: 'This is a raw file, not a finished photograph. Export a JPEG from it first.',
  cr2: 'This is a raw file, not a finished photograph. Export a JPEG from it first.',
  cr3: 'This is a raw file, not a finished photograph. Export a JPEG from it first.',
  nef: 'This is a raw file, not a finished photograph. Export a JPEG from it first.',
  arw: 'This is a raw file, not a finished photograph. Export a JPEG from it first.',
  raf: 'This is a raw file, not a finished photograph. Export a JPEG from it first.',
  orf: 'This is a raw file, not a finished photograph. Export a JPEG from it first.',
  rw2: 'This is a raw file, not a finished photograph. Export a JPEG from it first.',
};

/**
 * Why a file would not open, in terms of what to do about it.
 *
 * Called only after a decode has genuinely failed. "Unsupported format" tells someone
 * holding a photograph that plainly exists precisely nothing; naming their format and the
 * setting on their phone tells them everything.
 */
export function decodeAdvice(file) {
  const extension = extensionOf(file.name);
  const known = ADVICE[extension];
  if (known) {
    return `${file.name} could not be opened. ${known}`;
  }
  return `${file.name} could not be opened. `
    + (extension ? `This browser does not decode .${extension} files. ` : '')
    + 'Convert it to JPEG, PNG or MP4 and try again.';
}
