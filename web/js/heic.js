/**
 * HEIC and HEIF decoding, for the photographs an iPhone actually takes.
 *
 * WHY IT IS NEEDED
 *     An iPhone shoots HEIC by default and only Safari decodes it. Everywhere else
 *     `createImageBitmap` refuses, so a perfectly good photograph is unusable in the app
 *     that exists to look at photographs. Telling the operator to change a camera setting
 *     is a real answer, and it does nothing for the thousand pictures already taken.
 *
 * THE LICENCE, WHICH IS NOT A DETAIL
 *     libheif is **LGPL-3.0**, and that is a copyleft licence. It is a far milder one than
 *     the AGPL that rules out the obvious YOLO checkpoints - LGPL is satisfied by letting
 *     someone replace the library, and loading it as its own separate file from a CDN,
 *     which is exactly what happens below, is the ordinary way to do that. But it is a
 *     condition rather than a courtesy, so it is written here rather than buried in a
 *     lockfile: everything else in this application is Apache-2.0 or MIT, and this one is
 *     not.
 *
 * WHEN IT LOADS
 *     Only when a HEIC is actually dropped. It is a couple of megabytes of WebAssembly, and
 *     most inspections are JPEG.
 */

const CDN = 'https://cdn.jsdelivr.net/npm/libheif-js@1.23.2/libheif-wasm/libheif-bundle.mjs';

let base = CDN;
let libraryPromise = null;

export function configureHeic({ heicUrl } = {}) {
  if (typeof heicUrl === 'string' && heicUrl.length) base = heicUrl;
}

export function isHeic(file) {
  const name = (file?.name ?? '').toLowerCase();
  const type = (file?.type ?? '').toLowerCase();
  return type === 'image/heic' || type === 'image/heif'
    || name.endsWith('.heic') || name.endsWith('.heif');
}

async function library() {
  if (libraryPromise) return libraryPromise;
  libraryPromise = import(/* @vite-ignore */ base)
    .then((module) => (module.default ?? module)())
    .catch((error) => {
      libraryPromise = null;
      throw new Error(
        `The HEIC decoder could not be loaded (${error.message}). It is fetched on demand `
        + 'and needs one connection. Exporting the photograph as JPEG avoids it entirely.',
      );
    });
  return libraryPromise;
}

/**
 * Decode a HEIC file to something the rest of the app can draw.
 *
 * @returns {Promise<ImageBitmap>}
 */
export async function decodeHeic(file) {
  const heif = await library();
  const decoder = new heif.HeifDecoder();
  const images = decoder.decode(new Uint8Array(await file.arrayBuffer()));

  if (!images?.length) {
    throw new Error(
      `${file.name} is named like a HEIC but holds no image the decoder recognises.`,
    );
  }

  // A HEIC can hold a burst or a Live Photo. The first image is the one the phone shows,
  // and analysing the other frames of a burst as separate inspections would be surprising.
  const image = images[0];
  const width = image.get_width();
  const height = image.get_height();

  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  const imageData = ctx.createImageData(width, height);

  await new Promise((resolve, reject) => {
    image.display(imageData, (result) => {
      if (result) resolve(result);
      else reject(new Error(`${file.name} could not be decoded from HEIC.`));
    });
  });

  ctx.putImageData(imageData, 0, 0);
  return createImageBitmap(canvas);
}
