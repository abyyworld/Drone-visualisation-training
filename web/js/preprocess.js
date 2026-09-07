/**
 * Image -> model input tensor.
 *
 * Two different resize policies, because the two model types were trained differently:
 *
 *   letterbox() — for the detectors. Preserves aspect ratio and pads, exactly as Ultralytics
 *                 does at inference. Stretching instead would distort every box the model
 *                 learned, which is the kind of silent mismatch that makes a model look
 *                 broken when it is merely being fed wrong.
 *
 *   centerCrop() — for the gate classifier, matching standard ImageNet-style eval.
 */

// ImageNet statistics, used by the torchvision-pretrained gate backbone. The YOLO detectors
// take plain 0..1 input and must NOT be normalised this way.
const IMAGENET_MEAN = [0.485, 0.456, 0.406];
const IMAGENET_STD = [0.229, 0.224, 0.225];

function drawToCanvas(source, width, height) {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.imageSmoothingQuality = 'high';
  return { canvas, ctx };
}

/**
 * Resize preserving aspect ratio, padding the remainder.
 *
 * @returns {{data: Float32Array, width:number, height:number, scale:number, padX:number, padY:number}}
 *   `scale`/`padX`/`padY` are what {@link module:detect} needs to map boxes back to the
 *   original image coordinates.
 */
export function letterbox(source, size) {
  const sourceWidth = source.naturalWidth ?? source.width;
  const sourceHeight = source.naturalHeight ?? source.height;

  const scale = Math.min(size / sourceWidth, size / sourceHeight);
  const drawWidth = Math.round(sourceWidth * scale);
  const drawHeight = Math.round(sourceHeight * scale);
  const padX = Math.floor((size - drawWidth) / 2);
  const padY = Math.floor((size - drawHeight) / 2);

  const { ctx } = drawToCanvas(source, size, size);
  ctx.fillStyle = '#727272'; // Ultralytics' letterbox grey (114,114,114)
  ctx.fillRect(0, 0, size, size);
  ctx.drawImage(source, padX, padY, drawWidth, drawHeight);

  const { data: rgba } = ctx.getImageData(0, 0, size, size);
  const data = toCHW(rgba, size, size);

  return { data, width: size, height: size, scale, padX, padY };
}

/** Resize the short side then centre-crop — the standard classification eval transform. */
export function centerCrop(source, size) {
  const sourceWidth = source.naturalWidth ?? source.width;
  const sourceHeight = source.naturalHeight ?? source.height;

  const scale = size / Math.min(sourceWidth, sourceHeight);
  const drawWidth = Math.round(sourceWidth * scale);
  const drawHeight = Math.round(sourceHeight * scale);

  const { ctx } = drawToCanvas(source, size, size);
  ctx.drawImage(
    source,
    Math.round((size - drawWidth) / 2),
    Math.round((size - drawHeight) / 2),
    drawWidth,
    drawHeight,
  );

  const { data: rgba } = ctx.getImageData(0, 0, size, size);
  return { data: toCHW(rgba, size, size, IMAGENET_MEAN, IMAGENET_STD), width: size, height: size };
}

/**
 * Interleaved RGBA bytes -> planar CHW float32, optionally normalised.
 * ONNX expects channel-planar input; canvas gives channel-interleaved.
 */
function toCHW(rgba, width, height, mean, std) {
  const pixels = width * height;
  const out = new Float32Array(pixels * 3);

  for (let i = 0; i < pixels; i += 1) {
    const r = rgba[i * 4] / 255;
    const g = rgba[i * 4 + 1] / 255;
    const b = rgba[i * 4 + 2] / 255;

    if (mean) {
      out[i] = (r - mean[0]) / std[0];
      out[pixels + i] = (g - mean[1]) / std[1];
      out[pixels * 2 + i] = (b - mean[2]) / std[2];
    } else {
      out[i] = r;
      out[pixels + i] = g;
      out[pixels * 2 + i] = b;
    }
  }
  return out;
}

/** Decode a File into an ImageBitmap-like drawable, with a clear error on unsupported input. */
export async function loadImage(file) {
  if (!file.type.startsWith('image/')) {
    throw new Error(`${file.name} is not an image file.`);
  }
  try {
    return await createImageBitmap(file);
  } catch {
    throw new Error(`${file.name} could not be decoded — it may be corrupt or an unsupported format.`);
  }
}
