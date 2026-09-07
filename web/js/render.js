/**
 * Annotated-image rendering.
 *
 * Boxes are drawn at the image's native resolution and the canvas is scaled down by CSS,
 * so the annotated PNG a user downloads is full quality rather than whatever size the card
 * happened to be on screen. Line widths and font sizes scale with the image so a 4000px
 * drone frame does not end up with hairline boxes and unreadable labels.
 */

// Distinguishable at a glance and in both themes. Index by class id so a given class keeps
// its colour across every image in a batch.
const PALETTE = [
  '#E5484D', // red
  '#F76B15', // orange
  '#FFB224', // amber
  '#30A46C', // green
  '#0091FF', // blue
  '#8E4EC6', // purple
  '#E93D82', // pink
  '#00A2C7', // cyan
];

export function colorFor(classId) {
  return PALETTE[classId % PALETTE.length];
}

/**
 * Draw an image plus its detections onto a canvas.
 *
 * @param {HTMLCanvasElement} canvas
 * @param {ImageBitmap} image
 * @param {Array<{label:string, classId:number, confidence:number, box:number[]}>} detections
 */
export function drawDetections(canvas, image, detections) {
  canvas.width = image.width;
  canvas.height = image.height;

  const ctx = canvas.getContext('2d');
  ctx.drawImage(image, 0, 0);

  // Annotations are sized as a *fraction of the image*, not in absolute pixels, because the
  // canvas is displayed scaled-to-fit inside a card. A fixed 18px label is fine on a 640px
  // thumbnail and invisible on a 4000px drone frame shrunk to the same width; a proportional
  // one stays the same apparent size at any source resolution. The floors keep very small
  // images legible, where the proportion would otherwise round to nothing.
  const longEdge = Math.max(image.width, image.height);
  const lineWidth = Math.max(2, Math.round(longEdge / 250));
  const fontSize = Math.max(13, Math.round(longEdge / 30));
  const padding = Math.max(3, Math.round(fontSize / 3));

  ctx.font = `600 ${fontSize}px system-ui, -apple-system, "Segoe UI", sans-serif`;
  ctx.textBaseline = 'top';
  ctx.lineJoin = 'round';

  for (const detection of detections) {
    const [x1, y1, x2, y2] = detection.box;
    const color = colorFor(detection.classId);

    ctx.strokeStyle = color;
    ctx.lineWidth = lineWidth;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

    const text = `${detection.label} ${(detection.confidence * 100).toFixed(0)}%`;
    const labelWidth = ctx.measureText(text).width + padding * 2;
    const labelHeight = fontSize + padding * 2;

    // Keep the label inside the frame on all four edges - a detection against the right or
    // top border would otherwise have its label clipped away, which is exactly when the
    // reader most needs to know what it is.
    const labelX = Math.max(0, Math.min(x1 - lineWidth / 2, image.width - labelWidth));
    const labelY = y1 - labelHeight < 0 ? y1 : y1 - labelHeight;

    ctx.fillStyle = color;
    ctx.fillRect(labelX, labelY, labelWidth, labelHeight);

    ctx.fillStyle = '#FFFFFF';
    ctx.fillText(text, labelX + padding, labelY + padding);
  }
}

/** Canvas -> PNG blob, for download. */
export function toBlob(canvas) {
  return new Promise((resolve) => canvas.toBlob(resolve, 'image/png'));
}
