/**
 * Cutting a frame into pieces, and putting the findings back together.
 *
 * Pure arithmetic, in its own file so it can be tested without a browser. The worker does
 * the drawing and the detecting; the geometry and the merging are here, because they are
 * where the mistakes live. A tile mapped back one pixel wrong puts a box next to a person
 * instead of on them, and a merge that is too eager draws one box on two people while a
 * merge that is too shy counts one person twice.
 */

export const TILE_COLUMNS = 3;
export const TILE_ROWS = 2;
export const TILE_COUNT = TILE_COLUMNS * TILE_ROWS;

/** Tiles overlap, so a person standing on a seam is whole in at least one of them. */
export const TILE_OVERLAP = 0.18;

/** Above this overlap, two boxes of the same class are the same thing seen twice. */
export const MERGE_IOU = 0.55;

/**
 * Where tile `index` sits in a frame of this size.
 *
 * @returns {{x:number, y:number, width:number, height:number}}
 */
export function tileRegion(index, frameWidth, frameHeight) {
  const wrapped = ((index % TILE_COUNT) + TILE_COUNT) % TILE_COUNT;
  const column = wrapped % TILE_COLUMNS;
  const row = Math.floor(wrapped / TILE_COLUMNS);

  const tileWidth = frameWidth / TILE_COLUMNS;
  const tileHeight = frameHeight / TILE_ROWS;
  const padX = tileWidth * TILE_OVERLAP;
  const padY = tileHeight * TILE_OVERLAP;

  const x = Math.max(0, column * tileWidth - padX);
  const y = Math.max(0, row * tileHeight - padY);
  return {
    x,
    y,
    width: Math.min(frameWidth - x, tileWidth + padX * 2),
    height: Math.min(frameHeight - y, tileHeight + padY * 2),
  };
}

/** Intersection over union, for deciding whether two boxes are one thing. */
export function overlap(a, b) {
  const width = Math.min(a[2], b[2]) - Math.max(a[0], b[0]);
  const height = Math.min(a[3], b[3]) - Math.max(a[1], b[1]);
  if (width <= 0 || height <= 0) return 0;
  const intersection = width * height;
  const union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection;
  return union > 0 ? intersection / union : 0;
}

/**
 * Fold a tile's findings into the frame's, dropping anything already there.
 *
 * A tile overlaps its neighbours and covers ground the full-frame pass also looked at, so
 * the same person arrives twice. Keeping both would draw two boxes on one person and, far
 * worse, count them twice - which is the whole thing this system is trying not to do.
 *
 * Where they are the same, the more confident box wins, and that is usually the tile's: it
 * saw three times as many pixels of that person as the full frame did.
 */
export function merge(base, extra, threshold = MERGE_IOU) {
  const merged = base.map((finding) => ({ ...finding }));
  for (const candidate of extra) {
    let duplicate = false;
    for (const existing of merged) {
      if (existing.label !== candidate.label) continue;
      if (overlap(existing.box, candidate.box) > threshold) {
        if (candidate.confidence > existing.confidence) {
          existing.box = candidate.box;
          existing.confidence = candidate.confidence;
        }
        duplicate = true;
        break;
      }
    }
    if (!duplicate) merged.push({ ...candidate });
  }
  return merged;
}

/**
 * Map a box found inside a tile back to where it is in the whole frame.
 *
 * @param {number[]} box  in the tile's own drawn pixels
 * @param {{x:number, y:number}} region  where the tile was cut from
 * @param {number} scaleX  tile source pixels per drawn pixel
 * @param {number} scaleY
 */
export function toFrame(box, region, scaleX, scaleY) {
  return [
    region.x + box[0] * scaleX,
    region.y + box[1] * scaleY,
    region.x + box[2] * scaleX,
    region.y + box[3] * scaleY,
  ];
}

/** Every tile covers ground, and between them they cover all of it. */
export function coversFrame(frameWidth, frameHeight) {
  const covered = [];
  for (let i = 0; i < TILE_COUNT; i += 1) covered.push(tileRegion(i, frameWidth, frameHeight));
  return covered;
}
