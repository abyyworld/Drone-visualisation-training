/**
 * Cutting a frame into pieces, and putting the findings back together.
 *
 * Pure arithmetic, in its own file so it can be tested without a browser. The worker does
 * the drawing and the detecting; the geometry and the merging are here, because they are
 * where the mistakes live. A tile mapped back one pixel wrong puts a box next to a person
 * instead of on them, and a merge that is too eager draws one box on two people while a
 * merge that is too shy counts one person twice.
 */

// Two tiles, not six. Recall against tile count saturates at two once the model's input is
// 640 - three, four and six all measure 70% on the same frames - and two is what lets every
// person be looked at every cycle instead of every sixth one. Tiles.java carries the
// numbers; keep the two in step.
export const TILE_COLUMNS = 2;
export const TILE_ROWS = 1;
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
 * How much of a frame-pass box a tile box has to sit inside before the frame box is
 * treated as a blur over it rather than a rival for it.
 */
const CONTAINED = 0.7;

/**
 * Fold a tile's findings into the frame's.
 *
 * THE CASE THIS GETS WRONG IF IT IS DONE NAIVELY
 *     Two people standing close together come back from the full-frame pass as one wide
 *     box, because at that scale they are one blob of pixels. The tile, looking at the same
 *     ground three times larger, sees two. A merge that just asks "do these overlap?" then
 *     throws both tile boxes away in favour of the blob, and the app draws one box around
 *     two people and counts them as one. That was reported from a real flight and it is the
 *     exact opposite of what tiling is for.
 *
 *     So overlap alone does not decide it. A tile box that sits mostly *inside* a frame box
 *     is not a rival reading of the same thing, it is a closer reading of part of it, and
 *     the frame box is a blur over whatever the tile found. When that happens the frame box
 *     is dropped and every tile box in it is kept.
 *
 *     The two are treated as the same thing only when they are the same size and place -
 *     genuine double vision, one person seen by both passes - and then the more confident
 *     wins, which is usually the tile's.
 */
export function merge(base, extra, threshold = MERGE_IOU) {
  const merged = base.map((finding) => ({ ...finding }));
  const supersededByTile = new Set();

  for (const candidate of extra) {
    let duplicate = false;

    for (const existing of merged) {
      if (existing.label !== candidate.label) continue;
      if (supersededByTile.has(existing)) continue;

      // The same thing seen twice: same place, same size.
      if (overlap(existing.box, candidate.box) > threshold) {
        if (candidate.confidence > existing.confidence) {
          existing.box = candidate.box;
          existing.confidence = candidate.confidence;
        }
        duplicate = true;
        break;
      }

      // A closer look at part of it. The wide box was a blur over more than one thing, and
      // whatever the tile resolves inside it is the better answer.
      if (containment(candidate.box, existing.box) >= CONTAINED
        && area(candidate.box) < area(existing.box) * 0.75) {
        supersededByTile.add(existing);
      }
    }
    if (!duplicate) merged.push({ ...candidate });
  }

  return merged.filter((finding) => !supersededByTile.has(finding));
}

/** How much of `inner` lies inside `outer`, as a fraction of `inner`. */
export function containment(inner, outer) {
  const width = Math.min(inner[2], outer[2]) - Math.max(inner[0], outer[0]);
  const height = Math.min(inner[3], outer[3]) - Math.max(inner[1], outer[1]);
  if (width <= 0 || height <= 0) return 0;
  const size = area(inner);
  return size > 0 ? (width * height) / size : 0;
}

function area(box) {
  return (box[2] - box[0]) * (box[3] - box[1]);
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
