package world.abyy.droneinspection;

import java.util.ArrayList;
import java.util.List;

/**
 * Cutting a frame into pieces, and putting the findings back together.
 *
 * A port of web/js/tiles.js, kept in step with it.
 *
 * WHY A CROWD NEEDS THIS
 *     The detector's input is 448 pixels square. A 1920-wide frame is squeezed into that,
 *     so a person forty pixels tall in the original arrives nine pixels tall at the model.
 *     Nine pixels is below what any detector can find, which is why a crowd shot returns
 *     five people and not fifty: they are not being missed by a threshold, they are not
 *     being shown to the model at all.
 *
 *     A sixth of that frame is 640 across, which the model squeezes by 1.4 rather than 4.3,
 *     and the same person arrives at twenty-eight pixels. Same model, same weights, three
 *     times the size on the thing being looked for.
 *
 *     One tile per pass, cycling, rather than all six at once: six detections in a row is
 *     six times the latency, and the full-frame pass that runs every time is what keeps
 *     every track alive in between.
 */
final class Tiles {

    static final int COLUMNS = 3;
    static final int ROWS = 2;
    static final int COUNT = COLUMNS * ROWS;

    /** Tiles overlap, so a person standing on a seam is whole in at least one of them. */
    static final float OVERLAP = 0.18f;

    /** Above this overlap, two boxes of the same class are the same thing seen twice. */
    static final float MERGE_IOU = 0.55f;

    /** Where a tile sits in a frame: x, y, width, height. */
    static float[] region(int index, int frameWidth, int frameHeight) {
        int wrapped = ((index % COUNT) + COUNT) % COUNT;
        int column = wrapped % COLUMNS;
        int row = wrapped / COLUMNS;

        float tileWidth = frameWidth / (float) COLUMNS;
        float tileHeight = frameHeight / (float) ROWS;
        float padX = tileWidth * OVERLAP;
        float padY = tileHeight * OVERLAP;

        float x = Math.max(0f, column * tileWidth - padX);
        float y = Math.max(0f, row * tileHeight - padY);
        return new float[]{
                x, y,
                Math.min(frameWidth - x, tileWidth + padX * 2),
                Math.min(frameHeight - y, tileHeight + padY * 2),
        };
    }

    /** Intersection over union, for deciding whether two boxes are one thing. */
    static float overlap(float[] a, float[] b) {
        float width = Math.min(a[2], b[2]) - Math.max(a[0], b[0]);
        float height = Math.min(a[3], b[3]) - Math.max(a[1], b[1]);
        if (width <= 0 || height <= 0) {
            return 0f;
        }
        float intersection = width * height;
        float union = (a[2] - a[0]) * (a[3] - a[1])
                + (b[2] - b[0]) * (b[3] - b[1]) - intersection;
        return union > 0 ? intersection / union : 0f;
    }

    /**
     * Fold a tile's findings into the frame's, dropping anything already there.
     *
     * A tile overlaps its neighbours and covers ground the full-frame pass also looked at,
     * so the same person arrives twice. Keeping both would draw two boxes on one person
     * and, far worse, count them twice. Where they are the same, the more confident box
     * wins, and that is usually the tile's: it saw three times as many of their pixels.
     */
    static List<Finding> merge(List<Finding> base, List<Finding> extra) {
        List<Finding> merged = new ArrayList<>(base);
        for (Finding candidate : extra) {
            boolean duplicate = false;
            for (int i = 0; i < merged.size(); i++) {
                Finding existing = merged.get(i);
                if (!existing.label.equals(candidate.label)) {
                    continue;
                }
                if (overlap(box(existing), box(candidate)) > MERGE_IOU) {
                    if (candidate.confidence > existing.confidence) {
                        merged.set(i, candidate);
                    }
                    duplicate = true;
                    break;
                }
            }
            if (!duplicate) {
                merged.add(candidate);
            }
        }
        return merged;
    }

    private static float[] box(Finding finding) {
        return new float[]{finding.x0, finding.y0, finding.x1, finding.y1};
    }

    private Tiles() {
    }
}
