package world.abyy.droneinspection;

import java.util.ArrayList;
import java.util.List;

/**
 * Cutting a frame into pieces, and putting the findings back together.
 *
 * A port of web/js/tiles.js, kept in step with it.
 *
 * WHY A CROWD NEEDS THIS
 *     The detector's input is a fixed square. A 1920-wide frame squeezed into 640 of them
 *     turns a person forty pixels tall into thirteen, and a crowd shot comes back with five
 *     people rather than fifty: they are not being missed by a threshold, they are not
 *     being shown to the model at all.
 *
 *     Half that frame is 960 across, which the model squeezes by 1.5 rather than 3, and the
 *     same person arrives at twenty-four pixels. Same model, same weights, nearly twice the
 *     size on the thing being looked for.
 *
 * WHY TWO AND NOT SIX
 *     Six 320-pixel tiles was the old shape and it gave a person about the same pixels as
 *     this does. What it could not do is look at all of them at once: six tiles at the
 *     cycle rate meant a patch of ground got a look every sixth cycle, and everybody
 *     outside the current tile was coasting on a guess. Two tiles at 640 cost the same
 *     milliseconds and cover the whole frame every cycle.
 *
 *     Measured on four crowded VisDrone frames at the real cadence: 45% of the people
 *     reached and 1.92 numbers each, against 60% reached and 1.69 numbers each. Recall
 *     against tile count saturates at two - three, four and six tiles all measure 70% on a
 *     still frame - so the extra tiles were buying latency and nothing else.
 */
final class Tiles {

    static final int COLUMNS = 2;
    static final int ROWS = 1;
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
