package world.abyy.droneinspection;

import android.graphics.Bitmap;

/**
 * Recognising someone who has already been counted.
 *
 * A port of web/js/reid.js, and the two must stay in step: the tablet and the report of the
 * same flight should not disagree about how many people went past.
 *
 * The problem, in short. A tracker follows a thing while it can see it. The moment it
 * cannot - someone walks behind a van, the drone banks away and back, the model misses them
 * for two seconds - the track closes, and when they reappear they are a new number and the
 * running total goes up for a person already in it. Fly a route over a crowd and the count
 * stops being a count of people and becomes a count of reappearances.
 *
 * What is compared is what someone is wearing, as a coarse colour histogram over three
 * horizontal bands plus the whole box. The whole-box band is the drone's: the three spatial
 * bands assume a roughly side-on view, and that assumption weakens with every metre of
 * altitude, while which colours a person is wearing and in what proportion survives being
 * flown around.
 *
 * What it gets wrong: two people in the same dark jacket in the same light are one person
 * to this. That direction of error is the deliberate one. The count was already a floor,
 * because the detector misses anyone small, distant or overlapping; merging two people
 * keeps it a floor, and splitting one person into two would not.
 */
final class Reid {

    private static final int SPATIAL_BANDS = 3;
    private static final int WHOLE_BAND = 3;
    private static final int BANDS = 4;
    private static final int LEVELS = 4;
    private static final int BIN_COUNT = LEVELS * LEVELS * LEVELS;
    static final int SIZE = BANDS * BIN_COUNT;

    /** Whole box first, because it is what survives a change of viewing angle. */
    private static final float[] BAND_WEIGHTS = {0.12f, 0.28f, 0.15f, 0.45f};

    /**
     * Below this a box is too few pixels to describe. A person twelve pixels tall is four
     * pixels a band, and a histogram of four pixels matches almost anything. Refusing is
     * the right answer: they are then tracked and counted but never matched back, which
     * counts them again if they leave and return. A worse count, and an honest one.
     */
    private static final int MIN_BOX_WIDTH = 6;
    private static final int MIN_BOX_HEIGHT = 16;

    private Reid() {
    }

    /**
     * Build a colour signature for one box.
     *
     * @param pixels ARGB of the frame, row major
     * @param box    [x0, y0, x1, y1] in that frame's pixels
     * @return the signature, or null when the box is too small to say anything about
     */
    static float[] describe(int[] pixels, int width, int height, float[] box) {
        int x0 = Math.max(0, Math.round(box[0]));
        int y0 = Math.max(0, Math.round(box[1]));
        int x1 = Math.min(width, Math.round(box[2]));
        int y1 = Math.min(height, Math.round(box[3]));
        int boxWidth = x1 - x0;
        int boxHeight = y1 - y0;
        if (boxWidth < MIN_BOX_WIDTH || boxHeight < MIN_BOX_HEIGHT) {
            return null;
        }

        float[] signature = new float[SIZE];
        float[] counts = new float[BANDS];
        float bandHeight = boxHeight / (float) SPATIAL_BANDS;

        for (int y = y0; y < y1; y++) {
            int band = Math.min(SPATIAL_BANDS - 1, (int) ((y - y0) / bandHeight));
            for (int x = x0; x < x1; x++) {
                int argb = pixels[y * width + x];
                // Coarse on purpose: a finer histogram is more precise about the lighting
                // and less about the person, and the lighting moves with the drone.
                int r = (((argb >> 16) & 0xFF) * LEVELS) >> 8;
                int g = (((argb >> 8) & 0xFF) * LEVELS) >> 8;
                int b = ((argb & 0xFF) * LEVELS) >> 8;
                int bin = (r * LEVELS + g) * LEVELS + b;
                signature[band * BIN_COUNT + bin] += 1;
                counts[band] += 1;
                signature[WHOLE_BAND * BIN_COUNT + bin] += 1;
                counts[WHOLE_BAND] += 1;
            }
        }

        for (int band = 0; band < BANDS; band++) {
            if (counts[band] <= 0) {
                continue;
            }
            for (int bin = 0; bin < BIN_COUNT; bin++) {
                signature[band * BIN_COUNT + bin] /= counts[band];
            }
        }
        return signature;
    }

    /** Convenience for a bitmap the caller already has in hand. */
    static float[] describe(Bitmap frame, int[] scratch, float[] box) {
        int width = frame.getWidth();
        int height = frame.getHeight();
        if (scratch.length < width * height) {
            return null;
        }
        frame.getPixels(scratch, 0, width, 0, 0, width, height);
        return describe(scratch, width, height, box);
    }

    /**
     * How alike two signatures are, from 0 to 1.
     *
     * Histogram intersection: the fraction of the two distributions that overlaps. A band
     * partly occluded in one view loses its share and no more, rather than poisoning the
     * whole comparison the way a squared distance would.
     */
    static float similarity(float[] a, float[] b) {
        if (a == null || b == null || a.length != b.length) {
            return 0f;
        }
        float total = 0f;
        for (int band = 0; band < BANDS; band++) {
            float shared = 0f;
            int start = band * BIN_COUNT;
            for (int bin = 0; bin < BIN_COUNT; bin++) {
                float left = a[start + bin];
                float right = b[start + bin];
                shared += left < right ? left : right;
            }
            total += shared * BAND_WEIGHTS[band];
        }
        return total;
    }

    /**
     * Fold a fresh signature into the one held, weighted towards what is held.
     *
     * One frame of someone half behind a railing is a bad description of them; replacing
     * outright would let that frame become their identity, while averaging lets it
     * contribute and be outvoted.
     */
    static float[] blend(float[] held, float[] fresh) {
        if (held == null) {
            return fresh;
        }
        if (fresh == null) {
            return held;
        }
        float[] merged = new float[held.length];
        for (int i = 0; i < held.length; i++) {
            merged[i] = held[i] * 0.8f + fresh[i] * 0.2f;
        }
        return merged;
    }
}
