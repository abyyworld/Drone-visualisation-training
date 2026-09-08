package world.abyy.droneinspection;

import android.graphics.Bitmap;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.List;

/**
 * Flame and smoke on the drone's picture, computed rather than detected.
 *
 * A line-for-line port of web/js/firescan.js. The two must stay in step: the same frame
 * should produce the same regions in the browser and on the tablet, or the app is telling
 * an operator one thing and a report another. The constants below are therefore copied
 * exactly, and the test suite that pins the behaviour lives at tests/test_firescan.mjs.
 *
 * The reasoning, in short - the long version is in the JavaScript. The detector this ships
 * with is trained on COCO, which has no fire class. Every pretrained model for flame that
 * is reachable and accurate is AGPL, which this project cannot take. So flame and smoke are
 * computed from the published colour rules plus the part that actually decides it: fire
 * burns in place and churns inside its own outline, so the flame fraction of a cell changes
 * on nearly every frame. A red vehicle driving past changes it further, but only twice, and
 * holds perfectly still in between. The measure is therefore how often a cell changes, not
 * how far. What comes out is a region worth looking at, never a verdict, and never a
 * statement about a frame that has nothing marked on it.
 */
final class FireScan {

    static final String FLAME = "flame";
    static final String SMOKE = "smoke";

    private static final int WORK_WIDTH = 160;
    private static final int CELL = 8;
    private static final int HISTORY = 14;
    private static final int MIN_HISTORY = 4;
    private static final float EDGE_DECAY = 0.995f;

    private static final float R_MIN = 110f;
    private static final float S_T = 55f;
    private static final float CBCR_GAP = 40f;

    private static final float SMOKE_SPREAD = 26f;
    private static final float SMOKE_Y_MIN = 55f;
    private static final float SMOKE_Y_MAX = 245f;

    private static final float FLAME_RATIO_FLOOR = 0.25f;
    private static final float FLAME_FLICKER_FLOOR = 0.012f;
    private static final float FLAME_OCCUPANCY_FLOOR = 0.50f;
    private static final float FLAME_PRESENT = 0.15f;
    private static final float FLICKER_DELTA = 0.06f;
    private static final float FLICKER_RATE_FLOOR = 0.45f;

    private static final float SMOKE_RATIO_FLOOR = 0.35f;
    private static final float SMOKE_EDGE_DROP_FLOOR = 0.20f;
    private static final float SMOKE_MOVE_FLOOR = 0.35f;
    private static final float SMOKE_STILL_FLOOR = 0.50f;
    private static final float SMOKE_STILL_EDGE_MAX = 6f;

    private static final float STILL_CEILING_FLAME = 0.60f;
    private static final float STILL_CEILING_SMOKE = 0.45f;

    private static final int MIN_CELLS_FLAME = 2;
    private static final int MIN_CELLS_SMOKE = 6;
    private static final int MIN_SPAN_SMOKE = 2;

    /** One region, in coordinates normalised to the frame so any canvas can draw it. */
    static final class Region {
        final String label;
        final float confidence;
        final float x0;
        final float y0;
        final float x1;
        final float y1;
        final boolean temporal;

        Region(String label, float confidence, float x0, float y0, float x1, float y1,
               boolean temporal) {
            this.label = label;
            this.confidence = confidence;
            this.x0 = x0;
            this.y0 = y0;
            this.x1 = x1;
            this.y1 = y1;
            this.temporal = temporal;
        }
    }

    /** One frame's measurements for one cell. */
    private static final class Sample {
        final float flame;
        final float bright;

        Sample(float flame, float bright) {
            this.flame = flame;
            this.bright = bright;
        }
    }

    private int cols;
    private int rows;
    private List<Deque<Sample>> history = new ArrayList<>();
    private float[] edgeBase = new float[0];
    private int[] pixels = new int[0];
    private long frames;

    /** Forget every frame seen so far. Called whenever the stream changes. */
    void reset() {
        cols = 0;
        rows = 0;
        history = new ArrayList<>();
        edgeBase = new float[0];
        frames = 0;
    }

    long framesSeen() {
        return frames;
    }

    /**
     * Scan one frame.
     *
     * The bitmap is scaled down to the working width first. Everything here is measured in
     * regions rather than pixels, so 160 across is enough to find a plume while costing
     * about a millisecond - which is what lets it run beside the detector every frame
     * instead of taking a turn away from it.
     */
    List<Region> scan(Bitmap frame) {
        if (frame == null || frame.getWidth() < 2 || frame.getHeight() < 2) {
            return new ArrayList<>();
        }
        int w = Math.min(frame.getWidth(), WORK_WIDTH);
        int h = Math.max(1, Math.round(((float) frame.getHeight() / frame.getWidth()) * w));

        // Filtered, not nearest neighbour. Nearest neighbour discards exactly what is being
        // measured: a thin flame edge falling between sample points disappears, and the
        // fraction the whole method rests on comes out wrong.
        //
        // Nothing is held onto afterwards. createScaledBitmap hands back the source itself
        // when no scaling is needed, so keeping the result in a field and recycling it next
        // time would recycle a bitmap the caller still owns.
        Bitmap scaled = Bitmap.createScaledBitmap(frame, w, h, true);
        if (pixels.length != w * h) {
            pixels = new int[w * h];
        }
        scaled.getPixels(pixels, 0, w, 0, 0, w, h);
        if (scaled != frame) {
            scaled.recycle();
        }
        return regions(measure(w, h));
    }

    // -----------------------------------------------------------------------------------
    // Per-cell measurement
    // -----------------------------------------------------------------------------------

    private static final class Grid {
        float[] flame;
        float[] smoke;
        float[] bright;
        float[] edge;
    }

    private Grid measure(int width, int height) {
        int nextCols = Math.max(1, (int) Math.ceil(width / (double) CELL));
        int nextRows = Math.max(1, (int) Math.ceil(height / (double) CELL));
        if (nextCols != cols || nextRows != rows) {
            cols = nextCols;
            rows = nextRows;
            history = new ArrayList<>(cols * rows);
            for (int i = 0; i < cols * rows; i++) {
                history.add(new ArrayDeque<Sample>(HISTORY));
            }
            edgeBase = new float[cols * rows];
            java.util.Arrays.fill(edgeBase, -1f);
            frames = 0;
        }

        int count = width * height;
        float[] luma = new float[count];
        float[] cb = new float[count];
        float[] cr = new float[count];
        double sumY = 0;
        double sumCb = 0;
        double sumCr = 0;

        for (int i = 0; i < count; i++) {
            int argb = pixels[i];
            float r = (argb >> 16) & 0xFF;
            float g = (argb >> 8) & 0xFF;
            float b = argb & 0xFF;
            luma[i] = 0.299f * r + 0.587f * g + 0.114f * b;
            cb[i] = -0.168736f * r - 0.331264f * g + 0.5f * b + 128f;
            cr[i] = 0.5f * r - 0.418688f * g - 0.081312f * b + 128f;
            sumY += luma[i];
            sumCb += cb[i];
            sumCr += cr[i];
        }
        // Celik and Demirel's rules compare each pixel against the frame's own means, which
        // is what makes the test hold under a bright sky and at dusk alike.
        float meanY = (float) (sumY / count);
        float meanCb = (float) (sumCb / count);
        float meanCr = (float) (sumCr / count);

        Grid grid = new Grid();
        grid.flame = new float[cols * rows];
        grid.smoke = new float[cols * rows];
        grid.bright = new float[cols * rows];
        grid.edge = new float[cols * rows];
        float[] counts = new float[cols * rows];

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int i = y * width + x;
                int argb = pixels[i];
                float r = (argb >> 16) & 0xFF;
                float g = (argb >> 8) & 0xFF;
                float b = argb & 0xFF;
                int c = (y / CELL) * cols + (x / CELL);

                counts[c] += 1;
                grid.bright[c] += luma[i];

                float max = Math.max(r, Math.max(g, b));
                float min = Math.min(r, Math.min(g, b));
                float saturation = max > 0 ? ((max - min) / max) * 255f : 0f;

                boolean isFlame = r > g && g > b
                        && r >= R_MIN
                        && saturation >= ((255f - r) * S_T) / R_MIN
                        && luma[i] > cb[i] && cr[i] > cb[i]
                        && Math.abs(cb[i] - cr[i]) >= CBCR_GAP
                        && luma[i] >= meanY && cb[i] <= meanCb && cr[i] >= meanCr;

                if (isFlame) {
                    grid.flame[c] += 1;
                } else if (max - min <= SMOKE_SPREAD
                        && luma[i] >= SMOKE_Y_MIN && luma[i] <= SMOKE_Y_MAX) {
                    grid.smoke[c] += 1;
                }

                // Texture, as the plain gradient of brightness. Smoke pulls this down
                // because it veils the detail behind it, which is a more reliable cue than
                // its colour: plenty of things are grey, very few erase the background.
                float right = x + 1 < width ? luma[i + 1] : luma[i];
                float below = y + 1 < height ? luma[i + width] : luma[i];
                grid.edge[c] += Math.abs(right - luma[i]) + Math.abs(below - luma[i]);
            }
        }

        for (int c = 0; c < counts.length; c++) {
            float n = counts[c] == 0 ? 1 : counts[c];
            grid.flame[c] /= n;
            grid.smoke[c] /= n;
            grid.bright[c] /= n;
            grid.edge[c] /= n;
        }

        for (int c = 0; c < grid.flame.length; c++) {
            Deque<Sample> window = history.get(c);
            window.addLast(new Sample(grid.flame[c], grid.bright[c]));
            if (window.size() > HISTORY) {
                window.removeFirst();
            }
        }
        frames++;
        return grid;
    }

    // -----------------------------------------------------------------------------------
    // Scoring and region assembly
    // -----------------------------------------------------------------------------------

    private List<Region> regions(Grid grid) {
        float[] flameScore = new float[cols * rows];
        float[] smokeScore = new float[cols * rows];
        boolean temporal = false;

        for (int c = 0; c < cols * rows; c++) {
            Sample[] window = history.get(c).toArray(new Sample[0]);
            int seen = window.length;
            boolean hasTime = seen >= MIN_HISTORY;
            if (hasTime) {
                temporal = true;
            }

            float flameNow = grid.flame[c];
            float smokeNow = grid.smoke[c];
            float edgeNow = grid.edge[c];

            float flameMean = flameNow;
            float flicker = 0;
            float move = 0;
            float occupancy = flameNow >= FLAME_PRESENT ? 1 : 0;
            float flickerRate = 0;
            if (seen > 1) {
                float sum = 0;
                int held = 0;
                int changed = 0;
                for (int i = 0; i < seen; i++) {
                    sum += window[i].flame;
                    if (window[i].flame >= FLAME_PRESENT) {
                        held++;
                    }
                    if (i > 0) {
                        float delta = Math.abs(window[i].flame - window[i - 1].flame);
                        flicker += delta;
                        if (delta >= FLICKER_DELTA) {
                            changed++;
                        }
                        move += Math.abs(window[i].bright - window[i - 1].bright);
                    }
                }
                flameMean = sum / seen;
                occupancy = held / (float) seen;
                flickerRate = changed / (float) (seen - 1);
                flicker /= seen - 1;
                move /= seen - 1;
            }

            // Texture is compared against the cell's own decaying memory, not against the
            // frames in the flicker window. A plume that drifts in and then sits there
            // would otherwise become its own baseline within a second and stop being
            // reported, which is the moment an operator most needs the box.
            float prior = edgeBase[c] < 0 ? edgeNow : edgeBase[c] * EDGE_DECAY;
            float edgeDrop = prior > 0.5f ? clamp01((prior - edgeNow) / prior) : 0f;
            edgeBase[c] = Math.max(edgeNow, prior);

            if (hasTime) {
                // Rate, not amplitude. A red van crossing the shot swings a cell from no
                // flame colour to all of it and back, which is a bigger swing than fire
                // produces - but it does it twice, on the frames its edge crosses the cell,
                // and holds steady for the ten in between. Fire never holds steady.
                if (flameMean >= FLAME_RATIO_FLOOR
                        && flicker >= FLAME_FLICKER_FLOOR
                        && flickerRate >= FLICKER_RATE_FLOOR
                        && occupancy >= FLAME_OCCUPANCY_FLOOR) {
                    flameScore[c] = clamp01(
                            0.40f * clamp01(flameMean / 0.30f)
                            + 0.25f * clamp01(flicker / 0.05f)
                            + 0.25f * flickerRate
                            + 0.10f * clamp01(flameNow / 0.25f));
                }
                if (smokeNow >= SMOKE_RATIO_FLOOR
                        && edgeDrop >= SMOKE_EDGE_DROP_FLOOR
                        && move >= SMOKE_MOVE_FLOOR) {
                    smokeScore[c] = clamp01(
                            0.40f * clamp01(smokeNow / 0.60f)
                            + 0.35f * clamp01(edgeDrop / 0.50f)
                            + 0.25f * clamp01(move / 2f));
                }
            } else {
                // One frame, no history. Colour is all there is, so the floor is higher and
                // the ceiling lower: a still can never reach the confidence a flickering
                // region earns, because the evidence that separates them is not available.
                if (flameNow >= 0.15f) {
                    flameScore[c] = Math.min(STILL_CEILING_FLAME,
                            0.60f * clamp01(flameNow / 0.35f));
                }
                if (smokeNow >= SMOKE_STILL_FLOOR && edgeNow <= SMOKE_STILL_EDGE_MAX) {
                    smokeScore[c] = Math.min(STILL_CEILING_SMOKE,
                            0.45f * clamp01(smokeNow / 0.75f));
                }
            }
        }

        List<Region> found = new ArrayList<>();
        found.addAll(join(flameScore, MIN_CELLS_FLAME, FLAME, temporal));
        found.addAll(join(smokeScore, MIN_CELLS_SMOKE, SMOKE, temporal));
        return dropSky(found);
    }

    /**
     * Flood-fill the passing cells into rectangles.
     *
     * Four-connected on purpose. Eight-connected joins a flame region to an unrelated thing
     * touching it at one corner, and a box around both is worse than two boxes.
     */
    private List<Region> join(float[] score, int minCells, String label, boolean temporal) {
        boolean[] seen = new boolean[score.length];
        List<Region> out = new ArrayList<>();

        for (int start = 0; start < score.length; start++) {
            if (seen[start] || score[start] <= 0) {
                continue;
            }
            Deque<Integer> stack = new ArrayDeque<>();
            List<Integer> members = new ArrayList<>();
            stack.push(start);
            seen[start] = true;

            while (!stack.isEmpty()) {
                int at = stack.pop();
                members.add(at);
                int x = at % cols;
                int y = at / cols;
                int[] neighbours = {
                        x > 0 ? at - 1 : -1,
                        x + 1 < cols ? at + 1 : -1,
                        y > 0 ? at - cols : -1,
                        y + 1 < rows ? at + cols : -1,
                };
                for (int next : neighbours) {
                    if (next >= 0 && !seen[next] && score[next] > 0) {
                        seen[next] = true;
                        stack.push(next);
                    }
                }
            }
            if (members.size() < minCells) {
                continue;
            }

            int minX = cols;
            int minY = rows;
            int maxX = 0;
            int maxY = 0;
            float total = 0;
            for (int at : members) {
                int x = at % cols;
                int y = at / cols;
                minX = Math.min(minX, x);
                minY = Math.min(minY, y);
                maxX = Math.max(maxX, x);
                maxY = Math.max(maxY, y);
                total += score[at];
            }
            out.add(new Region(label, total / members.size(),
                    minX / (float) cols, minY / (float) rows,
                    (maxX + 1) / (float) cols, (maxY + 1) / (float) rows,
                    temporal));
        }
        return out;
    }

    /**
     * Throw away the smoke region that is actually the weather.
     *
     * Overcast, haze and a flat grey sky pass the colour test, and on a moving camera the
     * boundary where sky meets ground passes the movement test too. What they cannot do is
     * be shaped like a plume: that boundary is a hairline across the whole frame, and the
     * overcast above it hangs off the top edge.
     */
    private List<Region> dropSky(List<Region> found) {
        List<Region> flames = new ArrayList<>();
        for (Region region : found) {
            if (FLAME.equals(region.label)) {
                flames.add(region);
            }
        }

        List<Region> kept = new ArrayList<>();
        for (Region region : found) {
            if (!SMOKE.equals(region.label)) {
                kept.add(region);
                continue;
            }
            float width = region.x1 - region.x0;
            float height = region.y1 - region.y0;

            boolean nearFlame = false;
            for (Region flame : flames) {
                if (!(flame.x1 < region.x0 || region.x1 < flame.x0
                        || flame.y1 < region.y0 || region.y1 < flame.y0)) {
                    nearFlame = true;
                    break;
                }
            }
            // Flame in the same frame settles the argument: grey above a fire is smoke.
            if (nearFlame) {
                kept.add(region);
                continue;
            }
            if (width >= 0.90f && height <= 0.20f) {
                continue;
            }
            if (width < MIN_SPAN_SMOKE / (float) cols || height < MIN_SPAN_SMOKE / (float) rows) {
                continue;
            }
            if (region.y0 <= 1f / rows && width >= 0.80f) {
                continue;
            }
            if (width * height > 0.60f) {
                continue;
            }
            kept.add(region);
        }
        return kept;
    }

    private static float clamp01(float value) {
        if (value < 0) {
            return 0;
        }
        return value > 1 ? 1 : value;
    }
}
