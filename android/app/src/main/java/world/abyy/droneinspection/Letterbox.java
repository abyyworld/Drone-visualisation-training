package world.abyy.droneinspection;

/**
 * Where the video sits inside a view that is not its shape.
 *
 * WHY THIS IS ITS OWN FILE
 *     It was the same eight lines written twice, in GlPipeline.drawToDisplay and in
 *     OverlayView.onDraw, and the boxes are drawn by one against a picture placed by the
 *     other. When they disagreed, every box on screen was off its person by however wide the
 *     black bars were - systematically, in the same direction, for the whole flight, which
 *     reads as a detector that cannot aim rather than as a layout bug.
 *
 *     A test comparing two copies would have caught that after the fact. One copy cannot
 *     disagree with itself.
 *
 *     Letterboxed rather than stretched, because distances and shapes on the picture are
 *     what the operator is judging and a stretched frame quietly misrepresents both.
 */
public final class Letterbox {

    /**
     * The rectangle the video occupies: x, y, width, height, in the view's pixels.
     *
     * Public so tests/java/Cross.java can reach it: this arithmetic is checked against an
     * independently written copy of the contract rather than against itself.
     *
     * @param viewWidth   the surface being drawn into
     * @param viewHeight  the surface being drawn into
     * @param videoWidth  the picture's own size
     * @param videoHeight the picture's own size
     */
    public static int[] fit(int viewWidth, int viewHeight, int videoWidth, int videoHeight) {
        if (viewWidth <= 0 || viewHeight <= 0 || videoWidth <= 0 || videoHeight <= 0) {
            return new int[]{0, 0, Math.max(0, viewWidth), Math.max(0, viewHeight)};
        }
        float videoAspect = videoWidth / (float) videoHeight;
        float viewAspect = viewWidth / (float) viewHeight;
        int width;
        int height;
        if (viewAspect > videoAspect) {
            height = viewHeight;
            width = Math.round(viewHeight * videoAspect);
        } else {
            width = viewWidth;
            height = Math.round(viewWidth / videoAspect);
        }
        return new int[]{(viewWidth - width) / 2, (viewHeight - height) / 2, width, height};
    }

    private Letterbox() {
    }
}
