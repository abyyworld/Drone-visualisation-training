package world.abyy.droneinspection;

/**
 * Enough of Finding for Tiles and the Tracker to compile off-device.
 *
 * The real one carries JSON parsing and Android annotations, neither of which exists
 * outside an Android build. What those two actually touch is a label, a confidence, four
 * edges and an appearance signature, so that is what this has.
 */
public final class Finding {
    public final String label;
    public float confidence;
    public final float x0;
    public final float y0;
    public final float x1;
    public final float y1;
    public float[] signature;

    // Public where the real one is package-private, because the comparison harness lives in
    // the default package and has to be able to build a detection to feed in. Nothing else
    // about the shape differs.
    public Finding(String label, String certainty, String note,
                   float x0, float y0, float x1, float y1) {
        this.label = label;
        this.x0 = x0;
        this.y0 = y0;
        this.x1 = x1;
        this.y1 = y1;
    }
}
