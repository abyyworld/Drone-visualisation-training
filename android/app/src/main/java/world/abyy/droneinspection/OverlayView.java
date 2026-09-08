package world.abyy.droneinspection;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.DashPathEffect;
import android.graphics.Paint;
import android.graphics.Rect;
import android.graphics.RectF;
import android.util.AttributeSet;
import android.view.View;

import androidx.annotation.Nullable;

import java.util.ArrayList;
import java.util.List;

/**
 * Draws the findings over the video, and says how old they are.
 *
 * THE AGE IS NOT DECORATION
 *     These boxes come from a provider round trip that takes seconds, on a frame sampled
 *     seconds before that. On a moving drone over a moving crowd, a box a few seconds old
 *     is describing somewhere the camera is no longer pointed. Drawing it without saying so
 *     would present stale analysis as a live overlay, which is the most dangerous thing
 *     this screen could do, so the age of the boxes is rendered as prominently as the boxes.
 *
 *     When they get older than STALE_AFTER_MS they are dimmed and then dropped, because an
 *     old box on a new scene is worse than no box.
 */
public class OverlayView extends View {

    /** Matches PALETTE in web/js/render.js, so a label is the same colour in both. */
    private static final int[] PALETTE = {
            0xFFE5484D, 0xFFF76B15, 0xFFFFB224, 0xFF30A46C,
            0xFF0091FF, 0xFF8E4EC6, 0xFFE93D82, 0xFF00A2C7,
    };

    private static final long DIM_AFTER_MS = 4_000;

    /**
     * Above this many boxes, the labels come off.
     *
     * Almost nothing about a crowd costs more than a crowd of one: the model runs the same
     * convolutions over the same tensor whether the frame holds nobody or two hundred
     * people, and suppression and tracking are arithmetic on short lists. Drawing is the
     * exception. Every label measures its own text and then draws it, and at two hundred
     * boxes and sixty frames a second that is twelve thousand text measurements a second
     * on a handheld - which is real work, repeated, for something nobody can read anyway.
     *
     * Past this many, the boxes stay and the labels go. A number on a box is only useful
     * when there are few enough boxes to read one.
     */
    private static final int LABEL_LIMIT = 40;
    private static final long STALE_AFTER_MS = 15_000;

    private final Paint boxPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint labelPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint labelBackground = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint statusPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint statusBackground = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint firePaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint fireOutline = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Rect textBounds = new Rect();

    private List<Finding> findings = new ArrayList<>();
    private long findingsAt;
    private String status = "";

    // Two sources, drawn together and aged differently. Tracks come from the detector on
    // this device and are current to the last frame; findings come from a provider seconds
    // ago. Showing them with the same weight would present one as being as fresh as the
    // other, which is the whole thing this overlay exists not to do.
    private List<Tracker.Track> tracks = new ArrayList<>();
    // The size of the frame those tracks were found in. Their boxes are in that frame's
    // pixels, and this view is a different size - usually three times larger, because
    // detection runs on a 640-pixel copy of a 1080-pixel screen. Without these the boxes
    // draw at a third scale in the top-left corner.
    private int trackFrameWidth;
    private int trackFrameHeight;

    private List<FireScan.Region> fire = new ArrayList<>();

    public OverlayView(Context context, @Nullable AttributeSet attrs) {
        super(context, attrs);
        boxPaint.setStyle(Paint.Style.STROKE);
        labelPaint.setColor(Color.WHITE);
        labelBackground.setStyle(Paint.Style.FILL);
        statusPaint.setColor(Color.WHITE);
        statusBackground.setColor(0xCC000000);
        statusBackground.setStyle(Paint.Style.FILL);
        firePaint.setStyle(Paint.Style.FILL);
        fireOutline.setStyle(Paint.Style.STROKE);
        fireOutline.setPathEffect(new DashPathEffect(new float[]{16f, 10f}, 0f));
        setWillNotDraw(false);
    }

    /**
     * Replace the live tracks. Called on every detected frame.
     *
     * The frame size comes with them because the boxes are in its pixels, not this view's.
     */
    public void setTracks(List<Tracker.Track> next, int frameWidth, int frameHeight) {
        tracks = next == null ? new ArrayList<>() : next;
        trackFrameWidth = frameWidth;
        trackFrameHeight = frameHeight;
        invalidate();
    }

    /** Replace the flame and smoke regions. Already normalised, so no frame size needed. */
    public void setFire(List<FireScan.Region> next) {
        fire = next == null ? new ArrayList<>() : next;
        invalidate();
    }

    /** Replace what is drawn. An empty list clears the overlay. */
    public void setFindings(List<Finding> next) {
        findings = next == null ? new ArrayList<>() : next;
        findingsAt = System.currentTimeMillis();
        invalidate();
    }

    /** One line above the boxes: what the analyser is doing, or why it is not. */
    public void setStatus(String text) {
        status = text == null ? "" : text;
        invalidate();
    }

    public void clear() {
        findings = new ArrayList<>();
        tracks = new ArrayList<>();
        fire = new ArrayList<>();
        findingsAt = 0;
        invalidate();
    }

    /**
     * Draw onto an arbitrary canvas, for the recorder.
     *
     * The recorder composes each frame at the video's own resolution rather than the
     * screen's, so this takes explicit dimensions instead of using getWidth().
     */
    public void drawInto(Canvas canvas, int width, int height) {
        render(canvas, width, height);
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        render(canvas, getWidth(), getHeight());
    }

    /** Same hash as colourIndex() in web/js/vlm.js, so a label keeps one colour anywhere. */
    private static int colourIndex(String label) {
        int hash = 0;
        for (int i = 0; i < label.length(); i++) {
            hash = (hash * 31 + label.charAt(i)) % 4096;
        }
        return hash;
    }

    /**
     * Flame and smoke regions: a dashed outline and a wash, with no identity number.
     *
     * No number because these are not things being followed. A fire is not one object
     * moving through the shot, it is a region that grows, splits and dies, and giving it a
     * number would claim something about it that is not true. The percentage is the
     * scanner's own confidence, and the label is the class, so the box can be argued with.
     */
    private void drawFire(Canvas canvas, int width, int height, float stroke, float textSize) {
        for (FireScan.Region region : fire) {
            int colour = FireScan.SMOKE.equals(region.label) ? 0xFF9AA3AD : 0xFFFF5A1F;
            RectF box = new RectF(
                    region.x0 * width, region.y0 * height,
                    region.x1 * width, region.y1 * height);

            firePaint.setColor(colour);
            firePaint.setAlpha(40);
            canvas.drawRect(box, firePaint);

            fireOutline.setColor(colour);
            fireOutline.setStrokeWidth(stroke);
            canvas.drawRect(box, fireOutline);

            String text = region.label + "  " + Math.round(region.confidence * 100) + "%";
            labelPaint.setTextSize(textSize);
            labelPaint.getTextBounds(text, 0, text.length(), textBounds);
            float pad = stroke * 2;
            float top = Math.min(height - textBounds.height() - pad * 2, box.bottom + pad);

            labelBackground.setColor(colour);
            labelBackground.setAlpha(255);
            canvas.drawRect(box.left, top,
                    box.left + textBounds.width() + pad * 2, top + textBounds.height() + pad * 2,
                    labelBackground);
            labelPaint.setColor(Color.BLACK);
            labelPaint.setAlpha(255);
            canvas.drawText(text, box.left + pad, top + textBounds.height() + pad, labelPaint);
            labelPaint.setColor(Color.WHITE);
        }
    }

    /**
     * Synchronized, because this is now drawn from two threads.
     *
     * onDraw runs on the main thread for the screen; drawInto runs on the pipeline's thread
     * to compose a snapshot, and on the recorder's path to build the overlay texture. They
     * share the Paint objects and the text-bounds Rect above, and two threads using one
     * Paint at once produces boxes in the wrong colour and text in the wrong place, which
     * looks like a rendering bug and is a missing lock.
     */
    private synchronized void render(Canvas canvas, int width, int height) {
        long age = findingsAt == 0 ? 0 : System.currentTimeMillis() - findingsAt;
        boolean stale = findingsAt != 0 && age > STALE_AFTER_MS;

        float longEdge = Math.max(width, height);
        float stroke = Math.max(2f, longEdge / 260f);
        float textSize = Math.max(16f, longEdge / 38f);
        boxPaint.setStrokeWidth(stroke);
        labelPaint.setTextSize(textSize);
        statusPaint.setTextSize(textSize * 0.85f);

        // Flame and smoke underneath everything, so a person standing in front of a fire
        // still gets a solid box over the top of it.
        drawFire(canvas, width, height, stroke, textSize);

        // Live tracks next, at full weight, with the identity on the label. These are
        // current: they came from this device on the last frame it managed.
        float trackScaleX = trackFrameWidth > 0 ? width / (float) trackFrameWidth : 1f;
        float trackScaleY = trackFrameHeight > 0 ? height / (float) trackFrameHeight : 1f;
        boolean labelled = tracks.size() <= LABEL_LIMIT;
        for (Tracker.Track track : tracks) {
            int colour = PALETTE[colourIndex(track.label) % PALETTE.length];
            boxPaint.setColor(colour);
            boxPaint.setAlpha(track.coasted() ? 120 : 255);

            RectF box = new RectF(
                    track.box[0] * trackScaleX, track.box[1] * trackScaleY,
                    track.box[2] * trackScaleX, track.box[3] * trackScaleY);
            canvas.drawRect(box, boxPaint);

            if (!labelled) {
                // Too many to read. The box is the information at this density; the
                // numbers are in the status line.
                continue;
            }

            String text = track.label + " #" + track.id;
            labelPaint.getTextBounds(text, 0, text.length(), textBounds);
            float pad = stroke * 2;
            float top = Math.max(0, box.top - textBounds.height() - pad * 2);

            labelBackground.setColor(colour);
            labelBackground.setAlpha(track.coasted() ? 120 : 255);
            canvas.drawRect(box.left, top,
                    box.left + textBounds.width() + pad * 2, top + textBounds.height() + pad * 2,
                    labelBackground);
            labelPaint.setAlpha(track.coasted() ? 120 : 255);
            canvas.drawText(text, box.left + pad, top + textBounds.height() + pad, labelPaint);
        }

        if (!stale) {
            int alpha = findingsAt != 0 && age > DIM_AFTER_MS ? 130 : 255;
            for (Finding finding : findings) {
                int colour = PALETTE[finding.colourIndex() % PALETTE.length];
                boxPaint.setColor(colour);
                boxPaint.setAlpha(alpha);

                RectF box = new RectF(
                        finding.x0 * width, finding.y0 * height,
                        finding.x1 * width, finding.y1 * height);
                canvas.drawRect(box, boxPaint);

                String text = finding.label + "  " + finding.certainty;
                labelPaint.getTextBounds(text, 0, text.length(), textBounds);
                float pad = stroke * 2;
                float top = Math.max(0, box.top - textBounds.height() - pad * 2);

                labelBackground.setColor(colour);
                labelBackground.setAlpha(alpha);
                canvas.drawRect(box.left, top,
                        box.left + textBounds.width() + pad * 2, top + textBounds.height() + pad * 2,
                        labelBackground);
                labelPaint.setAlpha(alpha);
                canvas.drawText(text, box.left + pad, top + textBounds.height() + pad, labelPaint);
            }
        }

        String line = status;
        if (findingsAt != 0) {
            String ageText = stale
                    ? "boxes cleared, older than " + (STALE_AFTER_MS / 1000) + "s"
                    : "boxes " + (age / 1000) + "s old";
            line = line.isEmpty() ? ageText : line + "  ·  " + ageText;
        }
        if (line.isEmpty()) {
            return;
        }

        statusPaint.getTextBounds(line, 0, line.length(), textBounds);
        float pad = textSize * 0.4f;
        canvas.drawRect(0, 0, textBounds.width() + pad * 2, textBounds.height() + pad * 2,
                statusBackground);
        canvas.drawText(line, pad, textBounds.height() + pad, statusPaint);
    }
}
