package world.abyy.droneinspection;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

/**
 * Decoding a YOLO head, the Java half.
 *
 * The other half is web/js/yolo.js, and the two are run over the same numbers by
 * tests/java/cross_check.sh. That test is the point of this file being written the way it
 * is: getting this arithmetic wrong does not raise an error, it produces confident nonsense
 * or silence. Boxes an anchor-stride out land beside people rather than on them; a
 * transposed layout finds nothing at all, which looks exactly like a frame with nobody in
 * it. So the arithmetic is a line-for-line port, and anything that changes in one and not
 * the other fails the build.
 *
 * Everything here works in double, as JavaScript does, so the two agree to the last place
 * rather than to a tolerance.
 */
public final class Yolo {

    private Yolo() {
    }

    /** One decoded box, in whatever units the model works in. */
    public static final class Detection {
        public final int classId;
        public final float confidence;
        /** x0, y0, x1, y1. */
        public final double[] box;

        public Detection(int classId, float confidence, double[] box) {
            this.classId = classId;
            this.confidence = confidence;
            this.box = box;
        }
    }

    /**
     * Decode a classic YOLO head, channel-major: every anchor's x together, then every
     * anchor's y, and so on.
     *
     * @param values       the raw output
     * @param channels     4 + number of classes
     * @param anchors      how many boxes the head predicts
     * @param keepClasses  class ids worth keeping, or null for all of them
     */
    public static List<Detection> decodeHead(float[] values, int channels, int anchors,
                                             double confThreshold, double iouThreshold,
                                             Set<Integer> keepClasses) {
        int numClasses = channels - 4;
        if (numClasses < 1) {
            throw new IllegalArgumentException(
                    "a head of " + channels + " channels has no classes");
        }

        List<Detection> detections = new ArrayList<>();
        for (int i = 0; i < anchors; i++) {
            float bestScore = 0;
            int bestClass = -1;
            for (int c = 0; c < numClasses; c++) {
                // Filtered here rather than afterwards, so the suppression below never has
                // to consider a van at all - and so a van can never suppress a person
                // standing beside it.
                if (keepClasses != null && !keepClasses.contains(c)) {
                    continue;
                }
                float score = values[(4 + c) * anchors + i];
                if (score > bestScore) {
                    bestScore = score;
                    bestClass = c;
                }
            }
            if (bestClass < 0 || bestScore < confThreshold) {
                continue;
            }

            double cx = values[i];
            double cy = values[anchors + i];
            double w = values[anchors * 2 + i];
            double h = values[anchors * 3 + i];
            detections.add(new Detection(bestClass, bestScore, new double[]{
                    cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2,
            }));
        }

        return nms(detections, iouThreshold);
    }

    /**
     * Rearrange a per-anchor output into the channel-major one {@link #decodeHead} reads.
     *
     * WHY THIS EXISTS
     *     The same weights exported to ONNX and to TFLite do not come out the same way
     *     round. ONNX gives [1, 4+nc, anchors]; the TFLite converter usually gives
     *     [1, anchors, 4+nc]. Reading one as the other does not fail, it finds nothing,
     *     which is indistinguishable from an empty frame.
     *
     *     Transposing once and decoding one way is deliberate. A second decode path would
     *     be a second thing to get wrong and only one of them would be covered by the
     *     comparison against the JavaScript.
     *
     *     The cost is one pass over about thirty thousand floats, which is nothing beside
     *     the inference that produced them.
     */
    public static float[] toChannelMajor(float[] values, int anchors, int channels) {
        float[] out = new float[values.length];
        for (int anchor = 0; anchor < anchors; anchor++) {
            int row = anchor * channels;
            for (int channel = 0; channel < channels; channel++) {
                out[channel * anchors + anchor] = values[row + channel];
            }
        }
        return out;
    }

    /** Greedy per-class non-maximum suppression. */
    public static List<Detection> nms(List<Detection> detections, double iouThreshold) {
        List<Detection> kept = new ArrayList<>();

        // Insertion-ordered, because the JavaScript walks a Set built from the detection
        // order and the two outputs are compared line for line.
        Set<Integer> classes = new LinkedHashSet<>();
        for (Detection detection : detections) {
            classes.add(detection.classId);
        }

        for (int classId : classes) {
            List<Detection> candidates = new ArrayList<>();
            for (Detection detection : detections) {
                if (detection.classId == classId) {
                    candidates.add(detection);
                }
            }
            // Stable, like the JavaScript sort, so equal confidences keep their order.
            candidates.sort(Comparator.comparingDouble((Detection d) -> d.confidence)
                    .reversed());

            while (!candidates.isEmpty()) {
                Detection best = candidates.remove(0);
                kept.add(best);
                for (int i = candidates.size() - 1; i >= 0; i--) {
                    if (iou(best.box, candidates.get(i).box) > iouThreshold) {
                        candidates.remove(i);
                    }
                }
            }
        }
        return kept;
    }

    public static double iou(double[] a, double[] b) {
        double x0 = Math.max(a[0], b[0]);
        double y0 = Math.max(a[1], b[1]);
        double x1 = Math.min(a[2], b[2]);
        double y1 = Math.min(a[3], b[3]);

        double width = x1 - x0;
        double height = y1 - y0;
        if (width <= 0 || height <= 0) {
            return 0;
        }

        double intersection = width * height;
        double areaA = (a[2] - a[0]) * (a[3] - a[1]);
        double areaB = (b[2] - b[0]) * (b[3] - b[1]);
        double union = areaA + areaB - intersection;
        return union > 0 ? intersection / union : 0;
    }

    /** Undo a letterbox, back into the source picture's pixels, clamped to its bounds. */
    public static double[] unletterbox(double[] box, double scale, double padX, double padY,
                                       double width, double height) {
        return new double[]{
                Math.max(0, (box[0] - padX) / scale),
                Math.max(0, (box[1] - padY) / scale),
                Math.min(width, (box[2] - padX) / scale),
                Math.min(height, (box[3] - padY) / scale),
        };
    }

    /** Unmodifiable, for the callers that build a keep-set once and hold it. */
    public static Set<Integer> classes(int... ids) {
        Set<Integer> set = new LinkedHashSet<>();
        for (int id : ids) {
            set.add(id);
        }
        return Collections.unmodifiableSet(set);
    }
}
