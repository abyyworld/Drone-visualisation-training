package world.abyy.droneinspection;

import androidx.annotation.NonNull;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Turns per-frame detections into tracks that keep an identity across frames.
 *
 * A port of web/js/track.js, deliberately line for line: the same association rule, the
 * same constants, the same coasting. Two trackers that behave differently would mean the
 * live screen and the uploads screen disagree about how many people walked past, and the
 * one an operator saw in the field would be the one nobody could reproduce afterwards.
 *
 * Association is IoU first, then a centre-distance fallback measured against where the
 * track's own velocity says it should be. The fallback is not a nicety. Detection runs a
 * handful of times a second, so between two looks a walking person moves most of their own
 * width and the boxes then do not overlap at all - and IoU alone loses them and issues a
 * new number, which is what "person #1, then #2, then #3" looks like from the outside.
 */
public final class Tracker {

    private static final double MIN_IOU = 0.08;
    private static final double MAX_CENTRE_DRIFT = 1.6;
    private static final double MAX_SIZE_RATIO = 2.2;
    private static final int MAX_MISSES = 20;
    private static final int CONFIRM_AFTER = 2;
    private static final int MAX_PATH = 60;

    /** One thing being followed. */
    public static final class Track {
        public final int id;
        public final String label;
        public float[] box;
        public float confidence;
        public int seen;
        public int missed;
        public float[] velocity = {0f, 0f};
        public final List<float[]> path = new ArrayList<>();
        boolean counted;

        Track(int id, String label, float[] box, float confidence) {
            this.id = id;
            this.label = label;
            this.box = box;
            this.confidence = confidence;
            this.seen = 1;
            this.path.add(centre(box));
        }

        public boolean coasted() {
            return missed > 0;
        }
    }

    private final List<Track> tracks = new ArrayList<>();
    private final Map<String, Integer> everSeen = new HashMap<>();
    private int nextId = 1;

    /** Advance one frame. Returns the tracks worth drawing. */
    public List<Track> update(List<Finding> detections) {
        List<Pair> pairs = new ArrayList<>();
        for (Track track : tracks) {
            for (int i = 0; i < detections.size(); i++) {
                Finding detection = detections.get(i);
                // A car does not become a person between two frames, and allowing it lets
                // an identity jump across the frame.
                if (!track.label.equals(detection.label)) {
                    continue;
                }
                float[] box = boxOf(detection);
                double score = affinity(track, box);
                if (score > 0) {
                    pairs.add(new Pair(track, i, score, box));
                }
            }
        }
        pairs.sort(Comparator.comparingDouble((Pair p) -> p.score).reversed());

        Set<Track> usedTracks = new HashSet<>();
        Set<Integer> usedDetections = new HashSet<>();
        for (Pair pair : pairs) {
            if (usedTracks.contains(pair.track) || usedDetections.contains(pair.index)) {
                continue;
            }
            usedTracks.add(pair.track);
            usedDetections.add(pair.index);

            float[] previous = centre(pair.track.box);
            float[] now = centre(pair.box);
            pair.track.box = pair.box;
            pair.track.confidence = detections.get(pair.index).confidence;
            pair.track.missed = 0;
            pair.track.seen += 1;
            // Smoothed, so one noisy frame does not send the coasting prediction sideways.
            pair.track.velocity = new float[]{
                    pair.track.velocity[0] * 0.6f + (now[0] - previous[0]) * 0.4f,
                    pair.track.velocity[1] * 0.6f + (now[1] - previous[1]) * 0.4f,
            };
            pair.track.path.add(now);
            if (pair.track.path.size() > MAX_PATH) {
                pair.track.path.remove(0);
            }
        }

        for (int i = 0; i < detections.size(); i++) {
            if (usedDetections.contains(i)) {
                continue;
            }
            Finding detection = detections.get(i);
            tracks.add(new Track(nextId++, detection.label, boxOf(detection), detection.confidence));
        }

        for (Track track : tracks) {
            if (usedTracks.contains(track)) {
                continue;
            }
            track.missed += 1;
            track.box = new float[]{
                    track.box[0] + track.velocity[0], track.box[1] + track.velocity[1],
                    track.box[2] + track.velocity[0], track.box[3] + track.velocity[1],
            };
        }

        // Counted once, when a track becomes confirmed: not while it is a one-frame
        // flicker, and not again on every frame after.
        for (Track track : tracks) {
            if (track.seen == CONFIRM_AFTER && !track.counted) {
                track.counted = true;
                everSeen.merge(track.label, 1, Integer::sum);
            }
        }

        tracks.removeIf(t -> t.missed > MAX_MISSES);
        return open();
    }

    /** Tracks worth drawing: held back until seen enough times to not be a flicker. */
    public List<Track> open() {
        List<Track> confirmed = new ArrayList<>();
        for (Track track : tracks) {
            if (track.seen >= CONFIRM_AFTER) {
                confirmed.add(track);
            }
        }
        return confirmed;
    }

    /** How many of a label are being followed right now. */
    public int countOf(String label) {
        int n = 0;
        for (Track track : open()) {
            if (track.label.equals(label)) {
                n += 1;
            }
        }
        return n;
    }

    /** How many distinct ones have been seen since the start. Never goes down. */
    public int countSeen(String label) {
        Integer n = everSeen.get(label);
        return n == null ? 0 : n;
    }

    public void reset() {
        tracks.clear();
        everSeen.clear();
        nextId = 1;
    }

    // -----------------------------------------------------------------------------------

    private static final class Pair {
        final Track track;
        final int index;
        final double score;
        final float[] box;

        Pair(Track track, int index, double score, float[] box) {
            this.track = track;
            this.index = index;
            this.score = score;
            this.box = box;
        }
    }

    private static float[] boxOf(@NonNull Finding finding) {
        return new float[]{finding.x0, finding.y0, finding.x1, finding.y1};
    }

    private static float[] centre(float[] box) {
        return new float[]{(box[0] + box[2]) / 2f, (box[1] + box[3]) / 2f};
    }

    static double iou(float[] a, float[] b) {
        float x0 = Math.max(a[0], b[0]);
        float y0 = Math.max(a[1], b[1]);
        float x1 = Math.min(a[2], b[2]);
        float y1 = Math.min(a[3], b[3]);
        if (x1 <= x0 || y1 <= y0) {
            return 0;
        }
        double overlap = (double) (x1 - x0) * (y1 - y0);
        double areaA = (double) Math.max(0, a[2] - a[0]) * Math.max(0, a[3] - a[1]);
        double areaB = (double) Math.max(0, b[2] - b[0]) * Math.max(0, b[3] - b[1]);
        double union = areaA + areaB - overlap;
        return union <= 0 ? 0 : overlap / union;
    }

    /**
     * How strongly a detection belongs to a track. Zero means it does not.
     *
     * Overlap is the better evidence when there is any. When there is none, fall back to
     * how far the centre has moved relative to the size of the thing, measured from where
     * the track's velocity predicts it should be - with a size check, so a distant person
     * is not adopted by a nearby one's track.
     */
    private static double affinity(Track track, float[] box) {
        float[] predicted = {
                track.box[0] + track.velocity[0], track.box[1] + track.velocity[1],
                track.box[2] + track.velocity[0], track.box[3] + track.velocity[1],
        };

        double overlap = Math.max(iou(track.box, box), iou(predicted, box));
        if (overlap >= MIN_IOU) {
            return 1 + overlap;   // always beats any distance-only match
        }

        float tw = Math.abs(track.box[2] - track.box[0]);
        float th = Math.abs(track.box[3] - track.box[1]);
        float dw = Math.abs(box[2] - box[0]);
        float dh = Math.abs(box[3] - box[1]);
        if (tw <= 0 || th <= 0 || dw <= 0 || dh <= 0) {
            return 0;
        }

        double ratio = Math.max(Math.max(tw / dw, dw / tw), Math.max(th / dh, dh / th));
        if (ratio > MAX_SIZE_RATIO) {
            return 0;
        }

        float[] p = centre(predicted);
        float[] b = centre(box);
        double drift = Math.hypot(p[0] - b[0], p[1] - b[1])
                / Math.max(1, Math.hypot(tw, th) / 2);
        if (drift > MAX_CENTRE_DRIFT) {
            return 0;
        }
        return 1 - drift / MAX_CENTRE_DRIFT;
    }
}
