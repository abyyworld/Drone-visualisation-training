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

    /**
     * How long a track may coast before it is let go, in milliseconds.
     *
     * Frames were the wrong unit, and this is the bug that showed. Twenty missed frames is
     * about two and a half seconds at the eight detections a second a laptop manages, and
     * fourteen seconds at the 1.4 a tablet manages on a 363 ms model. A spurious box sat on
     * the screen, dashed and drifting, for a quarter of a minute after it had stopped being
     * detected. A second and a half is a person walking behind something and out the other
     * side; anything longer is a box describing the past.
     *
     * Kept in step with MAX_COAST_MS in web/js/track.js.
     */
    private static final long MAX_COAST_MS = 1500;

    /**
     * How alike two colour signatures must be to be the same person coming back.
     *
     * The same person in the same clothes under changing light lands around 0.7 to 0.9;
     * two different people in different clothes land around 0.2 to 0.5. This sits inside
     * that gap and nearer its lower half, deliberately: a missed match counts someone twice
     * and inflates the total, a wrong match merges two people and deflates it. The count is
     * already a floor, so deflating keeps it honest and inflating does not.
     *
     * Kept in step with REID_SIMILARITY in web/js/track.js.
     */
    private static final float REID_SIMILARITY = 0.62f;

    /** How long someone stays recognisable after leaving the frame: one route leg. */
    private static final long REID_WINDOW_MS = 5 * 60 * 1000L;

    /**
     * How many people can be remembered at once.
     *
     * A ceiling on how many distinct people a flight can count, not a detail: once it is
     * full the oldest are forgotten, and a forgotten person walking back into frame is
     * counted a second time. Two thousand signatures is about a megabyte and a half, which
     * is affordable even on a two-gigabyte controller.
     */
    private static final int REID_MAX_REMEMBERED = 2000;
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
        /** What this person looks like, for recognising them after they leave. */
        float[] signature;
        /** True when this identity has already been added to the running total. */
        boolean returned;
        /** When this track was last actually seen, rather than predicted. See MAX_COAST_MS. */
        long lastSeenAt;
        public float[] velocity = {0f, 0f};
        public final List<float[]> path = new ArrayList<>();
        boolean counted;

        Track(int id, String label, float[] box, float confidence, long now) {
            this.id = id;
            this.label = label;
            this.box = box;
            this.confidence = confidence;
            this.seen = 1;
            this.lastSeenAt = now;
            this.path.add(centre(box));
        }

        public boolean coasted() {
            return missed > 0;
        }
    }

    /** People seen and let go, so they are known when they come back. */
    private static final class Remembered {
        final int id;
        final String label;
        final float[] signature;
        final long lastSeen;

        Remembered(int id, String label, float[] signature, long lastSeen) {
            this.id = id;
            this.label = label;
            this.signature = signature;
            this.lastSeen = lastSeen;
        }
    }

    private final List<Remembered> remembered = new ArrayList<>();
    private final List<Track> tracks = new ArrayList<>();
    private final Map<String, Integer> everSeen = new HashMap<>();
    private int nextId = 1;

    /** Advance one frame. Returns the tracks worth drawing. */
    public List<Track> update(List<Finding> detections) {
        return update(detections, System.currentTimeMillis());
    }

    public List<Track> update(List<Finding> detections, long now) {
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
            float[] observedCentre = centre(pair.box);
            pair.track.box = pair.box;
            pair.track.confidence = detections.get(pair.index).confidence;
            pair.track.missed = 0;
            pair.track.seen += 1;
            pair.track.lastSeenAt = now;
            float[] fresh = detections.get(pair.index).signature;
            if (fresh != null) {
                // Kept current rather than frozen at first sight: someone turning around, or
                // the light changing as the drone moves, should update what they look like.
                pair.track.signature = Reid.blend(pair.track.signature, fresh);
            }
            // Smoothed, so one noisy frame does not send the coasting prediction sideways.
            pair.track.velocity = new float[]{
                    pair.track.velocity[0] * 0.6f + (observedCentre[0] - previous[0]) * 0.4f,
                    pair.track.velocity[1] * 0.6f + (observedCentre[1] - previous[1]) * 0.4f,
            };
            pair.track.path.add(observedCentre);
            if (pair.track.path.size() > MAX_PATH) {
                pair.track.path.remove(0);
            }
        }

        for (int i = 0; i < detections.size(); i++) {
            if (usedDetections.contains(i)) {
                continue;
            }
            Finding detection = detections.get(i);
            // Before issuing a new number, ask whether this is someone already known. A
            // track that closed because its subject walked behind something is not a
            // different person when they walk out the other side, and giving them a second
            // number is what turns a count of people into a count of reappearances.
            Remembered known = recognise(detection, now);
            Track track = new Track(known != null ? known.id : nextId++, detection.label,
                    boxOf(detection), detection.confidence, now);
            track.signature = detection.signature;
            if (known != null) {
                if (track.signature == null) {
                    track.signature = known.signature;
                }
                // Carried across, and this is the whole point: someone already counted is
                // not counted again when they come back.
                track.counted = true;
                track.returned = true;
                remembered.remove(known);
            }
            tracks.add(track);
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

        // Both bounds, and the time one is the one that matters. See MAX_COAST_MS.
        // Both bounds, and the time one is the one that matters. See MAX_COAST_MS. A track
        // that is being let go is put into the gallery on its way out.
        for (int i = tracks.size() - 1; i >= 0; i--) {
            Track track = tracks.get(i);
            if (track.missed > MAX_MISSES || now - track.lastSeenAt > MAX_COAST_MS) {
                remember(track, now);
                tracks.remove(i);
            }
        }
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

    /**
     * Is this detection someone already seen and let go?
     *
     * Only ever consulted for a brand new track. A detection that matched an open track is
     * that track, and no amount of colour similarity should overrule a box that is where
     * the last one was.
     */
    private Remembered recognise(Finding detection, long now) {
        if (detection.signature == null) {
            return null;
        }
        Remembered best = null;
        float bestScore = REID_SIMILARITY;
        for (Remembered entry : remembered) {
            if (!entry.label.equals(detection.label)) {
                continue;
            }
            if (now - entry.lastSeen > REID_WINDOW_MS) {
                continue;
            }
            float score = Reid.similarity(entry.signature, detection.signature);
            if (score >= bestScore) {
                bestScore = score;
                best = entry;
            }
        }
        return best;
    }

    /** Put a closing track into the gallery, so it can be recognised later. */
    private void remember(Track track, long now) {
        if (track.signature == null || !track.counted) {
            return;
        }
        remembered.add(new Remembered(track.id, track.label, track.signature, now));
        // Oldest out first: a gallery that grows without limit turns every new detection
        // into a linear scan of the whole flight.
        while (remembered.size() > REID_MAX_REMEMBERED) {
            remembered.remove(0);
        }
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
        remembered.clear();
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
