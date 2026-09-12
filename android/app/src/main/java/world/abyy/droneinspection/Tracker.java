package world.abyy.droneinspection;

import androidx.annotation.NonNull;

import java.util.ArrayList;
import java.util.Arrays;
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

    // Working out how far the whole picture moved between two looks. See estimateDrift.
    // The bin is coarse because the answer only has to be good enough to bring a box back
    // inside the gate above, and a coarse bin is what makes the vote decisive.
    private static final float DRIFT_BIN = 8f;
    // Counted in distinct tracks agreeing, not in pairs. Two tracks that both shifted by the
    // same amount are the camera moving; one track contributing several near-identical
    // offsets because the crowd around it is dense is not evidence of anything.
    private static final int DRIFT_MIN_VOTES = 2;
    private static final double DRIFT_MIN_SHARE = 0.2;
    // A ceiling on one cycle's worth of movement. Past this the frames have nothing to do
    // with each other and pairing them up would be invention rather than tracking.
    private static final float DRIFT_MAX = 400f;

    // Pairing something up when it is the only candidate. See unambiguousPairs.
    //
    // 1.4 comes from the geometry rather than from taste. Two people a gap apart, the camera
    // moving m per frame: from a track's old position its own new box is m away and its
    // neighbour's is gap minus m, so the ratio is (gap - m) / m. It only becomes genuinely
    // ambiguous when the camera has moved half the gap, and a threshold of R resolves
    // everything up to gap/(1 + R). This covers motion up to 42% of the gap between two
    // people, most of the way to the point where no rule could be right.
    private static final double LONELY_RATIO = 1.4;
    // Below every real affinity, so these are only ever used on what is left over.
    private static final double LONELY_SCORE = 1e-4;
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
     * WHY IT IS 1200 AND WAS 4000
     *     4000 was not chosen, it was forced. Under the old shape the detector looked at one
     *     sixth of the frame per cycle, so a person outside the current tile was not observed
     *     for six cycles running and a track that let go after a second and a half would drop
     *     them and renumber them on the next look. The window was stretched until that stopped
     *     happening, and the prose right above it went on saying a second and a half, because
     *     that is what it should be.
     *
     *     Every person is now looked at every cycle, so five missed looks in a row is five
     *     failures to detect somebody being looked straight at - which is a person who left,
     *     not a person waiting their turn. Measured over four crowded frames at the real
     *     cadence, dropping 4000 to 1200 moves the share of drawn boxes that are actually on a
     *     person from 74.7% to 80.8%, for two extra numbers issued across three flights. A
     *     coasting box IS the box sitting in the old place, so this is the same complaint the
     *     tile change answers, met from the other side.
     *
     * Kept in step with MAX_COAST_MS in web/js/track.js.
     */
    private static final long MAX_COAST_MS = 1200;

    /**
     * How recently a track must have been seen to count as being in view now.
     *
     * Holding an identity and being visible are two different questions, and one number was
     * answering both. This has to stay BELOW the coast window or the two questions collapse back
     * into one: a track would stop being reported as present at the same instant it is let go,
     * and the window where somebody is held without being counted as present - the window that
     * lets them keep their number through a couple of missed looks - would be empty.
     *
     * 750 ms is three cycles at the target period. Somebody not detected for three looks running
     * is not in view, whatever is still being held for them.
     */
    private static final long IN_VIEW_MS = 750;

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
    /**
     * Sightings before a track is given a number and added to the total.
     *
     * Raised from two, on evidence. Measured against VisDrone's own labels, about three
     * boxes in ten do not land on a labelled person: street furniture, mostly, which from
     * above is a small dark blob like everything else. At two sightings any of those that
     * survived a second look was issued a number and added to the total, so the total
     * climbed on things that were not people and the numbers on screen churned.
     *
     * It went to three when the detector was looking at one sixth of the frame per cycle,
     * because three sightings then meant three rounds of six tiles, four and a half seconds,
     * and a fourth really did cost people who are only ever seen briefly.
     *
     * The whole frame is looked at every cycle now, so a sighting is a cycle and four of
     * them is one second. Measured over four crowded frames at the real cadence, charging
     * every number to the person it spent the most frames on:
     *
     *     three sightings   49% of the people reached, 1.50 numbers each, 150 on nobody
     *     four              49% reached, 1.46 each, 149 on nobody
     *     five              48% reached, 1.44 each, 148 on nobody
     *
     * Four costs no reach at all and takes a number off about one person in twenty five;
     * five starts costing people. It does not fix the false boxes, which is a limit of the
     * model rather than of the tracking, but it stops them being counted as people.
     *
     * Kept in step with CONFIRM_AFTER in web/js/track.js.
     */
    private static final int CONFIRM_AFTER = 4;

    /**
     * How sure the detector has to be before a box may start a new identity.
     *
     * WHY STARTING AND CONTINUING ARE DIFFERENT QUESTIONS
     *     A weak box where nothing is being tracked is probably a bin. The same weak box
     *     landing where somebody already is, is almost certainly that person, seen badly for
     *     a moment. One number should not answer both.
     *
     *     So a detection below this can keep an existing track alive and can never create
     *     one. That lets the detector run at a low threshold without letting faint rubbish
     *     into the count. Measured against VisDrone's labels, boxes landing on a labelled
     *     person average 0.465 and the rest 0.330: far too overlapping to threshold in one
     *     frame, and separating cleanly once a track has to keep earning it.
     *
     *     This is ByteTrack's association, doing the same thing for the same reason.
     *
     * Measured over nine synthetic flights across a labelled frame, 1388 people between
     * them, the real detector run on every rendered frame so its misses and false positives
     * are all present:
     *
     *     this at 0.25, three sightings    707 people counted, 69% of boxes on a person
     *     this at 0.30, four sightings     533 people counted, 72%
     *     no such rule, four sightings     513 people counted, 72%
     *
     * Adjustable from the settings screen, because which of those is the right mistake
     * depends on the site.
     */
    private static final float NEW_TRACK_CONFIDENCE = 0.25f;

    /** Read by the thread that runs detection, written by the main one. */
    private volatile float newTrackConfidence = NEW_TRACK_CONFIDENCE;
    private static final int MAX_PATH = 60;

    /** One thing being followed. */
    public static final class Track {
        public final int id;

        /**
         * The number an operator reads off the box. Zero until this track is confirmed.
         *
         * NOT the id, and the difference is the whole point. id is spent the moment any box
         * arrives that no existing track wanted, which includes every flicker of gravel,
         * roof vent and shadow that never survives to a second look; it is bookkeeping and
         * it is meant to be thrown away. The overlay used to print it, so one real person
         * standing in a scene with sixty-six discarded flickers was labelled "person 67" and
         * the operator quite reasonably read that as sixty-seven people having been counted.
         *
         * This only ever moves when somebody is confirmed, so the highest number on screen
         * is the number of people counted. Kept in step with number in web/js/track.js.
         */
        public int number;
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
        /**
         * Where this track was last actually seen.
         *
         * Velocity is measured from here and never from box, which by then may have been
         * carried forward by the velocity on every cycle the track was missed. Kept in step
         * with lastObserved in web/js/track.js.
         */
        float[] lastObserved;
        public final List<float[]> path = new ArrayList<>();
        boolean counted;

        /**
         * Drawn, but not yet anybody.
         *
         * A detection too faint to start an identity used to be thrown away, so a person the
         * model was only half sure about had no box at all. It gets one now: the track is
         * created, drawn, and matched to like any other, and it simply cannot be numbered or
         * counted until a detection the tracker would have believed on its own agrees with
         * it. Drawing and counting are two decisions, and only the second one needs to be
         * careful.
         *
         * Measured on four crowded frames: a box on 55.2% of the people in view against
         * 43.4% before, with the repeat numbering moving 1.47 to 1.49.
         *
         * Kept in step with provisional in web/js/track.js.
         */
        boolean provisional;

        Track(int id, String label, float[] box, float confidence, long now) {
            this.id = id;
            this.label = label;
            this.box = box;
            this.confidence = confidence;
            this.seen = 1;
            this.lastSeenAt = now;
            this.lastObserved = centre(box);
            this.path.add(centre(box));
        }

        public boolean coasted() {
            return missed > 0;
        }
    }

    /** People seen and let go, so they are known when they come back. */
    private static final class Remembered {
        final int id;
        /** The number they were given, so they come back as themselves and not as a new one. */
        final int number;
        final String label;
        final float[] signature;
        final long lastSeen;

        Remembered(int id, int number, String label, float[] signature, long lastSeen) {
            this.id = id;
            this.number = number;
            this.label = label;
            this.signature = signature;
            this.lastSeen = lastSeen;
        }
    }

    private final List<Remembered> remembered = new ArrayList<>();
    private final List<Track> tracks = new ArrayList<>();

    /** The last measured movement of the whole picture. See estimateDrift. */
    private float[] lastDrift = {0f, 0f};
    private final Map<String, Integer> everSeen = new HashMap<>();
    private int nextId = 1;

    /** Numbers people actually see, issued at confirmation. See Track.number. */
    private int nextNumber = 1;

    /** Advance one frame. Returns the tracks worth drawing. */
    public List<Track> update(List<Finding> detections) {
        return update(detections, System.currentTimeMillis());
    }

    public List<Track> update(List<Finding> detections, long now) {
        // Before any pairing: how far the whole picture moved. See estimateDrift.
        float[] drift = estimateDrift(detections);
        lastDrift = drift;

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
                double score = affinity(track, box, drift);
                if (score > 0) {
                    pairs.add(new Pair(track, i, score, box));
                }
            }
        }
        // Appended rather than merged: they score below every real affinity, so the greedy
        // pass below only reaches them for a track and a detection nothing else wanted.
        pairs.addAll(unambiguousPairs(detections));
        // Somebody who has been counted always gets first refusal. A provisional track is a
        // box drawn over something that may be nothing, and letting it outbid a real track
        // for a detection - which it will, whenever the real track's box has drifted and the
        // faint one happens to sit closer - takes the observation away from a person, who
        // then misses, coasts, and eventually gets a second number. Measured: without this,
        // boxing the faint ones put the repeat numbering up from 1.47 per person to 1.63.
        // Score decides between equals; being real is not a score.
        pairs.sort(Comparator.comparing((Pair p) -> p.track.provisional)
                .thenComparing(Comparator.comparingDouble((Pair p) -> p.score).reversed()));

        Set<Track> usedTracks = new HashSet<>();
        Set<Integer> usedDetections = new HashSet<>();
        for (Pair pair : pairs) {
            if (usedTracks.contains(pair.track) || usedDetections.contains(pair.index)) {
                continue;
            }
            usedTracks.add(pair.track);
            usedDetections.add(pair.index);

            float[] observedCentre = centre(pair.box);
            // How many cycles since this track was last actually seen, not predicted.
            int coasted = Math.max(1, pair.track.missed + 1);
            pair.track.box = pair.box;
            pair.track.confidence = detections.get(pair.index).confidence;
            pair.track.missed = 0;
            pair.track.seen += 1;
            // A confident look is what turns a drawn box into somebody who can be numbered.
            if (pair.track.provisional
                    && detections.get(pair.index).confidence >= newTrackConfidence) {
                pair.track.provisional = false;
            }
            pair.track.lastSeenAt = now;
            float[] fresh = detections.get(pair.index).signature;
            if (fresh != null) {
                // Kept current rather than frozen at first sight: someone turning around, or
                // the light changing as the drone moves, should update what they look like.
                pair.track.signature = Reid.blend(pair.track.signature, fresh);
            }
            // Smoothed, so one noisy frame does not send the coasting prediction sideways.
            //
            // Measured from where the track was last SEEN, and divided by how many cycles
            // ago that was. It used to be measured against track.box, which on a missed
            // cycle had already been carried forward by this same velocity - so the
            // difference was the prediction's own leftover error rather than how far the
            // person went, and feeding that back in is a loop fighting itself. It
            // oscillates instead of settling whenever a track is looked at less often than
            // every cycle. Kept in step with web/js/track.js.
            float[] last = pair.track.lastObserved == null
                    ? observedCentre : pair.track.lastObserved;
            // A pairing made only because there was nothing else it could be is not
            // evidence about motion, and letting it set velocity is what flings a coasting
            // box across the frame. See unambiguousPairs and LONELY_SCORE.
            if (pair.score > LONELY_SCORE) {
                pair.track.velocity = new float[]{
                        pair.track.velocity[0] * 0.6f
                                + (observedCentre[0] - last[0]) / coasted * 0.4f,
                        pair.track.velocity[1] * 0.6f
                                + (observedCentre[1] - last[1]) / coasted * 0.4f,
                };
            }
            pair.track.lastObserved = observedCentre;
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

            // A box too weak to be somebody NEW. It may keep a person alive through a bad
            // moment and it may not invent one. See NEW_TRACK_CONFIDENCE. It is still drawn:
            // the track is created provisional, which can be seen and cannot be counted.
            boolean faint = detection.confidence > 0 && detection.confidence < newTrackConfidence;

            // Before issuing a new number, ask whether this is someone already known. A
            // track that closed because its subject walked behind something is not a
            // different person when they walk out the other side, and giving them a second
            // number is what turns a count of people into a count of reappearances.
            Remembered known = recognise(detection, now);
            Track track = new Track(known != null ? known.id : nextId++, detection.label,
                    boxOf(detection), detection.confidence, now);
            track.provisional = faint && known == null;
            track.signature = detection.signature;
            if (known != null) {
                if (track.signature == null) {
                    track.signature = known.signature;
                }
                // Carried across, and this is the whole point: someone already counted is
                // not counted again when they come back, and keeps the number they had.
                track.counted = true;
                track.number = known.number;
                track.returned = true;
                remembered.remove(known);
            }
            tracks.add(track);
            usedTracks.add(track);
        }

        // Newly created tracks count as used, because they were: the detection that made each
        // of them was seen on THIS frame. Without this they fall into the loop below and are
        // marked as having missed the very frame they were born on, which is wrong three ways.
        // Their miss count is permanently one too high, so the budget that decides when to let
        // them go is one short; `coasted()` is true from the first frame, so a brand-new box is
        // drawn dimmed and dashed as though it were a guess; and velocity is divided by a
        // coast that never happened.
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
            if (track.seen >= CONFIRM_AFTER && !track.counted && !track.provisional) {
                track.counted = true;
                if (track.number == 0) {
                    track.number = nextNumber++;
                }
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
            if (track.seen >= CONFIRM_AFTER && !track.provisional) {
                confirmed.add(track);
            }
        }
        return confirmed;
    }

    /**
     * Everything worth drawing a box around, which is more than everything worth numbering.
     *
     * WHY THESE ARE TWO QUESTIONS
     *     They used to be one, and it forced a choice nobody should have to make. Drawing
     *     only confirmed tracks left somebody who had just walked into frame with no box at
     *     all for four cycles, about a second, which reads as the detector not seeing them.
     *     Drawing everything the instant it appeared would have put a number on every
     *     flicker, and a number that appears and vanishes is worse than no number.
     *
     *     So: a box as soon as anything is detected, and a number only once it has agreed
     *     with itself. An unconfirmed track comes back with number 0 and OverlayView draws
     *     it without a label. Nothing is hidden from the operator, and nothing unproven is
     *     counted.
     *
     *     A track that is coasting and not yet confirmed is left out: it was seen once, it
     *     has not been seen since, and a box with nothing behind it drifting across the
     *     screen is the thing this whole application is written against.
     *
     * Kept in step with visible() in web/js/track.js.
     */
    public List<Track> visible() {
        List<Track> drawable = new ArrayList<>();
        for (Track track : tracks) {
            if (track.missed == 0 || track.seen >= CONFIRM_AFTER) {
                drawable.add(track);
            }
        }
        return drawable;
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
        remembered.add(new Remembered(track.id, track.number, track.label, track.signature, now));
        // Oldest out first: a gallery that grows without limit turns every new detection
        // into a linear scan of the whole flight.
        while (remembered.size() > REID_MAX_REMEMBERED) {
            remembered.remove(0);
        }
    }

    /** How many of a label are being followed right now. */
    public int countOf(String label) {
        long now = System.currentTimeMillis();
        int n = 0;
        for (Track track : open()) {
            if (track.label.equals(label) && !track.provisional
                    && now - track.lastSeenAt <= IN_VIEW_MS) {
                n += 1;
            }
        }
        return n;
    }

    /** How many distinct ones have been seen since the start. Never goes down. */
    /** How sure a box must be to start a new identity. See NEW_TRACK_CONFIDENCE. */
    public void setNewTrackConfidence(float confidence) {
        newTrackConfidence = confidence;
    }

    public int countSeen(String label) {
        Integer n = everSeen.get(label);
        return n == null ? 0 : n;
    }

    public void reset() {
        tracks.clear();
        remembered.clear();
        everSeen.clear();
        nextId = 1;
        nextNumber = 1;
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

    private static float[] shift(float[] box, float[] by) {
        return new float[]{box[0] + by[0], box[1] + by[1], box[2] + by[0], box[3] + by[1]};
    }

    /**
     * How far the whole picture moved, worked out before anything is matched.
     *
     * WHY THIS IS NOT OPTIONAL
     *     The camera is on a drone. When it moves, every box in the frame moves at once, by
     *     the same amount, and none of the people did anything. From altitude a person is
     *     about ten pixels across, so the centre gate above is worth roughly seventeen
     *     pixels, and a drone in forward flight covers far more than that between two looks
     *     a quarter of a second apart.
     *
     *     So every track fails its gate on the same frame, every one is let go, and the whole
     *     crowd is issued new numbers. Measured on a labelled frame panned under the tracker,
     *     with the dataset's own boxes so the detector could not be blamed: a still camera
     *     numbers 140 people as 140, and the same crowd under a camera moving twenty pixels a
     *     frame comes out as 320.
     *
     *     It cannot be recovered from a track's own velocity, and that is the trap. A track
     *     has to survive a frame to learn how fast it is moving; at these speeds none of them
     *     survive one, so none of them ever learn. The estimate has to come from the boxes
     *     themselves, before any of them are paired up.
     *
     * HOW
     *     Every track against every detection of its class, each pair offering the offset
     *     that would join them, and the offsets voted into coarse bins. A rigid translation
     *     puts one vote per person into the same bin; wrong pairings scatter across all the
     *     others. The winning bin is the drone's own motion, and the median inside it is the
     *     value.
     */
    private float[] estimateDrift(List<Finding> detections) {
        if (tracks.size() < 2 || detections.size() < 2) {
            return new float[]{0f, 0f};
        }

        // Keyed by binned offset. Each bucket keeps the offsets that landed in it and the set
        // of tracks that put them there, and it is the second of those that decides.
        Map<Long, Bucket> votes = new HashMap<>();
        Bucket best = null;
        for (Track track : tracks) {
            float tw = Math.abs(track.box[2] - track.box[0]);
            float th = Math.abs(track.box[3] - track.box[1]);
            if (tw <= 0 || th <= 0) {
                continue;
            }
            float[] from = centre(track.box);

            for (Finding detection : detections) {
                if (!track.label.equals(detection.label)) {
                    continue;
                }
                float[] box = boxOf(detection);
                float dw = Math.abs(box[2] - box[0]);
                float dh = Math.abs(box[3] - box[1]);
                if (dw <= 0 || dh <= 0) {
                    continue;
                }
                // Same reasoning as the size gate in affinity: a box twice the size is a
                // different person, and its offset says nothing about where the picture went.
                double ratio = Math.max(Math.max(tw / dw, dw / tw), Math.max(th / dh, dh / th));
                if (ratio > MAX_SIZE_RATIO) {
                    continue;
                }

                float[] to = centre(box);
                float dx = to[0] - from[0];
                float dy = to[1] - from[1];
                if (Math.abs(dx) > DRIFT_MAX || Math.abs(dy) > DRIFT_MAX) {
                    continue;
                }

                long key = (long) Math.round(dx / DRIFT_BIN) * 100000L
                        + Math.round(dy / DRIFT_BIN);
                Bucket bucket = votes.get(key);
                if (bucket == null) {
                    bucket = new Bucket();
                    votes.put(key, bucket);
                }
                bucket.offsets.add(new float[]{dx, dy});
                bucket.tracks.add(track);
                if (best == null || bucket.tracks.size() > best.tracks.size()) {
                    best = bucket;
                }
            }
        }

        // Enough of the frame has to agree, or this is not a translation, it is coincidence.
        if (best == null || best.tracks.size() < DRIFT_MIN_VOTES
                || best.tracks.size() < tracks.size() * DRIFT_MIN_SHARE) {
            return new float[]{0f, 0f};
        }
        return new float[]{median(best.offsets, 0), median(best.offsets, 1)};
    }

    /** One candidate offset, and which tracks voted for it. */
    private static final class Bucket {
        final List<float[]> offsets = new ArrayList<>();
        final Set<Track> tracks = new HashSet<>();
    }

    /**
     * Pairings that are obvious because there is nothing else they could be.
     *
     * WHY THE GATES ELSEWHERE ARE NOT ENOUGH
     *     The centre-distance gate exists to stop an identity jumping to a *competing*
     *     candidate. With one person in the frame there is no competitor, so it guards
     *     against a risk that is not there and refuses the only sensible answer. Measured,
     *     one person about ten pixels across with the camera moving twenty pixels a frame:
     *     every frame started a fresh track, none survived to be confirmed, and the screen
     *     showed a number that changed constantly or no box at all.
     *
     * WHAT MAKES A PAIRING OBVIOUS
     *     They are each other's nearest, and the runner-up on both sides is far enough
     *     behind to leave no real doubt. That is the ratio test used for matching image
     *     features, and it says the right thing here: distance alone is a poor reason to
     *     refuse a match, but distance relative to the next best candidate is a good one.
     *
     *     In a crowd the runner-up is close, the ratio fails, and this does nothing. In an
     *     empty scene there is no runner-up and it does all of the work.
     */
    private List<Pair> unambiguousPairs(List<Finding> detections) {
        List<Pair> pairs = new ArrayList<>();
        for (int i = 0; i < tracks.size(); i++) {
            Track track = tracks.get(i);
            int pick = nearestDetection(track, detections);
            if (pick < 0) {
                continue;
            }
            // And the same answer looking the other way, so two tracks cannot both claim one
            // detection just because it is the only thing near either of them.
            if (nearestTrack(boxOf(detections.get(pick)), detections.get(pick).label) != i) {
                continue;
            }
            pairs.add(new Pair(track, pick, LONELY_SCORE, boxOf(detections.get(pick))));
        }
        return pairs;
    }

    private int nearestDetection(Track track, List<Finding> detections) {
        float[] from = track.box;
        float fw = Math.abs(from[2] - from[0]);
        float fh = Math.abs(from[3] - from[1]);
        if (fw <= 0 || fh <= 0) {
            return -1;
        }
        float[] at = centre(from);
        int bestIndex = -1;
        double bestAway = Double.MAX_VALUE;
        double second = Double.MAX_VALUE;
        for (int i = 0; i < detections.size(); i++) {
            Finding detection = detections.get(i);
            if (!track.label.equals(detection.label)) {
                continue;
            }
            float[] box = boxOf(detection);
            double away = separation(at, fw, fh, box);
            if (away < 0) {
                continue;
            }
            if (away < bestAway) {
                second = bestAway;
                bestAway = away;
                bestIndex = i;
            } else if (away < second) {
                second = away;
            }
        }
        if (bestIndex < 0 || second < bestAway * LONELY_RATIO) {
            return -1;
        }
        return bestIndex;
    }

    private int nearestTrack(float[] box, String label) {
        float fw = Math.abs(box[2] - box[0]);
        float fh = Math.abs(box[3] - box[1]);
        if (fw <= 0 || fh <= 0) {
            return -1;
        }
        float[] at = centre(box);
        int bestIndex = -1;
        double bestAway = Double.MAX_VALUE;
        double second = Double.MAX_VALUE;
        for (int i = 0; i < tracks.size(); i++) {
            Track track = tracks.get(i);
            if (!track.label.equals(label)) {
                continue;
            }
            double away = separation(at, fw, fh, track.box);
            if (away < 0) {
                continue;
            }
            if (away < bestAway) {
                second = bestAway;
                bestAway = away;
                bestIndex = i;
            } else if (away < second) {
                second = away;
            }
        }
        if (bestIndex < 0 || second < bestAway * LONELY_RATIO) {
            return -1;
        }
        return bestIndex;
    }

    /** Centre distance, or a negative number when these two could not be the same thing. */
    private static double separation(float[] at, float fw, float fh, float[] other) {
        float ow = Math.abs(other[2] - other[0]);
        float oh = Math.abs(other[3] - other[1]);
        if (ow <= 0 || oh <= 0) {
            return -1;
        }
        if (Math.max(Math.max(fw / ow, ow / fw), Math.max(fh / oh, oh / fh)) > MAX_SIZE_RATIO) {
            return -1;
        }
        float[] to = centre(other);
        double dx = to[0] - at[0];
        double dy = to[1] - at[1];
        if (Math.abs(dx) > DRIFT_MAX || Math.abs(dy) > DRIFT_MAX) {
            return -1;
        }
        return Math.hypot(dx, dy);
    }

    private static float median(List<float[]> offsets, int axis) {
        float[] values = new float[offsets.size()];
        for (int i = 0; i < offsets.size(); i++) {
            values[i] = offsets.get(i)[axis];
        }
        Arrays.sort(values);
        return values[values.length / 2];
    }

    /**
     * How strongly a detection belongs to a track. Zero means it does not.
     *
     * Overlap is the better evidence when there is any. When there is none, fall back to
     * how far the centre has moved relative to the size of the thing - with a size check, so
     * a distant person is not adopted by a nearby one's track.
     *
     * Three guesses at where the track should be, and the best of them wins: where it was,
     * where its own velocity says it went, and where the whole picture went. Best-of rather
     * than a sum, because a track alive for a while has already absorbed the drone's motion
     * into its velocity, and adding the drift on top would carry it twice as far.
     */
    private static double affinity(Track track, float[] box, float[] drift) {
        float[] moved = shift(track.box, track.velocity);
        float[] drifted = shift(track.box, drift);

        double overlap = Math.max(iou(track.box, box),
                Math.max(iou(moved, box), iou(drifted, box)));
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

        float[] b = centre(box);
        double reach = Math.max(1, Math.hypot(tw, th) / 2);
        double closest = Double.MAX_VALUE;
        for (float[] guess : new float[][]{track.box, moved, drifted}) {
            float[] p = centre(guess);
            closest = Math.min(closest, Math.hypot(p[0] - b[0], p[1] - b[1]) / reach);
        }
        if (closest > MAX_CENTRE_DRIFT) {
            return 0;
        }
        return 1 - closest / MAX_CENTRE_DRIFT;
    }
}
