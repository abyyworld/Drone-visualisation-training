package world.abyy.droneinspection;

import android.Manifest;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.app.PictureInPictureParams;
import android.content.pm.PackageManager;
import android.content.res.Configuration;
import android.media.MediaScannerConnection;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.os.PowerManager;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.Looper;
import android.os.SystemClock;
import android.util.Rational;
import android.view.Surface;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.TextureView;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;

import androidx.activity.OnBackPressedCallback;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.annotation.OptIn;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.ContextCompat;
import androidx.media3.common.MediaItem;
import androidx.media3.common.PlaybackException;
import androidx.media3.common.Player;
import androidx.media3.common.VideoSize;
import androidx.media3.common.util.UnstableApi;
import androidx.media3.exoplayer.DefaultLoadControl;
import androidx.media3.exoplayer.ExoPlayer;
import androidx.media3.exoplayer.LoadControl;
import androidx.media3.exoplayer.rtsp.RtspMediaSource;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Locale;

/**
 * Live view of what the drone is seeing, with findings drawn over it.
 *
 * THE LINK
 *     The MK15 hands its video out as RTSP over its Ethernet/IP link, not over WiFi and not
 *     as a file (docs/HARDWARE.md). No browser can play RTSP, which is the entire reason
 *     this screen is native Android while the rest of the app is a web page.
 *
 * WHAT "REAL TIME" MEANS HERE, HONESTLY
 *     The video is real time. The boxes are not, and cannot be while the analysis is a
 *     provider API: one round trip is seconds, so frames are sampled on an interval and the
 *     overlay says how old its boxes are. Anything else would draw a stale box over a live
 *     picture and let it be read as current, which on a crowd screen is the worst thing
 *     this app could do. A model running on the device is what makes the boxes live, and
 *     that is what the accumulated inspections are for.
 *
 * THE RECORDING
 *     Boxes burned in, one file. See BoxRecorder for why it is one and why it is not at the
 *     stream's own frame rate.
 */
@OptIn(markerClass = UnstableApi.class)
public class LiveActivity extends AppCompatActivity {

    /** How often a composed frame is handed to the recorder. Matches BoxRecorder.FRAME_RATE. */
    private static final long RECORD_INTERVAL_MS = 1000 / BoxRecorder.FRAME_RATE;
    private static final int RECORD_LONG_EDGE = 1280;

    /**
     * The ceiling on the recorded picture when the GPU path is running.
     *
     * Higher than the fallback's, because there is no readback to pay for: the frame is
     * already on the GPU and the encoder is on the GPU, so recording at the stream's own
     * size costs nothing extra. The cap is only there because H.264 encoders on handheld
     * hardware have real limits, and going past them fails at configure() rather than
     * degrading.
     */
    private static final int RECORD_MAX_EDGE = 1920;

    /** A snapshot is a still to be looked at closely, so it is taken at full resolution. */
    private static final int SNAPSHOT_LONG_EDGE = 4096;

    /** The overlay layer's ceiling. See pushOverlay() for why it is below the video's. */
    private static final int OVERLAY_MAX_EDGE = 1280;

    /**
     * What the frame is scaled to before detection.
     *
     * The model works at 448 px, so handing it a 1080p frame only costs a bigger downscale
     * inside MediaPipe. 640 keeps small subjects resolvable without paying for pixels the
     * model throws away.
     */
    private static final int DETECT_LONG_EDGE = 640;

    /**
     * The size a region is fetched at for its close look.
     *
     * A sixth of a 1920-wide frame is about 750 pixels across, so asking for 640 of it is
     * very nearly one to one and the model sees real detail rather than an upscale.
     *
     * This used to work by reading the whole frame back at 1280 and cropping a sixth out of
     * it in Java, which cost the whole frame's bytes to use a sixth of them and left the
     * crop at 427 pixels of a picture already downscaled once. Two small reads - the whole
     * frame at 640 and one region at 640 - are less than half the bytes and the region is
     * sharper. See GlPipeline.requestRegion.
     */
    /**
     * How big a close look is read back at.
     *
     * The model's own input size, and not a pixel more, so nothing is read off the GPU that
     * the letterbox is only going to throw away again.
     *
     * WHY THE MODEL'S INPUT IS 640 AND NO LONGER 320
     *     The old shape squeezed a sixth of the frame into 320 and ran six of them. The
     *     same weights exported at 640 and given HALF the frame arrive at almost the same
     *     magnification on a person, cost about three times as much per look, and need only
     *     two looks to cover everything - so the milliseconds come out level and the whole
     *     frame is covered every single cycle instead of one sixth of it.
     *
     *     Measured on four crowded VisDrone frames, flying the real model at the real
     *     cadence, counting a number against the person it actually spent its frames on:
     *
     *         320, six tiles, 250 ms   reached 45% of the people, 1.92 numbers each
     *         640, two tiles, 250 ms   reached 60% of the people, 1.69 numbers each
     *
     *     The other three frames agree: 34 to 48, 25 to 36, 39 to 53 percent reached, with
     *     the repeat numbering flat or better in every one. Same weights, same file size,
     *     same milliseconds. The model was never weak; it was being starved of pixels and
     *     then asked to remember what it could not see for five cycles out of six.
     *
     * AND IT COSTS FEWER TRIPS OFF THE GPU, NOT MORE
     *     Worth writing down because the opposite is the obvious guess. A cycle reads 906
     *     thousand pixels now against 578 thousand before, but GlPipeline reads in strips of
     *     at most READ_CHUNK_PIXELS and paces one strip per rendered video frame, so what
     *     costs wall clock is the number of strips and not the number of pixels. A 320 tile
     *     is 320x181 and fills a fraction of one strip; a 640 tile is 640x529 and fills two.
     *
     *         before   whole frame 2 strips + six tiles at 1   =  8 strips, 267 ms at 30 fps
     *         now      whole frame 2 strips + two tiles at 2   =  6 strips, 200 ms
     *
     *     So the readback stopped being the thing that sets how fast a cycle can repeat.
     */
    private static final int TILE_LONG_EDGE = 640;

    /**
     * How much video the player is allowed to hold, in milliseconds.
     *
     * Deliberately tiny. Every millisecond buffered is a millisecond the picture is behind
     * the drone, and ExoPlayer's defaults hold 2.5 seconds before showing a first frame
     * because they are written for films over the internet rather than for a radio link to
     * a camera fifty metres away. Half a second is enough to absorb a dropped packet and
     * short enough that a pilot is looking at now.
     */
    private static final int LIVE_MIN_BUFFER_MS = 100;
    private static final int LIVE_MAX_BUFFER_MS = 500;

    /** Show the first frame as soon as there is one, and never wait after a break. */
    private static final int LIVE_START_MS = 0;
    private static final int LIVE_RESTART_MS = 0;

    /**
     * Floor on the gap between detections, so a fast tablet leaves the decoder some room.
     * The real gap is the length of the last detection, which is longer than this on
     * everything except a very quick device.
     */
    private static final long MIN_DETECT_GAP_MS = 40;

    /**
     * How often a detection cycle is allowed to START, however fast the device is.
     *
     * THE TABLET WAS COOKING, AND THIS IS WHY
     *     The loop used to begin the next cycle the moment the last one finished. On a
     *     controller that is a hundred per cent duty cycle on the CPU, for as long as the
     *     screen is open, and the MK15 is a sealed handheld with no fan. It gets hot, and
     *     then Android throttles it, and the throttled cycles take longer - so the delay
     *     that started as a heat problem looks like a software one and gets worse the
     *     longer you fly.
     *
     *     Five detections a second is not a compromise here. The tracker predicts between
     *     detections and the screen draws at its own rate, so the boxes move smoothly at
     *     any detection rate; what the rate actually decides is how quickly a *new* person
     *     is picked up, and a fifth of a second is faster than anyone can walk into frame
     *     and matter. Running four times that only spends the thermal budget that would
     *     otherwise keep the rate steady for the whole flight.
     */
    private static final long TARGET_PERIOD_MS = 200;

    /**
     * How often the whole frame is looked at, in cycles, when tiling is on.
     *
     * See onWholeFrame for the measurement behind it. Every cycle still reads the frame
     * back, because the flame scan and the appearance signatures both need those pixels;
     * this is only about which cycles pay for an inference on it.
     */
    private static final int WIDE_EVERY = 0;

    /**
     * How many tiles get a close look each cycle.
     *
     * All of them, which is now two rather than six. Nothing coasts: every person in the
     * frame is looked at every cycle, so a box that has not moved is a person who has not
     * moved rather than a box waiting its turn. That is the whole reason the tile count
     * came down and the model's input went up - see TILE_LONG_EDGE for the measurement.
     *
     * One tile per cycle was the original design and it meant a given patch of ground was
     * looked at once every six cycles, over a second at the target period. In between,
     * everybody outside that tile was coasting, which is the box that sits in the wrong
     * place until it jumps.
     */
    private static final int TILES_PER_CYCLE = Tiles.COUNT;

    /**
     * The lowest score a box needs to reach the tracker at all.
     *
     * Deliberately below anything that would be believed on its own. A box this weak cannot
     * start a track; it can only keep one alive. Costs about a millisecond and a half of
     * decoding against a cycle of two hundred, which is the price of the people who are only
     * ever seen faintly.
     */
    private static final float DETECT_FLOOR = 0.15f;

    /**
     * The share of the time detection may occupy. The rest is left for the video, the
     * encoder, and for the device to shed heat.
     */
    private static final float DUTY_CYCLE = 0.6f;

    /**
     * How long a requested frame may be outstanding before detection gives up on it.
     *
     * The GPU path serves a request from the next video frame to arrive. If the stream
     * stalls, no frame arrives, and without this the in-flight flag would stay set and
     * detection would never restart when the picture came back.
     */
    private static final long DETECT_TIMEOUT_MS = 3000;

    private final Handler handler = new Handler(Looper.getMainLooper());

    private SurfaceView videoSurface;
    private TextureView video;
    private OverlayView overlay;

    /**
     * The GPU path, and whether it came up.
     *
     * When it did, the decoder's frames never touch the CPU on their way to the screen or
     * to the encoder, and the model's frames are read back small and off the main thread.
     * When it did not, everything falls back to the TextureView and getBitmap(), which is
     * slower and always works. A field with a drone in the air is the wrong place to find
     * out that a device's GL driver is unusual.
     */
    private GlPipeline glPipeline;

    /**
     * The floating window, when one is out.
     *
     * While it is, the pipeline draws into its surface instead of the one on this screen, and
     * this activity is stopped rather than resumed - so onStop must not tear anything down
     * and the process must be held up. See PopoutWindow for why the system's own
     * picture-in-picture could not be made to do this.
     */
    private PopoutWindow popout;

    /** Whether the pipeline's picture is currently going to the floating window. */
    private boolean popoutOwnsDisplay;
    private boolean usingGl;
    private boolean surfaceReady;
    private boolean wantStream;
    private int videoPixelWidth = 1280;
    private int videoPixelHeight = 720;
    private int recordWidth;
    private int recordHeight;
    private TextView status;
    private Button recordButton;
    private Button snapshotButton;
    private Button backButton;

    private ExoPlayer player;
    private LiveAnalyser analyser;
    private BoxRecorder recorder;

    /**
     * Everything expensive runs here, and nothing expensive runs on the main thread.
     *
     * THIS IS THE WHOLE REASON THE RECORDING WAS JERKY
     *     Inference took a fifth of a second and ran on the main thread. The recorder is
     *     driven by a 66 ms tick on that same thread, so it simply could not fire while the
     *     model was thinking, and the encoder stamps frames with the wall clock: a gap in
     *     submitted frames is a frozen picture for exactly that long, not a dropped frame
     *     nobody notices. Between those, the provider's frame was JPEG compressed and base64
     *     encoded on the main thread every two seconds, which is a much longer stall and
     *     exactly the "frozen moment every other second" it looked like.
     *
     *     So the model, the flame scan and the tracker all live on this thread now. The main
     *     thread grabs frames, draws, and gets out of the way.
     */
    private HandlerThread workThread;
    private Handler work;

    /** Written on the work thread, read on the main one, hence volatile throughout. */
    private volatile NativeDetector detector;

    /**
     * The fire and smoke model, opened only for a wildfire flight.
     *
     * Null on any other flight, and null on a build with no fire model in it, which is not
     * a fault: FireScan needs no model and is what this app did for fire before there was
     * one. Both run in wildfire mode, because they fail differently. The model was measured
     * finding 50 of 120 published fire pictures and marking none of 150 real drone frames
     * (docs/metrics-wildfire.txt), so more than half the time it is the scan or nothing.
     */
    private volatile NativeDetector fireDetector;

    /**
     * How often the fire model gets a cycle, when one is loaded.
     *
     * Every one, and this used to be every other one on a reason that did not survive being
     * measured.
     *
     * WHAT THE OLD REASONING GOT WRONG
     *     It said alternating "costs each of them half the rate", and that a wildfire flight
     *     still has people in it so the people model cannot be given up. Neither is what the
     *     code does: the fire pass runs IN ADDITION to the person pass, never instead of it,
     *     so the person model was already running every cycle and alternating only ever
     *     halved the rate of the fire model. The trade being described was not the trade
     *     being made.
     *
     *     And measured, the thing being saved is small. On the same machine, one look each:
     *
     *         person, two tiles at 640    153 ms   (76.5 ms a tile)
     *         fire, whole frame at 320     25 ms
     *
     *     So alternating saved about 12 ms on a cycle of 166, under 8 percent, and bought it
     *     by halving how often the tablet looks for fire on a flight whose entire purpose is
     *     looking for fire - with a model that already misses more than half of what it is
     *     shown (docs/metrics-wildfire.txt). Eight percent of cycle time is the wrong thing
     *     to protect there, and the tracker absorbs the longer cycle by itself now, because
     *     its windows follow the cadence the device achieves rather than a fixed figure.
     *
     *     Memory was the other half of the old argument and it is not a constraint either.
     *     Both models held open at once measured 124 MB of a two gigabyte device: 85 MB for
     *     the person model and 39 MB for the fire one.
     *
     * AND BACK TO EVERY OTHER ONE, ON BETTER INFORMATION
     *     That change was right about the cost and wrong about what to spend it on. Moving
     *     the fire model to 640 turned out to be worth far more than looking twice as often,
     *     and it costs four times as much per look, so both together do not fit:
     *
     *         320, every cycle          25 ms a cycle
     *         640, every other cycle    50 ms a cycle
     *         640, every cycle         100 ms a cycle, on a person pass of 153
     *
     *     Measured on 311 labelled pictures, the same weights at 640 find 36 of 120 distant
     *     plumes where 320 finds 27, at the same false alarm rate, and beat the shipped
     *     setting on every source at once. See docs/metrics-wildfire.txt.
     *
     *     Halving the rate costs almost nothing HERE, and that is the part worth writing
     *     down, because it is not true of people. A person crosses a frame in a couple of
     *     seconds and a box that is a cycle late is a box in the wrong place. A plume stands
     *     in the same valley for minutes: looking every 500 ms instead of every 250 ms does
     *     not miss a fire, it notices it a quarter of a second later. Resolution is worth
     *     buying with rate for a slow thing, and it is not for a fast one.
     */
    private static final int FIRE_EVERY = 2;
    /**
     * Why the detector is not running, kept APART from lastError.
     *
     * lastError is shared with the stream, and the stream is on a radio link that drops
     * packets as a matter of course. A model that failed to load wrote its reason there and
     * the first RTSP hiccup overwrote it, leaving an operator with live video, no boxes, and
     * a stale message about the stream - which reads as "nobody in frame", the one
     * conclusion this application must never invite.
     */
    private volatile String detectorFailure = "";

    /**
     * Consecutive close looks that have thrown.
     *
     * One is a lost look and does not matter. Two in a row is a detector that is not working,
     * and in crowd mode there is no other path that would ever notice. See onRegion.
     */
    private int tileFailures;
    private static final int TILE_FAILURES_BEFORE_GIVING_UP = 2;

    /** A previous recording is still writing its index. See stopRecording. */
    private boolean finishing;

    private volatile boolean detectBusy;
    private volatile long detectRequestedAt;
    /** When the current cycle began, and the earliest the next one may. See TARGET_PERIOD_MS. */
    private volatile long cycleStartedAt;
    private volatile long nextCycleAt;

    /** The cycle time this device is actually managing, smoothed. See Tracker.setCadence. */
    private volatile long smoothedCycleMs;
    /**
     * A multiplier on the cycle period, raised when Android says the device is too hot.
     * One while it is comfortable; larger while it is not.
     */
    private volatile float thermalEase = 1f;
    private volatile boolean hot;
    private PowerManager.OnThermalStatusChangedListener thermalListener;
    private Button pipButton;
    /** Read on the work thread, set on the main one when the subject changes. */
    private volatile boolean scansForFire;
    /** Reused pixel buffer for the appearance signatures, so each frame is one allocation. */
    private int[] signatureScratch;
    /** Which tile gets the close look this pass. Cycles; see Tiles. */
    private int tileTurn;
    /** Counts detect cycles, so the wide pass can take every third one. See WIDE_EVERY. */
    private int cycleTurn;
    private volatile int peopleInView;
    private volatile int peopleSeen;
    private volatile long detectionsRun;

    /**
     * Only ever touched on the work thread. The tracker carries state between frames and
     * the counts are read off it, so letting the main thread ask it questions while the
     * work thread is updating it would be a race for the sake of a status line.
     */
    private final Tracker tracker = new Tracker();
    private final FireScan fireScan = new FireScan();

    private List<Finding> findings = new ArrayList<>();
    private String lastError = "";

    /**
     * The tally, written to a file beside the recording while one is running.
     *
     * Tied to recording rather than to the screen being open, so that starting a recording
     * is the one deliberate act that says "keep this flight". A live view left running on a
     * bench does not quietly fill the tablet.
     */
    private volatile FlightLog flightLog;

    /**
     * Where the last thing written went, kept on the status line.
     *
     * A toast is gone in four seconds and a path is the one part of it worth reading twice.
     * Telling an operator to "analyse it from the menu" answered a question nobody asked:
     * the question is where the file is, so that it can be copied off the tablet.
     */
    private String lastSaved = "";

    /**
     * Permission to put a flight where it can be found.
     *
     * Nothing depends on the answer. Refused, the files go to the app's own directory as
     * before and are still there for the analysis screen; they are simply harder to reach
     * with a file manager. A refusal must never stop somebody flying.
     */
    private final ActivityResultLauncher<String> storagePermission = registerForActivityResult(
            new ActivityResultContracts.RequestPermission(), granted -> refreshStatus());

    /** Save an annotated still each time the total passes another of these. */
    private static final int EVIDENCE_EVERY = 10;

    /**
     * And no more than this many per flight.
     *
     * A busy square could otherwise write a full resolution JPEG every few seconds for
     * twenty minutes, which is the sort of thing that fills a tablet during the one flight
     * you needed it for.
     */
    private static final int EVIDENCE_LIMIT = 20;

    private int evidenceSaved;
    private int evidenceAt;
    private long framesDropped;
    private volatile List<FireScan.Region> fireRegions = new ArrayList<>();
    private long detectStartedAt;

    /**
     * The live loop: grab a frame on the main thread, think about it on another.
     *
     * The tick itself does almost nothing: it reads one frame off the video surface and
     * hands it to the work thread. It never waits for the answer, and it never starts a
     * second frame while the first is still being looked at, so a slow device falls to a
     * lower detection rate rather than building a backlog it can never clear.
     *
     * The interval is fixed and short. It used to be "however long the last inference
     * took", which made sense when inference was on this thread and had to be paid for
     * here. It is not paid for here any more, so the only thing that matters is asking
     * often enough to catch the work thread the moment it goes idle.
     */
    private final Runnable detectTick = new Runnable() {
        @Override
        public void run() {
            long now = SystemClock.uptimeMillis();
            if (!detectBusy && now >= nextCycleAt) {
                detectBusy = true;
                cycleStartedAt = now;
                detectRequestedAt = System.currentTimeMillis();
                // The whole frame, small. This is the pass that keeps every track alive,
                // and it is deliberately cheap: it only has to find what is large enough to
                // survive a downscale.
                requestFrame(DETECT_LONG_EDGE, frame -> onWholeFrame(frame));
                handler.postDelayed(releaseDetect, DETECT_TIMEOUT_MS);
            }
            handler.postDelayed(this, MIN_DETECT_GAP_MS);
        }
    };

    /**
     * The whole frame has arrived. Detect in it, then ask for one region up close.
     *
     * Called on the pipeline's thread. The detection is handed to the work thread and the
     * region is asked for straight away, so the two arrive a video frame or two apart -
     * close enough that a distant person, which is what the region pass is for, has moved a
     * pixel or so between them.
     */
    private void onWholeFrame(Bitmap frame) {
        Handler worker = work;
        if (worker == null || !worker.post(() -> {
            NativeDetector current = detector;
            // One read of a field another thread can null out at any moment. Reading it
            // twice is a null check that was true and a call that is not.
            GlPipeline pipeline = glPipeline;
            boolean tiling = tilingWanted() && pipeline != null && usingGl;

            // The wide pass, only when it is worth its inference.
            //
            // WHY IT IS NOT RUN EVERY CYCLE ANY MORE
            //     Measured against VisDrone's labels over twelve of its busiest frames,
            //     1562 labelled people: the wide pass alone lands on 163 of them, and with
            //     the close looks merged in that becomes 730. A tenth against a half.
            //
            //     It is easy to see why. A 1920 frame is read back at 640 and then squeezed
            //     into the model's 320, a sixfold reduction, so from altitude there is
            //     almost nobody left in it big enough to find. It still cost a full
            //     inference every cycle, which was half the budget spent for a tenth of the
            //     result, and that inference is what set how fast the cycle could repeat.
            //
            //     So when tiling is on it runs every third cycle. The cycles in between are
            //     a close look and nothing else, which makes them cheaper, which means more
            //     of them a second, which brings each tile round again sooner. That gap was
            //     what forced the coast window up to four seconds.
            //
            //     Not skipped when tiling is off: for a turbine or a panel, which fill the
            //     frame, the wide pass is the entire job.
            // WIDE_EVERY of 0 means never, and must not reach the modulo: `% 0` throws.
            boolean wideNow = !tiling || (WIDE_EVERY > 0 && cycleTurn % WIDE_EVERY == 0);
            List<Finding> found = current == null || !wideNow
                    ? new ArrayList<>() : safeDetect(current, frame);
            cycleTurn++;

            if (!tiling) {
                // Kept until here: the flame scan and the appearance signatures both need
                // the pixels, and finishCycle is what recycles them.
                finishCycle(found, frame);
                return;
            }
            // Every tile, this cycle, rather than one of six and the rest next time.
            // See TILES_PER_CYCLE for what that was costing.
            Cycle cycle = new Cycle(found, TILES_PER_CYCLE);
            for (int i = 0; i < TILES_PER_CYCLE; i++) {
                // A region of the frame, rendered at its own resolution rather than cropped
                // out of a big readback. See GlPipeline.requestRegion.
                float[] region = Tiles.region(tileTurn++, videoPixelWidth, videoPixelHeight);
                pipeline.requestRegion(
                        region[0] / videoPixelWidth, region[1] / videoPixelHeight,
                        (region[0] + region[2]) / videoPixelWidth,
                        (region[1] + region[3]) / videoPixelHeight,
                        TILE_LONG_EDGE,
                        tile -> onRegion(tile, region, cycle, frame));
            }
        })) {
            frame.recycle();
            detectBusy = false;
        }
    }

    /**
     * What one detect cycle has gathered so far, across its close looks.
     *
     * Only ever touched on the work thread, which is where every readback callback is
     * posted, so the counting needs no locking.
     */
    private static final class Cycle {
        List<Finding> found;
        int outstanding;

        Cycle(List<Finding> found, int outstanding) {
            this.found = found;
            this.outstanding = outstanding;
        }
    }

    /** One close look has arrived. Merge it, and finish the cycle once they all have. */
    private void onRegion(Bitmap tile, float[] region, Cycle cycle, Bitmap frame) {
        Handler worker = work;
        if (worker == null || !worker.post(() -> {
            NativeDetector current = detector;
            if (current != null) {
                try {
                    cycle.found = Tiles.merge(cycle.found,
                            current.detectRegion(tile, inFrame(region, frame)));
                    tileFailures = 0;
                } catch (RuntimeException | Error failure) {
                    // NOT silent, and this catch is why.
                    //
                    // One lost look is nothing: the other tile still stands and the tracker
                    // coasts through a gap of one cycle without noticing. A detector that
                    // throws EVERY cycle is the opposite - it is this application's worst
                    // failure, because an empty list is indistinguishable from a frame with
                    // nobody in it.
                    //
                    // In crowd mode every inference comes through here. WIDE_EVERY is 0 and
                    // tiling is on, so the wide pass never runs and safeDetect - the only
                    // path that ever reported a dead interpreter - is never reached. A
                    // delegate fault or a native allocation failure in the 640 graph would
                    // have left the operator with live video, no boxes, a detection rate
                    // still counting up and a millisecond figure frozen at the last good
                    // value. Nothing on the screen would have changed.
                    if (++tileFailures >= TILE_FAILURES_BEFORE_GIVING_UP) {
                        detectorFailure = getString(R.string.detector_stopped,
                                String.valueOf(failure.getMessage()));
                        current.close();
                        detector = null;
                        tileFailures = 0;
                        handler.post(this::refreshStatus);
                    }
                }
            }
            tile.recycle();
            if (--cycle.outstanding <= 0) {
                finishCycle(cycle.found, frame);
            }
        })) {
            tile.recycle();
            // The frame belongs to the whole cycle, so it is only let go when the last of
            // its looks has failed. Recycling it on the first would pull it out from under
            // the others.
            if (--cycle.outstanding <= 0) {
                frame.recycle();
                detectBusy = false;
            }
        }
    }

    /**
     * The region rectangle, moved out of the video's pixels and into the analysed frame's.
     *
     * These are two different sizes and it matters. Tiles.region works in the video's own
     * pixels, because that is what it is dividing up, while the whole-frame pass reads back
     * at DETECT_LONG_EDGE and its findings are in that smaller picture's pixels. The two
     * lists are then merged and drawn against the analysed frame.
     *
     * Handing the region across unconverted put every close look three times too far out,
     * so it fell outside the frame and was dropped - which meant the tiling pass, the whole
     * point of which is the person too small to see in the wide shot, silently contributed
     * nothing at all. That is the distant person this application exists to catch.
     */
    private float[] inFrame(float[] region, Bitmap frame) {
        float toX = frame.getWidth() / (float) Math.max(1, videoPixelWidth);
        float toY = frame.getHeight() / (float) Math.max(1, videoPixelHeight);
        return new float[]{region[0] * toX, region[1] * toY, region[2] * toX, region[3] * toY};
    }

    private List<Finding> safeDetect(NativeDetector current, Bitmap frame) {
        try {
            return current.detect(frame);
        } catch (RuntimeException | Error failure) {
            detectorFailure = getString(R.string.detector_stopped,
                    String.valueOf(failure.getMessage()));
            current.close();
            detector = null;
            handler.post(this::refreshStatus);
            return new ArrayList<>();
        }
    }

    /**
     * Is there anything small enough in this subject to be worth a close look?
     *
     * A crowd is people who are a handful of pixels each, and a fire front has people at it.
     * A turbine or a panel fills the frame, and detecting a sixth of it at high resolution
     * finds nothing the whole frame did not.
     */
    private boolean tilingWanted() {
        String domain = Settings.domain(this);
        return "crowd".equals(domain) || "wildfire".equals(domain);
    }

    /** Undo a detection that was asked for and never arrived. See detectTick. */
    private final Runnable releaseDetect = new Runnable() {
        @Override
        public void run() {
            if (detectBusy && System.currentTimeMillis() - detectRequestedAt >= DETECT_TIMEOUT_MS) {
                detectBusy = false;
            }
        }
    };

    /**
     * Attach a colour signature to every person found.
     *
     * One read of the frame's pixels serves every box in it. Never allowed to throw:
     * without a signature someone is still tracked and still counted, they are just counted
     * again if they leave and come back, which is a worse number rather than no number.
     */
    private void signPeople(List<Finding> found, Bitmap frame) {
        boolean anyone = false;
        for (Finding finding : found) {
            if ("person".equals(finding.label)) {
                anyone = true;
                break;
            }
        }
        if (!anyone) {
            return;
        }
        try {
            int width = frame.getWidth();
            int height = frame.getHeight();
            if (signatureScratch == null || signatureScratch.length < width * height) {
                signatureScratch = new int[width * height];
            }
            frame.getPixels(signatureScratch, 0, width, 0, 0, width, height);
            for (Finding finding : found) {
                if (!"person".equals(finding.label)) {
                    continue;
                }
                // Already in this frame's pixels: MediaPipe's bounding boxes are, and the
                // tracker consumes them unscaled. Only the provider's findings arrive
                // normalised, and they never come through here.
                finding.signature = Reid.describe(signatureScratch, width, height, new float[]{
                        finding.x0, finding.y0, finding.x1, finding.y1,
                });
            }
        } catch (RuntimeException | OutOfMemoryError ignored) {
            // Tracked but not recognisable, which the tracker already handles.
        }
    }

    /**
     * Everything after detection: tracking, the flame scan, the signatures, and the boxes.
     *
     * On the work thread. The tracker is updated here and its counts read here, so the main
     * thread never asks it anything; what crosses back is a finished list of boxes and two
     * integers. Takes ownership of the frame and recycles it.
     */
    private void finishCycle(List<Finding> found, Bitmap frame) {
        long now = System.currentTimeMillis();
        int frameWidth = frame != null ? frame.getWidth() : videoPixelWidth;
        int frameHeight = frame != null ? frame.getHeight() : videoPixelHeight;

        // A colour signature per person, so someone who leaves the frame and comes back is
        // recognised rather than counted a second time. See Reid.
        if (frame != null) {
            signPeople(found, frame);
        }

        List<Tracker.Track> tracks = tracker.update(found, now);
        detectionsRun++;

        List<FireScan.Region> fire = new ArrayList<>();
        // Only when the operator says they are looking for fire. The scan looks for a flat,
        // desaturated region that loses its texture as things move across it: outdoors that
        // is smoke, indoors it is a painted wall with someone walking past. The pixels
        // cannot tell those apart; the person holding the controller can.
        if (scansForFire && frame != null) {
            try {
                List<float[]> occluders = new ArrayList<>();
                for (Tracker.Track track : tracks) {
                    occluders.add(new float[]{
                            track.box[0] / frameWidth, track.box[1] / frameHeight,
                            track.box[2] / frameWidth, track.box[3] / frameHeight,
                    });
                }
                fire = fireScan.scan(frame, occluders);
            } catch (RuntimeException | OutOfMemoryError ignored) {
                fire = new ArrayList<>();
            }

            // And the trained model, on the cycles it gets, added to what the scan found
            // rather than replacing it. Neither is a superset of the other: the model knows
            // what fire looks like and misses more than half of it, the scan knows what
            // fire does over time and cannot tell a plume from a painted wall. Two engines
            // marking the same fire twice is a smaller problem than one of them missing it.
            NativeDetector fireModel = fireDetector;
            if (fireModel != null && detectionsRun % FIRE_EVERY == 0) {
                try {
                    for (Finding marked : fireModel.detect(frame)) {
                        fire.add(new FireScan.Region(marked.label, marked.confidence,
                                marked.x0, marked.y0, marked.x1, marked.y1, false));
                    }
                } catch (RuntimeException | OutOfMemoryError ignored) {
                    // The scan's regions still stand; only this cycle's model pass is lost.
                }
            }
        }
        if (frame != null) {
            frame.recycle();
        }

        peopleInView = tracker.countOf("person");
        peopleSeen = tracker.countSeen("person");
        fireRegions = fire;

        // Written from this thread on purpose. It is file I/O, and the main thread is the
        // one drawing the video.
        FlightLog log = flightLog;
        if (log != null) {
            log.record(peopleInView, peopleSeen);
            if (peopleSeen >= evidenceAt + EVIDENCE_EVERY && evidenceSaved < EVIDENCE_LIMIT) {
                evidenceAt = peopleSeen - (peopleSeen % EVIDENCE_EVERY);
                evidenceSaved++;
                int at = peopleSeen;
                handler.post(() -> {
                    File still = new File(outputDirectory(),
                            "flight-" + timestamp() + "-" + at + "-people.jpg");
                    saveFrame(still, false);
                });
            }
        }

        // What gets DRAWN is wider than what gets counted. update() returns the confirmed
        // tracks, which is the right list for the fire scan's occluders above and for the
        // running total; the overlay gets visible(), which also includes somebody detected
        // on this very cycle and not yet proven. They are boxed immediately and numbered
        // only once they have earned it. See Tracker.visible().
        final List<Tracker.Track> finalTracks = tracker.visible();
        final List<FireScan.Region> finalFire = fire;
        // When the next cycle may start: never sooner than the target period, and never
        // sooner than the duty cycle allows given what this one actually cost. A device
        // that has been throttled to half speed therefore runs at half the rate rather than
        // at a hundred per cent of a slower CPU, which is how it climbs back out.
        long took = SystemClock.uptimeMillis() - cycleStartedAt;
        long period = Math.round(TARGET_PERIOD_MS * thermalEase);
        long wait = Math.max(0, Math.max(period, Math.round(took / DUTY_CYCLE)) - took);
        nextCycleAt = SystemClock.uptimeMillis() + wait;

        // Tell the tracker how fast this device is really going.
        //
        // Its coast window is three looks and its in-view window two, and both were written
        // down in milliseconds against a 250 ms cycle - which is what a desktop CPU managed
        // while the numbers were being tuned, not what this tablet does. On a controller
        // running at 600 ms a cycle, an 800 ms coast is barely one look: everybody would be
        // let go between consecutive looks at them and renumbered on the next, which is the
        // exact failure those numbers were tuned to avoid.
        //
        // Smoothed, because one slow cycle is a hiccup and the windows should follow the
        // trend rather than the noise. Thermal throttling moves it too, which is right: a
        // tablet that has slowed down is one where a person waits longer between looks.
        long cycle = took + wait;
        smoothedCycleMs = smoothedCycleMs <= 0 ? cycle
                : Math.round(smoothedCycleMs * 0.8 + cycle * 0.2);
        tracker.setCadence(smoothedCycleMs);

        handler.post(() -> {
            overlay.setTracks(finalTracks, frameWidth, frameHeight);
            overlay.setFire(finalFire);
            PopoutWindow window = popout;
            if (window != null) {
                window.overlay().setTracks(finalTracks, frameWidth, frameHeight);
                window.overlay().setFire(finalFire);
            }
            detectBusy = false;
            // The recording's overlay is a texture, re-uploaded only when the boxes change.
            pushOverlay();
            refreshStatus();
        });
    }

    private final Runnable analysisTick = new Runnable() {
        @Override
        public void run() {
            if (providerWorthRunning()) {
                captureAndAnalyse();
            }
            handler.postDelayed(this, Math.max(1, Settings.intervalSeconds(LiveActivity.this)) * 1000L);
        }
    };

    /**
     * The recorder's clock, on a fixed grid rather than a delay after the last frame.
     *
     * postDelayed(66) means 66 ms *after this tick finished*, so the grab and the overlay
     * draw are added to every interval and the recording runs slower than the frame rate it
     * is stamped with. postAtTime puts each frame on an absolute grid, so a tick that runs
     * late is followed by one that runs on time instead of pushing the whole run late.
     */
    private long nextRecordAt;

    private final Runnable recordTick = new Runnable() {
        @Override
        public void run() {
            captureForRecording();
            long now = SystemClock.uptimeMillis();
            nextRecordAt += RECORD_INTERVAL_MS;
            if (nextRecordAt <= now) {
                // Fallen behind by a whole frame or more: give up on catching up rather
                // than submitting a burst, which would only make the picture jump.
                nextRecordAt = now + RECORD_INTERVAL_MS;
            }
            handler.postAtTime(this, nextRecordAt);
        }
    };

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_live);

        // A controller screen that sleeps mid-flight is worse than useless.
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        watchTemperature();

        // Asked for once, up front, rather than at the moment someone presses record.
        //
        // Refusing it does not stop a recording: the service still runs and the file is
        // still written. What is lost is the line in the notification shade saying so, and
        // an app that records with nothing anywhere to show it is the thing that rule
        // exists to prevent - so it is worth asking before it matters.
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU
                && checkSelfPermission(android.Manifest.permission.POST_NOTIFICATIONS)
                    != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{android.Manifest.permission.POST_NOTIFICATIONS}, 1);
        }

        videoSurface = findViewById(R.id.video_surface);
        video = findViewById(R.id.video);
        overlay = findViewById(R.id.overlay);

        videoSurface.getHolder().addCallback(new SurfaceHolder.Callback() {
            @Override
            public void surfaceCreated(@NonNull SurfaceHolder holder) {
                if (popout != null) {
                    // The floating window has the picture, or is about to. Taking it back
                    // here would blank the window the operator is actually watching.
                    return;
                }
                if (glPipeline != null && glPipeline.isAlive()) {
                    // Coming back from the background with a recording still running: the
                    // pipeline never went away, it just had no screen.
                    glPipeline.attachDisplay(holder.getSurface(),
                            videoSurface.getWidth(), videoSurface.getHeight());
                    surfaceReady = true;
                    refreshStatus();
                    return;
                }
                openPipeline(holder.getSurface());
            }

            @Override
            public void surfaceChanged(@NonNull SurfaceHolder holder, int format, int w, int h) {
                if (popout != null) {
                    return;
                }
                if (glPipeline != null) {
                    glPipeline.setDisplaySize(w, h);
                }
            }

            @Override
            public void surfaceDestroyed(@NonNull SurfaceHolder holder) {
                if (popout != null) {
                    // This screen's surface is going because the app left the foreground,
                    // which is the normal case with the window out.
                    //
                    // Tested on the window rather than on whether it already owns the
                    // picture, and that distinction is the whole bug this avoids: the
                    // window's own surface is created a frame or two after the window is
                    // added, so this usually runs while the window owns nothing yet. Falling
                    // through would close the pipeline outright, and the window would come
                    // up black with nothing left able to fill it.
                    if (!popoutOwnsDisplay && glPipeline != null && glPipeline.isAlive()) {
                        glPipeline.detachDisplay();
                    }
                    surfaceReady = false;
                    return;
                }
                if (isRecording() && glPipeline != null) {
                    // Only the window goes. The context, the decoder's texture and the
                    // encoder's surface all stay, so the recording carries on with nobody
                    // watching it - which is the whole point of recording during a flight.
                    glPipeline.detachDisplay();
                    surfaceReady = false;
                    return;
                }
                closePipeline();
            }
        });
        status = findViewById(R.id.status);
        recordButton = findViewById(R.id.record);
        snapshotButton = findViewById(R.id.snapshot);
        backButton = findViewById(R.id.back);

        pipButton = findViewById(R.id.pip);
        pipButton.setOnClickListener(v -> enterSmallWindow(true));
        recordButton.setOnClickListener(v -> toggleRecording());
        snapshotButton.setOnClickListener(v -> takeSnapshot());
        backButton.setOnClickListener(v -> leave());
        findViewById(R.id.settings).setOnClickListener(v -> startActivity(
                new Intent(this, LiveSettingsActivity.class)));

        // Leaving mid-recording would abandon a half-written file. The button is hidden and
        // the system back gesture is swallowed until the recording is stopped, which is also
        // why stopping is the one thing the screen always lets you do.
        getOnBackPressedDispatcher().addCallback(this, new OnBackPressedCallback(true) {
            @Override
            public void handleOnBackPressed() {
                if (isRecording()) {
                    Toast.makeText(LiveActivity.this, R.string.stop_recording_first,
                            Toast.LENGTH_SHORT).show();
                    return;
                }
                setEnabled(false);
                getOnBackPressedDispatcher().onBackPressed();
            }
        });

        // The detector that ships with the app. It runs on every frame it can manage and
        // is what makes this screen live rather than a slideshow of provider answers.
        // Asked for once, on opening the screen, so that the first recording lands somewhere
        // findable rather than discovering the problem after the flight.
        if (Build.VERSION.SDK_INT <= Build.VERSION_CODES.P && !canWriteSharedStorage()) {
            storagePermission.launch(Manifest.permission.WRITE_EXTERNAL_STORAGE);
        }

        StringBuilder failure = new StringBuilder();
        detector = NativeDetector.open(this, failure);
        detectorFailure = detector == null ? failure.toString() : "";

        // Built only when it will be used. A WebView is tens of megabytes of a
        // two-gigabyte device, and in crowd mode with no key it does nothing whatsoever.
        if (!providerWorthRunning()) {
            analyser = null;
        } else {
        analyser = new LiveAnalyser(this, new LiveAnalyser.Listener() {
            @Override
            public void onReady() {
                refreshStatus();
            }

            @Override
            public void onFindings(String asset, String overall, List<Finding> next) {
                findings = next;
                lastError = "";
                overlay.setFindings(next);
                if (popout != null) {
                    popout.overlay().setFindings(next);
                }
                pushOverlay();
                refreshStatus();
            }

            @Override
            public void onError(String message) {
                lastError = message;
                refreshStatus();
            }
        });
        }
    }

    @Override
    protected void onStart() {
        super.onStart();
        // The stream cannot open until it is known which surface it is opening onto, and
        // that is not known until the SurfaceView has one. Whichever happens second starts
        // it; see openPipeline().
        wantStream = true;
        // Before any of the early returns below.
        //
        // This block used to sit at the bottom of onStart, after the return for a running
        // recording and the one for the floating window. Both of those are the normal way
        // back from the settings screen mid-flight, so the sensitivity slider, the
        // people-only switch and the domain were all read only when nothing was going on -
        // which is the one time nobody is adjusting them. An operator who moved the slider
        // while recording saw no change at all and concluded the detector was broken.
        applySettings();
        if (popout != null) {
            // Back on screen with the floating window out: nothing was ever stopped, so take
            // the picture back and leave the stream, the detector and any recording alone.
            closePopout(true);
            refreshStatus();
            return;
        }
        if (isRecording() && player != null) {
            // Came back to a recording that never stopped. Everything is still running.
            refreshStatus();
            return;
        }
        if (surfaceReady) {
            openStream();
        }

        // ONE work thread, however many times this path is entered.
        //
        // onStop returns early while a pop-out is open, leaving the thread and the detect
        // tick running, which is what keeps the boxes alive behind another app. onStart's
        // early returns test different conditions, and the pair can diverge: close the
        // floating window with its own X from inside the flight software and the listener
        // sets popout = null while this activity is still stopped. Coming back then falls
        // through to here with the first thread still running.
        //
        // Building a second one over it did three things. The first was never quit, so a
        // thread leaked on every pop-out cycle for the life of the process. The new thread's
        // first act is tracker.reset(), which clears the track list while the old thread may
        // still be inside tracker.update() - the tracker's whole contract is that one thread
        // touches it. And the running total silently restarted at zero on a stream that was
        // never torn down, so a crowd counted for twenty minutes went back to nothing
        // because somebody closed a window.
        if (workThread == null) {
            workThread = new HandlerThread("live-work");
            workThread.start();
            work = new Handler(workThread.getLooper());
        }
        // Reset on the thread that owns them, not from here. Both carry state between
        // frames, and clearing that state underneath a detection still in flight is the
        // kind of race that shows up once a month and never in a test.
        work.post(() -> {
            tracker.reset();
            // The flame scanner measures how a region changes over a window of frames.
            // Carrying one stream's window into the next would compare a frame against
            // something filmed somewhere else.
            fireScan.reset();
        });
        fireRegions = new ArrayList<>();
        peopleInView = 0;
        peopleSeen = 0;
        detectBusy = false;
        detectionsRun = 0;
        detectStartedAt = System.currentTimeMillis();
        // removeCallbacks first: the tick reposts itself, so entering here twice would leave
        // two chains polling. The body is gated on detectBusy so it would not double the
        // inference, but two chains that never end is not a thing to leave running.
        handler.removeCallbacks(detectTick);
        handler.removeCallbacks(analysisTick);
        handler.post(detectTick);
        // The provider still runs, on its slow interval, for what the on-device model
        // cannot see: fire, smoke, blade damage, soiling. None of those are COCO classes.
        handler.post(analysisTick);
    }

    /**
     * Take up what the settings screen was last told, whatever else is going on.
     *
     * Safe to call at any point: every one of these is a field or a setter on something
     * already built, so applying them does not disturb the stream, the detector or a
     * recording in progress.
     */
    private void applySettings() {
        scansForFire = "wildfire".equals(Settings.domain(this));
        openFireDetector();

        // Crowd mode counts people, so it looks for people. A car in a crowd shot is
        // another box, another track, and another chance to be wrong about whoever is
        // standing beside it.
        NativeDetector current = detector;
        if (current != null) {
            current.setPeopleOnly("crowd".equals(Settings.domain(this)));
            // The detector runs low on purpose, so a weak box still reaches the tracker and
            // can carry somebody through a frame they were barely seen in. What the slider
            // sets is the harder question: how sure a box has to be before it is allowed to
            // be somebody new. See Tracker.NEW_TRACK_CONFIDENCE.
            current.setConfidence(DETECT_FLOOR);
            tracker.setNewTrackConfidence(Settings.confidence(this));
        }
    }

    @Override
    protected void onStop() {
        super.onStop();

        // A recording carries on when the app leaves the screen, and nothing below happens.
        //
        // This used to release the player and stop the detector while deliberately leaving
        // the recorder running, which is the worst of both: the file kept running with no
        // frames arriving, so it held one frozen picture for however long the pilot was in
        // their flight software, and the boxes stopped moving because the detector had been
        // shut down. The recording could not be recovered either, because the pipeline was
        // torn down with the surface and never reattached to the encoder.
        //
        // RecordingService is what makes staying alive legal: without a foreground service
        // Android stops this activity and then kills the process, and the recording ends
        // silently with nobody finding out until they land.
        if (isRecording()) {
            return;
        }
        if (popout != null) {
            // Same reasoning, for the same reason: the floating window is showing this
            // stream to the operator right now. Releasing the player here would freeze it.
            return;
        }

        wantStream = false;
        handler.removeCallbacks(analysisTick);
        handler.removeCallbacks(detectTick);
        handler.removeCallbacks(releaseDetect);
        if (workThread != null) {
            // quitSafely, so a detection already running finishes and recycles its bitmap
            // rather than being cut off mid-frame. Then wait for it, briefly: onDestroy
            // closes the detector, and closing it underneath a detection still using it is
            // a native crash rather than an exception.
            HandlerThread quitting = workThread;
            workThread = null;
            work = null;
            quitting.quitSafely();
            try {
                quitting.join(1000);
            } catch (InterruptedException interrupted) {
                Thread.currentThread().interrupt();
            }
        }
        detectBusy = false;
        // The recording is deliberately not stopped here: a controller screen that blanks
        // for a moment must not silently end a recording that is still wanted. It is ended
        // by the button, or by the activity being destroyed.
        if (player != null) {
            player.release();
            player = null;
        }
    }

    /**
     * Open or close the fire model to match what the operator is flying for.
     *
     * Called on every start rather than once, because the domain is a setting and can change
     * between flights. A crowd flight should not be carrying a fire model's weights around
     * in 2 GB of RAM, and a wildfire flight that started life as a crowd one has to be able
     * to pick it up without restarting the stream.
     */
    private void openFireDetector() {
        if (!scansForFire) {
            closeFireDetector();
            return;
        }
        if (fireDetector != null) {
            return;
        }
        StringBuilder why = new StringBuilder();
        fireDetector = NativeDetector.openFire(this, why);
        // No message when it is absent. The scan runs either way, and a line saying a model
        // failed to load reads, on a fire screen, like a line about fire.
    }

    /**
     * Let the fire model go, ON THE THREAD THAT USES IT.
     *
     * WHY THIS IS NOT JUST closing.close()
     *     This is called from applySettings, which runs on the main thread and runs on every
     *     return to the screen - including while a detect cycle is in flight on the work
     *     thread. Nulling the field first does not help: onWholeFrame reads it into a local
     *     before calling detect, so the work thread can be inside Interpreter.run() on an
     *     interpreter this thread then frees underneath it.
     *
     *     That is a use-after-free in native code. It does not throw, it takes the process
     *     down, and it does so when the operator changes a setting mid-flight - which is
     *     exactly when they are least able to afford it.
     *
     *     Posting the close to the work thread makes it wait its turn behind any cycle
     *     already running, because that thread does one thing at a time. If there is no work
     *     thread there is nothing running on it either, so closing here is safe.
     */
    private void closeFireDetector() {
        NativeDetector closing = fireDetector;
        fireDetector = null;
        if (closing == null) {
            return;
        }
        Handler worker = work;
        if (worker == null || !worker.post(closing::close)) {
            closing.close();
        }
    }

    @Override
    protected void onDestroy() {
        // Before anything is released: the window is showing a pipeline that is about to be
        // torn down, and a floating window outliving its activity is one that cannot be shut.
        closePopout(false);
        stopWatchingTemperature();
        handler.removeCallbacksAndMessages(null);
        finishRecordingBeforeTeardown();
        if (analyser != null) {
            analyser.destroy();
            analyser = null;
        }
        closePipeline();
        // Both interpreters are freed on the work thread and this one WAITS for it, for the
        // same reason closeFireDetector posts: the work thread can be inside Interpreter.run()
        // right now, and freeing a TFLite interpreter under a running inference is a native
        // use-after-free that takes the process down rather than throwing.
        //
        // onStop has usually quit the thread by the time onDestroy runs, in which case there
        // is nothing to wait for. It has not when the activity is destroyed while a pop-out
        // is open, which is the path that made this worth doing.
        NativeDetector person = detector;
        detector = null;
        Handler worker = work;
        java.util.concurrent.CountDownLatch freed = new java.util.concurrent.CountDownLatch(1);
        boolean posted = worker != null && worker.post(() -> {
            if (person != null) {
                person.close();
            }
            freed.countDown();
        });
        if (!posted && person != null) {
            person.close();
        }
        closeFireDetector();
        if (posted) {
            try {
                // Bounded: a work thread wedged on a broken delegate must not stop the
                // activity being destroyed. Leaking an interpreter is survivable; hanging
                // teardown is not.
                freed.await(2, java.util.concurrent.TimeUnit.SECONDS);
            } catch (InterruptedException interrupted) {
                Thread.currentThread().interrupt();
            }
        }
        super.onDestroy();
    }

    // -----------------------------------------------------------------------------------
    // Stream
    // -----------------------------------------------------------------------------------

    private void openStream() {
        String uri = Settings.stream(this);
        // A buffer sized for a live link, not for a film.
        //
        // WHY THE PICTURE WAS BEHIND THE DRONE
        //     ExoPlayer's defaults are built for streaming video over the internet, where
        //     holding a few seconds in hand is what stops a film stuttering when the
        //     connection dips. They wait 2.5 seconds before showing anything and rebuild
        //     that cushion after every hiccup, and a cushion is exactly a delay: the
        //     picture on the screen is whatever the drone saw two and a half seconds ago.
        //
        //     Nothing about this link wants that. It is a dedicated radio to a camera on
        //     the same network, and a pilot needs to see what the drone is looking at now,
        //     not what it was looking at. Better to show the newest frame and occasionally
        //     stutter than to be reliably late.
        //
        //     So: start on the first frame, hold at most half a second, and never wait to
        //     accumulate anything after a break.
        LoadControl live = new DefaultLoadControl.Builder()
                .setBufferDurationsMs(
                        LIVE_MIN_BUFFER_MS, LIVE_MAX_BUFFER_MS,
                        LIVE_START_MS, LIVE_RESTART_MS)
                .setPrioritizeTimeOverSizeThresholds(true)
                .build();

        player = new ExoPlayer.Builder(this).setLoadControl(live).build();
        if (usingGl && glPipeline != null && glPipeline.videoInput() != null) {
            // The player draws into the pipeline's own SurfaceTexture rather than into a
            // view. That texture is what goes to the screen and to the encoder, which is
            // the whole point: one decode, two destinations, no copy through the CPU.
            player.setVideoSurface(glPipeline.videoInput());
        } else {
            player.setVideoTextureView(video);
        }

        // Forcing TCP rather than letting it negotiate UDP: the controller link drops
        // packets under load, and a UDP RTSP session degrades into a frozen picture with no
        // error, which is indistinguishable from the drone not moving.
        RtspMediaSource.Factory factory = new RtspMediaSource.Factory().setForceUseRtpTcp(true);
        player.setMediaSource(factory.createMediaSource(MediaItem.fromUri(Uri.parse(uri))));

        player.addListener(new Player.Listener() {
            @Override
            public void onPlayerError(@NonNull PlaybackException error) {
                lastError = getString(R.string.stream_failed, uri, String.valueOf(error.getMessage()));
                refreshStatus();
            }

            @Override
            public void onPlaybackStateChanged(int state) {
                refreshStatus();
            }

            @Override
            public void onVideoSizeChanged(@NonNull VideoSize size) {
                if (size.width <= 0 || size.height <= 0) {
                    return;
                }
                videoPixelWidth = size.width;
                videoPixelHeight = size.height;
                // The pipeline needs the real dimensions for two things: the buffer the
                // decoder writes into, and the aspect ratio it letterboxes to. Guessing
                // either produces a picture that is subtly the wrong shape, which on a feed
                // an operator is judging distances from is worse than an obvious fault.
                if (glPipeline != null) {
                    glPipeline.setVideoSize(size.width, size.height);
                }
                // The floating window opens before the stream's real shape is known, so it
                // corrects itself here rather than sitting letterboxed for the whole flight.
                if (popout != null) {
                    popout.setVideoSize(size.width, size.height);
                }
            }
        });
        player.prepare();
        player.setPlayWhenReady(true);
        refreshStatus();
    }

    /**
     * Listen to what the device says about its own temperature, and back off when it is hot.
     *
     * Android reports a thermal status, and it is worth more than any guess this code could
     * make: it comes from the sensors, it accounts for the sun on the back of a controller
     * on an airfield, and it changes before throttling does. Backing off on that signal
     * keeps the detection rate steady and predictable instead of letting it decay through a
     * flight as the chip is slowed underneath it.
     */
    private void watchTemperature() {
        // Android 9 is what the MK15 runs and this arrived in Android 10, so on the device
        // this was written for it never fires. That is not a reason to drop it - a newer
        // controller gets the benefit - but it does mean the thermal handling cannot be
        // relied on. What actually keeps the MK15 cool is TARGET_PERIOD_MS and DUTY_CYCLE
        // above, which need no API at all.
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.Q) {
            return;
        }
        PowerManager power = getSystemService(PowerManager.class);
        if (power == null) {
            return;
        }
        thermalListener = status -> {
            float ease;
            if (status >= PowerManager.THERMAL_STATUS_SEVERE) {
                ease = 4f;          // one detection a second, and no close look
            } else if (status >= PowerManager.THERMAL_STATUS_MODERATE) {
                ease = 2f;
            } else {
                ease = 1f;
            }
            thermalEase = ease;
            hot = status >= PowerManager.THERMAL_STATUS_SEVERE;
            handler.post(this::refreshStatus);
        };
        power.addThermalStatusListener(thermalListener);
    }

    private void stopWatchingTemperature() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.Q || thermalListener == null) {
            return;
        }
        PowerManager power = getSystemService(PowerManager.class);
        if (power != null) {
            power.removeThermalStatusListener(thermalListener);
        }
        thermalListener = null;
    }

    /**
     * Shrink to a small movable window and let another app have the screen.
     *
     * The activity is not stopped in this mode, it is only small, so the stream, the
     * detector and the recording all carry on exactly as they were - which is the point:
     * fly in the flight software with the drone's feed and its boxes in the corner.
     *
     * Entered on the button and automatically when the operator leaves for another app, so
     * it does not have to be remembered mid-flight.
     */
    private void enterSmallWindow(boolean mayAsk) {
        if (popout != null) {
            return;
        }
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M
                && !android.provider.Settings.canDrawOverlays(this)) {
            // Asked at the moment it is wanted rather than at startup, and only when the
            // press was deliberate. Leaving for the flight software is not the moment to put
            // a settings screen in front of a pilot, so that path takes the fixed window
            // instead, which is what the app did before any of this.
            if (mayAsk) {
                askForOverlayPermission();
                return;
            }
            enterSystemSmallWindow();
            return;
        }
        if (openPopout()) {
            return;
        }
        enterSystemSmallWindow();
    }

    /**
     * The window this app owns: draggable, resizable, and the only kind of either on this
     * tablet.
     *
     * @return true if it is on screen, false to fall back to the system's fixed one.
     */
    private boolean openPopout() {
        PopoutWindow window = new PopoutWindow(this, popoutListener,
                videoPixelWidth, videoPixelHeight);
        String refused = window.open();
        if (refused != null) {
            return false;
        }
        popout = window;
        // The same reason a recording needs it: with the window out this activity is stopped,
        // and a stopped activity's process can be killed at any moment. See RecordingService.
        RecordingService.start(this);
        Toast.makeText(this, R.string.popout_hint, Toast.LENGTH_LONG).show();
        // The pipeline moves to the window's surface when that surface arrives. This screen
        // steps aside now so the window is over whatever the operator switches to, which is
        // the point of pressing the button at all.
        moveTaskToBack(true);
        return true;
    }

    /**
     * Ask for the permission that makes a resizable window possible, once.
     *
     * There is no callback worth registering: the operator either comes back with it granted,
     * in which case the next press of the button opens the real window, or they do not, in
     * which case the button keeps working as it always did.
     */
    private void askForOverlayPermission() {
        Toast.makeText(this, R.string.popout_permission, Toast.LENGTH_LONG).show();
        try {
            startActivity(new Intent(
                    android.provider.Settings.ACTION_MANAGE_OVERLAY_PERMISSION,
                    Uri.parse("package:" + getPackageName())));
        } catch (RuntimeException noSuchScreen) {
            // Some builds do not carry that settings screen. Nothing to do but carry on
            // without the resizable window.
        }
    }

    /**
     * Take the picture back from the floating window and close it.
     *
     * @param handBack whether to put the picture back on this screen. False when the activity
     *                 is on its way out and there is no screen left to hand it to.
     */
    private void closePopout(boolean handBack) {
        PopoutWindow closing = popout;
        if (closing == null) {
            return;
        }
        popout = null;
        closing.close();
        if (popoutOwnsDisplay) {
            // The window's surface callback did not run, or ran before the pipeline existed.
            // Either way the pipeline must stop drawing into a surface that has gone.
            popoutOwnsDisplay = false;
            if (glPipeline != null && glPipeline.isAlive()) {
                glPipeline.detachDisplay();
            }
        }
        if (!isRecording()) {
            RecordingService.stop(this);
        }
        Surface here = handBack ? videoSurface.getHolder().getSurface() : null;
        if (glPipeline != null && glPipeline.isAlive() && here != null && here.isValid()) {
            glPipeline.attachDisplay(here, videoSurface.getWidth(), videoSurface.getHeight());
            surfaceReady = true;
        }
        refreshStatus();
    }

    /** What the floating window reports back. Everything arrives on the main thread. */
    private final PopoutWindow.Listener popoutListener = new PopoutWindow.Listener() {
        @Override
        public void onPopoutSurfaceCreated(Surface surface, int width, int height) {
            if (glPipeline == null || !glPipeline.isAlive()) {
                return;
            }
            // attachDisplay destroys the old window surface and creates one on this, so the
            // picture moves rather than being duplicated. The decoder, the detector, the
            // tracker and any recording never learn that anything happened.
            glPipeline.attachDisplay(surface, width, height);
            popoutOwnsDisplay = true;
            surfaceReady = false;
        }

        @Override
        public void onPopoutSurfaceChanged(int width, int height) {
            if (popoutOwnsDisplay && glPipeline != null) {
                // Resized. Without this the picture is drawn at the old size and stretched or
                // cropped to the new one.
                glPipeline.setDisplaySize(width, height);
            }
        }

        @Override
        public void onPopoutSurfaceDestroyed() {
            if (!popoutOwnsDisplay) {
                return;
            }
            popoutOwnsDisplay = false;
            if (glPipeline != null && glPipeline.isAlive()) {
                glPipeline.detachDisplay();
            }
        }

        @Override
        public void onPopoutTapped() {
            // Back to the full screen. onStart closes the window and takes the picture back.
            Intent back = new Intent(LiveActivity.this, LiveActivity.class);
            back.addFlags(Intent.FLAG_ACTIVITY_REORDER_TO_FRONT
                    | Intent.FLAG_ACTIVITY_NEW_TASK);
            startActivity(back);
        }

        @Override
        public void onPopoutClosed() {
            closePopout(true);
        }
    };

    /** The system's window: one size, chosen by Android, and the fallback when ours cannot run. */
    private void enterSystemSmallWindow() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) {
            Toast.makeText(this, R.string.pip_unsupported, Toast.LENGTH_LONG).show();
            return;
        }
        if (!getPackageManager().hasSystemFeature(PackageManager.FEATURE_PICTURE_IN_PICTURE)) {
            Toast.makeText(this, R.string.pip_unsupported, Toast.LENGTH_LONG).show();
            return;
        }
        try {
            // The video's own shape, so the window is not letterboxed. Android refuses
            // anything more extreme than about 2.39:1, hence the clamp.
            int width = Math.max(1, videoPixelWidth);
            int height = Math.max(1, videoPixelHeight);
            float ratio = width / (float) height;
            if (ratio > 2.39f) {
                height = Math.round(width / 2.39f);
            } else if (ratio < 0.42f) {
                width = Math.round(height * 0.42f);
            }
            enterPictureInPictureMode(new PictureInPictureParams.Builder()
                    .setAspectRatio(new Rational(width, height))
                    .build());
        } catch (RuntimeException refused) {
            Toast.makeText(this, R.string.pip_unsupported, Toast.LENGTH_LONG).show();
        }
    }

    @Override
    public void onUserLeaveHint() {
        super.onUserLeaveHint();
        // Leaving for another app while the stream is up shrinks the window rather than
        // hiding it. Without this, switching apps mid-flight loses sight of the drone.
        if (player != null && popout == null && !isInPictureInPictureMode()) {
            enterSmallWindow(false);
        }
    }

    @Override
    public void onPictureInPictureModeChanged(boolean small, @NonNull Configuration config) {
        super.onPictureInPictureModeChanged(small, config);
        // Nothing but the video and its boxes in a window that size. Buttons at that scale
        // are unhittable and would cover most of the picture.
        int visibility = small ? View.GONE : View.VISIBLE;
        recordButton.setVisibility(visibility);
        snapshotButton.setVisibility(visibility);
        status.setVisibility(visibility);
        findViewById(R.id.settings).setVisibility(visibility);
        pipButton.setVisibility(visibility);
        backButton.setVisibility(small || isRecording() ? View.GONE : View.VISIBLE);
    }

    // -----------------------------------------------------------------------------------
    // The video pipeline
    // -----------------------------------------------------------------------------------

    /**
     * Bring the GPU path up against the display surface, or fall back to the old one.
     *
     * The fallback is not a formality. This is the only code in the app that depends on a
     * particular device's GL driver behaving, and the cost of being wrong about that is a
     * black screen on a controller. So if anything at all goes wrong here, the TextureView
     * takes over and the screen keeps working, more slowly and with a line saying so.
     */
    private void openPipeline(Surface surface) {
        if (glPipeline != null) {
            return;
        }
        GlPipeline pipeline = new GlPipeline(message -> handler.post(() -> {
            // Delivered from the GL thread after something failed mid-flight. Falling back
            // now means tearing down the player and rebuilding it on the TextureView.
            lastError = getString(R.string.video_path_fallback, message);
            fallBackToTextureView();
        }));
        String problem = pipeline.start(surface, videoSurface.getWidth(), videoSurface.getHeight());
        if (problem != null) {
            pipeline.release();
            lastError = getString(R.string.video_path_fallback, problem);
            usingGl = false;
            videoSurface.setVisibility(View.GONE);
            video.setVisibility(View.VISIBLE);
        } else {
            glPipeline = pipeline;
            usingGl = true;
        }
        surfaceReady = true;
        if (wantStream && player == null) {
            openStream();
        }
        refreshStatus();
    }

    private void closePipeline() {
        surfaceReady = false;
        GlPipeline releasing = glPipeline;
        glPipeline = null;
        usingGl = false;
        if (releasing != null) {
            releasing.release();
        }
    }

    /** Something broke in GL while running. Rebuild the stream on the slow path. */
    private void fallBackToTextureView() {
        if (!usingGl) {
            return;
        }
        // Computed and then used, which it was not before.
        //
        // This path stopped the recording and never started another, so one GL fault ended
        // a flight's video permanently. The operator landed with a file covering the
        // seconds up to the fault and nothing after it, and was told nothing: the fallback
        // message talks about the video path and never mentions the recording.
        boolean wasRecording = isRecording();
        if (wasRecording) {
            stopRecording();
        }
        closePipeline();
        videoSurface.setVisibility(View.GONE);
        video.setVisibility(View.VISIBLE);
        if (player != null) {
            player.release();
            player = null;
        }
        openStream();
        if (wasRecording) {
            // Into a new file, on the slow path, because the GPU one has just been taken
            // away. The flight is split across two files, which is worth saying out loud
            // and is a great deal better than the second half not existing.
            //
            // Delayed, because openStream() has only just asked for the stream: the
            // fallback reads its frames off the TextureView and there is not one there yet.
            // Starting immediately would find no frame, say "no video yet", and record
            // nothing at all - which is the bug this is here to fix, in a new costume.
            lastError = getString(R.string.recording_continued);
            handler.postDelayed(() -> {
                if (recorder == null && player != null) {
                    startRecordingOnCpu();
                }
            }, 2000);
        }
        refreshStatus();
    }

    /**
     * One frame, however this device is getting them.
     *
     * On the GPU path the frame is rendered small into an offscreen buffer and read back on
     * the pipeline's thread, so the consumer is called there. On the fallback it is pulled
     * off the TextureView here, so the consumer is called on the main thread. Either way the
     * consumer owns the bitmap and must recycle it.
     */
    private void requestFrame(int longEdge, GlPipeline.FrameConsumer consumer) {
        GlPipeline pipeline = glPipeline;
        if (usingGl && pipeline != null) {
            pipeline.requestFrame(longEdge, consumer);
            return;
        }
        Bitmap frame = grabVideoFrame(longEdge);
        if (frame != null) {
            consumer.onFrame(frame);
        }
    }

    /**
     * Hand the current boxes to the pipeline, as pixels, for it to composite into the file.
     *
     * Only while recording, and only when the boxes have changed rather than once per frame.
     * That is the whole reason compositing in GL is cheaper than drawing onto every frame:
     * the overlay changes at detection rate, a few times a second, not at frame rate.
     */
    private void pushOverlay() {
        GlPipeline pipeline = glPipeline;
        if (!usingGl || pipeline == null || !isRecording() || recordWidth <= 0) {
            return;
        }
        try {
            // Not at the recording's own resolution, deliberately. At 1920x1080 that layer
            // is 8 MB, drawn on the main thread and uploaded to the GPU on the render
            // thread, several times a second - and the render thread is the one presenting
            // frames to the encoder, so every upload is a hitch in the file. The layer holds
            // outlines and short labels, which survive being scaled up by the GPU; at 1280
            // it is a third of the bytes and the boxes look the same.
            float scale = Math.min(1f, OVERLAY_MAX_EDGE / (float) Math.max(recordWidth, recordHeight));
            int width = Math.max(2, Math.round(recordWidth * scale));
            int height = Math.max(2, Math.round(recordHeight * scale));

            Bitmap layer = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
            overlay.drawInto(new Canvas(layer), width, height);
            pipeline.setOverlay(layer);
        } catch (OutOfMemoryError tooBig) {
            // The recording keeps its picture and loses its boxes, which is far better than
            // the recording ending.
            lastError = getString(R.string.overlay_too_big);
        }
    }

    // -----------------------------------------------------------------------------------
    // Frames
    // -----------------------------------------------------------------------------------

    /** The current video frame, or null while the surface has nothing on it yet. */
    @Nullable
    private Bitmap grabVideoFrame(int longEdge) {
        if (!video.isAvailable() || video.getWidth() == 0 || video.getHeight() == 0) {
            return null;
        }
        float scale = Math.min(1f, (float) longEdge / Math.max(video.getWidth(), video.getHeight()));
        int width = Math.max(2, Math.round(video.getWidth() * scale));
        int height = Math.max(2, Math.round(video.getHeight() * scale));
        return video.getBitmap(width, height);
    }

    /**
     * Is the slow provider pass worth running at all right now?
     *
     * In crowd mode it is not. Counting people is done on the device, frame by frame, and
     * a provider adds nothing to it: it cannot track, it cannot count across a flight, and
     * its answer is seconds old before it arrives. What it does add is a spike every two
     * seconds - a full-resolution frame pulled off the GPU, compressed and handed to a
     * WebView - on a sealed handheld that is already thermally limited.
     *
     * It earns its place on a turbine or a panel, where it describes damage no detector
     * here has a class for. Not here.
     */
    private boolean providerWorthRunning() {
        return Settings.canAnalyse() && !"crowd".equals(Settings.domain(this));
    }

    private void captureAndAnalyse() {
        if (analyser == null || !analyser.isReady() || analyser.isBusy()) {
            return;
        }
        if (!Settings.canAnalyse()) {
            refreshStatus();
            return;
        }
        String provider = Settings.provider(this);
        String model = Settings.model(this);
        String key = Settings.apiKey();
        String domain = Settings.domain(this);
        // The analyser takes ownership: it encodes on its own thread and recycles when it
        // is done. Recycling here would pull the bitmap out from under that encode.
        requestFrame(LiveAnalyser.MAX_EDGE, frame -> {
            LiveAnalyser current = analyser;
            if (current == null) {
                frame.recycle();
                return;
            }
            current.analyse(frame, provider, model, key, domain);
        });
        refreshStatus();
    }

    private void captureForRecording() {
        if (recorder == null || !recorder.isRunning()) {
            return;
        }
        if (recorder.failure() != null) {
            stopRecording();
            return;
        }
        Bitmap frame = grabVideoFrame(RECORD_LONG_EDGE);
        if (frame == null) {
            framesDropped++;
            return;
        }
        // The boxes go onto the frame itself, at the frame's resolution, so the recording
        // holds the same annotation the operator saw rather than a screen-sized approximation.
        Canvas canvas = new Canvas(frame);
        overlay.drawInto(canvas, frame.getWidth(), frame.getHeight());
        recorder.submit(frame);   // takes ownership
        refreshStatus();
    }

    // -----------------------------------------------------------------------------------
    // Buttons
    // -----------------------------------------------------------------------------------

    private boolean isRecording() {
        return recorder != null && recorder.isRunning();
    }

    private void toggleRecording() {
        if (isRecording()) {
            stopRecording();
        } else {
            startRecording();
        }
    }

    /**
     * Never start a second recording over a live one.
     *
     * The button's label and isRecording() used to be able to disagree: a recording that
     * died on its own cleared the recorder's running flag while the button went on saying
     * "Stop recording", so the next press took the start branch. That overwrote the field
     * holding the first recorder, and the file holding the flight was orphaned with no
     * index on it, unopenable, for good. recordingDied() now clears the field and resets
     * the button, and this refuses the case anyway.
     */
    private void startRecording() {
        if (recorder != null) {
            stopRecording();
            return;
        }
        // A second encoder while the first is still writing its index is two encoders on one
        // Snapdragon 660, and the file being finalised is the one that loses. Refused with a
        // reason rather than queued: the operator pressed a button and is owed an answer.
        if (finishing) {
            Toast.makeText(this, R.string.still_saving, Toast.LENGTH_SHORT).show();
            return;
        }
        if (usingGl && glPipeline != null) {
            startRecordingOnGpu();
        } else {
            startRecordingOnCpu();
        }
    }

    /**
     * The good path: the pipeline renders straight onto the encoder.
     *
     * Nothing is read back, nothing is uploaded, and the recording runs at the video's own
     * resolution rather than at whatever a readback could afford. The overlay is pushed as
     * a texture when the boxes change, and the encoder is drained after each frame the
     * pipeline presents.
     */
    private void startRecordingOnGpu() {
        // Even dimensions: H.264 encoders reject odd ones on a great many devices, and the
        // failure is an opaque IllegalStateException at configure() rather than a message.
        int longEdge = Math.max(videoPixelWidth, videoPixelHeight);
        float scale = Math.min(1f, RECORD_MAX_EDGE / (float) longEdge);
        recordWidth = Math.max(2, Math.round(videoPixelWidth * scale)) & ~1;
        recordHeight = Math.max(2, Math.round(videoPixelHeight * scale)) & ~1;

        String stamp = timestamp();
        File file = new File(outputDirectory(), "flight-" + stamp + ".mp4");
        BoxRecorder starting = new BoxRecorder(file, recordWidth, recordHeight, false);
        String problem = starting.startAndWait();
        if (problem != null) {
            lastError = getString(R.string.recording_failed, problem);
            starting.stop(null);
            refreshStatus();
            return;
        }
        recorder = starting;
        // Told when it dies, because on this path nothing else ever finds out: the CPU
        // fallback polls failure() on its own tick and this one has no tick to poll on.
        starting.setListener(this::recordingDied);
        framesDropped = 0;
        openFlightLog(stamp);
        glPipeline.startRecording(starting.input(), recordWidth, recordHeight,
                BoxRecorder.FRAME_RATE, starting::drainNow);

        recordButton.setText(R.string.stop_recording);
        backButton.setVisibility(View.GONE);
        // From here the process is allowed to keep running with nothing on screen. See
        // RecordingService.
        RecordingService.start(this);
        pushOverlay();
        refreshStatus();
    }

    /** The fallback: read each frame back, draw the boxes on it, hand it over. */
    private void startRecordingOnCpu() {
        Bitmap probe = grabVideoFrame(RECORD_LONG_EDGE);
        if (probe == null) {
            Toast.makeText(this, R.string.no_video_yet, Toast.LENGTH_SHORT).show();
            return;
        }
        recordWidth = probe.getWidth();
        recordHeight = probe.getHeight();
        probe.recycle();

        String stamp = timestamp();
        File file = new File(outputDirectory(), "flight-" + stamp + ".mp4");
        recorder = new BoxRecorder(file, recordWidth, recordHeight);
        recorder.setListener(this::recordingDied);
        recorder.start();
        framesDropped = 0;
        openFlightLog(stamp);
        nextRecordAt = SystemClock.uptimeMillis();
        handler.post(recordTick);

        recordButton.setText(R.string.stop_recording);
        backButton.setVisibility(View.GONE);
        RecordingService.start(this);
        refreshStatus();
    }

    private void stopRecording() {
        // The service is NOT released here, deliberately, though it used to be.
        //
        // Ending the recording does not end the work: the file is finalised afterwards, on
        // the recorder's own thread, and the last thing that happens there is the muxer
        // writing the file's index. Dropping the foreground service first left the process
        // killable across exactly that window, and a process killed there leaves the whole
        // flight on disk in a file no player will open. It is released in the callback
        // below instead, once the file is closed. Holding it a second longer is free.
        handler.removeCallbacks(recordTick);
        GlPipeline pipeline = glPipeline;
        if (pipeline != null) {
            // Detach the encoder surface before the encoder is torn down, or the pipeline
            // renders into a surface that has gone.
            pipeline.stopRecording();
        }
        FlightLog log = flightLog;
        flightLog = null;
        String tally = "";
        if (log != null) {
            log.close();
            publish(log.file());
            // Named in the toast because a file nobody knows about is a file nobody opens.
            tally = getString(R.string.and_the_log, log.file().getName(), log.highestSeen());
        }
        final String savedTally = tally;

        BoxRecorder closing = recorder;
        recorder = null;
        finishing = closing != null;
        recordButton.setText(R.string.start_recording);
        backButton.setVisibility(View.VISIBLE);

        if (closing == null) {
            finishing = false;
            return;
        }
        File file = closing.output();
        closing.stop(() -> handler.post(() -> {
            // Asked AFTER the file is closed, not before.
            //
            // This used to read failure() on the line above stop() and test that snapshot
            // in here. Every way finalising can fail happens after that snapshot was taken:
            // the end-of-stream drain, and the muxer write that puts an index on the file.
            // So the snapshot was always null, and the app always said it had saved - while
            // handing the operator a file that would not open.
            finishing = false;
            String problem = closing.failure();
            if (problem == null && !closing.wroteAnything()) {
                problem = getString(R.string.nothing_recorded);
            }
            if (problem != null) {
                lastError = getString(R.string.recording_failed, problem);
                Toast.makeText(this, lastError, Toast.LENGTH_LONG).show();
                if (!closing.wroteAnything()) {
                    // An empty file is worse than none: it sits in the gallery looking like
                    // a flight nobody can open.
                    file.delete();
                }
            } else {
                publish(file);
                lastSaved = getString(R.string.saved_to, file.getParent());
                Toast.makeText(this,
                        getString(R.string.recording_saved, file.getName(), file.getParent())
                                + savedTally,
                        Toast.LENGTH_LONG).show();
            }
            // Only if nobody has started recording again in the meantime.
            //
            // This callback runs on the recorder's own thread and can be seconds late: the
            // end-of-stream drain is bounded at two seconds and the muxer's index write
            // follows it. An operator who stops and immediately restarts has a NEW recorder
            // running by the time this fires, and this used to drop the foreground service
            // out from under it - leaving the second recording killable, with a notification
            // that had already gone.
            if (recorder == null && popout == null) {
                RecordingService.stop(this);
            }
            refreshStatus();
        }));
    }

    /**
     * Close the file, and WAIT for it, before anything else is torn down.
     *
     * This used to be recorder.stop(null) followed straight away by closePipeline(). stop()
     * only posts to the recorder's thread and returns, so the muxer was still writing the
     * file's index while closePipeline() destroyed the encoder's EGL surface from the GL
     * thread - two threads pulling down opposite ends of one buffer queue. Either the
     * process ended before the index was written, or the teardown threw. Both leave the
     * whole flight on disk in a file that will not open.
     *
     * So the pipeline is detached from the encoder first, and then this blocks until the
     * file is closed. Bounded, because onDestroy is not a place to hang: past the bound the
     * app is closing anyway and the recorder's own end-of-stream deadline has already run.
     */
    private void finishRecordingBeforeTeardown() {
        BoxRecorder sealing = recorder;
        recorder = null;
        if (sealing == null) {
            return;
        }
        GlPipeline pipeline = glPipeline;
        if (pipeline != null) {
            pipeline.stopRecording();
        }
        FlightLog log = flightLog;
        flightLog = null;
        if (log != null) {
            log.close();
            publish(log.file());
        }
        final java.util.concurrent.CountDownLatch closed =
                new java.util.concurrent.CountDownLatch(1);
        sealing.stop(closed::countDown);
        try {
            // Long enough for the encoder's own two second deadline plus the muxer write.
            closed.await(4, java.util.concurrent.TimeUnit.SECONDS);
        } catch (InterruptedException interrupted) {
            Thread.currentThread().interrupt();
        }
        if (sealing.wroteAnything()) {
            // Handed to the media scanner here too, which onDestroy never used to do, so a
            // flight ended by the app closing was invisible in the gallery and over USB.
            publish(sealing.output());
        } else {
            sealing.output().delete();
        }
        RecordingService.stop(this);
    }

    /**
     * A recording died on its own, mid-flight.
     *
     * Reached from BoxRecorder's failure listener, which fires the moment an encoder or a
     * muxer throws. Before this existed nothing was told: the recorder cleared its own
     * running flag, isRecording() went false, the status line quietly dropped its REC
     * counter, and the button went on saying "Stop recording". Pressing it then took the
     * start branch and opened a SECOND recording over the top of the first, so the file
     * holding the flight was abandoned for good.
     */
    private void recordingDied(String reason) {
        handler.post(() -> {
            BoxRecorder dead = recorder;
            if (dead == null) {
                return;
            }
            recorder = null;
            handler.removeCallbacks(recordTick);
            GlPipeline pipeline = glPipeline;
            if (pipeline != null) {
                pipeline.stopRecording();
            }
            FlightLog log = flightLog;
            flightLog = null;
            if (log != null) {
                log.close();
                publish(log.file());
            }
            recordButton.setText(R.string.start_recording);
            backButton.setVisibility(View.VISIBLE);
            // The recorder has already closed its own file by the time this runs, so
            // whatever was captured before the fault is playable and worth publishing.
            if (dead.wroteAnything()) {
                publish(dead.output());
                lastSaved = getString(R.string.saved_to, dead.output().getParent());
            } else {
                dead.output().delete();
            }
            lastError = getString(R.string.recording_failed, reason);
            Toast.makeText(this, lastError, Toast.LENGTH_LONG).show();
            // Same identity check as stopRecording: a recorder that died is not a reason
            // to pull the service out from under one that has since been started.
            if (recorder == null && popout == null) {
                RecordingService.stop(this);
            }
            refreshStatus();
        });
    }

    /**
     * Start a table of what this flight finds, beside the video it belongs to.
     *
     * The same name as the recording, so a flight is one video and one table rather than two
     * things to pair up afterwards. A log that will not open is reported and the flight
     * carries on: losing the numbers is a nuisance, losing the footage is the flight.
     */
    private void openFlightLog(String stamp) {
        evidenceSaved = 0;
        evidenceAt = 0;
        FlightLog log = new FlightLog(new File(outputDirectory(), "flight-" + stamp + ".csv"));
        if (log.failure() != null) {
            lastError = getString(R.string.log_failed, log.failure());
            return;
        }
        flightLog = log;
    }

    private void takeSnapshot() {
        saveFrame(new File(outputDirectory(), "frame-" + timestamp() + ".jpg"), true);
    }

    /**
     * One frame with its boxes drawn on, written to a file.
     *
     * @param announce whether to say so. The operator pressing the button wants to know it
     *                 worked; the automatic ones taken as the tally passes each landmark
     *                 would be a toast every few seconds, over the video, during a flight.
     */
    private void saveFrame(File file, boolean announce) {
        requestFrame(SNAPSHOT_LONG_EDGE, frame -> {
            // The boxes are drawn here, on whichever thread delivered the frame, and the
            // JPEG is written here too. Neither belongs on the main thread: compressing a
            // full-resolution frame is tens of milliseconds and writing it is file I/O, and
            // the button that started this is on the same thread as the video.
            try {
                overlay.drawInto(new Canvas(frame), frame.getWidth(), frame.getHeight());
                try (FileOutputStream out = new FileOutputStream(file)) {
                    frame.compress(Bitmap.CompressFormat.JPEG, 92, out);
                }
                if (announce) {
                    handler.post(() -> {
                        publish(file);
                        lastSaved = getString(R.string.saved_to, file.getParent());
                        Toast.makeText(this,
                                getString(R.string.snapshot_saved,
                                        file.getName(), file.getParent()),
                                Toast.LENGTH_LONG).show();
                        refreshStatus();
                    });
                }
            } catch (IOException | RuntimeException writing) {
                if (announce) {
                    handler.post(() -> Toast.makeText(this,
                            getString(R.string.snapshot_failed,
                                    String.valueOf(writing.getMessage())),
                            Toast.LENGTH_LONG).show());
                }
            } finally {
                frame.recycle();
            }
        });
    }


    private void leave() {
        if (isRecording()) {
            Toast.makeText(this, R.string.stop_recording_first, Toast.LENGTH_SHORT).show();
            return;
        }
        finish();
    }

    // -----------------------------------------------------------------------------------
    // Status
    // -----------------------------------------------------------------------------------

    private void refreshStatus() {
        StringBuilder line = new StringBuilder();
        line.append(Settings.domain(this));

        if (player != null && player.getPlaybackState() == Player.STATE_BUFFERING) {
            line.append("  ·  connecting to ").append(Settings.stream(this));
        }

        // The live half: what is being tracked right now, and how many distinct people
        // have gone past since the screen opened. Two different numbers, and confusing
        // them is the classic mistake.
        // One local read of a field the work thread can null out at any moment. Reading it
        // twice is a null check that was true and a call that is not.
        NativeDetector current = detector;
        // Said every time, not once. A detector that is not running is the most important
        // thing on this screen, and it used to be announced through lastError - a field the
        // stream shares, on a radio link that drops packets routinely. One reconnect message
        // and the operator was left with live video, no boxes, and nothing saying why.
        if (current == null && !detectorFailure.isEmpty()) {
            line.append("  ·  ").append(detectorFailure);
        }
        if (current != null) {
            float seconds = Math.max(1, System.currentTimeMillis() - detectStartedAt) / 1000f;
            line.append("  ·  ").append(String.format(java.util.Locale.UK, "%.1f", detectionsRun / seconds))
                    .append("/s, ").append(current.lastInferenceMillis()).append(" ms")
                    // Which delegate won the trial. Worth saying out loud: it is the single
                    // biggest thing deciding that millisecond figure, and it is decided on
                    // this device rather than declared here.
                    .append(", ").append(current.delegate());
            // The counts come off the work thread with the boxes. The main thread never
            // asks the tracker anything, because the tracker is being written to over there.
            line.append("  ·  ").append(getString(R.string.people_readout,
                    peopleInView, peopleSeen));
        }

        if (!lastSaved.isEmpty()) {
            line.append("  ·  ").append(lastSaved);
        }

        // Flame and smoke, when there is any. Regions, not fires: one fire seen as two
        // regions is two boxes and one fire, and the scanner cannot tell those apart, so
        // it does not claim to.
        int flame = 0;
        int smoke = 0;
        for (FireScan.Region region : fireRegions) {
            if (FireScan.FLAME.equals(region.label)) {
                flame++;
            } else {
                smoke++;
            }
        }
        if (flame > 0 || smoke > 0) {
            line.append("  ·  ").append(getString(R.string.fire_readout, flame, smoke));
        }

        // The slow half: a provider, for what the on-device model has no class for.
        if (!Settings.canAnalyse()) {
            line.append("  ·  ").append(getString(R.string.no_key_no_boxes));
        } else if (analyser != null && analyser.lastLatencyMillis() > 0) {
            line.append("  ·  ").append(analyser.lastLatencyMillis() / 1000f).append("s round trip");
        }

        if (hot) {
            line.append("  ·  ").append(getString(R.string.thermal_easing));
        }

        if (isRecording()) {
            line.append("  ·  REC ").append(recorder.elapsedMillis() / 1000).append("s");
            if (framesDropped > 0) {
                line.append(" (").append(framesDropped).append(" frames dropped)");
            }
        }
        if (!lastError.isEmpty()) {
            line.append("\n").append(lastError);
        }

        status.setText(line.toString());
        String marked = findings.isEmpty() ? "" : findings.size() + " marked by the provider";
        overlay.setStatus(marked);
        if (popout != null) {
            popout.overlay().setStatus(marked);
        }
    }

    /** The folder a flight's files go in, at the top of the tablet's storage. */
    private static final String FOLDER = "DroneInspection";

    /**
     * Where recordings, stills and tallies go.
     *
     * WHY NOT THE APP'S OWN DIRECTORY ANY MORE
     *     Because nobody could find them. getExternalFilesDir is the tidy answer: private,
     *     needs no permission, and removed with the app. It is also six levels down inside
     *     Android/data, which a file manager on a handheld either hides or makes very hard
     *     to reach. Files that exist and cannot be opened are the same as no files.
     *
     *     So a flight goes to /storage/emulated/0/DroneInspection, which is one tap from
     *     the top of internal storage and sits beside DCIM and Download. The app's own
     *     directory is still the fallback, for a device that will not allow the write.
     */
    private File outputDirectory() {
        if (canWriteSharedStorage()) {
            File shared = new File(Environment.getExternalStorageDirectory(), FOLDER);
            if (shared.isDirectory() || shared.mkdirs()) {
                return shared;
            }
        }
        File directory = getExternalFilesDir(Environment.DIRECTORY_MOVIES);
        if (directory == null) {
            directory = getFilesDir();
        }
        if (!directory.exists() && !directory.mkdirs()) {
            return getFilesDir();
        }
        return directory;
    }

    /**
     * Whether the shared folder is available to write to.
     *
     * Android 10 closed direct writes to shared storage, so from there this returns false
     * and files go back to the app's own directory. The controller this is flown with is
     * Android 9, where the permission below is all it takes.
     */
    private boolean canWriteSharedStorage() {
        if (Build.VERSION.SDK_INT > Build.VERSION_CODES.P) {
            return false;
        }
        return ContextCompat.checkSelfPermission(this,
                Manifest.permission.WRITE_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED;
    }

    /**
     * Tell the tablet a file has appeared, so it shows up without a reboot.
     *
     * A file written directly to shared storage is invisible to the gallery and to most
     * file managers until the media scanner has seen it, which is its own small trap: the
     * file is there, the folder looks empty, and it looks exactly like the save failed.
     */
    private void publish(File file) {
        try {
            MediaScannerConnection.scanFile(this, new String[]{file.getAbsolutePath()},
                    null, null);
        } catch (RuntimeException ignored) {
            // Worst case the file appears after the next reboot. Not worth a message.
        }
    }

    private static String timestamp() {
        return new SimpleDateFormat("yyyyMMdd-HHmmss", Locale.US).format(new Date());
    }
}
