package world.abyy.droneinspection;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.app.PictureInPictureParams;
import android.content.pm.PackageManager;
import android.content.res.Configuration;
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
import androidx.appcompat.app.AppCompatActivity;
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
    private volatile boolean detectBusy;
    private volatile long detectRequestedAt;
    /** When the current cycle began, and the earliest the next one may. See TARGET_PERIOD_MS. */
    private volatile long cycleStartedAt;
    private volatile long nextCycleAt;
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
            List<Finding> found = current == null
                    ? new ArrayList<>() : safeDetect(current, frame);

            if (!tilingWanted() || glPipeline == null || !usingGl) {
                // Kept until here: the flame scan and the appearance signatures both need
                // the pixels, and finishCycle is what recycles them.
                finishCycle(found, frame);
                return;
            }
            // A region of the frame, rendered at its own resolution rather than cropped out
            // of a big readback. See GlPipeline.requestRegion.
            float[] region = Tiles.region(tileTurn++, videoPixelWidth, videoPixelHeight);
            glPipeline.requestRegion(
                    region[0] / videoPixelWidth, region[1] / videoPixelHeight,
                    (region[0] + region[2]) / videoPixelWidth,
                    (region[1] + region[3]) / videoPixelHeight,
                    TILE_LONG_EDGE,
                    tile -> onRegion(tile, region, found, frame));
        })) {
            frame.recycle();
            detectBusy = false;
        }
    }

    /** The close look has arrived. Merge it with the whole frame and finish the cycle. */
    private void onRegion(Bitmap tile, float[] region, List<Finding> whole, Bitmap frame) {
        Handler worker = work;
        if (worker == null || !worker.post(() -> {
            NativeDetector current = detector;
            List<Finding> merged = whole;
            if (current != null) {
                try {
                    merged = Tiles.merge(whole, current.detectRegion(tile, inFrame(region, frame)));
                } catch (RuntimeException ignored) {
                    // The whole-frame findings still stand; only the close look is lost.
                }
            }
            tile.recycle();
            finishCycle(merged, frame);
        })) {
            tile.recycle();
            frame.recycle();
            detectBusy = false;
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
        } catch (RuntimeException failure) {
            lastError = getString(R.string.detector_stopped, String.valueOf(failure.getMessage()));
            current.close();
            detector = null;
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
        }
        if (frame != null) {
            frame.recycle();
        }

        peopleInView = tracker.countOf("person");
        peopleSeen = tracker.countSeen("person");
        fireRegions = fire;

        final List<Tracker.Track> finalTracks = tracks;
        final List<FireScan.Region> finalFire = fire;
        // When the next cycle may start: never sooner than the target period, and never
        // sooner than the duty cycle allows given what this one actually cost. A device
        // that has been throttled to half speed therefore runs at half the rate rather than
        // at a hundred per cent of a slower CPU, which is how it climbs back out.
        long took = SystemClock.uptimeMillis() - cycleStartedAt;
        long period = Math.round(TARGET_PERIOD_MS * thermalEase);
        nextCycleAt = SystemClock.uptimeMillis()
                + Math.max(0, Math.max(period, Math.round(took / DUTY_CYCLE)) - took);

        handler.post(() -> {
            overlay.setTracks(finalTracks, frameWidth, frameHeight);
            overlay.setFire(finalFire);
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
                if (glPipeline != null) {
                    glPipeline.setDisplaySize(w, h);
                }
            }

            @Override
            public void surfaceDestroyed(@NonNull SurfaceHolder holder) {
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
        pipButton.setOnClickListener(v -> enterSmallWindow());
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
        StringBuilder failure = new StringBuilder();
        detector = NativeDetector.open(this, failure);
        if (detector == null) {
            lastError = failure.toString();
        }

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
        if (isRecording() && player != null) {
            // Came back to a recording that never stopped. Everything is still running.
            refreshStatus();
            return;
        }
        if (surfaceReady) {
            openStream();
        }

        workThread = new HandlerThread("live-work");
        workThread.start();
        work = new Handler(workThread.getLooper());
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
        scansForFire = "wildfire".equals(Settings.domain(this));

        // Crowd mode counts people, so it looks for people. A car in a crowd shot is
        // another box, another track, and another chance to be wrong about whoever is
        // standing beside it.
        NativeDetector current = detector;
        if (current != null) {
            current.setPeopleOnly("crowd".equals(Settings.domain(this)));
        }
        handler.post(detectTick);
        // The provider still runs, on its slow interval, for what the on-device model
        // cannot see: fire, smoke, blade damage, soiling. None of those are COCO classes.
        handler.post(analysisTick);
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

        wantStream = false;
        handler.removeCallbacks(analysisTick);
        handler.removeCallbacks(detectTick);
        handler.removeCallbacks(releaseDetect);
        if (workThread != null) {
            // quitSafely, so a detection already running finishes and recycles its bitmap
            // rather than being cut off mid-frame. Then wait for it, briefly: onDestroy
            // closes the detector, and closing it underneath a detection still using it is
            // a native crash rather than an exception.
            HandlerThread finishing = workThread;
            workThread = null;
            work = null;
            finishing.quitSafely();
            try {
                finishing.join(1000);
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

    @Override
    protected void onDestroy() {
        stopWatchingTemperature();
        handler.removeCallbacksAndMessages(null);
        if (recorder != null) {
            recorder.stop(null);
            recorder = null;
        }
        if (analyser != null) {
            analyser.destroy();
            analyser = null;
        }
        closePipeline();
        if (detector != null) {
            detector.close();
            detector = null;
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
    private void enterSmallWindow() {
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
        if (player != null && !isInPictureInPictureMode()) {
            enterSmallWindow();
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
        GlPipeline finishing = glPipeline;
        glPipeline = null;
        usingGl = false;
        if (finishing != null) {
            finishing.release();
        }
    }

    /** Something broke in GL while running. Rebuild the stream on the slow path. */
    private void fallBackToTextureView() {
        if (!usingGl) {
            return;
        }
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

    private void startRecording() {
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

        File file = new File(outputDirectory(), "flight-" + timestamp() + ".mp4");
        BoxRecorder starting = new BoxRecorder(file, recordWidth, recordHeight, false);
        String problem = starting.startAndWait();
        if (problem != null) {
            lastError = getString(R.string.recording_failed, problem);
            starting.stop(null);
            refreshStatus();
            return;
        }
        recorder = starting;
        framesDropped = 0;
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

        File file = new File(outputDirectory(), "flight-" + timestamp() + ".mp4");
        recorder = new BoxRecorder(file, recordWidth, recordHeight);
        recorder.start();
        framesDropped = 0;
        nextRecordAt = SystemClock.uptimeMillis();
        handler.post(recordTick);

        recordButton.setText(R.string.stop_recording);
        backButton.setVisibility(View.GONE);
        RecordingService.start(this);
        refreshStatus();
    }

    private void stopRecording() {
        // Released first: the moment the recording ends, this process has no business
        // holding a notification or being exempt from being stopped.
        RecordingService.stop(this);
        handler.removeCallbacks(recordTick);
        GlPipeline pipeline = glPipeline;
        if (pipeline != null) {
            // Detach the encoder surface before the encoder is torn down, or the pipeline
            // renders into a surface that has gone.
            pipeline.stopRecording();
        }
        BoxRecorder finishing = recorder;
        recorder = null;
        recordButton.setText(R.string.start_recording);
        backButton.setVisibility(View.VISIBLE);

        if (finishing == null) {
            return;
        }
        File file = finishing.output();
        String problem = finishing.failure();
        finishing.stop(() -> handler.post(() -> {
            if (problem != null) {
                lastError = getString(R.string.recording_failed, problem);
            } else {
                Toast.makeText(this, getString(R.string.recording_saved, file.getName()),
                        Toast.LENGTH_LONG).show();
            }
            refreshStatus();
        }));
    }

    private void takeSnapshot() {
        File file = new File(outputDirectory(), "frame-" + timestamp() + ".jpg");
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
                handler.post(() -> Toast.makeText(this,
                        getString(R.string.snapshot_saved, file.getName()),
                        Toast.LENGTH_LONG).show());
            } catch (IOException | RuntimeException writing) {
                handler.post(() -> Toast.makeText(this,
                        getString(R.string.snapshot_failed, String.valueOf(writing.getMessage())),
                        Toast.LENGTH_LONG).show());
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
        overlay.setStatus(findings.isEmpty() ? "" : findings.size() + " marked by the provider");
    }

    /**
     * Where recordings and snapshots go.
     *
     * The app's own Movies directory: visible to the tablet's file manager and to the
     * analysis screen's file picker, which is the whole point of saving them, and removed
     * with the app rather than left behind on shared storage.
     */
    private File outputDirectory() {
        File directory = getExternalFilesDir(Environment.DIRECTORY_MOVIES);
        if (directory == null) {
            directory = getFilesDir();
        }
        if (!directory.exists() && !directory.mkdirs()) {
            return getFilesDir();
        }
        return directory;
    }

    private static String timestamp() {
        return new SimpleDateFormat("yyyyMMdd-HHmmss", Locale.US).format(new Date());
    }
}
