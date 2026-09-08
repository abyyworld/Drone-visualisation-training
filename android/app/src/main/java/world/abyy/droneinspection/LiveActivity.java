package world.abyy.droneinspection;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.Looper;
import android.os.SystemClock;
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
import androidx.media3.common.util.UnstableApi;
import androidx.media3.exoplayer.ExoPlayer;
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
     * What the frame is scaled to before detection.
     *
     * The model works at 448 px, so handing it a 1080p frame only costs a bigger downscale
     * inside MediaPipe. 640 keeps small subjects resolvable without paying for pixels the
     * model throws away.
     */
    private static final int DETECT_LONG_EDGE = 640;

    /**
     * Floor on the gap between detections, so a fast tablet leaves the decoder some room.
     * The real gap is the length of the last detection, which is longer than this on
     * everything except a very quick device.
     */
    private static final long MIN_DETECT_GAP_MS = 40;

    private final Handler handler = new Handler(Looper.getMainLooper());

    private TextureView video;
    private OverlayView overlay;
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
            if (!detectBusy) {
                Bitmap frame = grabVideoFrame(DETECT_LONG_EDGE);
                if (frame != null) {
                    detectBusy = true;
                    if (work == null || !work.post(() -> analyseFrame(frame))) {
                        frame.recycle();
                        detectBusy = false;
                    }
                }
            }
            handler.postDelayed(this, MIN_DETECT_GAP_MS);
        }
    };

    /**
     * Detection, tracking and the flame scan, all on the work thread.
     *
     * The tracker is updated here and its counts are read here, so the main thread never
     * asks it anything. What crosses back is a finished list of boxes and two integers.
     */
    private void analyseFrame(Bitmap frame) {
        List<Tracker.Track> tracks = null;
        List<FireScan.Region> fire;
        String failureText = null;
        int frameWidth = frame.getWidth();
        int frameHeight = frame.getHeight();

        NativeDetector current = detector;
        if (current != null) {
            try {
                tracks = tracker.update(current.detect(frame));
                detectionsRun++;
            } catch (RuntimeException failure) {
                failureText = getString(R.string.detector_stopped,
                        String.valueOf(failure.getMessage()));
                current.close();
                detector = null;
            }
        }

        // Its own catch, and outside the detector's null check on purpose. The two engines
        // are independent: if the model fails to load or dies, the flame and smoke scan
        // carries on, because it needs no model.
        try {
            fire = fireScan.scan(frame);
        } catch (RuntimeException | OutOfMemoryError ignored) {
            fire = new ArrayList<>();
        }
        frame.recycle();

        peopleInView = tracker.countOf("person");
        peopleSeen = tracker.countSeen("person");
        fireRegions = fire;

        final List<Tracker.Track> finalTracks = tracks;
        final List<FireScan.Region> finalFire = fire;
        final String finalFailure = failureText;
        handler.post(() -> {
            if (finalFailure != null) {
                lastError = finalFailure;
            }
            if (finalTracks != null) {
                overlay.setTracks(finalTracks, frameWidth, frameHeight);
            }
            overlay.setFire(finalFire);
            detectBusy = false;
            refreshStatus();
        });
    }

    private final Runnable analysisTick = new Runnable() {
        @Override
        public void run() {
            captureAndAnalyse();
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

        video = findViewById(R.id.video);
        overlay = findViewById(R.id.overlay);
        status = findViewById(R.id.status);
        recordButton = findViewById(R.id.record);
        snapshotButton = findViewById(R.id.snapshot);
        backButton = findViewById(R.id.back);

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
                refreshStatus();
            }

            @Override
            public void onError(String message) {
                lastError = message;
                refreshStatus();
            }
        });
    }

    @Override
    protected void onStart() {
        super.onStart();
        openStream();

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
        handler.post(detectTick);
        // The provider still runs, on its slow interval, for what the on-device model
        // cannot see: fire, smoke, blade damage, soiling. None of those are COCO classes.
        handler.post(analysisTick);
    }

    @Override
    protected void onStop() {
        super.onStop();
        handler.removeCallbacks(analysisTick);
        handler.removeCallbacks(detectTick);
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
        handler.removeCallbacksAndMessages(null);
        if (recorder != null) {
            recorder.stop(null);
            recorder = null;
        }
        if (analyser != null) {
            analyser.destroy();
            analyser = null;
        }
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
        player = new ExoPlayer.Builder(this).build();
        player.setVideoTextureView(video);

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
        });
        player.prepare();
        player.setPlayWhenReady(true);
        refreshStatus();
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

    private void captureAndAnalyse() {
        if (analyser == null || !analyser.isReady() || analyser.isBusy()) {
            return;
        }
        if (!Settings.canAnalyse()) {
            refreshStatus();
            return;
        }
        Bitmap frame = grabVideoFrame(LiveAnalyser.MAX_EDGE);
        if (frame == null) {
            return;
        }
        // The analyser takes ownership: it encodes on its own thread and recycles when it
        // is done. Recycling here would pull the bitmap out from under that encode.
        analyser.analyse(frame, Settings.provider(this), Settings.model(this),
                Settings.apiKey(), Settings.domain(this));
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
        Bitmap probe = grabVideoFrame(RECORD_LONG_EDGE);
        if (probe == null) {
            Toast.makeText(this, R.string.no_video_yet, Toast.LENGTH_SHORT).show();
            return;
        }
        int width = probe.getWidth();
        int height = probe.getHeight();
        probe.recycle();

        File file = new File(outputDirectory(), "flight-" + timestamp() + ".mp4");
        recorder = new BoxRecorder(file, width, height);
        recorder.start();
        framesDropped = 0;
        nextRecordAt = SystemClock.uptimeMillis();
        handler.post(recordTick);

        recordButton.setText(R.string.stop_recording);
        backButton.setVisibility(View.GONE);
        refreshStatus();
    }

    private void stopRecording() {
        handler.removeCallbacks(recordTick);
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
        Bitmap frame = grabVideoFrame(4096);
        if (frame == null) {
            Toast.makeText(this, R.string.no_video_yet, Toast.LENGTH_SHORT).show();
            return;
        }
        Canvas canvas = new Canvas(frame);
        overlay.drawInto(canvas, frame.getWidth(), frame.getHeight());

        File file = new File(outputDirectory(), "frame-" + timestamp() + ".jpg");
        try (FileOutputStream out = new FileOutputStream(file)) {
            frame.compress(Bitmap.CompressFormat.JPEG, 92, out);
            Toast.makeText(this, getString(R.string.snapshot_saved, file.getName()),
                    Toast.LENGTH_LONG).show();
        } catch (IOException writing) {
            Toast.makeText(this, getString(R.string.snapshot_failed, String.valueOf(writing.getMessage())),
                    Toast.LENGTH_LONG).show();
        } finally {
            frame.recycle();
        }
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
                    .append("/s, ").append(current.lastInferenceMillis()).append(" ms");
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
