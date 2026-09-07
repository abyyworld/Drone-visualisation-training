package world.abyy.droneinspection;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.os.Looper;
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

    private List<Finding> findings = new ArrayList<>();
    private String lastError = "";
    private long framesDropped;

    private final Runnable analysisTick = new Runnable() {
        @Override
        public void run() {
            captureAndAnalyse();
            handler.postDelayed(this, Math.max(1, Settings.intervalSeconds(LiveActivity.this)) * 1000L);
        }
    };

    private final Runnable recordTick = new Runnable() {
        @Override
        public void run() {
            captureForRecording();
            handler.postDelayed(this, RECORD_INTERVAL_MS);
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
        handler.post(analysisTick);
    }

    @Override
    protected void onStop() {
        super.onStop();
        handler.removeCallbacks(analysisTick);
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
        analyser.analyse(frame, Settings.provider(this), Settings.model(this),
                Settings.apiKey(), Settings.domain(this));
        frame.recycle();
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
        overlay.setStatus(findings.isEmpty() ? "" : findings.size() + " marked");
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
