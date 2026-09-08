package world.abyy.droneinspection;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.RectF;

import androidx.annotation.Nullable;

import com.google.mediapipe.framework.image.BitmapImageBuilder;
import com.google.mediapipe.framework.image.MPImage;
import com.google.mediapipe.tasks.components.containers.Category;
import com.google.mediapipe.tasks.components.containers.Detection;
import com.google.mediapipe.tasks.core.BaseOptions;
import com.google.mediapipe.tasks.core.Delegate;
import com.google.mediapipe.tasks.vision.core.RunningMode;
import com.google.mediapipe.tasks.vision.objectdetector.ObjectDetector;
import com.google.mediapipe.tasks.vision.objectdetector.ObjectDetectorResult;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * People and vehicles, found on the tablet, on every frame.
 *
 * WHY THIS IS HERE AND NOT IN THE WEBVIEW
 *     The live screen used to sample a frame every couple of seconds and send it to a
 *     provider. That is the best an API can do - one round trip is seconds - and it means
 *     no identity between frames, so no tracking, no counting, and an overlay that has to
 *     confess how old its boxes are. Detection on the device is what turns that into
 *     something that follows what it is looking at.
 *
 *     It could not go through the existing WebView bridge: getting a frame there means
 *     base64 through evaluateJavascript, which costs more than the inference does. So the
 *     model runs natively, against the same detector.tflite the web app uses, with the same
 *     tracker ported alongside it.
 *
 * CPU, DELIBERATELY
 *     The same reasoning as web/js/ondevice.js, and it is the most important line in the
 *     file. On the GPU delegate this detector returns an empty list on a photograph with a
 *     person filling half of it - no error, no warning, nothing - where CPU scores that
 *     same person at 0.98. A silent empty result is indistinguishable from a frame with
 *     nobody in it, which is the conclusion an operator would draw, and there is no way to
 *     tell the two apart at runtime.
 *
 * WHAT IT CANNOT SEE
 *     COCO classes. So people and vehicles, and none of fire, smoke, cracks or corrosion.
 *     Those still go to the provider on the slow interval, which is exactly what that
 *     interval is good for.
 */
public final class NativeDetector {

    /** Where the shared model sits inside the APK's assets. */
    private static final String MODEL_ASSET = "www/models/detector.tflite";

    /** Worth reporting from a drone. The other COCO classes are furniture and food. */
    private static final Set<String> KEEP = new HashSet<>(Arrays.asList(
            "person", "bicycle", "car", "motorcycle", "bus", "truck", "boat", "train",
            "airplane"));

    /**
     * How many real frames the GPU has to get right before it is trusted with the job.
     *
     * A frame where the CPU found nothing proves nothing either way, so only frames with
     * something in them are counted.
     */
    private static final int PROBE_FRAMES = 8;

    /** The GPU must find at least this share of what the CPU found, over the probe. */
    private static final float PROBE_RECALL = 0.9f;

    /** And be at least this much faster, or the risk buys nothing. */
    private static final float PROBE_SPEEDUP = 1.3f;

    private final ObjectDetector cpu;

    /**
     * The GPU detector, on trial.
     *
     * WHY IT IS ON TRIAL AND NOT SIMPLY USED
     *     The GPU delegate is worth two to nine times the speed on this class of hardware,
     *     which on a Snapdragon-660 controller is the difference between a third of a
     *     second a frame and a tenth. It is also the delegate that, in this project's web
     *     build, returned an *empty list* on a photograph of a person filling half the
     *     frame - no error, no warning, nothing. An empty list is indistinguishable from a
     *     frame with nobody in it, which is the one failure this application must never
     *     have, and there is no way to detect it from a single result.
     *
     *     So it is not chosen by declaration. Both run side by side on the first few real
     *     frames, and the GPU is adopted only if it finds what the CPU found and is
     *     genuinely faster. If it does not, it is closed and never used again, and the
     *     status line says which one is running.
     */
    @Nullable
    private ObjectDetector gpu;

    private boolean gpuAdopted;
    private int probedFrames;
    private int cpuFoundTotal;
    private int gpuFoundTotal;
    private long cpuNanosTotal;
    private long gpuNanosTotal;

    private long lastTimestamp = -1;
    // Written by whichever thread runs detection, read by the main thread for the status
    // line. One word, one writer, so volatile is the whole of the synchronisation needed.
    private volatile long lastInferenceMs;
    private volatile String delegateName = "CPU";

    private NativeDetector(ObjectDetector cpu, @Nullable ObjectDetector gpu) {
        this.cpu = cpu;
        this.gpu = gpu;
    }

    /** Which delegate is actually running, for the status line. */
    public String delegate() {
        return delegateName;
    }

    /**
     * Load the detector, or explain why it could not be loaded.
     *
     * @return null if it is unavailable, with the reason in {@code failure}
     */
    @Nullable
    public static NativeDetector open(Context context, StringBuilder failure) {
        ObjectDetector processor;
        try {
            processor = build(context, Delegate.CPU);
        } catch (RuntimeException problem) {
            failure.append("The on-device detector could not start: ")
                    .append(problem.getMessage() == null
                            ? problem.getClass().getSimpleName() : problem.getMessage());
            return null;
        }

        // Opened alongside, not instead. If it will not even load - which is common enough
        // on older drivers - that is simply the end of it and the CPU carries on.
        ObjectDetector accelerated = null;
        try {
            accelerated = build(context, Delegate.GPU);
        } catch (RuntimeException | Error unavailable) {
            accelerated = null;
        }
        return new NativeDetector(processor, accelerated);
    }

    private static ObjectDetector build(Context context, Delegate delegate) {
        BaseOptions base = BaseOptions.builder()
                .setModelAssetPath(MODEL_ASSET)
                .setDelegate(delegate)
                .build();

        ObjectDetector.ObjectDetectorOptions options =
                ObjectDetector.ObjectDetectorOptions.builder()
                        .setBaseOptions(base)
                        .setRunningMode(RunningMode.VIDEO)
                        .setScoreThreshold(0.35f)
                        // Per pass, and there are two passes a cycle. Sixty was a
                        // number for a scene with a few things in it; a crowd is not that.
                        // The cost of raising it is non-max suppression over more boxes,
                        // which is arithmetic on a list, while the cost of leaving it low
                        // is a count that silently stops climbing at a round number.
                        .setMaxResults(300)
                        .build();
        return ObjectDetector.createFromOptions(context, options);
    }

    /** Milliseconds the last frame took, for the status line and for pacing. */
    public long lastInferenceMillis() {
        return lastInferenceMs;
    }

    /**
     * Detect in one frame.
     *
     * Boxes come back in pixels of the bitmap passed in.
     */
    public List<Finding> detect(Bitmap frame) {
        long started = System.nanoTime();
        List<Finding> findings = detectIn(frame, 0f, 0f, 1f, 1f);
        lastInferenceMs = (System.nanoTime() - started) / 1_000_000;
        return findings;
    }

    /**
     * Detect in a picture of one region of the frame, with the boxes mapped back.
     *
     * The region arrives already rendered at its own resolution by GlPipeline, rather than
     * being cropped out of a big readback here. A sixth of a 1920-wide frame is about 750
     * pixels across and comes back at 640, so the model sees it at nearly one to one -
     * which is the entire reason for tiling. Cropping a sixth out of a 1280-wide readback
     * gave 427 pixels of a picture that had already been downscaled once, and stretching
     * that back up invents nothing.
     *
     * @param region x, y, width, height of the region in full-frame pixels
     */
    public List<Finding> detectRegion(Bitmap image, float[] region) {
        long started = System.nanoTime();
        float scaleX = region[2] / Math.max(1, image.getWidth());
        float scaleY = region[3] / Math.max(1, image.getHeight());
        List<Finding> findings = detectIn(image, region[0], region[1], scaleX, scaleY);
        lastInferenceMs = (System.nanoTime() - started) / 1_000_000;
        return findings;
    }

    /**
     * Run one frame, and while the GPU is on trial, run it on both and keep score.
     *
     * The trial costs a few frames of doing the work twice, at startup, once. What it buys
     * is the difference between believing a delegate's claim and having watched it agree
     * with a known-good answer on this device, on this stream, on real frames.
     */
    private ObjectDetectorResult run(MPImage image, long timestamp) {
        if (gpuAdopted && gpu != null) {
            return gpu.detectForVideo(image, timestamp);
        }

        long startedCpu = System.nanoTime();
        ObjectDetectorResult fromCpu = cpu.detectForVideo(image, timestamp);
        long cpuNanos = System.nanoTime() - startedCpu;

        if (gpu == null) {
            return fromCpu;
        }

        int found = fromCpu.detections().size();
        if (found == 0) {
            // Proves nothing either way. Both agreeing on an empty frame is exactly what a
            // broken delegate looks like.
            return fromCpu;
        }

        try {
            long startedGpu = System.nanoTime();
            // A timestamp of its own: this detector has its own video clock and rejects one
            // that does not advance.
            ObjectDetectorResult fromGpu = gpu.detectForVideo(image, timestamp);
            gpuNanosTotal += System.nanoTime() - startedGpu;
            gpuFoundTotal += fromGpu.detections().size();
        } catch (RuntimeException | Error broken) {
            closeGpu("GPU (failed)");
            return fromCpu;
        }

        cpuNanosTotal += cpuNanos;
        cpuFoundTotal += found;
        probedFrames++;

        if (probedFrames >= PROBE_FRAMES) {
            decide();
        }
        return fromCpu;
    }

    /** The verdict on the GPU, taken once, on the evidence. */
    private void decide() {
        boolean findsThem = gpuFoundTotal >= cpuFoundTotal * PROBE_RECALL;
        boolean faster = gpuNanosTotal > 0
                && cpuNanosTotal >= gpuNanosTotal * PROBE_SPEEDUP;

        if (findsThem && faster) {
            gpuAdopted = true;
            delegateName = "GPU";
            return;
        }
        // Named so the status line can say why. A delegate that is fast and blind is the
        // dangerous one, and it is worth being able to see that it was caught.
        closeGpu(findsThem ? "CPU (GPU no faster)" : "CPU (GPU missed people)");
    }

    private void closeGpu(String reason) {
        delegateName = reason;
        if (gpu != null) {
            try {
                gpu.close();
            } catch (RuntimeException | Error ignored) {
                // Closing a delegate that has already fallen over is not worth reporting.
            }
            gpu = null;
        }
        gpuAdopted = false;
    }

    /** One detector pass, with its boxes mapped back into the frame they came from. */
    private List<Finding> detectIn(Bitmap image1, float offsetX, float offsetY,
                                   float scaleX, float scaleY) {
        // MediaPipe rejects a timestamp that does not advance, and two calls can land in
        // the same millisecond on a fast device. The tile pass is a second call in the same
        // millisecond by construction, so this matters here rather than being belt braces.
        long timestamp = Math.max(lastTimestamp + 1, System.currentTimeMillis());
        lastTimestamp = timestamp;

        MPImage image = new BitmapImageBuilder(image1).build();
        ObjectDetectorResult result = run(image, timestamp);

        List<Finding> findings = new ArrayList<>();
        for (Detection detection : result.detections()) {
            List<Category> categories = detection.categories();
            if (categories.isEmpty()) {
                continue;
            }
            Category top = categories.get(0);
            String label = top.categoryName() == null
                    ? "" : top.categoryName().toLowerCase(java.util.Locale.ROOT);
            if (!KEEP.contains(label)) {
                continue;
            }
            RectF box = detection.boundingBox();
            findings.add(new Finding(
                    label,
                    certaintyOf(top.score()),
                    "",
                    offsetX + box.left * scaleX, offsetY + box.top * scaleY,
                    offsetX + box.right * scaleX, offsetY + box.bottom * scaleY));
            findings.get(findings.size() - 1).confidence = top.score();
        }
        return findings;
    }

    /**
     * A score turned into the same three words the provider engines use.
     *
     * The rest of the application talks in certainty bands rather than percentages, because
     * a vision model's number is not calibrated. This model's is, but showing a percentage
     * here and a band there would make the two engines look like they measure different
     * things when they are answering the same question.
     */
    private static String certaintyOf(float score) {
        if (score >= 0.75f) {
            return "high";
        }
        return score >= 0.5f ? "medium" : "low";
    }

    public void close() {
        cpu.close();
        if (gpu != null) {
            gpu.close();
            gpu = null;
        }
    }
}
