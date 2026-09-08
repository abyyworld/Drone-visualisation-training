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

    private final ObjectDetector detector;
    private long lastTimestamp = -1;
    // Written by whichever thread runs detection, read by the main thread for the status
    // line. One word, one writer, so volatile is the whole of the synchronisation needed.
    private volatile long lastInferenceMs;

    private NativeDetector(ObjectDetector detector) {
        this.detector = detector;
    }

    /**
     * Load the detector, or explain why it could not be loaded.
     *
     * @return null if it is unavailable, with the reason in {@code failure}
     */
    @Nullable
    public static NativeDetector open(Context context, StringBuilder failure) {
        try {
            BaseOptions base = BaseOptions.builder()
                    .setModelAssetPath(MODEL_ASSET)
                    .setDelegate(Delegate.CPU)
                    .build();

            ObjectDetector.ObjectDetectorOptions options =
                    ObjectDetector.ObjectDetectorOptions.builder()
                            .setBaseOptions(base)
                            .setRunningMode(RunningMode.VIDEO)
                            .setScoreThreshold(0.35f)
                            .setMaxResults(60)
                            .build();

            return new NativeDetector(ObjectDetector.createFromOptions(context, options));
        } catch (RuntimeException problem) {
            failure.append("The on-device detector could not start: ")
                    .append(problem.getMessage() == null
                            ? problem.getClass().getSimpleName() : problem.getMessage());
            return null;
        }
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

    /** One detector pass, with its boxes mapped back into the frame they came from. */
    private List<Finding> detectIn(Bitmap image1, float offsetX, float offsetY,
                                   float scaleX, float scaleY) {
        // MediaPipe rejects a timestamp that does not advance, and two calls can land in
        // the same millisecond on a fast device. The tile pass is a second call in the same
        // millisecond by construction, so this matters here rather than being belt braces.
        long timestamp = Math.max(lastTimestamp + 1, System.currentTimeMillis());
        lastTimestamp = timestamp;

        MPImage image = new BitmapImageBuilder(image1).build();
        ObjectDetectorResult result = detector.detectForVideo(image, timestamp);

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
        detector.close();
    }
}
