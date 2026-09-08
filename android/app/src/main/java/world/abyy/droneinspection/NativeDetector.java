package world.abyy.droneinspection;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Rect;

import androidx.annotation.Nullable;

import org.json.JSONArray;
import org.json.JSONObject;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Delegate;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.nnapi.NnApiDelegate;

import java.io.FileInputStream;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;

/**
 * People, found on the tablet, on every frame.
 *
 * WHY THIS IS HERE AND NOT IN THE WEBVIEW
 *     The live screen used to sample a frame every couple of seconds and send it to a
 *     provider. That is the best an API can do - one round trip is seconds - and it means no
 *     identity between frames, so nothing followed and nothing tallied, and an overlay that
 *     has to confess how old its boxes are. Detection on the device is what turns that into
 *     something that follows what it is looking at.
 *
 * WHY IT IS NOT MEDIAPIPE ANY MORE
 *     It was, running the same detector.tflite the web app ships: EfficientDet-Lite2, eighty
 *     COCO classes, trained on photographs taken by people standing on the ground, at 448
 *     pixels. Three things wrong at once for this job. A 1920-wide frame squeezed to 448 is
 *     a fourfold reduction, so somebody forty pixels tall in the air arrives nine pixels
 *     tall, and nothing in COCO is shot from above.
 *
 *     The model now is YOLO finetuned on VisDrone, which is aerial footage full of people a
 *     handful of pixels tall seen from overhead. That is this exact problem, already solved
 *     by somebody else and published. It is 2.8 MB against 7.2, and it runs at 320.
 *
 *     MediaPipe's ObjectDetector could not load it. That wrapper only accepts a model with
 *     the box decoding built into the graph, and a YOLO head is raw numbers: the decode has
 *     to happen afterwards. So the interpreter is driven directly here, and the decode lives
 *     in Yolo.java, which tests/java/cross_check.sh compares against the browser's copy on
 *     the same numbers.
 *
 * WHAT IT CANNOT SEE
 *     Fire, smoke, cracks, corrosion, soiling. Flame and smoke are FireScan's job, from
 *     colour and behaviour rather than from a model; the rest still go to the provider on
 *     its slow interval, which is what that interval is good for.
 */
public final class NativeDetector {

    /** The model, and the facts about it, both written by .github/workflows/model.yml. */
    private static final String MODEL_ASSET = "www/models/person-320-int8.tflite";
    private static final String META_ASSET = "www/models/person-320.json";

    /** Ultralytics' letterbox grey. The model was trained against padding this colour. */
    private static final int PAD = Color.rgb(114, 114, 114);

    private static final Paint FILTER = new Paint(Paint.FILTER_BITMAP_FLAG);

    /**
     * How many real frames an accelerator has to get right before it is trusted with the job.
     *
     * A frame where the CPU found nobody proves nothing either way, so only frames with
     * somebody in them are counted.
     */
    private static final int PROBE_FRAMES = 8;

    /** It must find at least this share of what the CPU found, over the probe. */
    private static final float PROBE_RECALL = 0.9f;

    /** And be at least this much faster, or the risk buys nothing. */
    private static final float PROBE_SPEEDUP = 1.3f;

    private final Interpreter cpu;

    /**
     * The accelerated interpreter, on trial.
     *
     * WHY IT IS ON TRIAL AND NOT SIMPLY USED
     *     This tablet's Snapdragon has a Hexagon DSP that runs an int8 model several times
     *     faster than its cores do, reachable through NNAPI, and a GPU behind that. Either
     *     is the difference between a third of a second a frame and a tenth.
     *
     *     They are also the delegates that, in this project's own web build, returned an
     *     *empty list* on a photograph of a person filling half the frame. No error, no
     *     warning, nothing. An empty list is indistinguishable from a frame with nobody in
     *     it, which is the one failure this application must never have, and there is no way
     *     to spot it from a single result.
     *
     *     So it is not chosen by declaration. Both run side by side on the first few real
     *     frames, and the accelerator is adopted only if it finds what the CPU found and is
     *     genuinely faster. Otherwise it is closed and never used again, and the status line
     *     says which one is running and why.
     */
    @Nullable
    private Interpreter accelerated;
    @Nullable
    private AutoCloseable acceleratedDelegate;
    private final String acceleratedName;

    private boolean adopted;
    private int probedFrames;
    private int cpuFoundTotal;
    private int acceleratedFoundTotal;
    private long cpuNanosTotal;
    private long acceleratedNanosTotal;

    /** From person-320.json, so the app and the model cannot disagree about the classes. */
    private final String[] labels;
    private final Set<Integer> personClasses;
    /**
     * Adjustable while flying, from the settings screen. Only ever read by the thread that
     * runs detection, and one word written by another, so volatile is the whole of it.
     */
    private volatile double confThreshold;
    private final double iouThreshold;
    /** The file beside the model would not parse, so its class names came from here. */
    private final boolean usingDefaults;

    private final int inputSize;
    /** True when the input is [1, 3, size, size] rather than [1, size, size, 3]. */
    private final boolean inputPlanar;
    private final DataType inputType;
    private final float inputScale;
    private final int inputZeroPoint;
    private final DataType outputType;
    private final float outputScale;
    private final int outputZeroPoint;
    private final int outputChannels;
    private final int outputAnchors;
    /** True when the output is [1, anchors, channels] rather than [1, channels, anchors]. */
    private final boolean outputPerAnchor;

    // Reused across frames. Allocating a megabyte of direct buffers per detection would be
    // the biggest thing this class did, on a device with 2 GB and a video decoder in it.
    private final ByteBuffer input;
    private final ByteBuffer output;
    private final Bitmap square;
    private final Canvas canvas;
    private final int[] pixels;
    private final float[] head;
    private final Rect target = new Rect();

    /**
     * What the box numbers are in: 1 for pixels of the model's own square, or the square's
     * size when they arrive as fractions of it. Measured at startup, never assumed.
     */
    private float boxScale = 1f;

    private volatile Set<Integer> keep;
    private volatile long lastInferenceMs;
    private volatile String delegateName = "CPU";

    private NativeDetector(Interpreter cpu, @Nullable Interpreter accelerated,
                           @Nullable AutoCloseable acceleratedDelegate, String acceleratedName,
                           String[] labels, Set<Integer> personClasses,
                           double confThreshold, double iouThreshold, boolean usingDefaults) {
        this.cpu = cpu;
        this.accelerated = accelerated;
        this.acceleratedDelegate = acceleratedDelegate;
        this.acceleratedName = acceleratedName;
        this.labels = labels;
        this.personClasses = personClasses;
        this.confThreshold = confThreshold;
        this.iouThreshold = iouThreshold;
        this.usingDefaults = usingDefaults;
        this.keep = personClasses;

        Tensor entry = cpu.getInputTensor(0);
        int[] entryShape = entry.shape();
        // Channels first or channels last, read off the tensor rather than assumed.
        //
        // The assumption cost a build. Nearly every TFLite vision model is [1, size, size,
        // 3], so this took the second dimension as the size - and this converter emits
        // [1, 3, 320, 320], so the size came out as 3. A three pixel square scaled up to
        // the model, the rest of the buffer left at zero, and a detector that finds nobody
        // and cannot say why. Which is the failure this whole file is written against.
        this.inputPlanar = entryShape[1] == 3 && entryShape[3] != 3;
        this.inputSize = inputPlanar ? entryShape[2] : entryShape[1];
        if (inputSize < 32) {
            throw new IllegalStateException(
                    "the model wants an input this app cannot read: "
                            + java.util.Arrays.toString(entryShape));
        }
        this.inputType = entry.dataType();
        Tensor.QuantizationParams entryQuant = entry.quantizationParams();
        // A scale of zero means the tensor is not quantised, whatever its type says.
        this.inputScale = entryQuant.getScale() == 0 ? 1f : entryQuant.getScale();
        this.inputZeroPoint = entryQuant.getZeroPoint();

        Tensor exit = cpu.getOutputTensor(0);
        int[] out = exit.shape();
        // [1, channels, anchors] or [1, anchors, channels]. Channels is four plus the class
        // count, so it is the small one; anchors runs to thousands. Decided by the shape
        // rather than by a setting, because a setting that disagrees with the file finds
        // nobody, and finding nobody looks exactly like an empty frame.
        this.outputPerAnchor = out[1] > out[2];
        this.outputChannels = outputPerAnchor ? out[2] : out[1];
        this.outputAnchors = outputPerAnchor ? out[1] : out[2];
        this.outputType = exit.dataType();
        Tensor.QuantizationParams exitQuant = exit.quantizationParams();
        this.outputScale = exitQuant.getScale() == 0 ? 1f : exitQuant.getScale();
        this.outputZeroPoint = exitQuant.getZeroPoint();

        this.input = ByteBuffer.allocateDirect(entry.numBytes()).order(ByteOrder.nativeOrder());
        this.output = ByteBuffer.allocateDirect(exit.numBytes()).order(ByteOrder.nativeOrder());
        this.square = Bitmap.createBitmap(inputSize, inputSize, Bitmap.Config.ARGB_8888);
        this.canvas = new Canvas(square);
        this.pixels = new int[inputSize * inputSize];
        this.head = new float[outputChannels * outputAnchors];
    }

    /**
     * How sure to be before marking something.
     *
     * Costs nothing to change. The model pass is the same work whatever this is; only the
     * decode after it sees more or fewer candidates, and across the whole usable range that
     * is a fraction of a millisecond against a cycle of two hundred.
     */
    public void setConfidence(double confidence) {
        confThreshold = confidence;
    }

    /** Look for people only, or for everything the model knows. */
    public void setPeopleOnly(boolean peopleOnly) {
        if (peopleOnly) {
            keep = personClasses;
            return;
        }
        Set<Integer> all = new LinkedHashSet<>();
        for (int i = 0; i < labels.length; i++) {
            all.add(i);
        }
        keep = all;
    }

    /**
     * What the detector is actually doing, for the status line.
     *
     * The delegate, the size it runs at, and whether the box numbers needed scaling. This
     * reads like more than a status line needs, and it is there because two builds went out
     * marking nobody and neither of them could say why. Whatever the next fault is, this
     * puts the state that decides it on the screen.
     */
    public String delegate() {
        return delegateName
                + " " + inputSize + "px"
                + (boxScale == 1f ? "" : " x" + (int) boxScale)
                + (usingDefaults ? " (default classes)" : "");
    }

    /** Milliseconds the last frame took, for the status line and for pacing. */
    public long lastInferenceMillis() {
        return lastInferenceMs;
    }

    /**
     * Load the detector, or explain why it could not be loaded.
     *
     * @return null if it is unavailable, with the reason in {@code failure}
     */
    @Nullable
    public static NativeDetector open(Context context, StringBuilder failure) {
        ByteBuffer model;
        try {
            model = mapAsset(context, MODEL_ASSET);
        } catch (Exception problem) {
            failure.append(describe(problem));
            return null;
        }

        // Defaults that match the model this app ships with, used when the file beside it
        // cannot be read.
        //
        // The metadata is a convenience: it names the classes and says which of them are
        // people. Refusing to detect anything because a 282 byte JSON file would not parse
        // is the wrong trade by a wide margin, and it is what used to happen. A detector
        // running on defaults still marks people; one that never opened marks nothing and
        // looks exactly like a street with nobody on it.
        String[] labels = {"person", "person", "bicycle", "car", "van", "truck", "tricycle",
                "awning-tricycle", "bus", "motor", "others"};
        Set<Integer> people = Yolo.classes(0, 1);
        double conf = 0.25;
        double iou = 0.45;
        boolean metadataFailed = false;
        // Parsed into its own variables and only adopted once all of it worked. Assigning
        // as it went would leave a half read file half applied: the classes to keep are read
        // after the names, so a file that failed in between gave an empty keep set, and an
        // empty keep set is a detector that runs perfectly and marks nobody.
        try {
            JSONObject meta = new JSONObject(new String(readAsset(context, META_ASSET), "UTF-8"));

            JSONArray names = meta.getJSONArray("labels");
            String[] readLabels = new String[names.length()];
            for (int i = 0; i < names.length(); i++) {
                readLabels[i] = names.getString(i).toLowerCase(Locale.ROOT);
            }

            // VisDrone separates a person standing or walking from a person in any other
            // pose. For this application those are one thing, and both are wanted.
            JSONArray wanted = meta.getJSONArray("keepClasses");
            Set<Integer> readPeople = new LinkedHashSet<>();
            for (int i = 0; i < wanted.length(); i++) {
                int id = wanted.getInt(i);
                if (id < 0 || id >= readLabels.length) {
                    throw new IllegalArgumentException("keepClasses " + id + " is not a class");
                }
                readPeople.add(id);
                readLabels[id] = "person";
            }
            if (readPeople.isEmpty()) {
                throw new IllegalArgumentException("nothing would be kept, so nothing marked");
            }

            labels = readLabels;
            people = readPeople;
            conf = meta.optDouble("confThreshold", 0.25);
            iou = meta.optDouble("iouThreshold", 0.45);
        } catch (Exception unreadable) {
            // Carry on with the defaults above. The status line says so, because a detector
            // that had to guess its own class names is worth knowing about even though it
            // still marks people.
            metadataFailed = true;
        }

        Interpreter processor;
        try {
            Interpreter.Options options = new Interpreter.Options();
            // Four of the eight cores. Taking all of them starves the video decoder and the
            // encoder, which are the two things that must not stutter.
            options.setNumThreads(4);
            processor = new Interpreter(model, options);
            processor.allocateTensors();
        } catch (RuntimeException | Error problem) {
            failure.append(describe(problem));
            return null;
        }

        // Opened alongside, not instead. NNAPI first: this model is int8, and int8 through
        // NNAPI is what reaches the Hexagon DSP, the one piece of silicon on this device
        // actually built for the job. The GPU is the fallback. If neither will even load,
        // which is common enough on a 2018 driver stack, that is simply the end of it and
        // the CPU carries on alone.
        Interpreter candidate = null;
        AutoCloseable candidateDelegate = null;
        String candidateName = "";
        for (String attempt : new String[]{"NNAPI", "GPU"}) {
            AutoCloseable delegate = null;
            try {
                delegate = "NNAPI".equals(attempt) ? new NnApiDelegate() : new GpuDelegate();
                Interpreter.Options options = new Interpreter.Options();
                options.addDelegate((Delegate) delegate);
                Interpreter built = new Interpreter(model, options);
                built.allocateTensors();
                candidate = built;
                candidateDelegate = delegate;
                candidateName = attempt;
                break;
            } catch (Exception | Error unavailable) {
                if (delegate != null) {
                    try {
                        delegate.close();
                    } catch (Exception | Error ignored) {
                        // A delegate that would not load is not going to close cleanly
                        // either, and neither failure is worth reporting.
                    }
                }
                candidate = null;
                candidateDelegate = null;
            }
        }

        NativeDetector detector = new NativeDetector(processor, candidate, candidateDelegate,
                candidateName, labels, people, conf, iou, metadataFailed);
        try {
            detector.calibrate();
        } catch (RuntimeException | Error problem) {
            detector.close();
            failure.append(describe(problem));
            return null;
        }
        return detector;
    }

    private static String describe(Throwable problem) {
        return "The on-device detector could not start: "
                + (problem.getMessage() == null
                ? problem.getClass().getSimpleName() : problem.getMessage());
    }

    /**
     * The model, mapped if the APK stored it whole and copied if it did not.
     *
     * `openFd` is the cheap path and the one that works today, because build.gradle keeps
     * tflite files uncompressed. It throws the moment that stops being true, so the fallback
     * reads the bytes instead: three megabytes of heap against a detector that will not
     * start is not a difficult trade.
     */
    private static ByteBuffer mapAsset(Context context, String path) throws Exception {
        try (AssetFileDescriptor descriptor = context.getAssets().openFd(path);
             FileInputStream stream = new FileInputStream(descriptor.getFileDescriptor())) {
            return stream.getChannel().map(FileChannel.MapMode.READ_ONLY,
                    descriptor.getStartOffset(), descriptor.getDeclaredLength());
        } catch (java.io.IOException compressed) {
            byte[] bytes = readAsset(context, path);
            ByteBuffer buffer = ByteBuffer.allocateDirect(bytes.length)
                    .order(ByteOrder.nativeOrder());
            buffer.put(bytes);
            buffer.rewind();
            return buffer;
        }
    }

    /**
     * Every byte of an asset, however it is stored.
     *
     * Not `available()` and one `read()`. This file is compressed inside the APK, where
     * `available()` is not a length and `read` fills as much of the array as it feels like.
     * A short read gives truncated JSON, which throws, which used to mean the detector
     * refused to start and the screen marked nobody, for a reason nothing displayed.
     */
    private static byte[] readAsset(Context context, String path) throws Exception {
        try (InputStream stream = context.getAssets().open(path)) {
            java.io.ByteArrayOutputStream all = new java.io.ByteArrayOutputStream();
            byte[] chunk = new byte[16 * 1024];
            int read;
            while ((read = stream.read(chunk)) > 0) {
                all.write(chunk, 0, read);
            }
            return all.toByteArray();
        }
    }

    /**
     * Detect in one frame.
     *
     * Boxes come back in pixels of the bitmap passed in.
     */
    public List<Finding> detect(Bitmap frame) {
        long started = System.nanoTime();
        List<Finding> findings =
                detectIn(frame, 0f, 0f, frame.getWidth(), frame.getHeight());
        lastInferenceMs = (System.nanoTime() - started) / 1_000_000;
        return findings;
    }

    /**
     * Detect in a picture of one region, with the boxes mapped back.
     *
     * The region arrives already rendered at its own resolution by GlPipeline rather than
     * being cropped out of a big readback. A sixth of a 1920-wide frame is about 750 pixels
     * across and comes back at 640, so the model sees it at nearly one to one, which is the
     * entire reason for tiling.
     *
     * @param region x, y, width, height of the region, in the same pixels the whole-frame
     *               findings come back in - not in the video's own pixels. The two lists are
     *               merged by the caller, so one of them being in a different space puts
     *               every close look several times too far out and loses it, which is what
     *               used to happen and is exactly the distant person tiling exists to catch.
     */
    public List<Finding> detectRegion(Bitmap image, float[] region) {
        long started = System.nanoTime();
        List<Finding> findings = detectIn(image, region[0], region[1], region[2], region[3]);
        lastInferenceMs = (System.nanoTime() - started) / 1_000_000;
        return findings;
    }

    /**
     * One pass: letterbox, run, decode, and map the boxes onto the target rectangle.
     *
     * @param spanX width the picture covers, in the coordinates the findings must come back in
     */
    private List<Finding> detectIn(Bitmap image, float offsetX, float offsetY,
                                   float spanX, float spanY) {
        // Letterboxed rather than stretched, because the model was trained that way and a
        // squashed person is a person it has not been shown.
        int width = image.getWidth();
        int height = image.getHeight();
        double scale = Math.min(inputSize / (double) width, inputSize / (double) height);
        int drawWidth = (int) Math.round(width * scale);
        int drawHeight = (int) Math.round(height * scale);
        int padX = (inputSize - drawWidth) / 2;
        int padY = (inputSize - drawHeight) / 2;

        canvas.drawColor(PAD);
        target.set(padX, padY, padX + drawWidth, padY + drawHeight);
        canvas.drawBitmap(image, null, target, FILTER);
        square.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize);

        fillInput();
        // Leaves `head` holding the numbers to decode, whichever interpreter produced them.
        run();

        List<Yolo.Detection> found = decodeHead();

        // The head works in the letterboxed square's pixels. Undo that back to the picture,
        // then place the picture inside the coordinates the caller asked for.
        double toTargetX = spanX / (double) width;
        double toTargetY = spanY / (double) height;

        List<Finding> findings = new ArrayList<>(found.size());
        for (Yolo.Detection detection : found) {
            double[] box = Yolo.unletterbox(detection.box, scale, padX, padY, width, height);
            String label = detection.classId < labels.length
                    ? labels[detection.classId] : "class_" + detection.classId;
            Finding finding = new Finding(label, certaintyOf(detection.confidence), "",
                    (float) (offsetX + box[0] * toTargetX),
                    (float) (offsetY + box[1] * toTargetY),
                    (float) (offsetX + box[2] * toTargetX),
                    (float) (offsetY + box[3] * toTargetY));
            finding.confidence = detection.confidence;
            findings.add(finding);
        }
        return findings;
    }

    private List<Yolo.Detection> decodeHead() {
        float[] channelMajor = outputPerAnchor
                ? Yolo.toChannelMajor(head, outputAnchors, outputChannels) : head;
        return Yolo.decodeHead(channelMajor, outputChannels, outputAnchors,
                confThreshold, iouThreshold, keep);
    }

    /** Pack the letterboxed square into whatever the model's input tensor wants. */
    private void fillInput() {
        input.rewind();
        boolean quantised = inputType == DataType.UINT8 || inputType == DataType.INT8;
        if (inputPlanar) {
            // Every red, then every green, then every blue. Interleaving them into a
            // channels-first tensor is not an error either: it is a picture of noise, and
            // a model shown noise reports an empty scene.
            for (int shift = 16; shift >= 0; shift -= 8) {
                for (int pixel : pixels) {
                    put((pixel >> shift) & 0xFF, quantised);
                }
            }
        } else {
            for (int pixel : pixels) {
                put((pixel >> 16) & 0xFF, quantised);
                put((pixel >> 8) & 0xFF, quantised);
                put(pixel & 0xFF, quantised);
            }
        }
        input.rewind();
    }

    private void put(int channel, boolean quantised) {
        if (quantised) {
            // The converter's own scale and zero point, read off the tensor rather than
            // assumed: an int8 export and a uint8 one want the same bytes shifted by 128,
            // and guessing gives the model a picture it has never seen.
            input.put(quantise(channel));
        } else {
            input.putFloat(channel / 255f);
        }
    }

    private byte quantise(int channel) {
        int value = Math.round((channel / 255f) / inputScale) + inputZeroPoint;
        // Clamped to the range the tensor's own type can hold. With the converter's scale
        // and zero point this never bites, but a value that ran past the end would wrap to
        // the opposite extreme - a white pixel arriving as black - and nothing downstream
        // could tell.
        int floor = inputType == DataType.INT8 ? -128 : 0;
        int ceiling = inputType == DataType.INT8 ? 127 : 255;
        return (byte) Math.max(floor, Math.min(ceiling, value));
    }

    /** Read the output back into floats, undoing quantisation and the box units. */
    private void readOutput() {
        output.rewind();
        if (outputType == DataType.FLOAT32) {
            for (int i = 0; i < head.length; i++) {
                head[i] = output.getFloat();
            }
        } else if (outputType == DataType.INT8) {
            for (int i = 0; i < head.length; i++) {
                head[i] = (output.get() - outputZeroPoint) * outputScale;
            }
        } else {
            for (int i = 0; i < head.length; i++) {
                head[i] = ((output.get() & 0xFF) - outputZeroPoint) * outputScale;
            }
        }
        output.rewind();
        scaleBoxes();
    }

    /**
     * Put the four box channels into the model square's pixels, whatever they arrived in.
     *
     * The same weights exported two ways do not agree about this. The ONNX gives centres and
     * sizes in pixels of the 320 square; the TFLite converter divides them all by 320 and
     * gives fractions. Nothing about the file says which, the shapes are identical, and
     * reading fractions as pixels is not an error: every box comes out a fraction of a pixel
     * across, in the top left corner, and the screen shows nothing at all while the detector
     * reports finding a hundred people.
     */
    private void scaleBoxes() {
        if (boxScale == 1f) {
            return;
        }
        if (outputPerAnchor) {
            for (int anchor = 0; anchor < outputAnchors; anchor++) {
                int row = anchor * outputChannels;
                for (int c = 0; c < 4; c++) {
                    head[row + c] *= boxScale;
                }
            }
        } else {
            for (int i = 0; i < 4 * outputAnchors; i++) {
                head[i] *= boxScale;
            }
        }
    }

    /**
     * Work out those units, once, by asking the model.
     *
     * The box centres come from the anchor grid rather than from the picture, so they span
     * the model's whole square no matter what it is shown. On this model a flat grey frame
     * gives centres reaching 316 when they are pixels and 0.99 when they are fractions, so
     * the two are two orders of magnitude apart and a middle threshold cannot be wrong by
     * accident. Measuring beats believing the file extension.
     */
    private void calibrate() {
        java.util.Arrays.fill(pixels, PAD);
        fillInput();
        output.rewind();
        cpu.run(input, output);
        readOutput();

        float largest = 0;
        for (int anchor = 0; anchor < outputAnchors; anchor++) {
            for (int c = 0; c < 2; c++) {
                float value = outputPerAnchor
                        ? head[anchor * outputChannels + c] : head[c * outputAnchors + anchor];
                largest = Math.max(largest, value);
            }
        }
        boxScale = largest <= inputSize / 8f ? inputSize : 1f;
    }

    /**
     * Run one frame, leaving the numbers to decode in {@code head}.
     *
     * While the accelerator is on trial this runs both and keeps score, and the CPU's answer
     * is the one left behind: the trial must not be able to change what the operator sees.
     * It costs a few frames of doing the work twice, at startup, once. What it buys is the
     * difference between believing a delegate's claim and having watched it agree with a
     * known-good answer on this device, on this stream, on real frames.
     */
    private void run() {
        if (adopted && accelerated != null) {
            output.rewind();
            accelerated.run(input, output);
            readOutput();
            return;
        }

        long startedCpu = System.nanoTime();
        output.rewind();
        cpu.run(input, output);
        long cpuNanos = System.nanoTime() - startedCpu;
        readOutput();

        if (accelerated == null) {
            return;
        }

        int found = decodeHead().size();
        if (found == 0) {
            // Proves nothing either way. Both agreeing on an empty frame is exactly what a
            // broken delegate looks like.
            return;
        }
        float[] fromCpu = head.clone();

        int foundAccelerated;
        try {
            long startedAccelerated = System.nanoTime();
            input.rewind();
            output.rewind();
            accelerated.run(input, output);
            acceleratedNanosTotal += System.nanoTime() - startedAccelerated;
            readOutput();
            foundAccelerated = decodeHead().size();
        } catch (RuntimeException | Error broken) {
            closeAccelerated(acceleratedName + " (failed)");
            System.arraycopy(fromCpu, 0, head, 0, head.length);
            return;
        }

        acceleratedFoundTotal += foundAccelerated;
        cpuFoundTotal += found;
        cpuNanosTotal += cpuNanos;
        probedFrames++;
        System.arraycopy(fromCpu, 0, head, 0, head.length);

        if (probedFrames >= PROBE_FRAMES) {
            decide();
        }
    }

    /** The verdict, taken once, on the evidence. */
    private void decide() {
        boolean findsThem = acceleratedFoundTotal >= cpuFoundTotal * PROBE_RECALL;
        boolean faster = acceleratedNanosTotal > 0
                && cpuNanosTotal >= acceleratedNanosTotal * PROBE_SPEEDUP;

        if (findsThem && faster) {
            adopted = true;
            delegateName = acceleratedName;
            return;
        }
        // Named so the status line can say why. A delegate that is fast and blind is the
        // dangerous one, and it is worth being able to see that it was caught.
        closeAccelerated(findsThem
                ? "CPU (" + acceleratedName + " no faster)"
                : "CPU (" + acceleratedName + " missed people)");
    }

    private void closeAccelerated(String reason) {
        delegateName = reason;
        if (accelerated != null) {
            try {
                accelerated.close();
            } catch (RuntimeException | Error ignored) {
                // Closing an interpreter that has already fallen over is not worth reporting.
            }
            accelerated = null;
        }
        if (acceleratedDelegate != null) {
            try {
                acceleratedDelegate.close();
            } catch (Exception | Error ignored) {
                // As above.
            }
            acceleratedDelegate = null;
        }
        adopted = false;
    }

    /**
     * A score turned into the same three words the provider engines use.
     *
     * The rest of the application talks in certainty bands rather than percentages, because
     * showing a percentage here and a band there would make the two engines look like they
     * measure different things when they are answering the same question.
     */
    private static String certaintyOf(float score) {
        if (score >= 0.75f) {
            return "high";
        }
        return score >= 0.5f ? "medium" : "low";
    }

    public void close() {
        cpu.close();
        closeAccelerated(delegateName);
        square.recycle();
    }
}
