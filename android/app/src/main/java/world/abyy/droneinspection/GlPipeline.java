package world.abyy.droneinspection;

import android.graphics.Bitmap;
import android.graphics.SurfaceTexture;
import android.opengl.EGL14;
import android.opengl.EGLConfig;
import android.opengl.EGLContext;
import android.opengl.EGLDisplay;
import android.opengl.EGLExt;
import android.opengl.EGLSurface;
import android.opengl.GLES11Ext;
import android.opengl.GLES20;
import android.opengl.GLUtils;
import android.os.Handler;
import android.os.HandlerThread;
import android.view.Surface;

import androidx.annotation.Nullable;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.ArrayDeque;

/**
 * The video path, on the GPU, from the decoder to the screen and to the encoder.
 *
 * WHAT THIS REPLACED, AND WHY
 *     Every frame used to make a round trip through the CPU. TextureView.getBitmap() pulled
 *     it out of the GPU, the boxes were drawn onto it with a Canvas, and glTexImage2D pushed
 *     the whole thing back to the GPU for the encoder. At 1280x720 that is 3.7 MB down and
 *     3.7 MB up, fifteen times a second: about 110 MB/s of memory traffic to draw a few
 *     rectangles, and a readback stall on the thread that does it.
 *
 *     Here the frame never leaves the GPU. The decoder writes into a SurfaceTexture, which
 *     is an external texture, and that texture is drawn twice: once to the screen, once to
 *     the encoder's own surface with the overlay composited on top. The overlay is uploaded
 *     only when it changes, which is at detection rate rather than frame rate.
 *
 * WHAT STILL COMES BACK TO THE CPU, AND WHY IT HAS TO
 *     The model needs pixels. A frame for detection is rendered into a small offscreen
 *     buffer at 640 across and read back from there, which is a sixth of the pixels the old
 *     path moved and happens on this thread rather than the one drawing the screen. The
 *     provider's frame and a snapshot go the same way, at their own sizes, when asked for.
 *
 * THREADING
 *     Everything after start() happens on this class's own thread. The only methods safe to
 *     call from elsewhere are the ones documented as such, and they all work by posting.
 *     Callbacks arrive on this thread, not the main one, which is deliberate: what they hand
 *     over is a Bitmap that is about to be worked on, and bouncing it through the main thread
 *     first would put the work back where this class exists to take it away from.
 */
final class GlPipeline implements SurfaceTexture.OnFrameAvailableListener {

    /** A frame produced on request. Delivered on the pipeline's thread. */
    interface FrameConsumer {
        void onFrame(Bitmap frame);
    }

    interface Listener {
        /** GL could not be set up or has died. Delivered on the pipeline's thread. */
        void onFailed(String message);
    }

    private static final int EGL_RECORDABLE_ANDROID = 0x3142;

    /** Most requests than this waiting means the consumer is behind; the oldest go. */
    private static final int MAX_PENDING = 3;

    private static final String VERTEX = ""
            + "attribute vec4 aPosition;\n"
            + "attribute vec4 aTexCoord;\n"
            + "uniform mat4 uTexMatrix;\n"
            + "varying vec2 vTexCoord;\n"
            + "void main() {\n"
            + "  gl_Position = aPosition;\n"
            + "  vTexCoord = (uTexMatrix * aTexCoord).xy;\n"
            + "}\n";

    // samplerExternalOES, not sampler2D. The decoder's output is not an ordinary texture:
    // it is whatever native format the hardware decoder produces, and this is the only
    // sampler that can read it without a conversion pass.
    private static final String FRAGMENT_EXTERNAL = ""
            + "#extension GL_OES_EGL_image_external : require\n"
            + "precision mediump float;\n"
            + "varying vec2 vTexCoord;\n"
            + "uniform samplerExternalOES uTexture;\n"
            + "void main() {\n"
            + "  gl_FragColor = texture2D(uTexture, vTexCoord);\n"
            + "}\n";

    private static final String FRAGMENT_2D = ""
            + "precision mediump float;\n"
            + "varying vec2 vTexCoord;\n"
            + "uniform sampler2D uTexture;\n"
            + "void main() {\n"
            + "  gl_FragColor = texture2D(uTexture, vTexCoord);\n"
            + "}\n";

    private static final float[] QUAD = {
        -1f, -1f,  0f, 0f,
         1f, -1f,  1f, 0f,
        -1f,  1f,  0f, 1f,
         1f,  1f,  1f, 1f,
    };

    // The same quad with its vertical flipped. Used for the offscreen pass, because
    // glReadPixels reads bottom row first and an unflipped render comes back upside down.
    private static final float[] QUAD_FLIPPED = {
        -1f,  1f,  0f, 0f,
         1f,  1f,  1f, 0f,
        -1f, -1f,  0f, 1f,
         1f, -1f,  1f, 1f,
    };

    private static final float[] IDENTITY = {
        1f, 0f, 0f, 0f,
        0f, 1f, 0f, 0f,
        0f, 0f, 1f, 0f,
        0f, 0f, 0f, 1f,
    };

    private final Listener listener;
    private final HandlerThread thread = new HandlerThread("gl-pipeline");
    private final Handler gl;
    private final ArrayDeque<Request> requests = new ArrayDeque<>();
    private final float[] textureMatrix = new float[16];

    private EGLDisplay eglDisplay = EGL14.EGL_NO_DISPLAY;
    private EGLContext eglContext = EGL14.EGL_NO_CONTEXT;
    private EGLConfig eglConfig;
    private EGLSurface displaySurface = EGL14.EGL_NO_SURFACE;
    private EGLSurface encoderSurface = EGL14.EGL_NO_SURFACE;

    private SurfaceTexture videoTexture;
    private Surface videoInput;
    private int videoTextureId;
    private int overlayTextureId;
    private boolean overlayLoaded;

    private int externalProgram;
    private int flatProgram;
    private FloatBuffer quad;
    private FloatBuffer quadFlipped;

    private int frameBuffer;
    private int frameTexture;
    private int frameWidth;
    private int frameHeight;
    private ByteBuffer readback;

    private int displayWidth;
    private int displayHeight;
    private int videoWidth = 1280;
    private int videoHeight = 720;

    private int recordWidth;
    private int recordHeight;
    private long recordIntervalNanos;
    private long recordStartedNanos;
    private long nextRecordNanos;
    private Runnable onRecordedFrame;

    private volatile boolean alive;
    private volatile boolean started;
    private volatile long framesRendered;

    GlPipeline(Listener listener) {
        this.listener = listener;
        thread.start();
        gl = new Handler(thread.getLooper());
    }

    // ----------------------------------------------------------------- from other threads

    /**
     * Bring the pipeline up against a display surface, and block until it has tried.
     *
     * Blocking, unlike everything else here, because the caller needs the answer: it has to
     * know whether to hand the player this pipeline's surface or fall back to the old path,
     * and it cannot start playback until it knows.
     *
     * @return null on success, or why it failed.
     */
    @Nullable
    String start(Surface display, int width, int height) {
        final String[] problem = new String[1];
        final Object done = new Object();
        synchronized (done) {
            gl.post(() -> {
                try {
                    openEgl();
                    openSurface(display, width, height);
                    openPrograms();
                    openVideoTexture();
                    alive = true;
                    started = true;
                } catch (Exception | Error opening) {
                    problem[0] = String.valueOf(opening.getMessage());
                    releaseQuietly();
                }
                synchronized (done) {
                    done.notifyAll();
                }
            });
            try {
                done.wait(5000);
            } catch (InterruptedException interrupted) {
                Thread.currentThread().interrupt();
                return "interrupted while starting the video pipeline";
            }
        }
        if (!started && problem[0] == null) {
            return "the video pipeline did not start within five seconds";
        }
        return problem[0];
    }

    /** The surface to give the player. Null until {@link #start} has succeeded. */
    @Nullable
    Surface videoInput() {
        return videoInput;
    }

    /** How many frames have reached the screen. The caller uses it as a liveness check. */
    long framesRendered() {
        return framesRendered;
    }

    /** The display surface changed size. */
    void setDisplaySize(int width, int height) {
        gl.post(() -> {
            displayWidth = width;
            displayHeight = height;
        });
    }

    /** The player worked out the video's real dimensions. */
    void setVideoSize(int width, int height) {
        if (width <= 0 || height <= 0) {
            return;
        }
        gl.post(() -> {
            videoWidth = width;
            videoHeight = height;
            if (videoTexture != null) {
                videoTexture.setDefaultBufferSize(width, height);
            }
        });
    }

    /**
     * Start writing composed frames to an encoder's input surface.
     *
     * @param onFrame run on this thread after each frame is handed over, so the caller can
     *                drain the encoder without this class knowing what an encoder is.
     */
    void startRecording(Surface encoderInput, int width, int height, int fps, Runnable onFrame) {
        gl.post(() -> {
            try {
                encoderSurface = createWindowSurface(encoderInput);
                recordWidth = width;
                recordHeight = height;
                recordIntervalNanos = 1_000_000_000L / Math.max(1, fps);
                recordStartedNanos = System.nanoTime();
                nextRecordNanos = recordStartedNanos;
                onRecordedFrame = onFrame;
            } catch (Exception | Error opening) {
                encoderSurface = EGL14.EGL_NO_SURFACE;
                fail("could not attach to the encoder: " + opening.getMessage());
            }
        });
    }

    void stopRecording() {
        gl.post(() -> {
            onRecordedFrame = null;
            if (encoderSurface != EGL14.EGL_NO_SURFACE) {
                EGL14.eglDestroySurface(eglDisplay, encoderSurface);
                encoderSurface = EGL14.EGL_NO_SURFACE;
            }
        });
    }

    /**
     * Replace the overlay drawn into the recording. Takes ownership of the bitmap.
     *
     * Called when the boxes change rather than when a frame arrives, which is a few times a
     * second rather than fifteen, and is the only reason compositing the overlay in GL is
     * cheaper than drawing it onto every frame with a Canvas.
     */
    void setOverlay(Bitmap overlay) {
        gl.post(() -> {
            try {
                if (overlayTextureId == 0) {
                    overlayTextureId = createTexture(GLES20.GL_TEXTURE_2D);
                }
                GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, overlayTextureId);
                GLUtils.texImage2D(GLES20.GL_TEXTURE_2D, 0, overlay, 0);
                overlayLoaded = true;
            } catch (Exception | Error uploading) {
                overlayLoaded = false;
            } finally {
                overlay.recycle();
            }
        });
    }

    /**
     * Ask for one frame as a Bitmap, at most `longEdge` across.
     *
     * Served from the next video frame to arrive, not from the last one drawn, so what comes
     * back is current. Requests do not queue up without limit: if the consumer is slower than
     * the source, the oldest are dropped, because an old frame is worth less than a new one
     * and holding both is how a live view turns into a backlog.
     */
    void requestFrame(int longEdge, FrameConsumer consumer) {
        gl.post(() -> {
            while (requests.size() >= MAX_PENDING) {
                requests.removeFirst();
            }
            requests.addLast(new Request(longEdge, consumer));
        });
    }

    void release() {
        alive = false;
        gl.post(this::releaseQuietly);
        thread.quitSafely();
        try {
            thread.join(1000);
        } catch (InterruptedException interrupted) {
            Thread.currentThread().interrupt();
        }
    }

    // --------------------------------------------------------------------- the frame path

    @Override
    public void onFrameAvailable(SurfaceTexture texture) {
        gl.post(this::drawFrame);
    }

    private void drawFrame() {
        if (!alive || videoTexture == null) {
            return;
        }
        try {
            makeCurrent(displaySurface);
            videoTexture.updateTexImage();
            videoTexture.getTransformMatrix(textureMatrix);

            drawToDisplay();
            drawToEncoder();
            serveOneRequest();
            framesRendered++;
        } catch (Exception | Error drawing) {
            fail("the video pipeline stopped: " + drawing.getMessage());
        }
    }

    /** The screen: the video letterboxed into whatever shape the view is. */
    private void drawToDisplay() {
        if (displayWidth <= 0 || displayHeight <= 0) {
            return;
        }
        makeCurrent(displaySurface);
        GLES20.glDisable(GLES20.GL_BLEND);
        GLES20.glClearColor(0f, 0f, 0f, 1f);
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT);

        // Letterboxed rather than stretched. A drone feed with the wrong aspect ratio is
        // worse than black bars: distances and shapes on it are what the operator is
        // judging, and a stretched picture quietly misrepresents both.
        float videoAspect = videoWidth / (float) videoHeight;
        float viewAspect = displayWidth / (float) displayHeight;
        int width;
        int height;
        if (viewAspect > videoAspect) {
            height = displayHeight;
            width = Math.round(displayHeight * videoAspect);
        } else {
            width = displayWidth;
            height = Math.round(displayWidth / videoAspect);
        }
        GLES20.glViewport((displayWidth - width) / 2, (displayHeight - height) / 2, width, height);
        drawExternal(quad, textureMatrix);
        EGL14.eglSwapBuffers(eglDisplay, displaySurface);
    }

    /** The file: the video at its own size, with the boxes burnt into it. */
    private void drawToEncoder() {
        if (encoderSurface == EGL14.EGL_NO_SURFACE) {
            return;
        }
        long now = System.nanoTime();
        if (now < nextRecordNanos) {
            return;   // this frame is not due; the encoder is written at its own rate
        }
        nextRecordNanos += recordIntervalNanos;
        if (nextRecordNanos <= now) {
            // A whole frame behind or worse. Do not try to catch up: submitting a burst
            // makes the picture jump, which is the thing being avoided.
            nextRecordNanos = now + recordIntervalNanos;
        }

        makeCurrent(encoderSurface);
        GLES20.glDisable(GLES20.GL_BLEND);
        GLES20.glViewport(0, 0, recordWidth, recordHeight);
        drawExternal(quad, textureMatrix);

        if (overlayLoaded) {
            // Android bitmaps arrive with their alpha premultiplied, so the source factor is
            // ONE rather than SRC_ALPHA. With SRC_ALPHA the boxes come out washed out over
            // anything dark, which looks like a rendering fault and is a blend mode.
            GLES20.glEnable(GLES20.GL_BLEND);
            GLES20.glBlendFunc(GLES20.GL_ONE, GLES20.GL_ONE_MINUS_SRC_ALPHA);
            drawFlat(quad, overlayTextureId);
            GLES20.glDisable(GLES20.GL_BLEND);
        }

        EGLExt.eglPresentationTimeANDROID(eglDisplay, encoderSurface, now - recordStartedNanos);
        EGL14.eglSwapBuffers(eglDisplay, encoderSurface);

        Runnable drain = onRecordedFrame;
        if (drain != null) {
            drain.run();
        }
    }

    /** The model's frame: rendered small, offscreen, and read back. */
    private void serveOneRequest() {
        Request request = requests.pollFirst();
        if (request == null) {
            return;
        }
        int longEdge = Math.max(16, request.longEdge);
        float scale = Math.min(1f, longEdge / (float) Math.max(videoWidth, videoHeight));
        int width = Math.max(2, Math.round(videoWidth * scale));
        int height = Math.max(2, Math.round(videoHeight * scale));

        Bitmap frame = null;
        try {
            ensureFrameBuffer(width, height);
            GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, frameBuffer);
            GLES20.glDisable(GLES20.GL_BLEND);
            GLES20.glViewport(0, 0, width, height);
            // The flipped quad, because glReadPixels starts at the bottom row and an
            // unflipped render therefore comes back upside down.
            drawExternal(quadFlipped, textureMatrix);

            readback.rewind();
            GLES20.glReadPixels(0, 0, width, height,
                    GLES20.GL_RGBA, GLES20.GL_UNSIGNED_BYTE, readback);
            readback.rewind();

            frame = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
            frame.copyPixelsFromBuffer(readback);
        } catch (Exception | Error reading) {
            if (frame != null) {
                frame.recycle();
                frame = null;
            }
        } finally {
            GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, 0);
        }
        if (frame != null) {
            request.consumer.onFrame(frame);
        }
    }

    // ------------------------------------------------------------------------ GL plumbing

    private void drawExternal(FloatBuffer geometry, float[] matrix) {
        GLES20.glUseProgram(externalProgram);
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, videoTextureId);
        GLES20.glUniform1i(GLES20.glGetUniformLocation(externalProgram, "uTexture"), 0);
        GLES20.glUniformMatrix4fv(
                GLES20.glGetUniformLocation(externalProgram, "uTexMatrix"), 1, false, matrix, 0);
        drawQuad(externalProgram, geometry);
    }

    private void drawFlat(FloatBuffer geometry, int texture) {
        GLES20.glUseProgram(flatProgram);
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, texture);
        GLES20.glUniform1i(GLES20.glGetUniformLocation(flatProgram, "uTexture"), 0);
        GLES20.glUniformMatrix4fv(
                GLES20.glGetUniformLocation(flatProgram, "uTexMatrix"), 1, false, IDENTITY, 0);
        drawQuad(flatProgram, geometry);
    }

    private void drawQuad(int program, FloatBuffer geometry) {
        int position = GLES20.glGetAttribLocation(program, "aPosition");
        int coordinate = GLES20.glGetAttribLocation(program, "aTexCoord");

        geometry.position(0);
        GLES20.glVertexAttribPointer(position, 2, GLES20.GL_FLOAT, false, 16, geometry);
        GLES20.glEnableVertexAttribArray(position);
        geometry.position(2);
        GLES20.glVertexAttribPointer(coordinate, 2, GLES20.GL_FLOAT, false, 16, geometry);
        GLES20.glEnableVertexAttribArray(coordinate);

        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4);
        GLES20.glDisableVertexAttribArray(position);
        GLES20.glDisableVertexAttribArray(coordinate);
    }

    private void openEgl() {
        eglDisplay = EGL14.eglGetDisplay(EGL14.EGL_DEFAULT_DISPLAY);
        if (eglDisplay == EGL14.EGL_NO_DISPLAY) {
            throw new IllegalStateException("no EGL display");
        }
        int[] version = new int[2];
        if (!EGL14.eglInitialize(eglDisplay, version, 0, version, 1)) {
            throw new IllegalStateException("eglInitialize failed");
        }
        // EGL_RECORDABLE_ANDROID is what makes a config usable as an encoder's input. A
        // config chosen without it can render to the screen perfectly and produce a green or
        // empty video file, with no error anywhere to say why.
        int[] attributes = {
            EGL14.EGL_RED_SIZE, 8,
            EGL14.EGL_GREEN_SIZE, 8,
            EGL14.EGL_BLUE_SIZE, 8,
            EGL14.EGL_ALPHA_SIZE, 8,
            EGL14.EGL_RENDERABLE_TYPE, EGL14.EGL_OPENGL_ES2_BIT,
            EGL_RECORDABLE_ANDROID, 1,
            EGL14.EGL_NONE,
        };
        EGLConfig[] configs = new EGLConfig[1];
        int[] found = new int[1];
        if (!EGL14.eglChooseConfig(eglDisplay, attributes, 0, configs, 0, 1, found, 0)
                || found[0] <= 0) {
            throw new IllegalStateException("no recordable EGL config");
        }
        eglConfig = configs[0];
        eglContext = EGL14.eglCreateContext(eglDisplay, eglConfig, EGL14.EGL_NO_CONTEXT,
                new int[]{EGL14.EGL_CONTEXT_CLIENT_VERSION, 2, EGL14.EGL_NONE}, 0);
        if (eglContext == null || eglContext == EGL14.EGL_NO_CONTEXT) {
            throw new IllegalStateException("eglCreateContext failed");
        }
    }

    private void openSurface(Surface display, int width, int height) {
        displaySurface = createWindowSurface(display);
        displayWidth = width;
        displayHeight = height;
        makeCurrent(displaySurface);
    }

    private EGLSurface createWindowSurface(Surface surface) {
        EGLSurface created = EGL14.eglCreateWindowSurface(eglDisplay, eglConfig, surface,
                new int[]{EGL14.EGL_NONE}, 0);
        if (created == null || created == EGL14.EGL_NO_SURFACE) {
            throw new IllegalStateException("eglCreateWindowSurface failed");
        }
        return created;
    }

    private void makeCurrent(EGLSurface surface) {
        if (!EGL14.eglMakeCurrent(eglDisplay, surface, surface, eglContext)) {
            throw new IllegalStateException("eglMakeCurrent failed");
        }
    }

    private void openPrograms() {
        externalProgram = link(VERTEX, FRAGMENT_EXTERNAL);
        flatProgram = link(VERTEX, FRAGMENT_2D);
        quad = toBuffer(QUAD);
        quadFlipped = toBuffer(QUAD_FLIPPED);
    }

    private void openVideoTexture() {
        videoTextureId = createTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES);
        videoTexture = new SurfaceTexture(videoTextureId);
        videoTexture.setDefaultBufferSize(videoWidth, videoHeight);
        videoTexture.setOnFrameAvailableListener(this, gl);
        videoInput = new Surface(videoTexture);
    }

    private void ensureFrameBuffer(int width, int height) {
        if (frameBuffer != 0 && width == frameWidth && height == frameHeight) {
            return;
        }
        deleteFrameBuffer();

        int[] handles = new int[1];
        GLES20.glGenFramebuffers(1, handles, 0);
        frameBuffer = handles[0];
        frameTexture = createTexture(GLES20.GL_TEXTURE_2D);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, frameTexture);
        GLES20.glTexImage2D(GLES20.GL_TEXTURE_2D, 0, GLES20.GL_RGBA, width, height, 0,
                GLES20.GL_RGBA, GLES20.GL_UNSIGNED_BYTE, null);
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, frameBuffer);
        GLES20.glFramebufferTexture2D(GLES20.GL_FRAMEBUFFER, GLES20.GL_COLOR_ATTACHMENT0,
                GLES20.GL_TEXTURE_2D, frameTexture, 0);
        int status = GLES20.glCheckFramebufferStatus(GLES20.GL_FRAMEBUFFER);
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, 0);
        if (status != GLES20.GL_FRAMEBUFFER_COMPLETE) {
            deleteFrameBuffer();
            throw new IllegalStateException("offscreen buffer incomplete: " + status);
        }
        frameWidth = width;
        frameHeight = height;
        readback = ByteBuffer.allocateDirect(width * height * 4).order(ByteOrder.nativeOrder());
    }

    private void deleteFrameBuffer() {
        if (frameBuffer != 0) {
            GLES20.glDeleteFramebuffers(1, new int[]{frameBuffer}, 0);
            frameBuffer = 0;
        }
        if (frameTexture != 0) {
            GLES20.glDeleteTextures(1, new int[]{frameTexture}, 0);
            frameTexture = 0;
        }
        frameWidth = 0;
        frameHeight = 0;
        readback = null;
    }

    private static int createTexture(int target) {
        int[] handles = new int[1];
        GLES20.glGenTextures(1, handles, 0);
        GLES20.glBindTexture(target, handles[0]);
        GLES20.glTexParameteri(target, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR);
        GLES20.glTexParameteri(target, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR);
        GLES20.glTexParameteri(target, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE);
        GLES20.glTexParameteri(target, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE);
        return handles[0];
    }

    private static int link(String vertexSource, String fragmentSource) {
        int vertex = compile(GLES20.GL_VERTEX_SHADER, vertexSource);
        int fragment = compile(GLES20.GL_FRAGMENT_SHADER, fragmentSource);
        int program = GLES20.glCreateProgram();
        GLES20.glAttachShader(program, vertex);
        GLES20.glAttachShader(program, fragment);
        GLES20.glLinkProgram(program);
        int[] linked = new int[1];
        GLES20.glGetProgramiv(program, GLES20.GL_LINK_STATUS, linked, 0);
        if (linked[0] == 0) {
            String why = GLES20.glGetProgramInfoLog(program);
            GLES20.glDeleteProgram(program);
            throw new IllegalStateException("shader link failed: " + why);
        }
        GLES20.glDeleteShader(vertex);
        GLES20.glDeleteShader(fragment);
        return program;
    }

    private static int compile(int type, String source) {
        int shader = GLES20.glCreateShader(type);
        GLES20.glShaderSource(shader, source);
        GLES20.glCompileShader(shader);
        int[] compiled = new int[1];
        GLES20.glGetShaderiv(shader, GLES20.GL_COMPILE_STATUS, compiled, 0);
        if (compiled[0] == 0) {
            String why = GLES20.glGetShaderInfoLog(shader);
            GLES20.glDeleteShader(shader);
            throw new IllegalStateException("shader compile failed: " + why);
        }
        return shader;
    }

    private static FloatBuffer toBuffer(float[] values) {
        FloatBuffer buffer = ByteBuffer.allocateDirect(values.length * 4)
                .order(ByteOrder.nativeOrder()).asFloatBuffer();
        buffer.put(values).position(0);
        return buffer;
    }

    private void fail(String message) {
        alive = false;
        listener.onFailed(message);
    }

    private void releaseQuietly() {
        alive = false;
        started = false;
        try {
            deleteFrameBuffer();
        } catch (Exception | Error ignored) {
            // Tearing down, and a failure here has nowhere useful to go.
        }
        if (videoInput != null) {
            videoInput.release();
            videoInput = null;
        }
        if (videoTexture != null) {
            videoTexture.release();
            videoTexture = null;
        }
        if (eglDisplay != EGL14.EGL_NO_DISPLAY) {
            EGL14.eglMakeCurrent(eglDisplay, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_SURFACE,
                    EGL14.EGL_NO_CONTEXT);
            if (encoderSurface != EGL14.EGL_NO_SURFACE) {
                EGL14.eglDestroySurface(eglDisplay, encoderSurface);
                encoderSurface = EGL14.EGL_NO_SURFACE;
            }
            if (displaySurface != EGL14.EGL_NO_SURFACE) {
                EGL14.eglDestroySurface(eglDisplay, displaySurface);
                displaySurface = EGL14.EGL_NO_SURFACE;
            }
            if (eglContext != EGL14.EGL_NO_CONTEXT) {
                EGL14.eglDestroyContext(eglDisplay, eglContext);
                eglContext = EGL14.EGL_NO_CONTEXT;
            }
            EGL14.eglTerminate(eglDisplay);
            eglDisplay = EGL14.EGL_NO_DISPLAY;
        }
    }

    private static final class Request {
        final int longEdge;
        final FrameConsumer consumer;

        Request(int longEdge, FrameConsumer consumer) {
            this.longEdge = longEdge;
            this.consumer = consumer;
        }
    }
}
