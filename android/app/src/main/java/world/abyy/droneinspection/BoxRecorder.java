package world.abyy.droneinspection;

import android.graphics.Bitmap;
import android.media.MediaCodec;
import android.media.MediaCodecInfo;
import android.media.MediaFormat;
import android.media.MediaMuxer;
import android.opengl.EGL14;
import android.opengl.EGLConfig;
import android.opengl.EGLContext;
import android.opengl.EGLDisplay;
import android.opengl.EGLExt;
import android.opengl.EGLSurface;
import android.opengl.GLES20;
import android.opengl.GLUtils;
import android.os.Handler;
import android.os.HandlerThread;
import android.view.Surface;

import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

/**
 * Records what is on screen, boxes included, to an MP4.
 *
 * WHAT IS SAVED, AND WHY ONLY ONE FILE
 *     The annotated video only. Keeping the clean original too would need a second encoder
 *     running at the same time on a handheld, and it is not what the recording is for: this
 *     file exists to be reviewed and to be re-uploaded to the analysis screen afterwards,
 *     and both of those want the boxes.
 *
 * TWO WAYS FRAMES GET HERE
 *     Preferred, and what runs when the GPU pipeline is up: nothing arrives here at all.
 *     GlPipeline is handed input() and renders the decoder's texture and the overlay
 *     straight onto it, so the frame never leaves the GPU. This class is then only an
 *     encoder and a muxer, and drainNow() is called after each frame is presented.
 *
 *     Fallback, when GL could not be brought up: the composed frame is read back as a
 *     Bitmap and handed to submit(), which uploads it to a small EGL context of this
 *     class's own and draws it onto the encoder's surface. That readback is a real cost,
 *     which is why the fallback records at FRAME_RATE rather than at the stream's own rate.
 *
 * THREADING
 *     Everything after submit() happens on this class's own thread, because an EGL context
 *     belongs to the thread that made it. submit() copies nothing and takes ownership of the
 *     bitmap it is given, so the caller must not draw into it again.
 */
public final class BoxRecorder {

    public static final int FRAME_RATE = 15;
    private static final String MIME = "video/avc";
    private static final int I_FRAME_INTERVAL_SECONDS = 1;
    private static final int TIMEOUT_US = 10_000;

    private final HandlerThread thread = new HandlerThread("box-recorder");
    private final Handler handler;
    private final int width;
    private final int height;
    private final File output;

    /**
     * True when this class composes frames itself from bitmaps, false when something else
     * renders onto input(). The EGL context below exists only in the first case: bringing up
     * a second one alongside GlPipeline's would be two contexts fighting over one encoder.
     */
    private final boolean composes;

    private MediaCodec encoder;
    private Surface inputSurface;
    private MediaMuxer muxer;
    private int trackIndex = -1;
    private boolean muxerStarted;
    private long startedAtNanos;
    private volatile boolean running;
    private volatile String failure;

    // EGL
    private EGLDisplay eglDisplay = EGL14.EGL_NO_DISPLAY;
    private EGLContext eglContext = EGL14.EGL_NO_CONTEXT;
    private EGLSurface eglSurface = EGL14.EGL_NO_SURFACE;
    private int program;
    private int textureId;
    private FloatBuffer vertices;

    private static final String VERTEX_SHADER =
            "attribute vec4 aPosition;\n"
            + "attribute vec2 aTexCoord;\n"
            + "varying vec2 vTexCoord;\n"
            + "void main() {\n"
            + "  gl_Position = aPosition;\n"
            + "  vTexCoord = aTexCoord;\n"
            + "}\n";

    private static final String FRAGMENT_SHADER =
            "precision mediump float;\n"
            + "varying vec2 vTexCoord;\n"
            + "uniform sampler2D uTexture;\n"
            + "void main() {\n"
            + "  gl_FragColor = texture2D(uTexture, vTexCoord);\n"
            + "}\n";

    /** The fallback: this recorder composes frames from bitmaps handed to submit(). */
    public BoxRecorder(File output, int width, int height) {
        this(output, width, height, true);
    }

    public BoxRecorder(File output, int width, int height, boolean composes) {
        this.composes = composes;
        this.output = output;
        // H.264 encoders reject odd dimensions on a great many devices, and the failure is
        // an opaque IllegalStateException at configure() rather than anything readable.
        this.width = width & ~1;
        this.height = height & ~1;
        thread.start();
        handler = new Handler(thread.getLooper());
    }

    public boolean isRunning() {
        return running;
    }

    /** Non-null once something has gone wrong; the screen shows it rather than staying quiet. */
    public String failure() {
        return failure;
    }

    public File output() {
        return output;
    }

    /** Milliseconds of video written so far. */
    public long elapsedMillis() {
        return startedAtNanos == 0 ? 0 : (System.nanoTime() - startedAtNanos) / 1_000_000;
    }

    public void start() {
        handler.post(this::openEverything);
    }

    /**
     * Start, and wait for the encoder to exist.
     *
     * The GL path needs input() the moment this returns, and a Surface that does not exist
     * yet is not something a caller can be handed and told to try again later.
     *
     * @return null on success, or why it failed.
     */
    public String startAndWait() {
        final Object done = new Object();
        synchronized (done) {
            handler.post(() -> {
                openEverything();
                synchronized (done) {
                    done.notifyAll();
                }
            });
            try {
                done.wait(5000);
            } catch (InterruptedException interrupted) {
                Thread.currentThread().interrupt();
                return "interrupted while starting the recorder";
            }
        }
        if (failure != null) {
            return failure;
        }
        return running ? null : "the recorder did not start within five seconds";
    }

    private void openEverything() {
        try {
            openEncoder();
            if (composes) {
                openGl();
            }
            running = true;
            startedAtNanos = System.nanoTime();
        } catch (Exception opening) {
            failure = describe(opening);
            releaseQuietly();
        }
    }

    /**
     * The encoder's input surface, for something else to render onto.
     *
     * Only meaningful after startAndWait() has returned null, and only when this recorder
     * was built with composes = false.
     */
    public Surface input() {
        return inputSurface;
    }

    /**
     * Take whatever the encoder has produced and write it out.
     *
     * Called from the GL thread after each frame is presented, and posted rather than run
     * there: the muxer writes to a file, and file writes do not belong on the thread that
     * has to be ready for the next video frame.
     */
    public void drainNow() {
        handler.post(() -> {
            if (!running) {
                return;
            }
            try {
                drain(false);
            } catch (Exception encoding) {
                failure = describe(encoding);
                running = false;
            }
        });
    }

    /**
     * Hand over one composed frame.
     *
     * Dropped silently if the recorder is not running or is already behind. Dropping a frame
     * is the correct response to being behind: queueing them would grow without bound and
     * end the recording in an out-of-memory kill, which loses the whole file rather than one
     * frame of it.
     */
    public void submit(Bitmap frame) {
        if (!running || frame == null) {
            if (frame != null) {
                frame.recycle();
            }
            return;
        }
        boolean posted = handler.post(() -> {
            try {
                if (running) {
                    drawFrame(frame);
                    drain(false);
                }
            } catch (Exception encoding) {
                failure = describe(encoding);
                running = false;
            } finally {
                frame.recycle();
            }
        });
        if (!posted) {
            frame.recycle();
        }
    }

    /** Finish the file. The callback runs on this recorder's thread. */
    public void stop(Runnable onFinished) {
        handler.post(() -> {
            try {
                if (running) {
                    running = false;
                    encoder.signalEndOfInputStream();
                    drain(true);
                }
            } catch (Exception stopping) {
                failure = describe(stopping);
            } finally {
                releaseQuietly();
                if (onFinished != null) {
                    onFinished.run();
                }
                thread.quitSafely();
            }
        });
    }

    // -----------------------------------------------------------------------------------
    // Encoder
    // -----------------------------------------------------------------------------------

    private void openEncoder() throws IOException {
        MediaFormat format = MediaFormat.createVideoFormat(MIME, width, height);
        format.setInteger(MediaFormat.KEY_COLOR_FORMAT,
                MediaCodecInfo.CodecCapabilities.COLOR_FormatSurface);
        // Enough for a 720p review copy without filling a controller's storage on a long
        // flight: roughly 45 MB per minute.
        format.setInteger(MediaFormat.KEY_BIT_RATE, 6_000_000);
        format.setInteger(MediaFormat.KEY_FRAME_RATE, FRAME_RATE);
        format.setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, I_FRAME_INTERVAL_SECONDS);

        encoder = MediaCodec.createEncoderByType(MIME);
        encoder.configure(format, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE);
        inputSurface = encoder.createInputSurface();
        encoder.start();

        muxer = new MediaMuxer(output.getAbsolutePath(), MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4);
    }

    private void drain(boolean endOfStream) {
        MediaCodec.BufferInfo info = new MediaCodec.BufferInfo();
        while (true) {
            int status = encoder.dequeueOutputBuffer(info, endOfStream ? TIMEOUT_US : 0);
            if (status == MediaCodec.INFO_TRY_AGAIN_LATER) {
                if (!endOfStream) {
                    return;
                }
                continue;
            }
            if (status == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                if (muxerStarted) {
                    throw new IllegalStateException("the encoder changed format twice");
                }
                trackIndex = muxer.addTrack(encoder.getOutputFormat());
                muxer.start();
                muxerStarted = true;
                continue;
            }
            if (status < 0) {
                continue;
            }

            ByteBuffer encoded = encoder.getOutputBuffer(status);
            if ((info.flags & MediaCodec.BUFFER_FLAG_CODEC_CONFIG) != 0) {
                // Codec configuration, already carried in the track format.
                info.size = 0;
            }
            if (info.size != 0 && muxerStarted && encoded != null) {
                encoded.position(info.offset);
                encoded.limit(info.offset + info.size);
                muxer.writeSampleData(trackIndex, encoded, info);
            }
            encoder.releaseOutputBuffer(status, false);

            if ((info.flags & MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0) {
                return;
            }
        }
    }

    // -----------------------------------------------------------------------------------
    // GL
    // -----------------------------------------------------------------------------------

    private void openGl() {
        eglDisplay = EGL14.eglGetDisplay(EGL14.EGL_DEFAULT_DISPLAY);
        if (eglDisplay == EGL14.EGL_NO_DISPLAY) {
            throw new IllegalStateException("no EGL display");
        }
        int[] version = new int[2];
        if (!EGL14.eglInitialize(eglDisplay, version, 0, version, 1)) {
            throw new IllegalStateException("eglInitialize failed");
        }

        int[] attributes = {
                EGL14.EGL_RED_SIZE, 8,
                EGL14.EGL_GREEN_SIZE, 8,
                EGL14.EGL_BLUE_SIZE, 8,
                EGL14.EGL_ALPHA_SIZE, 8,
                EGL14.EGL_RENDERABLE_TYPE, EGL14.EGL_OPENGL_ES2_BIT,
                // Tells EGL this config will be used with a MediaCodec input surface.
                EGLExt.EGL_RECORDABLE_ANDROID, 1,
                EGL14.EGL_NONE,
        };
        EGLConfig[] configs = new EGLConfig[1];
        int[] found = new int[1];
        if (!EGL14.eglChooseConfig(eglDisplay, attributes, 0, configs, 0, 1, found, 0)
                || found[0] == 0) {
            throw new IllegalStateException("no recordable EGL config");
        }

        eglContext = EGL14.eglCreateContext(eglDisplay, configs[0], EGL14.EGL_NO_CONTEXT,
                new int[]{EGL14.EGL_CONTEXT_CLIENT_VERSION, 2, EGL14.EGL_NONE}, 0);
        if (eglContext == EGL14.EGL_NO_CONTEXT) {
            throw new IllegalStateException("eglCreateContext failed");
        }
        eglSurface = EGL14.eglCreateWindowSurface(eglDisplay, configs[0], inputSurface,
                new int[]{EGL14.EGL_NONE}, 0);
        if (eglSurface == EGL14.EGL_NO_SURFACE) {
            throw new IllegalStateException("eglCreateWindowSurface failed");
        }
        if (!EGL14.eglMakeCurrent(eglDisplay, eglSurface, eglSurface, eglContext)) {
            throw new IllegalStateException("eglMakeCurrent failed");
        }

        program = buildProgram();

        // Full-screen quad. The texture coordinates are flipped vertically because a Bitmap
        // has its origin at the top left and GL has it at the bottom left; without this the
        // recording comes out upside down.
        float[] data = {
                -1f, -1f, 0f, 1f,
                 1f, -1f, 1f, 1f,
                -1f,  1f, 0f, 0f,
                 1f,  1f, 1f, 0f,
        };
        vertices = ByteBuffer.allocateDirect(data.length * 4)
                .order(ByteOrder.nativeOrder()).asFloatBuffer();
        vertices.put(data).position(0);

        int[] textures = new int[1];
        GLES20.glGenTextures(1, textures, 0);
        textureId = textures[0];
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, textureId);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE);
    }

    private int buildProgram() {
        int vertex = compile(GLES20.GL_VERTEX_SHADER, VERTEX_SHADER);
        int fragment = compile(GLES20.GL_FRAGMENT_SHADER, FRAGMENT_SHADER);
        int handle = GLES20.glCreateProgram();
        GLES20.glAttachShader(handle, vertex);
        GLES20.glAttachShader(handle, fragment);
        GLES20.glLinkProgram(handle);

        int[] linked = new int[1];
        GLES20.glGetProgramiv(handle, GLES20.GL_LINK_STATUS, linked, 0);
        if (linked[0] == 0) {
            String log = GLES20.glGetProgramInfoLog(handle);
            GLES20.glDeleteProgram(handle);
            throw new IllegalStateException("shader link failed: " + log);
        }
        GLES20.glDeleteShader(vertex);
        GLES20.glDeleteShader(fragment);
        return handle;
    }

    private int compile(int type, String source) {
        int shader = GLES20.glCreateShader(type);
        GLES20.glShaderSource(shader, source);
        GLES20.glCompileShader(shader);
        int[] compiled = new int[1];
        GLES20.glGetShaderiv(shader, GLES20.GL_COMPILE_STATUS, compiled, 0);
        if (compiled[0] == 0) {
            String log = GLES20.glGetShaderInfoLog(shader);
            GLES20.glDeleteShader(shader);
            throw new IllegalStateException("shader compile failed: " + log);
        }
        return shader;
    }

    private void drawFrame(Bitmap frame) {
        GLES20.glViewport(0, 0, width, height);
        GLES20.glClearColor(0f, 0f, 0f, 1f);
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT);

        GLES20.glUseProgram(program);
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, textureId);
        GLUtils.texImage2D(GLES20.GL_TEXTURE_2D, 0, frame, 0);
        GLES20.glUniform1i(GLES20.glGetUniformLocation(program, "uTexture"), 0);

        int position = GLES20.glGetAttribLocation(program, "aPosition");
        int texCoord = GLES20.glGetAttribLocation(program, "aTexCoord");

        vertices.position(0);
        GLES20.glVertexAttribPointer(position, 2, GLES20.GL_FLOAT, false, 16, vertices);
        GLES20.glEnableVertexAttribArray(position);
        vertices.position(2);
        GLES20.glVertexAttribPointer(texCoord, 2, GLES20.GL_FLOAT, false, 16, vertices);
        GLES20.glEnableVertexAttribArray(texCoord);

        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4);
        GLES20.glDisableVertexAttribArray(position);
        GLES20.glDisableVertexAttribArray(texCoord);

        // The presentation timestamp is wall-clock since start, not a frame counter times a
        // nominal interval. Frames get dropped when the tablet is busy, and a counter would
        // silently stretch a thirty-second recording into a minute of slow motion.
        EGLExt.eglPresentationTimeANDROID(eglDisplay, eglSurface,
                System.nanoTime() - startedAtNanos);
        EGL14.eglSwapBuffers(eglDisplay, eglSurface);
    }

    // -----------------------------------------------------------------------------------
    // Teardown
    // -----------------------------------------------------------------------------------

    private static String describe(Exception problem) {
        String message = problem.getMessage();
        return problem.getClass().getSimpleName() + (message == null ? "" : ": " + message);
    }

    private void releaseQuietly() {
        running = false;
        if (eglDisplay != EGL14.EGL_NO_DISPLAY) {
            EGL14.eglMakeCurrent(eglDisplay, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_SURFACE,
                    EGL14.EGL_NO_CONTEXT);
            if (eglSurface != EGL14.EGL_NO_SURFACE) {
                EGL14.eglDestroySurface(eglDisplay, eglSurface);
            }
            if (eglContext != EGL14.EGL_NO_CONTEXT) {
                EGL14.eglDestroyContext(eglDisplay, eglContext);
            }
            EGL14.eglReleaseThread();
            EGL14.eglTerminate(eglDisplay);
        }
        eglDisplay = EGL14.EGL_NO_DISPLAY;
        eglContext = EGL14.EGL_NO_CONTEXT;
        eglSurface = EGL14.EGL_NO_SURFACE;

        if (inputSurface != null) {
            inputSurface.release();
            inputSurface = null;
        }
        if (encoder != null) {
            try {
                encoder.stop();
            } catch (Exception ignored) {
                // Already stopped, or never started. Nothing useful to do either way.
            }
            encoder.release();
            encoder = null;
        }
        if (muxer != null) {
            try {
                if (muxerStarted) {
                    muxer.stop();
                }
            } catch (Exception ignored) {
                // A muxer with no samples throws on stop; the file is simply empty.
            }
            muxer.release();
            muxer = null;
        }
        muxerStarted = false;
    }
}
