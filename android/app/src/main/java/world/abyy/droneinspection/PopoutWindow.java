package world.abyy.droneinspection;

import android.annotation.SuppressLint;
import android.content.Context;
import android.graphics.PixelFormat;
import android.graphics.Point;
import android.os.Build;
import android.view.ContextThemeWrapper;
import android.view.Display;
import android.view.LayoutInflater;
import android.view.MotionEvent;
import android.view.ScaleGestureDetector;
import android.view.Surface;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;
import android.view.ViewConfiguration;
import android.view.WindowManager;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

/**
 * The drone's feed in a window the operator can drag and resize, over whatever else is running.
 *
 * WHY THIS EXISTS RATHER THAN PICTURE-IN-PICTURE
 *     The app already had a small floating window, and it was the system's:
 *     enterPictureInPictureMode, which takes an aspect ratio and nothing else. Android picks
 *     the size, and on Android 9 there is one size and no way for anyone to change it.
 *     Pinch-to-resize of a picture-in-picture window arrived in Android 11. The controller
 *     this flies on is Android 9, so no combination of PictureInPictureParams produces a
 *     window that can be made bigger. The only way to get one is to stop asking the system
 *     for a window and add our own.
 *
 *     So this is a WindowManager overlay: our view hierarchy, our layout parameters, and
 *     therefore our size. It needs SYSTEM_ALERT_WINDOW, which is why the permission is asked
 *     for at the moment the button is pressed, and why refusing it falls back to the old
 *     picture-in-picture path rather than leaving the button dead.
 *
 * WHAT IT DOES NOT DO
 *     It does not decode, detect or record anything. It is a second surface for a pipeline
 *     that is already running: LiveActivity hands GlPipeline this window's surface instead of
 *     the one on the live screen, and hands it back when the window closes. Nothing about the
 *     stream, the tracking or a recording in progress changes, which is the point - the
 *     picture moves, the work does not restart.
 *
 * KEEPING THE PROCESS ALIVE
 *     Also not this class's job, and worth saying because it is the part that would fail
 *     silently. With the system's window the activity stayed resumed; with ours it is stopped
 *     the moment another app takes the screen, and a stopped activity's process is killable.
 *     LiveActivity holds RecordingService up for as long as this window is open, the same way
 *     it does for a recording, and skips the teardown in onStop.
 */
final class PopoutWindow {

    /** What the window needs from whoever opened it. All delivered on the main thread. */
    interface Listener {
        /** A surface to draw the video into. */
        void onPopoutSurfaceCreated(Surface surface, int width, int height);

        /** The window was resized, so the picture has a new size to fit. */
        void onPopoutSurfaceChanged(int width, int height);

        /** The surface is going. Nothing may still be drawing into it when this returns. */
        void onPopoutSurfaceDestroyed();

        /** The picture was tapped: come back to the full screen. */
        void onPopoutTapped();

        /** The window was closed from its own button. */
        void onPopoutClosed();
    }

    /**
     * Small enough to keep out of the way, large enough that a box on a person is still a box
     * and not three pixels. Below about this the window stops being useful as a view of
     * anything and becomes a badge.
     */
    private static final int MIN_WIDTH_DP = 140;

    /**
     * A window wider than the screen cannot be moved back off it with the drag bar, because
     * the bar goes with it. Bounded to the screen for that reason rather than for looks.
     */
    private static final float MAX_SCREEN_FRACTION = 1.0f;

    private final Context context;
    private final Listener listener;
    private final WindowManager windows;
    private final WindowManager.LayoutParams params = new WindowManager.LayoutParams();

    private final View root;
    private final SurfaceView video;
    private final OverlayView overlay;
    private final ScaleGestureDetector pinch;

    private final int minWidth;
    private final int screenWidth;
    private final int screenHeight;
    private final int touchSlop;

    /** The video's shape, so resizing never stretches the picture. */
    private float aspect;

    private boolean added;

    @SuppressLint("ClickableViewAccessibility")
    PopoutWindow(Context context, Listener listener, int videoWidth, int videoHeight) {
        this.context = context.getApplicationContext();
        this.listener = listener;
        this.windows = (WindowManager) this.context.getSystemService(Context.WINDOW_SERVICE);
        this.aspect = videoWidth > 0 && videoHeight > 0
                ? videoWidth / (float) videoHeight : 16 / 9f;

        float density = this.context.getResources().getDisplayMetrics().density;
        this.minWidth = Math.round(MIN_WIDTH_DP * density);
        this.touchSlop = ViewConfiguration.get(this.context).getScaledTouchSlop();

        Point size = new Point();
        Display display = windows.getDefaultDisplay();
        display.getSize(size);
        this.screenWidth = Math.max(minWidth, size.x);
        this.screenHeight = Math.max(minWidth, size.y);

        // Themed explicitly. The application context carries no theme, and inflating a view
        // without one resolves attributes against nothing.
        root = LayoutInflater.from(new ContextThemeWrapper(this.context,
                R.style.Theme_DroneInspection)).inflate(R.layout.view_popout, null);
        video = root.findViewById(R.id.popout_video);
        overlay = root.findViewById(R.id.popout_overlay);

        video.getHolder().addCallback(new SurfaceHolder.Callback() {
            @Override
            public void surfaceCreated(@NonNull SurfaceHolder holder) {
                listener.onPopoutSurfaceCreated(holder.getSurface(),
                        video.getWidth(), video.getHeight());
            }

            @Override
            public void surfaceChanged(@NonNull SurfaceHolder holder, int format, int w, int h) {
                listener.onPopoutSurfaceChanged(w, h);
            }

            @Override
            public void surfaceDestroyed(@NonNull SurfaceHolder holder) {
                // Told before the surface goes, and the pipeline waits for its GL thread, so
                // nothing is still drawing into a surface Android has taken back.
                listener.onPopoutSurfaceDestroyed();
            }
        });

        pinch = new ScaleGestureDetector(this.context,
                new ScaleGestureDetector.SimpleOnScaleGestureListener() {
                    @Override
                    public boolean onScale(@NonNull ScaleGestureDetector detector) {
                        resizeTo(Math.round(params.width * detector.getScaleFactor()));
                        return true;
                    }
                });

        root.findViewById(R.id.popout_drag).setOnTouchListener(new DragListener());
        root.findViewById(R.id.popout_grip).setOnTouchListener(new GripListener());
        root.findViewById(R.id.popout_close).setOnClickListener(view -> {
            close();
            listener.onPopoutClosed();
        });
        video.setOnTouchListener(new BodyListener());
    }

    /** The window's own overlay, so the boxes are drawn on the small picture too. */
    OverlayView overlay() {
        return overlay;
    }

    /**
     * Put the window on screen, at the size and place it was last left.
     *
     * @return null if it opened, or why it could not, which is always the overlay permission.
     */
    @Nullable
    String open() {
        if (added) {
            return null;
        }
        params.type = Build.VERSION.SDK_INT >= Build.VERSION_CODES.O
                ? WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY
                : WindowManager.LayoutParams.TYPE_PHONE;
        // NOT_FOCUSABLE so the app underneath keeps the keyboard and every key press: this is
        // a view of the drone, not something to type into, and stealing focus from the flight
        // software mid-flight would be the worst thing this window could do.
        params.flags = WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE
                | WindowManager.LayoutParams.FLAG_LAYOUT_NO_LIMITS;
        params.format = PixelFormat.TRANSLUCENT;
        params.gravity = android.view.Gravity.TOP | android.view.Gravity.START;

        int width = clampWidth(Settings.popoutWidth(context, screenWidth / 3));
        params.width = width;
        params.height = heightFor(width);
        params.x = clampX(Settings.popoutX(context, screenWidth - width - 24));
        params.y = clampY(Settings.popoutY(context, 24));

        try {
            windows.addView(root, params);
            added = true;
            return null;
        } catch (RuntimeException refused) {
            // The permission was revoked between the check and here, or the manufacturer's
            // build refuses overlays outright. Either way the caller falls back.
            return refused.getMessage() == null ? "the window was refused" : refused.getMessage();
        }
    }

    /** Take the window down, remembering where it was. */
    void close() {
        if (!added) {
            return;
        }
        added = false;
        Settings.savePopout(context, params.x, params.y, params.width);
        try {
            windows.removeView(root);
        } catch (RuntimeException alreadyGone) {
            // Removed underneath us, which is not worth crashing a flight over.
        }
    }

    boolean isOpen() {
        return added;
    }

    /**
     * The video turned out to be a different shape than assumed.
     *
     * The stream's size is not known until the first frames arrive, so the window opens on a
     * guess and corrects itself here rather than sitting letterboxed for the whole flight.
     */
    void setVideoSize(int width, int height) {
        if (width <= 0 || height <= 0) {
            return;
        }
        float next = width / (float) height;
        if (Math.abs(next - aspect) < 0.01f) {
            return;
        }
        aspect = next;
        resizeTo(params.width);
    }

    // -----------------------------------------------------------------------------------
    // Moving and resizing
    // -----------------------------------------------------------------------------------

    private void resizeTo(int width) {
        if (!added) {
            return;
        }
        int next = clampWidth(width);
        if (next == params.width) {
            return;
        }
        params.width = next;
        params.height = heightFor(next);
        params.x = clampX(params.x);
        params.y = clampY(params.y);
        apply();
    }

    private void moveTo(int x, int y) {
        if (!added) {
            return;
        }
        params.x = clampX(x);
        params.y = clampY(y);
        apply();
    }

    private void apply() {
        try {
            windows.updateViewLayout(root, params);
        } catch (RuntimeException gone) {
            added = false;
        }
    }

    private int heightFor(int width) {
        int height = Math.round(width / Math.max(0.1f, aspect));
        if (height > screenHeight) {
            // A tall video on a short screen: fit the height and let the width follow, or the
            // controls at the bottom corner end up off the bottom of the screen.
            height = screenHeight;
        }
        return Math.max(1, height);
    }

    private int clampWidth(int width) {
        int most = Math.round(screenWidth * MAX_SCREEN_FRACTION);
        int tallest = Math.round(screenHeight * MAX_SCREEN_FRACTION * aspect);
        return Math.max(minWidth, Math.min(width, Math.min(most, tallest)));
    }

    private int clampX(int x) {
        return Math.max(0, Math.min(x, Math.max(0, screenWidth - params.width)));
    }

    private int clampY(int y) {
        return Math.max(0, Math.min(y, Math.max(0, screenHeight - params.height)));
    }

    /** Drag the strip at the top: the window follows the finger. */
    private final class DragListener implements View.OnTouchListener {
        private int startX;
        private int startY;
        private float fromRawX;
        private float fromRawY;

        @Override
        @SuppressLint("ClickableViewAccessibility")
        public boolean onTouch(View view, MotionEvent event) {
            switch (event.getActionMasked()) {
                case MotionEvent.ACTION_DOWN:
                    startX = params.x;
                    startY = params.y;
                    fromRawX = event.getRawX();
                    fromRawY = event.getRawY();
                    return true;
                case MotionEvent.ACTION_MOVE:
                    moveTo(startX + Math.round(event.getRawX() - fromRawX),
                            startY + Math.round(event.getRawY() - fromRawY));
                    return true;
                case MotionEvent.ACTION_UP:
                case MotionEvent.ACTION_CANCEL:
                    Settings.savePopout(context, params.x, params.y, params.width);
                    return true;
                default:
                    return false;
            }
        }
    }

    /** Drag the corner: the window grows and shrinks, keeping the video's shape. */
    private final class GripListener implements View.OnTouchListener {
        private int startWidth;
        private float fromRawX;
        private float fromRawY;

        @Override
        @SuppressLint("ClickableViewAccessibility")
        public boolean onTouch(View view, MotionEvent event) {
            switch (event.getActionMasked()) {
                case MotionEvent.ACTION_DOWN:
                    startWidth = params.width;
                    fromRawX = event.getRawX();
                    fromRawY = event.getRawY();
                    return true;
                case MotionEvent.ACTION_MOVE:
                    // Either axis grows it. Dragging down-right on a wide video is the natural
                    // gesture, and taking whichever moved further means the finger does not
                    // have to travel along the diagonal to be understood.
                    float dx = event.getRawX() - fromRawX;
                    float dy = (event.getRawY() - fromRawY) * aspect;
                    resizeTo(startWidth + Math.round(Math.abs(dx) > Math.abs(dy) ? dx : dy));
                    return true;
                case MotionEvent.ACTION_UP:
                case MotionEvent.ACTION_CANCEL:
                    Settings.savePopout(context, params.x, params.y, params.width);
                    return true;
                default:
                    return false;
            }
        }
    }

    /** The picture itself: pinch to resize, tap to go back to the full screen. */
    private final class BodyListener implements View.OnTouchListener {
        private float downX;
        private float downY;
        private boolean moved;

        @Override
        @SuppressLint("ClickableViewAccessibility")
        public boolean onTouch(View view, MotionEvent event) {
            pinch.onTouchEvent(event);
            switch (event.getActionMasked()) {
                case MotionEvent.ACTION_DOWN:
                    downX = event.getRawX();
                    downY = event.getRawY();
                    moved = false;
                    return true;
                case MotionEvent.ACTION_MOVE:
                    if (Math.hypot(event.getRawX() - downX, event.getRawY() - downY) > touchSlop) {
                        moved = true;
                    }
                    return true;
                case MotionEvent.ACTION_UP:
                    if (!moved && !pinch.isInProgress()) {
                        listener.onPopoutTapped();
                    } else {
                        Settings.savePopout(context, params.x, params.y, params.width);
                    }
                    return true;
                default:
                    return false;
            }
        }
    }
}
