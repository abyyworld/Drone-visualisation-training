package world.abyy.droneinspection;

import android.content.Context;
import android.graphics.Bitmap;
import android.os.Handler;
import android.os.Looper;
import android.util.Base64;
import android.webkit.JavascriptInterface;
import android.webkit.WebResourceRequest;
import android.webkit.WebResourceResponse;
import android.webkit.WebView;
import android.webkit.WebViewClient;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.webkit.WebViewAssetLoader;

import org.json.JSONException;
import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.util.List;

/**
 * Sends a frame to a vision model and hands back the findings.
 *
 * WHY THIS IS A WEBVIEW AND NOT JAVA
 *     The provider adapters, the prompt handling, the JSON repair and the box normalisation
 *     all exist already, in web/js/vlm.js, tested by tests/test_vlm.mjs. Writing them again
 *     in Java would be a second implementation of the part of this system most likely to
 *     drift, and the whole reason the prompt lives in a data file is that two copies of it
 *     grade the same picture differently.
 *
 *     So this holds an invisible WebView running web/live-bridge.html, which imports the
 *     same module the uploads screen does. It costs a few megabytes of memory and buys one
 *     implementation of everything that matters.
 *
 * ONE AT A TIME
 *     A request in flight blocks the next one. A provider round trip takes seconds; letting
 *     them overlap would mean paying for frames whose answers arrive out of order and
 *     overwrite each other with older boxes.
 */
public final class LiveAnalyser {

    /**
     * What the analysed frame is scaled to before sending. Matches maxEdge in
     * web/prompts/inspection.json: every provider downsamples above roughly this, so
     * sending more costs upload time and tokens and buys no accuracy.
     */
    public static final int MAX_EDGE = 1568;
    private static final int JPEG_QUALITY = 80;

    public interface Listener {
        void onFindings(String asset, String overall, List<Finding> findings);

        void onError(String message);

        void onReady();
    }

    private final WebView webView;
    private final Handler main = new Handler(Looper.getMainLooper());
    private final Listener listener;

    private boolean ready;
    private boolean busy;
    private int token;
    private long lastLatencyMillis;
    private long requestStartedAt;

    @SuppressWarnings("SetJavaScriptEnabled")
    public LiveAnalyser(Context context, Listener listener) {
        this.listener = listener;

        WebViewAssetLoader loader = new WebViewAssetLoader.Builder()
                .setDomain("appassets.androidplatform.net")
                .addPathHandler("/assets/", new WebViewAssetLoader.AssetsPathHandler(context))
                .build();

        webView = new WebView(context);
        webView.getSettings().setJavaScriptEnabled(true);
        webView.getSettings().setAllowFileAccess(false);
        webView.getSettings().setAllowContentAccess(false);
        webView.setWebViewClient(new WebViewClient() {
            @Override
            public WebResourceResponse shouldInterceptRequest(WebView view, WebResourceRequest request) {
                return loader.shouldInterceptRequest(request.getUrl());
            }
        });
        webView.addJavascriptInterface(new Bridge(), "Android");
        webView.loadUrl("https://appassets.androidplatform.net/assets/www/live-bridge.html");
    }

    public boolean isReady() {
        return ready;
    }

    public boolean isBusy() {
        return busy;
    }

    /** Round-trip time of the last completed request, for the status line. */
    public long lastLatencyMillis() {
        return lastLatencyMillis;
    }

    /**
     * Analyse one frame, unless a request is already in flight.
     *
     * @return true if the frame was sent, false if it was dropped
     */
    public boolean analyse(Bitmap frame, String provider, String model, String apiKey, String domain) {
        if (!ready || busy || frame == null) {
            return false;
        }
        busy = true;
        requestStartedAt = System.currentTimeMillis();
        int current = ++token;

        String dataUrl = "data:image/jpeg;base64," + encode(frame);
        String script = "window.analyseFrame("
                + current + ","
                + quote(dataUrl) + ","
                + quote(provider) + ","
                + quote(model) + ","
                + quote(apiKey) + ","
                + quote(domain) + ")";
        webView.evaluateJavascript(script, null);
        return true;
    }

    public void destroy() {
        webView.destroy();
    }

    /** Downscale, JPEG, base64. Scaling here rather than in JS keeps the string small. */
    private static String encode(Bitmap frame) {
        int longEdge = Math.max(frame.getWidth(), frame.getHeight());
        Bitmap scaled = frame;
        if (longEdge > MAX_EDGE) {
            float scale = (float) MAX_EDGE / longEdge;
            scaled = Bitmap.createScaledBitmap(frame,
                    Math.max(1, Math.round(frame.getWidth() * scale)),
                    Math.max(1, Math.round(frame.getHeight() * scale)), true);
        }
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        scaled.compress(Bitmap.CompressFormat.JPEG, JPEG_QUALITY, out);
        if (scaled != frame) {
            scaled.recycle();
        }
        return Base64.encodeToString(out.toByteArray(), Base64.NO_WRAP);
    }

    private static String quote(@Nullable String value) {
        return JSONObject.quote(value == null ? "" : value);
    }

    /** Called from JavaScript. Every method arrives on a WebView thread, not the main one. */
    private final class Bridge {

        @JavascriptInterface
        public void onBridgeReady() {
            main.post(() -> {
                ready = true;
                listener.onReady();
            });
        }

        @JavascriptInterface
        public void onAnalysis(int replyToken, @NonNull String payload) {
            main.post(() -> {
                // A late reply from a request we have moved on from is discarded rather than
                // drawn: its boxes describe a frame two frames ago.
                if (replyToken != token) {
                    return;
                }
                busy = false;
                lastLatencyMillis = System.currentTimeMillis() - requestStartedAt;
                try {
                    JSONObject json = new JSONObject(payload);
                    String error = json.optString("error", "");
                    if (!error.isEmpty()) {
                        listener.onError(error);
                        return;
                    }
                    listener.onFindings(
                            json.optString("asset", "neither"),
                            json.optString("overall", ""),
                            Finding.parse(json.optJSONArray("findings")));
                } catch (JSONException malformed) {
                    listener.onError("The analyser returned something unreadable.");
                }
            });
        }
    }
}
