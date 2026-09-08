package world.abyy.droneinspection;

import android.content.Intent;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.webkit.ValueCallback;
import android.Manifest;
import android.content.pm.PackageManager;
import android.webkit.PermissionRequest;
import android.webkit.WebChromeClient;
import android.webkit.WebResourceRequest;
import android.webkit.WebResourceResponse;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import android.widget.Button;
import android.widget.FrameLayout;
import android.widget.Toast;

import android.view.Gravity;
import android.view.ViewGroup;

import androidx.activity.OnBackPressedCallback;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.core.content.ContextCompat;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.webkit.ServiceWorkerClientCompat;
import androidx.webkit.ServiceWorkerControllerCompat;
import androidx.webkit.WebViewAssetLoader;
import androidx.webkit.WebViewFeature;

/**
 * The whole application: a WebView showing the bundled copy of the web app.
 *
 * This is the Analyse half of the app: photographs and video that already exist, in, and a
 * report out. The Camera half is LiveActivity, and it is native for the reason set out
 * there. MenuActivity chooses between them.
 *
 * WHY AN APK AT ALL
 *     The same page installs from the browser as a home-screen app, and on a tablet with a
 *     current Chrome that is the better route - it updates itself. This exists for the
 *     tablets where that is not available: a controller with a locked-down stock browser,
 *     no Play Store, or a fleet where the app has to be pushed rather than visited. It is
 *     the same web app, sideloadable as a file.
 *
 * WHY NOT file://
 *     The obvious way to show bundled HTML is loadUrl("file:///android_asset/..."), and it
 *     would break most of this app. A file:// page is not a secure context, so
 *     createImageBitmap, the module graph and the crypto the runtime touches are all
 *     degraded or unavailable; its origin is opaque, so every fetch to a provider API is a
 *     cross-origin request from "null" that CORS cannot approve; and ES module imports are
 *     blocked outright.
 *
 *     WebViewAssetLoader solves all of that by serving the same files over
 *     https://appassets.androidplatform.net/assets/ - a real https origin, so a secure
 *     context with a stable origin that provider CORS accepts. The domain is reserved by
 *     Android and never resolves on the network, so nothing leaves the device to load it.
 *
 * WHAT IT DOES NOT DO
 *     No native inference, no background service, no analytics, no storage permission. The
 *     only permission is INTERNET, and that is needed solely by the provider engines. With
 *     the on-device engine and no connection, this app never opens a socket.
 */
public class MainActivity extends AppCompatActivity {

    private static final String ORIGIN = "https://appassets.androidplatform.net";
    private static final String START_URL = ORIGIN + "/assets/www/index.html";

    private WebView webView;
    private WebViewAssetLoader assetLoader;

    /** Held between launching the picker and its result, for <input type="file">. */
    @Nullable
    private ValueCallback<Uri[]> pendingFileCallback;

    /**
     * Asks Android for the camera when the page first wants it.
     *
     * The page's own request is denied while this runs, because a WebView permission
     * request cannot be held open across a system dialog. The operator presses Start again
     * once granted, which is one extra tap and no ambiguity about what happened.
     */
    private final ActivityResultLauncher<String> cameraPermission = registerForActivityResult(
            new ActivityResultContracts.RequestPermission(),
            granted -> Toast.makeText(this,
                    granted ? R.string.camera_granted : R.string.camera_denied,
                    Toast.LENGTH_LONG).show());

    private final ActivityResultLauncher<Intent> filePicker = registerForActivityResult(
            new ActivityResultContracts.StartActivityForResult(),
            result -> {
                if (pendingFileCallback == null) {
                    return;
                }
                // A cancelled picker must still deliver null, or the <input> stays wedged
                // and the user cannot open the dialog a second time.
                pendingFileCallback.onReceiveValue(urisFrom(result.getResultCode(), result.getData()));
                pendingFileCallback = null;
            });

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        assetLoader = new WebViewAssetLoader.Builder()
                .setDomain("appassets.androidplatform.net")
                .addPathHandler("/assets/", new WebViewAssetLoader.AssetsPathHandler(this))
                .build();

        webView = new WebView(this);
        configure(webView.getSettings());

        // The web app draws its own full-width header, so a native toolbar above it would
        // cost a strip of a tablet screen that is showing photographs. A small button in the
        // corner is enough to get back to the menu, and it is the only native chrome here.
        FrameLayout root = new FrameLayout(this);
        root.addView(webView, new FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT));

        Button menu = new Button(this);
        menu.setText(R.string.back_to_menu);
        menu.setOnClickListener(v -> finish());
        FrameLayout.LayoutParams menuLayout = new FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT);
        menuLayout.gravity = Gravity.BOTTOM | Gravity.END;
        int margin = Math.round(getResources().getDisplayMetrics().density * 12);
        menuLayout.setMargins(margin, margin, margin, margin);
        root.addView(menu, menuLayout);

        setContentView(root);

        webView.setWebViewClient(new WebViewClient() {
            @Override
            public WebResourceResponse shouldInterceptRequest(WebView view, WebResourceRequest request) {
                return assetLoader.shouldInterceptRequest(request.getUrl());
            }

            @Override
            public boolean shouldOverrideUrlLoading(WebView view, WebResourceRequest request) {
                Uri url = request.getUrl();
                if (ORIGIN.equals(url.getScheme() + "://" + url.getAuthority())) {
                    return false;   // our own pages stay in the WebView
                }
                // Anything else is a link out. Hand it to the system browser rather than
                // letting the app become a general-purpose browser with no address bar.
                try {
                    startActivity(new Intent(Intent.ACTION_VIEW, url));
                } catch (Exception ignored) {
                    Toast.makeText(MainActivity.this, R.string.no_browser, Toast.LENGTH_SHORT).show();
                }
                return true;
            }
        });

        webView.setWebChromeClient(new WebChromeClient() {
            /**
             * Let the page open the camera for live tracking.
             *
             * A WebView denies this by default, silently, and the page then reports that the
             * camera was refused - which reads as the tablet having no camera. Granted only
             * for video and only once Android itself has granted it to the app, so the
             * system permission is still the thing in charge.
             */
            @Override
            public void onPermissionRequest(PermissionRequest request) {
                boolean wantsCamera = false;
                for (String resource : request.getResources()) {
                    if (PermissionRequest.RESOURCE_VIDEO_CAPTURE.equals(resource)) {
                        wantsCamera = true;
                    }
                }
                boolean allowed = ContextCompat.checkSelfPermission(
                        MainActivity.this, Manifest.permission.CAMERA)
                        == PackageManager.PERMISSION_GRANTED;

                if (wantsCamera && allowed) {
                    request.grant(new String[]{PermissionRequest.RESOURCE_VIDEO_CAPTURE});
                } else if (wantsCamera) {
                    request.deny();
                    cameraPermission.launch(Manifest.permission.CAMERA);
                } else {
                    // Audio and everything else. This app never needs a microphone.
                    request.deny();
                }
            }

            @Override
            public boolean onShowFileChooser(WebView view,
                                             ValueCallback<Uri[]> callback,
                                             FileChooserParams params) {
                if (pendingFileCallback != null) {
                    pendingFileCallback.onReceiveValue(null);
                }
                pendingFileCallback = callback;

                boolean many = params.getMode() == FileChooserParams.MODE_OPEN_MULTIPLE;
                for (Intent attempt : pickers(many)) {
                    try {
                        filePicker.launch(attempt);
                        return true;
                    } catch (Exception unavailable) {
                        // This device has nothing that answers that particular intent. Try
                        // the next, which asks for less.
                    }
                }
                pendingFileCallback = null;
                Toast.makeText(MainActivity.this, R.string.no_picker, Toast.LENGTH_LONG).show();
                return false;
            }
        });

        // A service worker's own fetches do not pass through WebViewClient, so without this
        // they would miss the asset loader entirely and fail. sw.js is not shipped in the
        // APK (see app/build.gradle), but a WebView that has one registered from a previous
        // install would otherwise serve a broken app, and this keeps that case working.
        if (WebViewFeature.isFeatureSupported(WebViewFeature.SERVICE_WORKER_BASIC_USAGE)) {
            ServiceWorkerControllerCompat.getInstance().setServiceWorkerClient(
                    new ServiceWorkerClientCompat() {
                        @Override
                        public WebResourceResponse shouldInterceptRequest(@NonNull WebResourceRequest request) {
                            return assetLoader.shouldInterceptRequest(request.getUrl());
                        }
                    });
        }

        getOnBackPressedDispatcher().addCallback(this, new OnBackPressedCallback(true) {
            @Override
            public void handleOnBackPressed() {
                if (webView.canGoBack()) {
                    webView.goBack();
                } else {
                    setEnabled(false);
                    getOnBackPressedDispatcher().onBackPressed();
                }
            }
        });

        if (savedInstanceState == null) {
            webView.loadUrl(START_URL);
        } else {
            webView.restoreState(savedInstanceState);
        }
    }


    /**
     * Ways to ask this device for a file, best first.
     *
     * WHY NOT FileChooserParams.createIntent()
     *     Because on the controller this app is flown with, it opened a picker in which
     *     nothing could be selected. The page asks for `image/*,video/*` and then a long
     *     tail of bare extensions, so that a phone's own formats are offered rather than
     *     only the ones this machine has a MIME type for. The WebView turns that accept
     *     list into MIME types for the intent, and the extensions Android 9 cannot resolve
     *     - heic and the action-camera ones among them - come through as entries that match
     *     no file at all. The picker then filters everything away and there is nothing to
     *     tap. It is not an error and nothing is logged.
     *
     *     So the intent is built here instead, from what this application actually accepts,
     *     which is pictures and video and nothing else. The accept list on the page is left
     *     alone: in a real browser it is useful, and it is still what decides which files
     *     the page will take once they arrive.
     *
     *     The ladder exists because a rugged controller is not a phone. It may have no
     *     gallery, and its file manager may answer one of these intents and not the others,
     *     so each rung asks for less than the one above.
     */
    private java.util.List<Intent> pickers(boolean many) {
        java.util.List<Intent> options = new java.util.ArrayList<>();

        // Both families at once needs a wildcard type with the real list beside it; setting
        // the type to image/* would hide every video.
        String[] wanted = {"image/*", "video/*"};

        Intent document = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        document.addCategory(Intent.CATEGORY_OPENABLE);
        document.setType("*/*");
        document.putExtra(Intent.EXTRA_MIME_TYPES, wanted);
        document.putExtra(Intent.EXTRA_ALLOW_MULTIPLE, many);
        options.add(document);

        Intent content = new Intent(Intent.ACTION_GET_CONTENT);
        content.addCategory(Intent.CATEGORY_OPENABLE);
        content.setType("*/*");
        content.putExtra(Intent.EXTRA_MIME_TYPES, wanted);
        content.putExtra(Intent.EXTRA_ALLOW_MULTIPLE, many);
        options.add(Intent.createChooser(content, getString(R.string.choose_files)));

        // Last resort: everything, with no filtering at all. A device whose picker mishandles
        // EXTRA_MIME_TYPES still shows the files, and the page rejects what it cannot read
        // with a message rather than silently.
        Intent anything = new Intent(Intent.ACTION_GET_CONTENT);
        anything.addCategory(Intent.CATEGORY_OPENABLE);
        anything.setType("*/*");
        anything.putExtra(Intent.EXTRA_ALLOW_MULTIPLE, many);
        options.add(anything);

        return options;
    }

    private void configure(WebSettings settings) {
        settings.setJavaScriptEnabled(true);
        settings.setDomStorageEnabled(true);

        // The page loads nothing from the filesystem and nothing from another origin's
        // files. Leaving these off closes the classic WebView file-disclosure holes, and
        // the asset loader means nothing needs them.
        settings.setAllowFileAccess(false);
        settings.setAllowContentAccess(false);
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.R) {
            settings.setAllowFileAccessFromFileURLs(false);
            settings.setAllowUniversalAccessFromFileURLs(false);
        }

        // Frame extraction seeks a <video> the user never pressed play on. Without this it
        // silently produces no frames on some builds, which reads as an empty clip.
        settings.setMediaPlaybackRequiresUserGesture(false);

        // The page is responsive and already sets a viewport; let it use the real width
        // rather than the desktop default, or every control renders at a third of its size.
        settings.setUseWideViewPort(true);
        settings.setLoadWithOverviewMode(true);
        settings.setSupportZoom(true);
        settings.setBuiltInZoomControls(true);
        settings.setDisplayZoomControls(false);
    }

    private static Uri[] urisFrom(int resultCode, @Nullable Intent data) {
        if (resultCode != RESULT_OK || data == null) {
            return null;
        }
        if (data.getClipData() != null) {
            android.content.ClipData clip = data.getClipData();
            Uri[] uris = new Uri[clip.getItemCount()];
            for (int i = 0; i < uris.length; i++) {
                uris[i] = clip.getItemAt(i).getUri();
            }
            return uris;
        }
        return data.getData() == null ? null : new Uri[]{data.getData()};
    }

    @Override
    protected void onSaveInstanceState(@NonNull Bundle outState) {
        super.onSaveInstanceState(outState);
        // A rotation or a memory trim must not throw away a batch of results that took real
        // money to produce.
        webView.saveState(outState);
    }

    @Override
    protected void onDestroy() {
        if (pendingFileCallback != null) {
            pendingFileCallback.onReceiveValue(null);
            pendingFileCallback = null;
        }
        webView.destroy();
        super.onDestroy();
    }
}
