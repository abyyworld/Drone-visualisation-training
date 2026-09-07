package world.abyy.droneinspection;

import android.content.Context;
import android.content.SharedPreferences;

/**
 * What the live screen needs to know, held for the life of the process.
 *
 * The stream address and the analysis cadence are saved: they are configuration, an
 * operator sets them once, and retyping an RTSP URL on a handheld in daylight is its own
 * kind of hazard.
 *
 * The API key is not saved, and that is deliberate. It is held in memory and dies with the
 * process, the same rule the web app follows. A key written to a controller that several
 * people fly is a key all of them have; typing it once per session is the price of that not
 * being true. If that trade is wrong for a given operation, run tools/vlm_inspect.py on a
 * laptop instead, where the key lives in an environment variable under one account.
 */
public final class Settings {

    /** SIYI's documented default for the MK15 video link. See docs/HARDWARE.md. */
    public static final String DEFAULT_STREAM = "rtsp://192.168.144.25:8554/main.264";

    private static final String PREFS = "live";
    private static final String KEY_STREAM = "stream";
    private static final String KEY_DOMAIN = "domain";
    private static final String KEY_INTERVAL = "intervalSeconds";
    private static final String KEY_PROVIDER = "provider";
    private static final String KEY_MODEL = "model";

    /**
     * Seconds between frames sent for analysis.
     *
     * Not a frame rate. A provider round trip is seconds, so this is a sampling interval,
     * and the screen says how old the boxes on it are rather than implying they are live.
     * Two seconds is a reasonable default for a crowd, where the thing being watched
     * changes over tens of seconds, not milliseconds.
     */
    public static final int DEFAULT_INTERVAL_SECONDS = 2;

    /** Not persisted. See the class comment. */
    private static String apiKey = "";

    private Settings() {
    }

    private static SharedPreferences prefs(Context context) {
        return context.getApplicationContext().getSharedPreferences(PREFS, Context.MODE_PRIVATE);
    }

    public static String stream(Context context) {
        return prefs(context).getString(KEY_STREAM, DEFAULT_STREAM);
    }

    public static String domain(Context context) {
        return prefs(context).getString(KEY_DOMAIN, "crowd");
    }

    public static String provider(Context context) {
        return prefs(context).getString(KEY_PROVIDER, "anthropic");
    }

    public static String model(Context context) {
        return prefs(context).getString(KEY_MODEL, "claude-opus-5");
    }

    public static int intervalSeconds(Context context) {
        return prefs(context).getInt(KEY_INTERVAL, DEFAULT_INTERVAL_SECONDS);
    }

    public static void save(Context context, String stream, String domain, String provider,
                            String model, int intervalSeconds) {
        prefs(context).edit()
                .putString(KEY_STREAM, stream.trim())
                .putString(KEY_DOMAIN, domain)
                .putString(KEY_PROVIDER, provider)
                .putString(KEY_MODEL, model.trim())
                .putInt(KEY_INTERVAL, Math.max(1, intervalSeconds))
                .apply();
    }

    public static String apiKey() {
        return apiKey;
    }

    public static void setApiKey(String value) {
        // Trimmed for the same reason the web app trims it: every provider answers a
        // trailing newline out of a password manager with a flat 401 that reads like a
        // wrong key.
        apiKey = value == null ? "" : value.trim();
    }

    public static boolean canAnalyse() {
        return !apiKey.isEmpty();
    }
}
