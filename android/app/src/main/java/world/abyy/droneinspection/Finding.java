package world.abyy.droneinspection;

import androidx.annotation.NonNull;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.util.ArrayList;
import java.util.List;

/**
 * One thing the model reported, with its box as fractions of the frame.
 *
 * Fractions rather than pixels because three different sizes are in play at once: the video
 * is whatever the drone sends, the frame that was analysed is a downscale of it, and the
 * view it is drawn over is whatever the tablet's screen gives. A pixel here would be a pixel
 * of the wrong one of those.
 */
public final class Finding {

    public final String label;
    public final String certainty;
    public final String note;
    public final float x0;
    public final float y0;
    public final float x1;
    public final float y1;

    Finding(String label, String certainty, String note, float x0, float y0, float x1, float y1) {
        this.label = label;
        this.certainty = certainty;
        this.note = note;
        this.x0 = x0;
        this.y0 = y0;
        this.x1 = x1;
        this.y1 = y1;
    }

    /** A stable colour per label, matching colourIndex() in web/js/vlm.js. */
    public int colourIndex() {
        int hash = 0;
        for (int i = 0; i < label.length(); i++) {
            hash = (hash * 31 + label.charAt(i)) % 4096;
        }
        return hash;
    }

    @NonNull
    @Override
    public String toString() {
        return label + " (" + certainty + ")";
    }

    /** Parse the array the bridge page sends back. Malformed entries are skipped. */
    public static List<Finding> parse(JSONArray array) {
        List<Finding> findings = new ArrayList<>();
        if (array == null) {
            return findings;
        }
        for (int i = 0; i < array.length(); i++) {
            try {
                JSONObject item = array.getJSONObject(i);
                JSONArray box = item.getJSONArray("box");
                findings.add(new Finding(
                        item.optString("label", "finding"),
                        item.optString("certainty", "medium"),
                        item.optString("note", ""),
                        (float) box.getDouble(0), (float) box.getDouble(1),
                        (float) box.getDouble(2), (float) box.getDouble(3)));
            } catch (JSONException malformed) {
                // One unusable entry is not a reason to drop the rest of the frame.
            }
        }
        return findings;
    }
}
