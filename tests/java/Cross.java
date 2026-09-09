import android.graphics.Bitmap;
import world.abyy.droneinspection.Finding;
import world.abyy.droneinspection.Tracker;
import world.abyy.droneinspection.Yolo;
import java.lang.reflect.*;
import java.util.*;

/** The same painted frames as tests/test_firescan.mjs, through the Java port. */
public class Cross {
  static final int W = 160, H = 120;

  interface Painter { int[] rgb(int x, int y, int frame); }

  static Bitmap frame(Painter p, int f) {
    int[] px = new int[W * H];
    for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
      int[] c = p.rgb(x, y, f);
      px[y * W + x] = 0xFF000000 | (clamp(c[0]) << 16) | (clamp(c[1]) << 8) | clamp(c[2]);
    }
    return new Bitmap(W, H, px);
  }
  static int clamp(int v) { return v < 0 ? 0 : v > 255 ? 255 : v; }

  static int[] ground(int x, int y) {
    int noise = ((x * 7 + y * 13) % 11) * 6;
    return new int[]{28 + noise, 58 + noise, 22 + (noise >> 1)};
  }
  static boolean inside(int x, int y, int[] b) { return x >= b[0] && x < b[2] && y >= b[1] && y < b[3]; }
  static final int[] FIRE = {40, 60, 80, 100};

  public static void main(String[] args) throws Exception {
    run("flame", 20, (x, y, f) -> {
      if (!inside(x, y, FIRE)) return ground(x, y);
      double height = 22 + 14 * Math.sin(f * 1.1 + x * 0.35);
      if (y < FIRE[3] - height) return ground(x, y);
      boolean ember = ((x * 3 + y * 5 + f * 7) % 9) == 0;
      return ember ? new int[]{120, 45, 12} : new int[]{255, 140, 30};
    });
    run("static-panel", 20, (x, y, f) ->
        inside(x, y, FIRE) ? new int[]{255, 140, 30} : ground(x, y));
    run("moving-van", 20, (x, y, f) -> {
      int[] box = {10 + f * 4, 60, 50 + f * 4, 90};
      return inside(x, y, box) ? new int[]{255, 140, 30} : ground(x, y);
    });
    run("sunset", 20, (x, y, f) -> {
      if (y >= 55) return ground(x, y);
      double drift = f * 0.4;
      int noise = ((x + y * 3 + f) % 5) - 2;
      return new int[]{(int) (252 - drift) + noise, 138 - y + noise, 40 + y + noise};
    });
    run("smoke", 24, (x, y, f) -> {
      int arrived = Math.max(0, f - 5);
      int top = Math.max(0, 100 - arrived * 6);
      boolean inPlume = f > 5 && y >= top && y < 100 && Math.abs(x - 80) < 30 + ((y + f) % 7);
      if (!inPlume) return ground(x, y);
      int grey = 168 + ((x + y * 2 + f * 9) % 5);
      return new int[]{grey, grey + 2, grey - 1};
    });
    run("overcast", 24, (x, y, f) -> {
      int wobble = (f % 3) - 1;
      if (y < 60 + wobble) { int grey = 176 + ((x + f) % 4); return new int[]{grey, grey, grey + 1}; }
      return ground(x, y);
    });
    run("nothing", 20, (x, y, f) -> ground(x + f, y));
    run("office", 24, (x, y, f) -> {
      int shift = f * 3;
      if (Math.abs(x - (30 + shift)) < 26 && y > 30) return new int[]{46, 40, 38};
      int wall = 190 + ((x + y) % 2);
      return new int[]{wall, wall + 1, wall - 1};
    });
    tiles();
    run("still", 1, (x, y, f) -> {
      if (!inside(x, y, FIRE)) return ground(x, y);
      boolean burning = ((x * 3 + y * 5) % 10) > 2;
      return burning ? new int[]{255, 140, 30} : new int[]{120, 40, 10};
    });
    yolo();
    tracking();
  }

  /**
   * The tracker under a camera that is moving, against the same run in JavaScript.
   *
   * A crowd standing still while the whole picture slides under them is the drone's own
   * motion with nothing else mixed in, and it is what used to make the numbering churn.
   * Both implementations have to issue the same identities for it, or the tablet and the
   * report of the same footage count the same crowd differently.
   */
  static void tracking() throws Exception {
    for (int speed : new int[]{0, 20, 45}) {
      Tracker tracker = new Tracker();
      long clock = 1000;
      for (int step = 0; step < 8; step++) {
        List<Finding> crowd = new ArrayList<>();
        for (int i = 0; i < 30; i++) {
          float x = (i % 6) * 90 - step * speed;
          float y = (float) Math.floor(i / 6.0) * 120;
          crowd.add(new Finding("person", "high", "", x, y, x + 40, y + 80));
        }
        tracker.update(crowd, clock);
        clock += 250;
      }
      System.out.println(String.format(Locale.UK, "track-pan-%d: %d", speed,
          tracker.countSeen("person")));
    }
  }

  /** The tile grid and the merge, printed so the JavaScript can be compared against it. */
  static void tiles() throws Exception {
    Class<?> cls = Class.forName("world.abyy.droneinspection.Tiles");
    Method region = cls.getDeclaredMethod("region", int.class, int.class, int.class);
    region.setAccessible(true);
    Method overlap = cls.getDeclaredMethod("overlap", float[].class, float[].class);
    overlap.setAccessible(true);

    StringBuilder line = new StringBuilder("tiles:");
    for (int i = 0; i < 6; i++) {
      float[] r = (float[]) region.invoke(null, i, 1920, 1080);
      line.append(String.format(Locale.UK, " [%.1f %.1f %.1f %.1f]", r[0], r[1], r[2], r[3]));
    }
    System.out.println(line);

    float[] a = {100, 100, 140, 190};
    float[] b = {104, 98, 144, 188};
    float[] far = {400, 100, 440, 190};
    System.out.println(String.format(Locale.UK, "overlap: %.4f %.4f",
        (float) overlap.invoke(null, a, b), (float) overlap.invoke(null, a, far)));
  }

  /**
   * The YOLO decode, against the same numbers web/js/yolo.js is given.
   *
   * The head is built from arithmetic rather than read from a file so that both languages
   * can produce the identical array without one shipping the other anything. The mixing is
   * written to wrap the same way in both: JavaScript's >>> converts to a 32-bit unsigned
   * first, which is what a Java int already does.
   */
  static void yolo() throws Exception {
    final int CHANNELS = 15, ANCHORS = 2100;
    float[] head = new float[CHANNELS * ANCHORS];
    for (int i = 0; i < head.length; i++) head[i] = value(i);

    System.out.println("yolo-all: " + said(Yolo.decodeHead(
        head, CHANNELS, ANCHORS, 0.25, 0.45, null)));
    System.out.println("yolo-people: " + said(Yolo.decodeHead(
        head, CHANNELS, ANCHORS, 0.25, 0.45, Yolo.classes(0, 1))));

    float[] perAnchor = new float[CHANNELS * ANCHORS];
    for (int i = 0; i < perAnchor.length; i++) perAnchor[i] = value(i * 7 + 3);
    System.out.println("yolo-transposed: " + said(Yolo.decodeHead(
        Yolo.toChannelMajor(perAnchor, ANCHORS, CHANNELS),
        CHANNELS, ANCHORS, 0.25, 0.45, Yolo.classes(0, 1))));

    double[] a = {100, 100, 140, 190}, b = {104, 98, 144, 188}, far = {400, 100, 440, 190};
    System.out.println(String.format(Locale.UK, "yolo-iou: %.6f %.6f %.6f",
        Yolo.iou(a, b), Yolo.iou(a, far), Yolo.iou(a, a)));

    double[] back = Yolo.unletterbox(new double[]{12, 30, 300, 290}, 0.3333, 0, 46.5, 1920, 1080);
    System.out.println(String.format(Locale.UK, "yolo-unletterbox: %.6f %.6f %.6f %.6f",
        back[0], back[1], back[2], back[3]));
  }

  static float value(int i) {
    return (float) (((i * 1103515245 + 12345) >>> 8 & 0xffff) / 65535.0);
  }

  /** The count, then the first six, which is enough to catch an ordering change. */
  static String said(List<Yolo.Detection> found) {
    StringBuilder out = new StringBuilder().append(found.size()).append(' ');
    for (int i = 0; i < Math.min(6, found.size()); i++) {
      Yolo.Detection d = found.get(i);
      if (i > 0) out.append(" | ");
      out.append(String.format(Locale.UK, "%d:%.6f[%.6f %.6f %.6f %.6f]",
          d.classId, d.confidence, d.box[0], d.box[1], d.box[2], d.box[3]));
    }
    return out.toString();
  }

  static void run(String name, int frames, Painter p) throws Exception {
    Class<?> cls = Class.forName("world.abyy.droneinspection.FireScan");
    Constructor<?> ctor = cls.getDeclaredConstructor();
    ctor.setAccessible(true);
    Object scan = ctor.newInstance();
    Method m = cls.getDeclaredMethod("scan", Bitmap.class);
    m.setAccessible(true);

    List<?> out = new ArrayList<>();
    for (int f = 0; f < frames; f++) out = (List<?>) m.invoke(scan, frame(p, f));

    List<String> said = new ArrayList<>();
    for (Object r : out) {
      Class<?> rc = r.getClass();
      said.add(String.format(Locale.UK, "%s %.2f [%.2f %.2f %.2f %.2f]",
          field(rc, r, "label"), field(rc, r, "confidence"),
          field(rc, r, "x0"), field(rc, r, "y0"), field(rc, r, "x1"), field(rc, r, "y1")));
    }
    Collections.sort(said);
    System.out.println(name + ": " + (said.isEmpty() ? "-" : String.join(" | ", said)));
  }

  static Object field(Class<?> c, Object o, String n) throws Exception {
    Field f = c.getDeclaredField(n); f.setAccessible(true); return f.get(o);
  }
}
