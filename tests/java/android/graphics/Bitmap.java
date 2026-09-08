package android.graphics;
/** Enough of Bitmap to run FireScan off-device, so the Java can be compared with the JS. */
public class Bitmap {
  private final int w, h; private final int[] px;
  public Bitmap(int w, int h, int[] px) { this.w = w; this.h = h; this.px = px; }
  public int getWidth() { return w; }
  public int getHeight() { return h; }
  public void getPixels(int[] out, int off, int stride, int x, int y, int rw, int rh) {
    System.arraycopy(px, 0, out, 0, rw * rh);
  }
  public void recycle() { }
  public static Bitmap createScaledBitmap(Bitmap src, int w, int h, boolean filter) {
    if (src.w == w && src.h == h) return src;
    int[] out = new int[w * h];
    for (int yy = 0; yy < h; yy++) for (int xx = 0; xx < w; xx++) {
      int x0 = xx * src.w / w, x1 = Math.max(x0 + 1, (xx + 1) * src.w / w);
      int y0 = yy * src.h / h, y1 = Math.max(y0 + 1, (yy + 1) * src.h / h);
      long r = 0, g = 0, b = 0, n = 0;
      for (int sy = y0; sy < y1 && sy < src.h; sy++) for (int sx = x0; sx < x1 && sx < src.w; sx++) {
        int p = src.px[sy * src.w + sx];
        r += (p >> 16) & 0xFF; g += (p >> 8) & 0xFF; b += p & 0xFF; n++;
      }
      out[yy * w + xx] = 0xFF000000 | ((int) (r / n) << 16) | ((int) (g / n) << 8) | (int) (b / n);
    }
    return new Bitmap(w, h, out);
  }
}
