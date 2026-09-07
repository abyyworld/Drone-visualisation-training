#!/usr/bin/env python3
"""Draw the app icons for web/icons/.

Writes both sets from one drawing: the web app's icons in web/icons/, and the Android
launcher icons in android/app/src/main/res/mipmap-*/. One mark, so the browser tab, the
home-screen shortcut and the sideloaded APK are not three slightly different logos.

Drawn from a script rather than committed as opaque binaries, so the mark can be changed by
editing 25 lines instead of opening an image editor, and so nothing in the repo is a blob
nobody can regenerate.

A rotor over a horizon: three blades, an amber hub, a darker band for the ground. It has to
be legible at 48 CSS pixels on a tablet home screen, which is the only size that matters -
detail below that is wasted, so there is none.

The maskable variants keep everything inside the middle 80% of the canvas, because Android
launchers crop a maskable icon to whatever shape the theme uses, most often a circle.

    pip install Pillow
    python3 tools/make_icons.py
"""

import math
import sys
from pathlib import Path

TEAL = (31, 78, 92)      # matches --brand in web/css/app.css and theme-color in index.html
GROUND = (20, 56, 66)
PALE = (234, 242, 244)
AMBER = (241, 196, 15)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "web" / "icons"
ANDROID_RES = ROOT / "android" / "app" / "src" / "main" / "res"
SIZES = (192, 512)

# Android launcher densities. The legacy square icon and the round variant are both drawn
# with the maskable safe zone, because a launcher that crops to a circle or a squircle will
# otherwise cut the blade tips off.
MIPMAP = {"mdpi": 48, "hdpi": 72, "xhdpi": 96, "xxhdpi": 144, "xxxhdpi": 192}


def draw(size: int, maskable: bool):
    from PIL import Image, ImageDraw

    image = Image.new("RGBA", (size, size), TEAL + (255,))
    canvas = ImageDraw.Draw(image)
    centre = size / 2
    radius = size * (0.30 if maskable else 0.36)

    canvas.rectangle([0, centre + radius * 0.75, size, size], fill=GROUND + (255,))

    for degrees in (90, 210, 330):
        angle = math.radians(degrees)
        tip = (centre + radius * math.cos(angle), centre - radius * math.sin(angle))
        left = (centre + radius * 0.16 * math.cos(angle + 1.9),
                centre - radius * 0.16 * math.sin(angle + 1.9))
        right = (centre + radius * 0.16 * math.cos(angle - 1.9),
                 centre - radius * 0.16 * math.sin(angle - 1.9))
        canvas.polygon([tip, left, right], fill=PALE + (255,))

    hub = radius * 0.13
    canvas.ellipse([centre - hub, centre - hub, centre + hub, centre + hub], fill=AMBER + (255,))
    return image


def main():
    try:
        import PIL  # noqa: F401
    except ImportError:
        print("Needs Pillow: pip install Pillow", file=sys.stderr)
        return 1

    OUT.mkdir(parents=True, exist_ok=True)
    for size in SIZES:
        draw(size, maskable=False).save(OUT / f"icon-{size}.png")
        draw(size, maskable=True).save(OUT / f"icon-{size}-maskable.png")
        print(f"icon-{size}.png, icon-{size}-maskable.png")
    print(f"Written to {OUT}")

    if not ANDROID_RES.is_dir():
        print(f"No {ANDROID_RES}; skipping the launcher icons.")
        return 0

    for density, pixels in MIPMAP.items():
        folder = ANDROID_RES / f"mipmap-{density}"
        folder.mkdir(parents=True, exist_ok=True)
        icon = draw(pixels, maskable=True)
        icon.save(folder / "ic_launcher.png")
        icon.save(folder / "ic_launcher_round.png")
        print(f"mipmap-{density}/ic_launcher.png ({pixels}px)")
    print(f"Written to {ANDROID_RES}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
