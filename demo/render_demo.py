"""Run the full pipeline over a video file and render the overlay to an MP4.

This is the offline twin of the live system: same ingest, same temporal filter,
same incident log, same drawing rules as the tablet overlay. What it does not
use is WebRTC -- it burns the boxes into an output file instead.

Two reasons it exists. It is the fallback for a live demonstration, where a
recorded run cannot fail on stage. And it is the honest artefact: it writes an
incident log alongside the video, so what the model saw can be compared with
what was there afterwards, frame by frame.

The drawing follows the same rules as app/js/overlay.js, and they are field
rules rather than aesthetics: class is encoded by line style as well as
colour, because colour-blind users exist and red already means something else
on an incident; confidence is printed and also drawn as a bar, because a 0.3
and a 0.9 must not look alike; and nothing is ever drawn to indicate an
absence of fire.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from demo.simple_detector import detect
from station.core.config import SourceConfig, TemporalConfig
from station.core.types import Detection, FrameDetections, ModelInfo
from station.incidentlog import IncidentLogWriter
from station.inference.temporal import TemporalFilter
from station.ingest import open_source

# 3x5 bitmap glyphs -- enough for the labels, and it avoids a font dependency.
# 5x5 bitmap glyphs. The first attempt used 3x5 to save space and M/N were
# indistinguishable, which turned the footer into "MOT A CERTIFIED SYSTEM" --
# the one line on screen that has to be unambiguous.
GLYPHS = {
    "A": ("01110", "10001", "11111", "10001", "10001"),
    "B": ("11110", "10001", "11110", "10001", "11110"),
    "C": ("01111", "10000", "10000", "10000", "01111"),
    "D": ("11110", "10001", "10001", "10001", "11110"),
    "E": ("11111", "10000", "11110", "10000", "11111"),
    "F": ("11111", "10000", "11110", "10000", "10000"),
    "G": ("01111", "10000", "10011", "10001", "01111"),
    "H": ("10001", "10001", "11111", "10001", "10001"),
    "I": ("11111", "00100", "00100", "00100", "11111"),
    "J": ("00011", "00001", "00001", "10001", "01110"),
    "K": ("10001", "10010", "11100", "10010", "10001"),
    "L": ("10000", "10000", "10000", "10000", "11111"),
    "M": ("10001", "11011", "10101", "10001", "10001"),
    "N": ("10001", "11001", "10101", "10011", "10001"),
    "O": ("01110", "10001", "10001", "10001", "01110"),
    "P": ("11110", "10001", "11110", "10000", "10000"),
    "Q": ("01110", "10001", "10101", "10010", "01101"),
    "R": ("11110", "10001", "11110", "10010", "10001"),
    "S": ("01111", "10000", "01110", "00001", "11110"),
    "T": ("11111", "00100", "00100", "00100", "00100"),
    "U": ("10001", "10001", "10001", "10001", "01110"),
    "V": ("10001", "10001", "10001", "01010", "00100"),
    "W": ("10001", "10001", "10101", "11011", "10001"),
    "X": ("10001", "01010", "00100", "01010", "10001"),
    "Y": ("10001", "01010", "00100", "00100", "00100"),
    "Z": ("11111", "00010", "00100", "01000", "11111"),
    "0": ("01110", "10011", "10101", "11001", "01110"),
    "1": ("00100", "01100", "00100", "00100", "01110"),
    "2": ("01110", "10001", "00110", "01000", "11111"),
    "3": ("11111", "00010", "00110", "10001", "01110"),
    "4": ("00010", "00110", "01010", "11111", "00010"),
    "5": ("11111", "10000", "11110", "00001", "11110"),
    "6": ("00110", "01000", "11110", "10001", "01110"),
    "7": ("11111", "00001", "00010", "00100", "01000"),
    "8": ("01110", "10001", "01110", "10001", "01110"),
    "9": ("01110", "10001", "01111", "00010", "01100"),
    ".": ("00000", "00000", "00000", "00000", "00100"),
    " ": ("00000", "00000", "00000", "00000", "00000"),
    "-": ("00000", "00000", "11111", "00000", "00000"),
    "/": ("00001", "00010", "00100", "01000", "10000"),
    ":": ("00000", "00100", "00000", "00100", "00000"),
}
GLYPH_W = 5

# Drawn into BGR frames, so these are stated in BGR to match.
FIRE_BGR = (30, 158, 255)
SMOKE_BGR = (255, 205, 150)


def draw_text(img: np.ndarray, text: str, x: int, y: int, scale: int = 3,
              rgb: tuple[int, int, int] = (255, 255, 255)) -> None:
    h, w, _ = img.shape
    cx = x
    for ch in text.upper():
        pat = GLYPHS.get(ch, GLYPHS[" "])
        for ry, row in enumerate(pat):
            for rx, bit in enumerate(row):
                if bit != "1":
                    continue
                px0, py0 = cx + rx * scale, y + ry * scale
                px1, py1 = min(w, px0 + scale), min(h, py0 + scale)
                if px0 < w and py0 < h:
                    img[max(0, py0):py1, max(0, px0):px1] = rgb
        cx += (GLYPH_W + 1) * scale


def draw_rect(img: np.ndarray, x0: int, y0: int, x1: int, y1: int,
              rgb: tuple[int, int, int], thickness: int = 3, dash: int = 0) -> None:
    h, w, _ = img.shape
    x0, x1 = max(0, min(w - 1, x0)), max(0, min(w - 1, x1))
    y0, y1 = max(0, min(h - 1, y0)), max(0, min(h - 1, y1))
    for t in range(thickness):
        for x in range(x0, x1 + 1):
            if dash and ((x // dash) % 2):
                continue
            if y0 + t < h:
                img[y0 + t, x] = rgb
            if y1 - t >= 0:
                img[y1 - t, x] = rgb
        for y in range(y0, y1 + 1):
            if dash and ((y // dash) % 2):
                continue
            if x0 + t < w:
                img[y, x0 + t] = rgb
            if x1 - t >= 0:
                img[y, x1 - t] = rgb


def draw_detection(img: np.ndarray, det: Detection) -> None:
    h, w, _ = img.shape
    x0, y0 = int(det.box.x1 * w), int(det.box.y1 * h)
    x1, y1 = int(det.box.x2 * w), int(det.box.y2 * h)
    fire = det.cls == "fire"
    rgb = FIRE_BGR if fire else SMOKE_BGR
    # Solid for fire, dashed for smoke: the class is legible without relying on
    # colour, and it also signals that a smoke boundary is approximate.
    draw_rect(img, x0, y0, x1, y1, rgb, thickness=3 if fire else 2, dash=0 if fire else 9)

    label = f"{det.cls} {det.conf:.2f}".replace("0.", ".")
    ty = max(0, y0 - 22)
    tw = len(label) * (GLYPH_W + 1) * 3 + 8
    img[ty:min(h, ty + 20), x0:min(w, x0 + tw)] = (
        0.25 * img[ty:min(h, ty + 20), x0:min(w, x0 + tw)]).astype(np.uint8)
    draw_text(img, label, x0 + 4, ty + 3, scale=3, rgb=rgb)

    # Confidence bar: a 0.3 and a 0.9 must not look alike at a glance.
    bar_w = int((x1 - x0) * det.conf)
    if y1 + 6 < h and bar_w > 0:
        img[y1 + 3:min(h, y1 + 7), x0:min(w, x0 + bar_w)] = rgb


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="demo/assets/wildfire_demo.mp4")
    ap.add_argument("--output", default="demo/assets/wildfire_demo_overlay.mp4")
    ap.add_argument("--incident-dir", default="demo/assets/incident-demo")
    ap.add_argument("--n", type=int, default=3, help="temporal filter: confirm on n of m")
    ap.add_argument("--m", type=int, default=5)
    args = ap.parse_args()

    import av

    tf = TemporalFilter(TemporalConfig(n=args.n, m=args.m))
    model = ModelInfo(name="demo-colour-threshold", version="0.0.0-not-a-model",
                      conf_threshold=0.34)

    src = open_source(SourceConfig(type="file", uri=args.input))
    container = None
    stream = None
    raw_total = confirmed_total = frames_with_boxes = 0

    with src, IncidentLogWriter(args.incident_dir, station_name="demo station",
                                source_uri=args.input, model=model) as log:
        for frame in src:
            img = np.ascontiguousarray(frame.image)
            if container is None:
                container = av.open(args.output, mode="w")
                stream = container.add_stream("libx264", rate=25)
                stream.width, stream.height = img.shape[1], img.shape[0]
                stream.pix_fmt = "yuv420p"
                stream.options = {"crf": "20", "preset": "medium"}

            raw = detect(img)
            raw_total += len(raw)
            confirmed = tf.update(raw, frame.pts)
            confirmed_total += len(confirmed)
            if confirmed:
                frames_with_boxes += 1

            for det in confirmed:
                draw_detection(img, det)

            # Header carries model identity and the last-inference time, so a
            # frozen pipeline is visible rather than silently showing old boxes.
            img[0:34] = (0.25 * img[0:34]).astype(np.uint8)
            draw_text(img, f"T {frame.pts:6.2f}S   MODEL {model.name}", 8, 9, 3, (235, 235, 235))
            img[-34:] = (0.22 * img[-34:]).astype(np.uint8)
            draw_text(img, "SITUATIONAL-AWARENESS AID - NOT A CERTIFIED SYSTEM",
                      8, img.shape[0] - 23, 3, (225, 225, 225))

            # Every frame is logged, including the ones with nothing on them.
            log.write(FrameDetections(frame_id=frame.frame_id, pts=frame.pts,
                                      detections=tuple(confirmed), model=model,
                                      source_id=args.input))

            for packet in stream.encode(av.VideoFrame.from_ndarray(img, format="bgr24")):
                container.mux(packet)

        n_frames = frame.frame_id + 1

    for packet in stream.encode():
        container.mux(packet)
    container.close()

    print(f"wrote {args.output}")
    print(f"  frames: {n_frames}, frames showing boxes: {frames_with_boxes}")
    print(f"  raw detections: {raw_total} -> after {args.n}-of-{args.m} filter: {confirmed_total} "
          f"({raw_total - confirmed_total} suppressed)")
    print(f"  incident log: {args.incident_dir}")


if __name__ == "__main__":
    main()
