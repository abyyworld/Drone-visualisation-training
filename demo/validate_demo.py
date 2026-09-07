"""Run the false-negative validation workflow end to end on the demo clip.

This exists to prove the *harness* works before it is pointed at real footage.
It builds a YOLO-format split from the clip's ground truth, scores the
stand-in detector against it with ``tools/evaluate.py``, and writes the miss
list. Every step is the one you will run on the department's own video --
only the input changes.

What it does NOT do is tell you anything about detecting real fire. The clip is
synthetic and the detector is a colour threshold; the numbers below describe
that pairing and nothing else. ``docs/VALIDATION.md`` is the real procedure.

The condition tags are the point worth copying. Recall averaged over a whole
clip hides exactly the cases that matter -- a model that finds a well-developed
fire every time and never sees the first ninety seconds scores well and is
useless. Tagging frames by condition and reading recall per tag is what
surfaces that, and it is why evaluate.py takes --conditions at all.
"""

from __future__ import annotations

import argparse
import json
import struct
import subprocess
import sys
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from demo.simple_detector import detect
from station.core.config import SourceConfig
from station.core.types import CLASSES
from station.ingest import open_source


def write_png(path: Path, bgr: np.ndarray) -> None:
    """Minimal RGB8 PNG writer, so this needs no image library."""
    rgb = bgr[..., ::-1]
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 4))
        + chunk(b"IEND", b"")
    )


def conditions_for(boxes: list[dict], pts: float, ignite_pts: float) -> list[str]:
    """Tag a frame by the conditions that make detection hard.

    These are stand-ins for the real ones in docs/VALIDATION.md (night, thin
    smoke on bright sky, smouldering, under canopy). The mechanism is what
    transfers: tag the hard cases, then read recall per tag rather than overall.
    """
    tags: list[str] = []
    if not boxes:
        tags.append("no_target")
        return tags
    if pts - ignite_pts < 4.0:
        # The first seconds: small, faint, and the most valuable to catch.
        tags.append("early_fire")
    for b in boxes:
        area = (b["box"][2] - b["box"][0]) * (b["box"][3] - b["box"][1])
        if b["cls"] == "smoke" and area > 0.25:
            tags.append("large_plume")
        if area < 0.01:
            tags.append("small_target")
    return sorted(set(tags))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", default="demo/assets/wildfire_demo.mp4")
    ap.add_argument("--truth", default="demo/assets/wildfire_demo.truth.json")
    ap.add_argument("--out", default="/tmp/wildfire-validation")
    ap.add_argument("--every", type=int, default=5, help="sample every Nth frame")
    args = ap.parse_args()

    truth = json.loads(Path(args.truth).read_text())
    frames = {f["frame_id"]: f for f in truth["frames"]}
    ignite = next((f["pts"] for f in truth["frames"] if f["boxes"]), 0.0)

    out = Path(args.out)
    img_dir, lbl_dir = out / "images" / "val", out / "labels" / "val"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    preds_path, cond_path = out / "predictions.jsonl", out / "conditions.json"
    conditions: dict[str, list[str]] = {}
    n = 0

    with open_source(SourceConfig(type="file", uri=args.video)) as src, \
         preds_path.open("w", encoding="utf-8") as preds:
        for frame in src:
            if frame.frame_id % args.every:
                continue
            gt = frames.get(frame.frame_id)
            if gt is None:
                continue
            name = f"frame_{frame.frame_id:05d}"
            rel = f"images/val/{name}.png"
            img = np.ascontiguousarray(frame.image)
            write_png(img_dir / f"{name}.png", img)

            # Ground truth as YOLO: class cx cy w h, normalised.
            lines = []
            for b in gt["boxes"]:
                x1, y1, x2, y2 = b["box"]
                lines.append(
                    f"{CLASSES.index(b['cls'])} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} "
                    f"{x2 - x1:.6f} {y2 - y1:.6f}"
                )
            (lbl_dir / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))

            preds.write(json.dumps({
                "image": rel,
                "detections": [
                    {"cls": d.cls, "conf": d.conf, "box": [round(v, 5) for v in d.box.as_tuple()]}
                    for d in detect(img)
                ],
            }) + "\n")
            conditions[rel] = conditions_for(gt["boxes"], gt["pts"], ignite)
            n += 1

    cond_path.write_text(json.dumps(conditions, indent=1))
    (out / "data.yaml").write_text(
        f"path: {out}\ntrain: images/val\nval: images/val\n"
        f"nc: {len(CLASSES)}\nnames: [{', '.join(CLASSES)}]\n"
    )
    print(f"built {n} frames at {out}\n")

    cmd = [
        sys.executable, "tools/evaluate.py",
        "--data", str(out / "data.yaml"),
        "--predictions", str(preds_path),
        "--conditions", str(cond_path),
        "--miss-list", str(out / "misses.json"),
        "--json", str(out / "report.json"),
    ]
    print("$ " + " ".join(cmd) + "\n")
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
