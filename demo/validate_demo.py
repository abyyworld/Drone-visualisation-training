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
from station.core.config import SourceConfig, TemporalConfig
from station.core.types import CLASSES
from station.inference.temporal import TemporalFilter
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
    ap.add_argument("--n", type=int, default=3, help="temporal filter: confirm on n of m")
    ap.add_argument("--m", type=int, default=5)
    ap.add_argument(
        "--flicker", type=float, default=0.0, metavar="P",
        help=(
            "drop each raw detection with probability P, modelling a model that "
            "flickers frame to frame. The demo clip's target is perfectly stable, "
            "so without this the filter has nothing to suppress and the "
            "comparison shows no difference in either direction."
        ),
    )
    ap.add_argument("--seed", type=int, default=0, help="seed for --flicker")
    args = ap.parse_args()

    truth = json.loads(Path(args.truth).read_text())
    frames = {f["frame_id"]: f for f in truth["frames"]}
    ignite = next((f["pts"] for f in truth["frames"] if f["boxes"]), 0.0)

    out = Path(args.out)
    img_dir, lbl_dir = out / "images" / "val", out / "labels" / "val"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    raw_path = out / "predictions-raw.jsonl"
    filtered_path = out / "predictions-filtered.jsonl"
    cond_path = out / "conditions.json"
    conditions: dict[str, list[str]] = {}
    n = 0

    # The temporal filter has to see EVERY frame, not just the sampled ones:
    # it ages tracks per frame, and feeding it one frame in five would age
    # every track out five times too fast and make the comparison meaningless.
    # So it runs over the whole sequence and only its output on sampled frames
    # is recorded.
    tf = TemporalFilter(TemporalConfig(n=args.n, m=args.m))
    rng = np.random.default_rng(args.seed)

    with open_source(SourceConfig(type="file", uri=args.video)) as src, \
         raw_path.open("w", encoding="utf-8") as raw_out, \
         filtered_path.open("w", encoding="utf-8") as filt_out:
        for frame in src:
            img = np.ascontiguousarray(frame.image)
            raw = detect(img)
            if args.flicker > 0:
                # Applied before both arms so they see the same input: the
                # question is what the filter does with a flickering model, not
                # whether two different detectors disagree.
                raw = [d for d in raw if rng.random() >= args.flicker]
            confirmed = tf.update(raw, frame.pts)

            if frame.frame_id % args.every:
                continue
            gt = frames.get(frame.frame_id)
            if gt is None:
                continue

            name = f"frame_{frame.frame_id:05d}"
            rel = f"images/val/{name}.png"
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

            for dets, handle in ((raw, raw_out), (confirmed, filt_out)):
                handle.write(json.dumps({
                    "image": rel,
                    "detections": [
                        {"cls": d.cls, "conf": d.conf, "box": [round(v, 5) for v in d.box.as_tuple()]}
                        for d in dets
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

    def score(preds: Path, tag: str) -> dict:
        """Run tools/evaluate.py over one prediction file and load its report."""
        report = out / f"report-{tag}.json"
        cmd = [
            sys.executable, "tools/evaluate.py",
            "--data", str(out / "data.yaml"),
            "--predictions", str(preds),
            "--conditions", str(cond_path),
            "--miss-list", str(out / f"misses-{tag}.json"),
            "--json", str(report),
            "--quiet",
        ]
        subprocess.call(cmd)
        return json.loads(report.read_text())

    raw_report = score(raw_path, "raw")
    filt_report = score(filtered_path, "filtered")

    def at_operating(report: dict) -> dict | None:
        conf = report.get("operating_conf")
        for row in report.get("thresholds", []):
            if abs(row.get("conf", -1) - conf) < 1e-9:
                return row
        return None

    a, b = at_operating(raw_report), at_operating(filt_report)
    if a is None or b is None:
        print("could not locate the operating point in both reports")
        raise SystemExit(1)

    print("=" * 78)
    print(f"WHAT THE {args.n}-OF-{args.m} TEMPORAL FILTER COSTS IN RECALL")
    print("=" * 78)
    print()
    print("The filter suppresses flicker by requiring a detection to persist. That")
    print("is the single biggest accuracy lever in the pipeline, and it is bought")
    print("with recall: anything it holds back is a target the operator is not")
    print("shown. This is the price, measured rather than assumed.")
    print()
    print(f"  {'':22s} {'raw':>12s} {'filtered':>12s} {'delta':>10s}")

    def row(label: str, x: dict | None, y: dict | None) -> None:
        if not x or not y or x.get("recall") is None or y.get("recall") is None:
            return
        d = y["recall"] - x["recall"]
        flag = "" if d >= -0.02 else "   <-- look here"
        print(f"  {label:22s} {x['recall']:>12.3f} {y['recall']:>12.3f} {d:>+10.3f}{flag}")

    row("overall recall", a.get("overall"), b.get("overall"))
    for cls in CLASSES:
        row(f"  {cls}", a.get("per_class", {}).get(cls), b.get("per_class", {}).get(cls))
    for bucket in ("tiny", "small", "medium", "large"):
        row(f"  {bucket} targets", a.get("per_bucket", {}).get(bucket), b.get("per_bucket", {}).get(bucket))
    for tag in sorted(set(a.get("per_tag", {})) | set(b.get("per_tag", {}))):
        row(f"  {tag}", a.get("per_tag", {}).get(tag), b.get("per_tag", {}).get(tag))

    print()
    delta = b["overall"]["tp"] - a["overall"]["tp"]
    if delta < 0:
        print(f"  The filter withheld {-delta} target(s) the detector had already found.")
        print("  Whether that is worth the stability it buys is a judgement for the")
        print("  department, not a default: raise n/m for calmer boxes, lower it to")
        print("  surface weaker evidence sooner.")
    elif delta > 0:
        print(f"  The filter RECOVERED {delta} target(s) the detector dropped, by")
        print("  coasting a confirmed track through frames the model missed. On a")
        print("  flickering model that is the filter earning its place: the operator")
        print("  sees a steady box instead of one blinking in and out.")
    else:
        print("  The filter changed nothing at this operating point.")
        if not args.flicker:
            print("  This clip's target never flickers, so there is nothing to suppress")
            print("  or recover. Re-run with --flicker 0.3 to model a model that does.")
    print()
    print(f"  full reports: {out}/report-raw.json, {out}/report-filtered.json")
    print(f"  miss lists:   {out}/misses-raw.json, {out}/misses-filtered.json")
    print()
    print("  Synthetic clip, colour-threshold stand-in: these numbers describe that")
    print("  pairing and nothing else. docs/VALIDATION.md is the real procedure.")


if __name__ == "__main__":
    main()
