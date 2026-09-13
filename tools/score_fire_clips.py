#!/usr/bin/env python3
"""The fire model over CLIPS rather than photographs.

    python3 tools/score_fire_clips.py --clips frames/ --report docs/metrics-fire-video.txt

WHY A CLIP IS SCORED AS A WHOLE
    An operator does not watch one frame. A fire that is marked on one frame in five is
    found, and a model that marks a different third of every clip is still telling them
    there is a fire there. So the question asked here is "was this fire ever marked, in ten
    seconds of looking at it", which is what the tablet actually gets to do.

    Per-frame recall is reported beside it, because the two answer different things: the
    clip figure says whether an operator is told, the frame figure says how steady the box
    would be once they look.

INPUT
    One directory per clip, each holding that clip's frames in filename order. The frames
    are extracted at the cycle the device detects at, not at the video's own frame rate:
    the tablet sees about four frames a second, and scoring all thirty would credit it with
    looks it never gets.
"""
import argparse, json, pathlib, sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import evaluate_fire                                    # noqa: E402  the decode, shared

ROOT = HERE.parent
THRESHOLDS = (0.10, 0.15, 0.20, 0.25, 0.35, 0.50)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", required=True, help="a directory of per-clip frame folders")
    ap.add_argument("--model", default=str(ROOT / "web/models/wildfire-640.onnx"))
    ap.add_argument("--manifest", default=str(ROOT / "web/models/manifest.json"))
    ap.add_argument("--subject", default="wildfire")
    ap.add_argument("--report", help="append the table here as well as printing it")
    args = ap.parse_args()

    spec = evaluate_fire.spec_for(args.manifest, args.subject)
    detect = evaluate_fire.Detector(args.model, spec)
    shipped = spec.get("confThreshold", 0.25)

    clips = sorted(p for p in pathlib.Path(args.clips).iterdir() if p.is_dir())
    if not clips:
        raise SystemExit(f"no clip folders in {args.clips}")

    lines = []

    def say(text=""):
        print(text, flush=True)
        lines.append(text)

    say(f"    model      {pathlib.Path(args.model).name}")
    say(f"    threshold  {shipped:.2f}, as shipped")
    say(f"    clips      {len(clips)}")
    say()
    say(f"    {'clip':<34}{'frames':>7}{'marked':>8}{'best':>7}  {'found at':>8}")

    best_by_clip = []
    frames_total = frames_marked = 0
    for clip in clips:
        frames = evaluate_fire.pictures(clip)
        best = 0.0
        marked = 0
        for frame in frames:
            found = detect(frame, 0.05)
            top = max((f["conf"] for f in found), default=0.0)
            best = max(best, top)
            if top >= shipped:
                marked += 1
        frames_total += len(frames)
        frames_marked += marked
        best_by_clip.append(best)
        found_at = next((f"{t:.2f}" for t in THRESHOLDS if best >= t), "-")
        say(f"    {clip.name[:33]:<34}{len(frames):>7}{marked:>8}{best:>7.2f}  {found_at:>8}")

    say()
    say(f"    {'threshold':>10}{'clips found':>14}{'frames marked':>16}")
    for threshold in THRESHOLDS:
        clips_found = sum(1 for b in best_by_clip if b >= threshold)
        say(f"    {threshold:>10.2f}{f'{clips_found}/{len(clips)}':>14}"
            f"{f'{frames_marked if threshold == shipped else chr(45)}':>16}")
    say()
    say(f"    At the shipped {shipped:.2f}: {sum(1 for b in best_by_clip if b >= shipped)}"
        f" of {len(clips)} clips ever marked, and {frames_marked} of {frames_total} frames.")
    say("    A clip with nothing marked has not been cleared of anything; it failed to")
    say("    meet a threshold.")

    if args.report:
        with open(args.report, "a", encoding="utf-8") as out:
            out.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
