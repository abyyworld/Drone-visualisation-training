#!/usr/bin/env python3
"""Does the Python signature agree with the one the app computes?

    python3 tools/check_reid_port.py

tools/fly_video.py has to build the same colour signature as web/js/reid.js, because the
tracker uses it to decide that somebody who walked behind a van is the same person coming
out the other side. If the two disagree, every re-identification number measured on real
footage is about a mechanism the app does not have.

So: the same random crops through both, compared bin by bin. A port that drifts fails here
rather than quietly reporting a better count than the tablet can achieve.
"""
import json, pathlib, subprocess, sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from fly_video import describe  # noqa: E402

RNG = np.random.default_rng(20260912)
CASES = [
    (40, 90, [2, 3, 38, 88]),      # an ordinary person-shaped box
    (24, 24, [0, 0, 24, 24]),      # the whole tile
    (60, 40, [10, 5, 55, 38]),     # wider than tall
    (10, 14, [0, 0, 10, 14]),      # just above the floor
    (9, 7, [0, 0, 9, 7]),          # below it: both must decline to describe
    (50, 50, [-5, -5, 60, 60]),    # a box running off every edge
    (32, 64, [7, 9, 31, 63]),      # an odd-sized band split
]

script = """
import { readFileSync } from 'node:fs';
import { describe } from '../web/js/reid.js';
// Through a file rather than argv: a few megapixels of test data is past what a command
// line will carry, and the failure for that is a confusing OSError rather than a clear one.
const cases = JSON.parse(readFileSync(process.argv[2], 'utf8'));
const out = [];
for (const { w, h, pixels, box } of cases) {
  const signature = describe(Uint8ClampedArray.from(pixels), w, h, box);
  out.push(signature ? Array.from(signature).map((v) => Math.round(v * 1e6) / 1e6) : null);
}
console.log(JSON.stringify(out));
"""

payload, mine = [], []
for width, height, box in CASES:
    frame = RNG.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    rgba = np.dstack([frame, np.full((height, width, 1), 255, dtype=np.uint8)])
    payload.append({"w": width, "h": height, "box": box,
                    "pixels": rgba.reshape(-1).tolist()})
    mine.append(describe(frame, box))

runner = ROOT / "tools" / ".reid_check.mjs"
cases_file = ROOT / "tools" / ".reid_cases.json"
runner.write_text(script)
cases_file.write_text(json.dumps(payload))
try:
    result = subprocess.run(["node", str(runner), str(cases_file)],
                            capture_output=True, text=True, cwd=ROOT / "tools", check=True)
finally:
    runner.unlink(missing_ok=True)
    cases_file.unlink(missing_ok=True)
theirs = json.loads(result.stdout)

problems = []
for (width, height, box), a, b in zip(CASES, mine, theirs):
    where = f"{width}x{height} box {box}"
    if (a is None) != (b is None):
        problems.append(f"{where}: python {'declined' if a is None else 'described'}, "
                        f"javascript {'declined' if b is None else 'described'}")
        continue
    if a is None:
        print(f"  ok  {where}: both decline, too few pixels to describe")
        continue
    worst = max(abs(x - y) for x, y in zip(a, b))
    if worst > 1e-5:
        problems.append(f"{where}: worst bin differs by {worst:.6f}")
    else:
        print(f"  ok  {where}: {len(a)} bins agree, worst difference {worst:.1e}")

if problems:
    print("\nThe Python port and web/js/reid.js disagree:")
    for p in problems:
        print(f"  {p}")
    print("\nEvery re-identification number measured with this port is about a mechanism "
          "the app does not have. Fix the port before trusting any of them.")
    raise SystemExit(1)
print("\nthe port agrees with the app on every case")
