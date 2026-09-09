#!/usr/bin/env python3
"""Is this model reading its input, or does it answer the same for every picture?

WHY THIS EXISTS
    A fire model was fetched, converted, committed, described in the manifest and shipped,
    and it returns a score between 0.0123 and 0.0155 for black, white, random noise, a
    flame-coloured block, sixty real drone photographs and fifty frames of the fire clip
    alike. It could not mark fire because it could not mark anything, and every check in
    the conversion workflow passed: the graph ran, the output was a shape the app decodes,
    the head matched the label list. The one step that ran the model ran it on a blank
    image and printed the shape.

WHAT IT ASKS, AND WHAT IT DELIBERATELY DOES NOT
    Only whether the answer CHANGES when the picture changes.

    It does not ask what the model finds. That question needs footage of the subject, which
    for most of these subjects does not exist here, and judging it wrongly is expensive: an
    earlier version of the workflow's check demanded people in a street photograph and
    would have thrown away the aerial model that is the whole point. Responsiveness needs
    none of that. A model that is quiet on a picture still moves; one that is dead does not.

WHY THE PICTURES ARE WHAT THEY ARE
    Measured, not chosen. Flat colour blocks alone do not separate anything - the person
    model spreads only 0.022 over five of them, which is barely over the floor. Real
    photographs are what carry it:

        pictures                                person   solar   wildfire
        five flat colour blocks                  0.022   0.129      0.001
        those plus one real photograph           0.262   0.158      0.003
        two real photographs plus three patterns 0.528   0.766      0.002

    So the two photographs are required and the generated patterns are there to widen the
    span, not to carry it. Without a photograph this reports that it could not judge, which
    is not the same as passing.

    Both live models clear the floor by more than twenty five times and the dead one sits
    nearly ten times under it, so this is not a close call being decided by a threshold.

USAGE
    python3 tools/model_liveness.py web/models/person-320.onnx --labels 11
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import urllib.request

# Eight times below the worst live model measured, seven times above the dead one.
FLOOR = 0.02

# Two real photographs, from the same place the conversion workflow already fetches its
# test picture. Ordinary photographs of ordinary things: what matters is that they are
# photographs rather than that they contain anything in particular.
PHOTOGRAPHS = {
    "bus.jpg": "https://raw.githubusercontent.com/ultralytics/ultralytics/main/"
               "ultralytics/assets/bus.jpg",
    "zidane.jpg": "https://raw.githubusercontent.com/ultralytics/ultralytics/main/"
                  "ultralytics/assets/zidane.jpg",
}


def patterns():
    """Generated pictures, to widen the span. Never enough on their own."""
    import numpy as np

    rng = np.random.default_rng(7)
    noise = (rng.random((240, 320, 3)) * 255).astype(np.uint8)
    checker = ((np.indices((240, 320)).sum(0) % 32 < 16).astype(np.uint8)[:, :, None]
               * np.uint8([255, 180, 40]))
    flame = np.zeros((240, 320, 3), np.uint8)
    flame[80:170, 100:220] = [255, 140, 20]
    return [("noise", noise), ("checker", checker), ("flame block", flame)]


def photographs(cache):
    """The real photographs, fetched once into `cache`. Empty if none could be had."""
    import numpy as np
    from PIL import Image

    found = []
    cache = pathlib.Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    for name, url in PHOTOGRAPHS.items():
        path = cache / name
        if not path.exists():
            try:
                request = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
                with urllib.request.urlopen(request, timeout=60) as response:
                    path.write_bytes(response.read())
            except Exception as problem:
                print(f"  could not fetch {name}: {problem}", file=sys.stderr)
                continue
        try:
            found.append((name, np.asarray(Image.open(path).convert("RGB"))))
        except Exception as problem:
            print(f"  could not read {name}: {problem}", file=sys.stderr)
    return found


def letterbox(array, size):
    """The app's own preprocessing: fit inside a square, pad with grey 114."""
    import numpy as np
    from PIL import Image

    image = Image.fromarray(array)
    scale = min(size / image.width, size / image.height)
    dw, dh = round(image.width * scale), round(image.height * scale)
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    canvas.paste(image.resize((dw, dh), Image.BILINEAR), ((size - dw) // 2, (size - dh) // 2))
    return np.asarray(canvas, np.float32).transpose(2, 0, 1)[None] / 255.0


def scores(session, labels, size, images):
    """The best class score each picture draws out of the model."""
    name = session.get_inputs()[0].name
    out = []
    for _, array in images:
        head = session.run(None, {name: letterbox(array, size)})[0][0]
        if head.shape[0] > head.shape[1]:
            head = head.T                  # per-anchor rows rather than channel-major
        # The real class channels only. A segmentation export carries mask coefficients
        # after them, and coefficients are not scores: unbounded, so read as scores they
        # mark every frame at over 1.0.
        out.append(float(head[4:4 + labels].max()))
    return out


def judge(session, labels, size, cache):
    """(spread, per-picture scores, whether there was enough to judge with)."""
    images = photographs(cache)
    enough = len(images) >= 1
    images = images + patterns()
    measured = scores(session, labels, size, images)
    named = list(zip([n for n, _ in images], measured))
    return (max(measured) - min(measured)) if measured else 0.0, named, enough


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model")
    parser.add_argument("--labels", type=int, required=True,
                        help="how many real classes the head carries, per the manifest")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--cache", default=".liveness")
    args = parser.parse_args(argv)

    import onnxruntime as ort

    session = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    spread, named, enough = judge(session, args.labels, args.imgsz, args.cache)
    for name, score in named:
        print(f"  {name:16} {score:.4f}")
    print(f"spread across {len(named)} pictures: {spread:.5f}")

    if not enough:
        # Not a pass. Flat and generated pictures alone do not separate a quiet model from
        # a dead one, so saying nothing is the only honest answer.
        print("::warning::No photograph could be fetched, so this could not be judged. "
              "Generated pictures alone do not separate a quiet model from a dead one.")
        return 0
    if spread < FLOOR:
        print(f"::error::These weights give the same answer for every picture (spread "
              f"{spread:.5f}, floor {FLOOR}). That is not a detector being quiet, it is one "
              f"that is not reading its input, and it would ship as a model that marks "
              f"nothing while the app reports it as working.")
        return 1
    print("the model responds to what it is shown")
    return 0


if __name__ == "__main__":
    sys.exit(main())
