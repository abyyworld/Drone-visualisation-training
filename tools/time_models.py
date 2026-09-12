#!/usr/bin/env python3
"""How long each of these detectors takes, on THIS machine, interleaved.

    python3 tools/time_models.py web/models/person-640.onnx web/models/experiments/x.onnx

WHY INTERLEAVED, AND WHY IN ONE RUN
    A candidate is rejected or adopted on what it costs per look, and that cost was being
    read across two CI runs on two different runners. Those are not comparable: the same
    weights measured 20.9 ms on one runner and 45.1 ms on another machine on the same day,
    which is enough to reverse a verdict. One of those readings made a slower model look
    twice as fast as the one it was competing with.

    So both models are timed here, on one machine, in one process, alternating between them
    so that a busy moment lands on both rather than on whichever went first. What comes out
    is a RATIO, which is the only part that carries to the tablet: the MK15's absolute
    milliseconds are its own and are measured on the device.
"""
import argparse, pathlib, statistics, time
import numpy as np
import onnxruntime as ort


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="+")
    ap.add_argument("--rounds", type=int, default=15)
    # Two, because that is what the tablet gives it: the work thread is one of a small
    # number of cores and the rest of them are decoding video.
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()

    sessions = {}
    for name in args.models:
        if not pathlib.Path(name).exists():
            raise SystemExit(f"no such model: {name}")
        options = ort.SessionOptions()
        options.intra_op_num_threads = args.threads
        options.inter_op_num_threads = 1
        sessions[name] = ort.InferenceSession(name, options,
                                              providers=["CPUExecutionProvider"])

    frame = {}
    for name, session in sessions.items():
        shape = [d if isinstance(d, int) else 1 for d in session.get_inputs()[0].shape]
        frame[name] = np.random.rand(*shape).astype(np.float32)
        session.run(None, {session.get_inputs()[0].name: frame[name]})

    taken = {name: [] for name in sessions}
    for _ in range(args.rounds):
        for name, session in sessions.items():
            started = time.perf_counter()
            session.run(None, {session.get_inputs()[0].name: frame[name]})
            taken[name].append((time.perf_counter() - started) * 1000)

    first = None
    print(f"  {'detector':<34}{'ms a look':>12}{'against the first':>20}")
    for name in sessions:
        median = statistics.median(taken[name])
        first = median if first is None else first
        print(f"  {pathlib.Path(name).name:<34}{median:>12.1f}{median / first:>19.2f}x")
    print("\n  One machine, one process, alternating. The ratio is the part that carries;")
    print("  the milliseconds are this machine's and not the tablet's.")


if __name__ == "__main__":
    main()
