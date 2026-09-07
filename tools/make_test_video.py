#!/usr/bin/env python3
"""Generate a short synthetic test clip with a ground-truth sidecar.

The clip comes from :class:`station.ingest.synthetic_source.SyntheticSource`,
which is the one video source in this system that knows exactly where its
target is. That is the whole point: the sidecar written next to the video is
not an annotation somebody made, it is the geometry the renderer drew from, so
overlay alignment, the N-of-M temporal filter and the staleness rule can be
asserted against an exact answer instead of eyeballed.

What it is not: a fire simulation. The target is a bright warm blob on a
sky/ground gradient. It exercises geometry, timing and the wire format. Any
statement about a real model's behaviour derived from this clip would be
meaningless, and this file deliberately does not make it easy to pretend
otherwise -- the sidecar labels itself as generated geometry.

Works with no encoder present. If PyAV is installed the output is a real
video file; if it is not, the same frames are written as packed BGR24 with a
sidecar describing the layout, or as PNGs written by a ~20-line encoder built
on ``zlib``. Every path produces the same ground truth, so a test fixture can
be regenerated on a machine with nothing installed.

A frame whose ``boxes`` list is empty is a frame on which the generator drew
no target. In the ``--present`` ranges that is by construction; nothing in
this file, or in the sidecar it writes, should ever be read as a statement
about a scene.

Example:
    $ python3 tools/make_test_video.py --out build/overlay_fixture.mp4 \\
          --frames 150 --fps 30 --present 40-110 --wire-jsonl build/truth.jsonl
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import zlib
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Imported at module scope on purpose: these pull in numpy and station.core and
# nothing else. Every decoder and encoder in this file is imported lazily.
from station.core.types import (  # noqa: E402
    CLASS_FIRE,
    BBox,
    Detection,
    FrameDetections,
    ModelInfo,
)
from station.ingest.synthetic_source import (  # noqa: E402
    SyntheticScene,
    SyntheticSource,
    synthetic_config,
)

__all__ = [
    "TRUTH_SIDECAR_VERSION",
    "TruthFrame",
    "Generated",
    "generate",
    "write_video",
    "write_raw",
    "write_png_sequence",
    "write_sidecar",
    "write_wire_jsonl",
    "main",
]

#: Bumped when the sidecar's shape changes. A fixture and the test that reads
#: it are usually committed months apart.
TRUTH_SIDECAR_VERSION = 1

#: Identity stamped on the wire-format truth stream. Not a model name: nothing
#: in an incident log may ever record generated geometry under the identity of
#: a model that was supposed to have produced it.
TRUTH_MODEL = ModelInfo(
    name="synthetic-ground-truth",
    version=f"sidecar-v{TRUTH_SIDECAR_VERSION}",
    classes=(CLASS_FIRE,),
)


@dataclass(frozen=True, slots=True)
class Generated:
    """What :func:`generate` produced.

    ``output_fps`` is separate from ``scene.fps`` and the distinction matters:
    with ``--target-fps`` the ingest layer drops frames, so the emitted frames
    step through the media timeline at the *output* rate. Encoding them at the
    scene rate would produce a file whose playback clock disagrees with the pts
    in the sidecar, which is precisely the drift the overlay fixture exists to
    detect.
    """

    images: list[np.ndarray]
    truth: list["TruthFrame"]
    scene: SyntheticScene
    source_uri: str
    output_fps: float


class TruthFrame(dict):
    """One frame's ground truth, as it appears in the sidecar.

    A plain dict subclass so it serialises directly, with the keys documented
    in one place: ``frame_id``, ``pts`` (media seconds) and ``boxes`` (a list
    of ``{"cls", "box"}`` with ``box`` normalised ``[x1, y1, x2, y2]``).
    """


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


# ---------------------------------------------------------------- generation


def generate(
    *,
    frames: int,
    fps: float,
    width: int,
    height: int,
    seed: int,
    present: str | None = None,
    target_fps: float | None = None,
    scene_overrides: dict[str, Any] | None = None,
) -> Generated:
    """Render frames and their ground truth from the synthetic source.

    Frames are pulled through :class:`SyntheticSource` rather than straight off
    :class:`SyntheticScene`, so the pts values in the sidecar are produced by
    the same clock the live pipeline uses -- including the effect of
    ``target_fps`` decimation. A fixture whose timestamps were computed a
    different way from the pipeline's would quietly pass overlay-alignment
    tests that the real path fails.

    Args:
        frames: Number of frames to generate before decimation.
        fps: Scene frame rate; fixes the media timeline (``pts = index / fps``).
        width: Frame width in pixels.
        height: Frame height in pixels.
        seed: Scene seed; the same seed always gives the same pixels.
        present: Half-open index ranges in which the target exists, e.g.
            ``"40-110,150-180"``. ``None`` means present throughout.
        target_fps: Rate cap applied by the ingest layer, as for a real source.
        scene_overrides: Extra :class:`SyntheticScene` parameters (``radius``,
            ``noise``, ``jitter``, ``amp_x``, ``period_x``, ...).

    Returns:
        A :class:`Generated` holding HxWx3 uint8 BGR frames, the per-frame
        ground truth, the scene, its reproducing URI and the emitted frame rate.

    Raises:
        station.ingest.base.SourceConfigError: A scene parameter is out of range.
    """
    params: dict[str, Any] = {
        "width": width,
        "height": height,
        "fps": fps,
        "seed": seed,
        "frames": frames,
    }
    if present:
        params["present"] = present
    params.update(scene_overrides or {})

    cfg = synthetic_config(target_fps=target_fps, **params)
    images: list[np.ndarray] = []
    truth: list[TruthFrame] = []
    with SyntheticSource(cfg) as source:
        scene = source.scene
        uri = source.info.uri
        output_fps = source.info.output_fps
        for frame in source:
            images.append(frame.image)
            box = source.ground_truth_for(frame)
            boxes = [] if box is None else [{"cls": CLASS_FIRE, "box": box.to_wire()}]
            truth.append(
                TruthFrame(frame_id=frame.frame_id, pts=round(frame.pts, 6), boxes=boxes)
            )
    return Generated(
        images=images, truth=truth, scene=scene, source_uri=uri, output_fps=output_fps
    )


# ------------------------------------------------------------------- writers


def write_video(
    path: Path,
    images: Sequence[np.ndarray],
    fps: float,
    *,
    codec: str = "libx264",
    crf: int = 20,
    pix_fmt: str = "yuv420p",
) -> dict[str, Any]:
    """Encode frames to a video file with PyAV.

    Args:
        path: Output file; the container is chosen from the extension.
        images: HxWx3 uint8 BGR frames.
        fps: Frame rate written into the container.
        codec: Encoder name. ``libx264`` unless the build lacks it.
        crf: x264 quality, lower is better. 20 keeps the blob's edge crisp
            enough that a detector run on the clip sees roughly the geometry
            the sidecar claims.
        pix_fmt: Pixel format. ``yuv420p`` for compatibility with browsers and
            with every decoder in this project.

    Returns:
        A dict describing what was written, for the sidecar.

    Raises:
        RuntimeError: PyAV is missing or the encoder is unavailable. The caller
            falls back to a codec-free format rather than failing the run --
            but the error names the package, because a silent fallback to raw
            frames would surprise somebody expecting an .mp4.
    """
    try:
        import av  # noqa: PLC0415 -- lazy by design; the fallback needs no codec.
    except ImportError as exc:
        raise RuntimeError(
            "PyAV is not installed, so no video can be encoded.\n"
            "  pip install av\n"
            "Use --format raw or --format png to produce a fixture without an encoder."
        ) from exc

    if not images:
        raise RuntimeError("no frames to encode")
    height, width = images[0].shape[:2]
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        container = av.open(str(path), mode="w")
    except Exception as exc:
        raise RuntimeError(f"could not open {path} for writing: {exc}") from exc
    try:
        # Rate as an exact Fraction where possible: a container rate of
        # 29.999 turns pts into a slowly drifting approximation, and this
        # fixture exists to test timestamp handling.
        stream = container.add_stream(codec, rate=_exact_rate(fps))
        stream.width, stream.height = width, height
        stream.pix_fmt = pix_fmt
        if codec in ("libx264", "libx265"):
            stream.options = {"crf": str(crf), "preset": "medium"}
        for image in images:
            frame = av.VideoFrame.from_ndarray(image, format="bgr24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():  # flush the encoder's lookahead
            container.mux(packet)
    except Exception as exc:
        container.close()
        raise RuntimeError(f"encoding with {codec!r} failed: {exc}") from exc
    finally:
        try:
            container.close()
        except Exception:
            pass

    return {
        "kind": "video",
        "path": path.name,
        "codec": codec,
        "pix_fmt": pix_fmt,
        "crf": crf if codec in ("libx264", "libx265") else None,
    }


def _exact_rate(fps: float) -> Any:
    """Container rate as a Fraction when fps is clean, else the float."""
    from fractions import Fraction  # noqa: PLC0415 -- stdlib, but only needed here.

    fraction = Fraction(fps).limit_denominator(1001)
    return fraction if abs(float(fraction) - fps) < 1e-9 else fps


def verify_video(path: Path, expected_frames: int) -> tuple[bool, str]:
    """Decode the written file back and count its frames.

    An encoder that silently dropped the tail would leave a fixture whose
    sidecar promises frames the file does not contain, and every test built on
    it would fail somewhere far away from the cause.

    Args:
        path: The file just written.
        expected_frames: How many frames were handed to the encoder.

    Returns:
        ``(ok, message)``. ``ok`` is True when the count matches; a missing
        PyAV returns True with a message saying the check was not run, because
        a fallback path that cannot verify is not the same as a failure.
    """
    try:
        import av  # noqa: PLC0415 -- lazy by design.
    except ImportError:
        return True, "PyAV absent; the written file was not read back"
    try:
        with av.open(str(path)) as container:
            decoded = sum(1 for _ in container.decode(video=0))
    except Exception as exc:
        return False, f"the file could not be decoded back: {exc}"
    if decoded != expected_frames:
        return False, f"encoded {expected_frames} frames but decoded {decoded}"
    return True, f"decoded {decoded} frames, matching the sidecar"


def write_raw(path: Path, images: Sequence[np.ndarray]) -> dict[str, Any]:
    """Write packed BGR24 frames with no container and no codec.

    The universal fallback: a flat file of ``frames * height * width * 3``
    bytes that ``numpy.memmap`` can open directly. The sidecar carries the
    shape, so nothing has to be guessed.

    Args:
        path: Output file.
        images: HxWx3 uint8 BGR frames.

    Returns:
        A dict describing the layout, for the sidecar.
    """
    height, width = images[0].shape[:2]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        for image in images:
            fh.write(np.ascontiguousarray(image).tobytes())
    return {
        "kind": "raw",
        "path": path.name,
        "pix_fmt": "bgr24",
        "layout": "frames x height x width x 3, uint8, C-contiguous, no padding",
        "numpy": (
            f"np.memmap(path, dtype=np.uint8, mode='r')"
            f".reshape(-1, {height}, {width}, 3)"
        ),
    }


def _png_bytes(image: np.ndarray) -> bytes:
    """Encode one BGR frame as a PNG using only ``zlib``.

    Pillow would do this in one line, and Pillow is not installed on the
    machine this has to run on. PNG's non-interlaced truecolour form is a
    zlib stream of scanlines with a filter byte, so it is ~15 lines.
    """
    rgb = np.ascontiguousarray(image[:, :, ::-1])  # BGR (the project's order) -> RGB
    height, width = rgb.shape[:2]
    # Filter type 0 (None) per scanline: bigger files, no filter heuristics to
    # get wrong, and these fixtures are seconds long.
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(height))

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )


def write_png_sequence(directory: Path, images: Sequence[np.ndarray]) -> dict[str, Any]:
    """Write one PNG per frame into a directory. Needs no image library.

    Args:
        directory: Output directory, created if absent.
        images: HxWx3 uint8 BGR frames.

    Returns:
        A dict describing the sequence, for the sidecar.
    """
    directory.mkdir(parents=True, exist_ok=True)
    for index, image in enumerate(images):
        (directory / f"frame_{index:05d}.png").write_bytes(_png_bytes(image))
    return {
        "kind": "png_sequence",
        "path": directory.name,
        "pattern": "frame_%05d.png",
        "pix_fmt": "rgb24",
    }


def write_sidecar(
    path: Path,
    media: dict[str, Any],
    truth: Sequence[TruthFrame],
    scene: SyntheticScene,
    source_uri: str,
    *,
    fps: float,
    output_fps: float,
    target_fps: float | None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Write the ground-truth sidecar JSON.

    Args:
        path: Sidecar path.
        media: The writer's description of what was produced.
        truth: Per-frame ground truth.
        scene: The scene that generated it, recorded in full so the clip can be
            regenerated byte-for-byte from the sidecar alone.
        source_uri: The ``synthetic:?...`` URI that reproduces the scene.
        fps: Scene frame rate, before decimation.
        output_fps: Rate the frames were actually emitted at, which is the rate
            the media file was written at and the rate the pts values step by.
        target_fps: Decimation requested, if any.
        extra: Additional top-level fields.
    """
    payload: dict[str, Any] = {
        "version": TRUTH_SIDECAR_VERSION,
        "generator": "tools/make_test_video.py",
        "created": _now_iso(),
        "kind": "generated geometry, not annotated footage",
        "media": media,
        "width": scene.width,
        "height": scene.height,
        # Three rates, because conflating them is how an overlay fixture ends
        # up with a playback clock that disagrees with its own timestamps.
        "fps": output_fps,
        "scene_fps": fps,
        "output_fps": output_fps,
        "target_fps": target_fps,
        "frames": len(truth),
        "classes": [CLASS_FIRE],
        "box_convention": "normalised [x1, y1, x2, y2], origin top-left, same as station.core.types.BBox",
        "pts_convention": "media-timeline seconds, zero-based, as station.ingest.base.Frame.pts",
        "empty_boxes_mean": (
            "the generator drew no target on that frame; it is a fact about this file "
            "and carries no meaning about any scene"
        ),
        "source_uri": source_uri,
        "scene": asdict(scene),
        "truth": list(truth),
    }
    if extra:
        payload.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_wire_jsonl(path: Path, truth: Sequence[TruthFrame], source_uri: str) -> None:
    """Write the ground truth as wire-format ``detections`` messages.

    Lets a fixture drive the tablet's overlay code, or the incident-log reader,
    through exactly the parser the live path uses. Confidence is 1.0 and the
    model identity is ``synthetic-ground-truth``: these are the answers, not a
    model's opinion, and an incident log must never confuse the two.

    Args:
        path: Output JSONL file.
        truth: Per-frame ground truth from :func:`generate`.
        source_uri: Recorded as ``source_id`` on each message.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for entry in truth:
            detections = tuple(
                Detection(
                    cls=box["cls"],
                    conf=1.0,
                    box=BBox.from_wire(box["box"]),
                )
                for box in entry["boxes"]
            )
            message = FrameDetections(
                frame_id=int(entry["frame_id"]),
                pts=float(entry["pts"]),
                detections=detections,
                model=TRUTH_MODEL,
                source_id=source_uri,
            )
            fh.write(message.to_json() + "\n")


# ---------------------------------------------------------------------- cli


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        prog="make_test_video.py",
        description=(
            "Generate a short synthetic test clip from station.ingest.synthetic_source, "
            "with a JSON sidecar of exact ground-truth boxes. Falls back to raw frames or "
            "a PNG sequence when no encoder is installed."
        ),
        epilog=(
            "The clip is a moving bright blob on a gradient, not a fire. It exists so that "
            "overlay alignment, the N-of-M temporal filter and the staleness rule can be "
            "asserted against known geometry. Nothing measured on it says anything about "
            "how a model behaves on real footage."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    out = parser.add_argument_group("output")
    out.add_argument(
        "--out",
        type=Path,
        default=Path("test_video.mp4"),
        help="Output path. The extension picks the container (default: test_video.mp4).",
    )
    out.add_argument(
        "--format",
        choices=("auto", "video", "raw", "png"),
        default="auto",
        help="auto encodes video if PyAV is present and falls back to raw (default: auto).",
    )
    out.add_argument(
        "--truth",
        type=Path,
        default=None,
        help="Sidecar path (default: the output path with a .truth.json suffix).",
    )
    out.add_argument(
        "--wire-jsonl",
        type=Path,
        default=None,
        help="Also write the truth as wire-format 'detections' messages, one per line.",
    )
    out.add_argument("--force", action="store_true", help="Overwrite an existing output.")
    out.add_argument("--quiet", action="store_true", help="Print nothing on success.")

    clip = parser.add_argument_group("clip")
    clip.add_argument("--frames", type=int, default=150, help="Frames to generate (default: 150).")
    clip.add_argument(
        "--seconds",
        type=float,
        default=None,
        help="Clip length in seconds; overrides --frames.",
    )
    clip.add_argument("--fps", type=float, default=30.0, help="Scene frame rate (default: 30).")
    clip.add_argument(
        "--target-fps",
        type=float,
        default=None,
        help="Decimate to this rate through the ingest layer, as a real source would.",
    )
    clip.add_argument("--width", type=int, default=640, help="Frame width (default: 640).")
    clip.add_argument("--height", type=int, default=360, help="Frame height (default: 360).")
    clip.add_argument("--seed", type=int, default=0, help="Scene seed (default: 0).")
    clip.add_argument(
        "--present",
        default=None,
        metavar="RANGES",
        help="Frame ranges holding the target, e.g. 40-110,150-180. Default: every frame.",
    )

    scene = parser.add_argument_group("scene")
    scene.add_argument("--radius", type=float, default=None, help="Target radius as a fraction of height.")
    scene.add_argument("--noise", type=float, default=None, help="Background noise sigma, 0..255 levels.")
    scene.add_argument("--jitter", type=float, default=None, help="Per-frame centre jitter, fraction of height.")
    scene.add_argument("--amp-x", type=float, default=None, dest="amp_x", help="Horizontal travel amplitude.")
    scene.add_argument("--amp-y", type=float, default=None, dest="amp_y", help="Vertical travel amplitude.")
    scene.add_argument("--period-x", type=float, default=None, dest="period_x", help="Horizontal sweep period, s.")
    scene.add_argument("--period-y", type=float, default=None, dest="period_y", help="Vertical sweep period, s.")

    encode = parser.add_argument_group("encoding")
    encode.add_argument("--codec", default="libx264", help="PyAV encoder name (default: libx264).")
    encode.add_argument("--crf", type=int, default=20, help="x264/x265 quality, lower is better (default: 20).")
    encode.add_argument("--pix-fmt", default="yuv420p", help="Encoder pixel format (default: yuv420p).")
    return parser


def _scene_overrides(args: argparse.Namespace) -> dict[str, Any]:
    keys = ("radius", "noise", "jitter", "amp_x", "amp_y", "period_x", "period_y")
    return {key: getattr(args, key) for key in keys if getattr(args, key) is not None}


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point.

    Returns:
        0 on success, 1 when the output could not be written or fails
        verification, 2 on bad arguments.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    frames = args.frames
    if args.seconds is not None:
        if args.seconds <= 0:
            parser.error("--seconds must be positive")
        frames = max(1, int(round(args.seconds * args.fps)))
    if frames <= 0:
        parser.error("--frames must be positive")

    output = args.out
    if args.format == "png" and output.suffix:
        output = output.with_suffix("")
    if output.exists() and not args.force:
        print(f"make_test_video: {output} exists; pass --force to overwrite", file=sys.stderr)
        return 2

    def emit(text: str) -> None:
        if not args.quiet:
            print(text)

    try:
        produced = generate(
            frames=frames,
            fps=args.fps,
            width=args.width,
            height=args.height,
            seed=args.seed,
            present=args.present,
            target_fps=args.target_fps,
            scene_overrides=_scene_overrides(args),
        )
    except Exception as exc:
        print(f"make_test_video: {exc}", file=sys.stderr)
        return 2

    images, truth = produced.images, produced.truth
    scene, source_uri = produced.scene, produced.source_uri
    # The encoded frame rate is the rate frames actually come out at, so the
    # container's clock and the sidecar's pts describe the same timeline.
    output_fps = produced.output_fps

    if not images:
        print("make_test_video: the source produced no frames", file=sys.stderr)
        return 1

    media: dict[str, Any]
    notes: list[str] = []
    want_video = args.format in ("auto", "video")
    if want_video:
        try:
            media = write_video(
                output, images, output_fps, codec=args.codec, crf=args.crf, pix_fmt=args.pix_fmt
            )
        except RuntimeError as exc:
            if args.format == "video":
                print(f"make_test_video: {exc}", file=sys.stderr)
                return 1
            # auto: fall back, loudly. A fixture is more useful than a failure,
            # but the operator must not think they got an mp4.
            fallback = output.with_suffix(".raw")
            notes.append(f"no encoder available ({exc.args[0].splitlines()[0]}); wrote raw frames")
            emit(f"make_test_video: {notes[-1]}")
            media = write_raw(fallback, images)
            output = fallback
    elif args.format == "raw":
        media = write_raw(output.with_suffix(".raw"), images)
        output = output.with_suffix(".raw")
    else:
        media = write_png_sequence(output, images)

    media.update(
        {
            "width": scene.width,
            "height": scene.height,
            "fps": output_fps,
            "scene_fps": args.fps,
            "frames": len(images),
        }
    )

    status = 0
    if media["kind"] == "video":
        ok, message = verify_video(output, len(images))
        notes.append(message)
        if not ok:
            print(f"make_test_video: {message}", file=sys.stderr)
            status = 1
        else:
            emit(f"verified: {message}")

    truth_path = args.truth or Path(str(output) + ".truth.json")
    write_sidecar(
        truth_path,
        media,
        truth,
        scene,
        source_uri,
        fps=args.fps,
        output_fps=output_fps,
        target_fps=args.target_fps,
        extra={"notes": notes} if notes else None,
    )
    if args.wire_jsonl:
        write_wire_jsonl(args.wire_jsonl, truth, source_uri)

    with_target = sum(1 for entry in truth if entry["boxes"])
    rate = f"{output_fps:g} fps"
    if abs(output_fps - args.fps) > 1e-9:
        rate += f" (decimated from {args.fps:g})"
    emit(f"wrote {output}  ({len(images)} frames, {scene.width}x{scene.height}, {rate})")
    emit(f"wrote {truth_path}  ({with_target} frames carry a target box)")
    if args.wire_jsonl:
        emit(f"wrote {args.wire_jsonl}  (wire-format detections, conf 1.0, model {TRUTH_MODEL.name})")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
