#!/usr/bin/env python3
"""Export trained weights for deployment, and refuse to ship a class swap.

The export itself is a thin wrapper over ultralytics. The reason this file
exists is the assertion around it.

``station.core.types.CLASSES`` is ``("fire", "smoke")`` and the order is
load-bearing: it is the class-index order the station assumes when it turns a
model's integer class outputs into wire labels. If an exported model emits
smoke as index 0, everything downstream keeps working perfectly. Boxes appear
in the right places, confidences look normal, the incident log fills up, the
tablets draw. Every label is simply wrong -- a flame front reported as a smoke
plume and a plume reported as flame -- and nothing in the system can detect it,
because there is no second opinion anywhere in the pipeline to disagree with.
An operator would have to notice by looking at the video, which is precisely
the moment the overlay was meant to help with.

So: the class order is checked against the wire contract before the export
runs, and the exported artifact's own embedded metadata is checked again
afterwards. A verification failure moves the artifact aside rather than
leaving a plausible-looking file where a deploy script might find it.

Every check on the exported artifact is dependency-free where it can be:
a ``.torchscript`` file is a zip, a TensorRT ``.engine`` from ultralytics
carries a length-prefixed JSON header, and OpenVINO writes ``metadata.yaml``.
Only ONNX prefers a real parser, and even that degrades to a byte scan. The
point is that verification must be possible on the machine doing the deploying,
which is rarely the machine with the full training stack.

``ultralytics`` and ``torch`` are imported lazily, inside the function that
needs them, so ``--verify`` and ``--help`` run on a laptop with neither.

Example:
    $ python3 tools/export.py --weights runs/train/best.pt --format onnx \\
          --imgsz 640 --out models/yolo11s-fire.onnx
    $ python3 tools/export.py --verify models/yolo11s-fire.onnx
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import struct
import sys
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from station.core.types import CLASSES  # noqa: E402 -- sys.path set up above.

__all__ = [
    "SUPPORTED_FORMATS",
    "ClassOrderCheck",
    "check_class_order",
    "read_exported_classes",
    "export_weights",
    "main",
]

#: Formats this tool will produce. Deliberately short: each one needs its
#: metadata read back, and a format whose class order cannot be verified after
#: export is a format this tool should not be handing to a deployment.
SUPPORTED_FORMATS: dict[str, str] = {
    "onnx": "portable graph; the default for a station without TensorRT",
    "torchscript": "torch-only runtime, no ONNX toolchain needed",
    "engine": "TensorRT engine; fastest, and tied to the exact GPU and driver it was built on",
    "openvino": "Intel CPU/iGPU runtime, for a station with no discrete GPU",
}

#: Used only if ``station.inference.runner`` cannot be imported. Kept in sync
#: by hand; the runner's table is the authority, because the runner is what
#: actually maps a model's class names at inference time, and a check that
#: disagrees with it would pass models the station then mislabels.
_FALLBACK_ALIASES: dict[str, str] = {
    "fire": "fire",
    "flame": "fire",
    "flames": "fire",
    "wildfire": "fire",
    "smoke": "smoke",
    "smoky": "smoke",
    "smog": "smoke",
}

_STATUS_OK = "ok"
_STATUS_ALIASED = "aliased"
_STATUS_PERMUTED = "permuted"
_STATUS_MISMATCH = "mismatch"
_STATUS_UNKNOWN = "unknown"


def _alias_map() -> dict[str, str]:
    """The station's own class-alias table, or a hand-kept copy of it."""
    try:
        from station.inference import runner  # noqa: PLC0415 -- imports no heavy deps.

        table = getattr(runner, "_CLASS_ALIASES", None)
        if isinstance(table, dict) and table:
            return {str(k).lower(): str(v).lower() for k, v in table.items()}
    except Exception:  # pragma: no cover - only when run outside the repo
        pass
    return dict(_FALLBACK_ALIASES)


@dataclass(slots=True)
class ClassOrderCheck:
    """The verdict on one model's class order."""

    status: str
    found: tuple[str, ...]
    expected: tuple[str, ...]
    source: str
    messages: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when the model may be deployed against this wire contract."""
        return self.status in (_STATUS_OK, _STATUS_ALIASED)

    @property
    def fatal(self) -> bool:
        """True when the model's outputs would be mislabelled by the station."""
        return self.status in (_STATUS_PERMUTED, _STATUS_MISMATCH)

    def to_json(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "found": list(self.found),
            "expected": list(self.expected),
            "source": self.source,
            "messages": list(self.messages),
        }

    def render(self) -> str:
        head = f"class order [{self.status.upper()}] from {self.source}"
        body = [f"  found    {list(self.found)}", f"  expected {list(self.expected)}"]
        body.extend(f"  {m}" for m in self.messages)
        return "\n".join([head, *body])


def normalise_names(names: Any) -> tuple[str, ...]:
    """Turn an ultralytics ``names`` value into an index-ordered tuple.

    Args:
        names: A list, tuple, or ``{index: name}`` mapping. Ultralytics uses
            all three depending on the version and the export format.

    Returns:
        Lowercase, stripped class names in class-index order.

    Raises:
        ValueError: The mapping has non-integer keys or gaps, so no unambiguous
            index order exists. Guessing one is exactly the mistake this whole
            module exists to prevent.
    """
    if names is None:
        return ()
    if isinstance(names, dict):
        try:
            indices = {int(k): str(v) for k, v in names.items()}
        except (TypeError, ValueError) as exc:
            raise ValueError(f"class names mapping has non-integer keys: {names!r}") from exc
        expected_keys = set(range(len(indices)))
        if set(indices) != expected_keys:
            raise ValueError(
                f"class indices are not contiguous from 0: got {sorted(indices)}"
            )
        return tuple(indices[i].strip().lower() for i in range(len(indices)))
    if isinstance(names, (list, tuple)):
        return tuple(str(v).strip().lower() for v in names)
    raise ValueError(f"unusable class names of type {type(names).__name__}: {names!r}")


def check_class_order(
    names: Any, source: str, expected: Sequence[str] = CLASSES
) -> ClassOrderCheck:
    """Compare a model's class order with the wire contract.

    Args:
        names: The model's class names, in any of the forms ultralytics uses.
        source: Where the names came from, for the message ("weights",
            "exported onnx", ...).
        expected: The wire class order; defaults to
            :data:`station.core.types.CLASSES`.

    Returns:
        A :class:`ClassOrderCheck`. ``status`` is one of ``ok`` (identical),
        ``aliased`` (differs only by a name the station's runner already maps),
        ``permuted`` (same classes, wrong order -- the catastrophic case),
        ``mismatch`` (different classes entirely), or ``unknown`` (no names
        could be read).
    """
    expected_tuple = tuple(str(c).strip().lower() for c in expected)
    try:
        found = normalise_names(names)
    except ValueError as exc:
        return ClassOrderCheck(_STATUS_UNKNOWN, (), expected_tuple, source, [str(exc)])

    if not found:
        return ClassOrderCheck(
            _STATUS_UNKNOWN,
            (),
            expected_tuple,
            source,
            ["no class names could be read, so the class order cannot be verified"],
        )

    if found == expected_tuple:
        return ClassOrderCheck(_STATUS_OK, found, expected_tuple, source)

    aliases = _alias_map()
    mapped = tuple(aliases.get(name, name) for name in found)
    if mapped == expected_tuple:
        return ClassOrderCheck(
            _STATUS_ALIASED,
            found,
            expected_tuple,
            source,
            [
                f"names differ from the wire contract but map onto it: {list(found)} -> "
                f"{list(mapped)}",
                "station.inference.runner applies the same mapping at inference time, so this",
                "is safe. Renaming the classes in the training data would remove the",
                "indirection and one more place for the two tables to drift apart.",
            ],
        )

    if sorted(mapped) == sorted(expected_tuple):
        return ClassOrderCheck(
            _STATUS_PERMUTED,
            found,
            expected_tuple,
            source,
            [
                "The classes are right and the ORDER IS WRONG. Exporting this would produce a",
                "model that reports fire as smoke and smoke as fire, on every frame, with",
                "correct boxes and plausible confidences. Nothing downstream can detect it.",
                "Fix the class order in the training dataset's data.yaml (and renumber the",
                "label files to match), retrain or re-index the head, and export again.",
                "Do not 'fix' this by swapping station.core.types.CLASSES: that file is the",
                "contract the tablet app and the incident log are written against.",
            ],
        )

    extra = [n for n in mapped if n not in expected_tuple]
    absent = [n for n in expected_tuple if n not in mapped]
    return ClassOrderCheck(
        _STATUS_MISMATCH,
        found,
        expected_tuple,
        source,
        [
            f"classes present in the model but not on the wire: {extra}" if extra else "",
            f"classes the wire expects but the model lacks: {absent}" if absent else "",
            "The station would receive class labels it has no rendering rule for, and the",
            "missing classes would simply never appear on any tablet.",
        ],
    )


# ------------------------------------------------------- reading back metadata


def read_exported_classes(path: Path) -> tuple[Any, str]:
    """Read the class names embedded in an exported artifact.

    Ultralytics stamps its metadata (class names, imgsz, task, stride) into
    every export format. Reading it back is the only way to know that the file
    on disk agrees with the weights it came from -- an export can silently
    reorder outputs, and a hand-edited or hand-copied artifact can be anything
    at all.

    Args:
        path: The exported file, or the directory for formats that export to
            one (OpenVINO).

    Returns:
        ``(names, how)`` where ``names`` is whatever was found (list, dict or
        ``None``) and ``how`` describes the method used, for the report.
    """
    if path.is_dir():
        return _read_openvino_metadata(path)
    suffix = path.suffix.lower()
    if suffix == ".onnx":
        return _read_onnx_metadata(path)
    if suffix in (".torchscript", ".pt") and zipfile.is_zipfile(path):
        return _read_torchscript_metadata(path)
    if suffix in (".engine", ".plan"):
        return _read_engine_metadata(path)
    return None, f"no metadata reader for {suffix or 'this path'}"


def _read_onnx_metadata(path: Path) -> tuple[Any, str]:
    """ONNX metadata_props, via onnx, then onnxruntime, then a byte scan."""
    try:
        import onnx  # noqa: PLC0415 -- optional, lazy by design.

        model = onnx.load(str(path), load_external_data=False)
        for prop in model.metadata_props:
            if prop.key == "names":
                return _parse_names_literal(prop.value), "onnx metadata_props"
        return None, "onnx metadata_props (no 'names' entry)"
    except ImportError:
        pass
    except Exception as exc:
        return None, f"onnx failed to parse the file: {exc}"

    try:
        import onnxruntime  # noqa: PLC0415 -- optional, lazy by design.

        session = onnxruntime.InferenceSession(
            str(path), providers=["CPUExecutionProvider"]
        )
        meta = session.get_modelmeta().custom_metadata_map or {}
        if "names" in meta:
            return _parse_names_literal(meta["names"]), "onnxruntime metadata"
        return None, "onnxruntime metadata (no 'names' entry)"
    except ImportError:
        pass
    except Exception as exc:
        return None, f"onnxruntime failed to load the file: {exc}"

    # Last resort: metadata_props are stored as plain UTF-8 inside the
    # protobuf, so the dict literal can be recovered without a parser. Marked
    # as best-effort in the report because a byte scan can find the right text
    # in the wrong field.
    try:
        blob = path.read_bytes()
    except OSError as exc:
        return None, f"unreadable: {exc}"
    match = re.search(rb"names.{0,16}?(\{\s*0\s*:\s*['\"].{0,4096}?\})", blob, re.S)
    if match:
        try:
            return _parse_names_literal(match.group(1).decode("utf-8", "replace")), (
                "raw byte scan (best effort; install onnx for a real parse)"
            )
        except ValueError:
            pass
    return None, "no ONNX parser available and the byte scan found nothing"


def _parse_names_literal(value: Any) -> Any:
    """Ultralytics stores names as ``str(dict)``; recover the dict."""
    if not isinstance(value, str):
        return value
    text = value.strip()
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"could not parse embedded class names {text[:80]!r}: {exc}") from exc


def _read_torchscript_metadata(path: Path) -> tuple[Any, str]:
    """TorchScript archives are zips; ultralytics adds ``extra/config.txt``."""
    try:
        with zipfile.ZipFile(path) as archive:
            candidates = [n for n in archive.namelist() if n.endswith("config.txt")]
            for name in candidates:
                text = archive.read(name).decode("utf-8", "replace").strip()
                if not text:
                    continue
                try:
                    meta = json.loads(text)
                except json.JSONDecodeError:
                    meta = ast.literal_eval(text)
                if isinstance(meta, dict) and "names" in meta:
                    return _parse_names_literal(meta["names"]), f"torchscript {name}"
    except (OSError, zipfile.BadZipFile, ValueError, SyntaxError) as exc:
        return None, f"could not read the torchscript archive: {exc}"
    return None, "torchscript archive has no config.txt with class names"


def _read_engine_metadata(path: Path) -> tuple[Any, str]:
    """A ultralytics TensorRT engine begins with a length-prefixed JSON blob."""
    try:
        with path.open("rb") as fh:
            raw = fh.read(4)
            if len(raw) < 4:
                return None, "engine file is truncated"
            length = struct.unpack("<I", raw)[0]
            # Sanity bound: a real metadata header is a few hundred bytes. A
            # bare engine (exported by trtexec, say) starts with its own magic
            # and this length would be nonsense.
            if not 0 < length < 1 << 20:
                return None, "engine has no ultralytics metadata header"
            meta = json.loads(fh.read(length).decode("utf-8", "replace"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        return None, f"could not read the engine header: {exc}"
    if isinstance(meta, dict) and "names" in meta:
        return _parse_names_literal(meta["names"]), "tensorrt engine header"
    return None, "engine header has no class names"


def _read_openvino_metadata(path: Path) -> tuple[Any, str]:
    """OpenVINO exports are a directory containing ``metadata.yaml``."""
    meta_file = path / "metadata.yaml"
    if not meta_file.is_file():
        return None, f"no metadata.yaml in {path}"
    try:
        import yaml  # noqa: PLC0415 -- PyYAML, only needed for this format.

        meta = yaml.safe_load(meta_file.read_text(encoding="utf-8")) or {}
    except ImportError:
        return None, "PyYAML is not installed, so metadata.yaml could not be read"
    except Exception as exc:
        return None, f"could not read metadata.yaml: {exc}"
    if isinstance(meta, dict) and "names" in meta:
        return meta["names"], "openvino metadata.yaml"
    return None, "metadata.yaml has no class names"


# ------------------------------------------------------------------- exporting


def load_source_names(weights: Path) -> tuple[Any, str]:
    """Read the class names from a ``.pt`` checkpoint.

    Prefers ultralytics, which is authoritative. Falls back to the checkpoint's
    own zip container, so the pre-export check still runs on a machine that has
    torch but not ultralytics, or neither.

    Args:
        weights: The ``.pt`` file.

    Returns:
        ``(names, how)``; ``names`` is ``None`` when nothing could be read.
    """
    try:
        from ultralytics import YOLO  # noqa: PLC0415 -- lazy by design.

        return YOLO(str(weights)).names, "ultralytics checkpoint"
    except ImportError:
        pass
    except Exception as exc:
        return None, f"ultralytics could not open the checkpoint: {exc}"
    if zipfile.is_zipfile(weights):
        names, how = _read_torchscript_metadata(weights)
        if names is not None:
            return names, how
    return None, "ultralytics is not installed and the checkpoint carries no readable metadata"


def export_weights(
    weights: Path,
    fmt: str,
    *,
    imgsz: int = 640,
    half: bool = False,
    int8: bool = False,
    dynamic: bool = False,
    simplify: bool = True,
    opset: int | None = None,
    batch: int = 1,
    device: str = "auto",
    workspace: float | None = None,
    nms: bool = False,
) -> Path:
    """Run the ultralytics export.

    Args:
        weights: Source ``.pt`` file.
        fmt: One of :data:`SUPPORTED_FORMATS`.
        imgsz: Export input size. Must match the size the station runs at
            (``inference.imgsz``): a model exported at 1280 and fed 640 will
            silently lose the small, distant targets that matter most.
        half: FP16 weights. Meaningful for TensorRT and CUDA ONNX only.
        int8: INT8 quantisation. Needs a calibration set; accuracy loss falls
            hardest on small, low-contrast targets, so measure recall by box
            size with ``tools/evaluate.py`` before and after.
        dynamic: Dynamic input shapes.
        simplify: Run the ONNX simplifier.
        opset: ONNX opset; ``None`` lets ultralytics choose.
        batch: Export batch size. The station infers one frame at a time.
        device: ``auto``, or an explicit torch device.
        workspace: TensorRT workspace size in GiB.
        nms: Bake NMS into the exported graph.

    Returns:
        The path ultralytics wrote.

    Raises:
        RuntimeError: ultralytics is missing, or the export failed. Never
            swallowed -- a half-finished export left on disk is a deployment
            hazard, so the caller is told and the file is not blessed.
    """
    try:
        from ultralytics import YOLO  # noqa: PLC0415 -- lazy by design.
    except ImportError as exc:
        raise RuntimeError(
            "ultralytics is not installed, so nothing can be exported.\n"
            "  pip install ultralytics\n"
            "(--verify on an already-exported artifact does not need it.)"
        ) from exc

    from station.inference.runner import resolve_device  # noqa: PLC0415

    kwargs: dict[str, Any] = {
        "format": fmt,
        "imgsz": imgsz,
        "half": half,
        "int8": int8,
        "dynamic": dynamic,
        "batch": batch,
        "device": resolve_device(device),
        "nms": nms,
    }
    if fmt == "onnx":
        kwargs["simplify"] = simplify
        if opset is not None:
            kwargs["opset"] = opset
    if fmt == "engine" and workspace is not None:
        kwargs["workspace"] = workspace

    try:
        produced = YOLO(str(weights)).export(**kwargs)
    except Exception as exc:
        raise RuntimeError(f"ultralytics export failed: {exc}") from exc
    if not produced:
        raise RuntimeError("ultralytics export returned no path")
    return Path(str(produced))


def write_sidecar(
    artifact: Path,
    weights: Path,
    fmt: str,
    imgsz: int,
    checks: Sequence[ClassOrderCheck],
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write the provenance sidecar next to the exported artifact.

    ``ModelInfo`` in the wire protocol carries ``weights_sha`` so that an
    incident log can be replayed against the model that produced it. That only
    works if the deployed artifact can be traced back to a checkpoint, and an
    exported ``.onnx`` has no room for the story. Hence a JSON file beside it.

    Args:
        artifact: The exported file or directory.
        weights: The source checkpoint.
        fmt: Export format.
        imgsz: Export input size.
        checks: The class-order checks that were run.
        extra: Additional fields to record.

    Returns:
        Path to the sidecar written.
    """
    try:
        from station.inference.runner import weights_sha  # noqa: PLC0415

        sha = weights_sha(weights)
    except Exception:
        sha = None

    payload: dict[str, Any] = {
        "tool": "tools/export.py",
        "version": 1,
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "artifact": artifact.name,
        "format": fmt,
        "imgsz": imgsz,
        "source_weights": str(weights),
        "source_weights_sha": sha,
        "wire_classes": list(CLASSES),
        "class_order_checks": [c.to_json() for c in checks],
    }
    try:
        import ultralytics  # noqa: PLC0415

        payload["ultralytics_version"] = getattr(ultralytics, "__version__", "unknown")
    except Exception:
        payload["ultralytics_version"] = None
    if extra:
        payload.update(extra)

    sidecar = artifact.with_name(artifact.name + ".wildfire.json")
    sidecar.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return sidecar


def quarantine(artifact: Path) -> Path:
    """Move a failed artifact aside so no deploy script can pick it up.

    Renaming rather than deleting: the file is evidence about what went wrong,
    and someone will want to look at it. What it must not be is a plausible
    model file sitting at the path a deployment expects.
    """
    target = artifact.with_name(artifact.name + ".REJECTED")
    counter = 1
    while target.exists():
        target = artifact.with_name(f"{artifact.name}.REJECTED.{counter}")
        counter += 1
    artifact.rename(target)
    return target


# ---------------------------------------------------------------------- cli


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    formats = "\n".join(f"  {name:<12} {desc}" for name, desc in SUPPORTED_FORMATS.items())
    parser = argparse.ArgumentParser(
        prog="export.py",
        description=(
            "Export trained wildfire-watch weights, asserting that the exported model's "
            "class order matches station.core.types.CLASSES before and after the export."
        ),
        epilog=(
            f"formats:\n{formats}\n\n"
            "A class-order swap between fire and smoke produces a model that works perfectly "
            "and labels everything backwards, and no other check in this system can see it. "
            "That is why this tool refuses rather than warns, and why a failed verification "
            "moves the artifact to <name>.REJECTED."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--weights", metavar="PT", type=Path, help="Trained .pt checkpoint to export.")
    parser.add_argument(
        "--verify",
        metavar="ARTIFACT",
        type=Path,
        help="Check an already-exported artifact's class order and exit. Needs no ultralytics.",
    )
    parser.add_argument(
        "--format",
        default="onnx",
        choices=sorted(SUPPORTED_FORMATS),
        help="Export format (default: onnx).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Export input size; must match inference.imgsz on the station (default: 640).",
    )
    parser.add_argument(
        "--out",
        metavar="PATH",
        type=Path,
        help="Move the exported artifact here (default: leave it beside the weights).",
    )
    parser.add_argument("--device", default="auto", help="Torch device for the export (default: auto).")
    parser.add_argument("--half", action="store_true", help="Export FP16 weights (TensorRT/CUDA ONNX).")
    parser.add_argument(
        "--int8",
        action="store_true",
        help="INT8 quantisation. Re-measure recall by box size afterwards; small targets suffer most.",
    )
    parser.add_argument("--dynamic", action="store_true", help="Dynamic input shapes.")
    parser.add_argument("--no-simplify", action="store_true", help="Skip the ONNX simplifier.")
    parser.add_argument("--opset", type=int, default=None, help="ONNX opset (default: ultralytics' choice).")
    parser.add_argument("--batch", type=int, default=1, help="Export batch size (default: 1).")
    parser.add_argument("--workspace", type=float, default=None, help="TensorRT workspace, GiB.")
    parser.add_argument("--nms", action="store_true", help="Bake NMS into the exported graph.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the pre-export class-order check and stop without exporting.",
    )
    parser.add_argument(
        "--allow-unverified",
        action="store_true",
        help="Proceed when the exported artifact's class order cannot be read back "
        "(no parser available). The pre-export check still has to pass.",
    )
    parser.add_argument("--json", metavar="FILE", type=Path, help="Write the result as JSON.")
    parser.add_argument("--quiet", action="store_true", help="Print only failures.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point.

    Returns:
        0 on a verified export, 1 on any class-order failure, 2 on bad input or
        a missing dependency.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.weights is None and args.verify is None:
        parser.error("one of --weights or --verify is required")

    checks: list[ClassOrderCheck] = []
    result: dict[str, Any] = {"tool": "export", "version": 1, "wire_classes": list(CLASSES)}

    def emit(text: str) -> None:
        if not args.quiet:
            print(text)

    # -- verify-only mode -------------------------------------------------
    if args.verify is not None:
        if not args.verify.exists():
            print(f"export: artifact not found: {args.verify}", file=sys.stderr)
            return 2
        names, how = read_exported_classes(args.verify)
        check = check_class_order(names, f"exported artifact ({how})")
        checks.append(check)
        emit(check.render())
        result["mode"] = "verify"
        result["artifact"] = str(args.verify)
        result["checks"] = [c.to_json() for c in checks]
        if args.json:
            args.json.write_text(json.dumps(result, indent=2), encoding="utf-8")
        if check.fatal:
            print("\nREFUSED: this artifact would mislabel every detection.", file=sys.stderr)
            return 1
        if check.status == _STATUS_UNKNOWN and not args.allow_unverified:
            print(
                "\nUNVERIFIED: the artifact's class order could not be read, so it has not "
                "been checked.\n  pip install onnx   (or pass --allow-unverified to accept "
                "the risk deliberately)",
                file=sys.stderr,
            )
            return 1
        if check.status == _STATUS_UNKNOWN:
            # --allow-unverified was passed. Say what actually happened rather
            # than "OK": an unchecked artifact is not a verified one, and this
            # line is what someone will paste into a deployment ticket.
            emit(
                "\nUNVERIFIED, accepted by --allow-unverified: no class order was read from "
                "this artifact. Nothing here says it is correct."
            )
        else:
            emit("\nOK: class order matches the wire contract.")
        return 0

    # -- export mode ------------------------------------------------------
    weights: Path = args.weights
    if not weights.is_file():
        print(f"export: weights file not found: {weights}", file=sys.stderr)
        return 2

    names, how = load_source_names(weights)
    pre = check_class_order(names, f"source weights ({how})")
    checks.append(pre)
    emit(pre.render())
    if pre.fatal:
        print(
            "\nREFUSED: not exporting. The checkpoint's class order does not match "
            "station.core.types.CLASSES, and exporting it would produce a model whose "
            "labels are wrong in a way nothing downstream can detect.",
            file=sys.stderr,
        )
        return 1
    if pre.status == _STATUS_UNKNOWN and not args.allow_unverified:
        print(
            "\nREFUSED: the checkpoint's class names could not be read, so the class order "
            "is unverified.\n  pip install ultralytics   (or pass --allow-unverified)",
            file=sys.stderr,
        )
        return 1

    if args.dry_run:
        emit("\n--dry-run: pre-export check passed; nothing was exported.")
        result["mode"] = "dry-run"
        result["checks"] = [c.to_json() for c in checks]
        if args.json:
            args.json.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return 0

    try:
        artifact = export_weights(
            weights,
            args.format,
            imgsz=args.imgsz,
            half=args.half,
            int8=args.int8,
            dynamic=args.dynamic,
            simplify=not args.no_simplify,
            opset=args.opset,
            batch=args.batch,
            device=args.device,
            workspace=args.workspace,
            nms=args.nms,
        )
    except RuntimeError as exc:
        print(f"export: {exc}", file=sys.stderr)
        return 2

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        artifact = Path(str(artifact)).replace(args.out)
    emit(f"\nexported {artifact}")

    exported_names, exported_how = read_exported_classes(artifact)
    post = check_class_order(exported_names, f"exported artifact ({exported_how})")
    checks.append(post)
    emit(post.render())

    result["mode"] = "export"
    result["artifact"] = str(artifact)
    result["format"] = args.format
    result["imgsz"] = args.imgsz
    result["checks"] = [c.to_json() for c in checks]

    if post.fatal:
        moved = quarantine(artifact)
        result["quarantined_to"] = str(moved)
        if args.json:
            args.json.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(
            f"\nREFUSED: the exported artifact's class order is wrong. Moved to {moved} so it "
            "cannot be deployed by accident.",
            file=sys.stderr,
        )
        return 1
    if post.status == _STATUS_UNKNOWN and not args.allow_unverified:
        moved = quarantine(artifact)
        result["quarantined_to"] = str(moved)
        if args.json:
            args.json.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(
            f"\nUNVERIFIED: could not read the exported artifact's class order ({exported_how}). "
            f"Moved to {moved}.\n  Install a reader (pip install onnx) and re-run, or pass "
            "--allow-unverified to accept an unchecked artifact deliberately.",
            file=sys.stderr,
        )
        return 1

    sidecar = write_sidecar(artifact, weights, args.format, args.imgsz, checks)
    result["sidecar"] = str(sidecar)
    if args.json:
        args.json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    emit(f"provenance written to {sidecar}")
    emit(
        "\nOK: class order verified before and after export. Set inference.weights to this "
        f"artifact and keep inference.imgsz at {args.imgsz}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
