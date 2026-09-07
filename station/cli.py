"""``python -m station`` -- the operator's interface to the ground station.

Four subcommands, and the split is about when they get used:

``run``     the deployment. Opens the source, loads the model, starts the
            pipeline, issues a certificate if needed, and serves the PWA to the
            tablets over HTTPS.
``certs``   issue or inspect the station's self-signed certificate. Separate
            because operators need to do it once, in advance, on a bench, and
            then install the certificate on the tablets.
``check``   validate the config and report which optional dependencies this
            laptop actually has. Meant to be run *before* an incident: the
            failure mode this exists to prevent is discovering at 3am that the
            laptop that was reimaged last month no longer has aiortc.
``replay``  re-serve a recorded incident through the real pipeline and the real
            PWA, for training and after-action review.

Everything printed here obeys the safety invariants in
``station/core/safety.py``. The banner and the status lines report **pipeline
liveness** -- is the model running, is it keeping up, where is the record being
written. Nothing printed by this program describes the scene, and no output
here should ever be readable as "there is nothing there". The absence of boxes
is the absence of evidence, and this is a situational-awareness aid rather than
a certified detection device.

Heavy dependencies are probed, not imported, so ``check`` runs on a laptop with
nothing installed and tells the operator what is missing instead of crashing.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import logging
import os
import signal
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from station.core.config import Config, load_config, validate
from station.core.safety import PRODUCT_DESCRIPTOR

__all__ = ["main", "build_parser"]

log = logging.getLogger("station")

PROGRAM = "python -m station"

#: The single sentence that has to survive every UI rewrite. Printed on every
#: start, because the person reading it is about to point a crew somewhere.
SAFETY_LINE = (
    "Boxes appear when the model produces them. Their absence is the absence\n"
    "of evidence, not evidence of absence. This is a "
    f"{PRODUCT_DESCRIPTOR}\n"
    "over video an operator is already watching, not a certified device."
)


@dataclass(frozen=True, slots=True)
class Dependency:
    """One optional dependency, and what the station loses without it."""

    module: str
    #: Distribution name for ``pip install`` and for the version lookup, when
    #: it differs from the import name (``cv2`` -> ``opencv-python``).
    distribution: str
    purpose: str

    @property
    def install_hint(self) -> str:
        return f"pip install {self.distribution}"


#: Everything the station can use, in the order an operator should care about.
OPTIONAL_DEPENDENCIES: tuple[Dependency, ...] = (
    Dependency("numpy", "numpy", "frame buffers; every decoder needs it"),
    Dependency("yaml", "PyYAML", "reading config.yaml"),
    Dependency("cryptography", "cryptography", "issuing the station's TLS certificate"),
    Dependency("aiohttp", "aiohttp", "serving the PWA and the signalling endpoint"),
    Dependency("aiortc", "aiortc", "WebRTC video track and the detections data channel"),
    Dependency("av", "av", "RTSP/RTMP/file decoding and incident video recording"),
    Dependency("cv2", "opencv-python", "capture-card ingest and fallback decode/record"),
    Dependency("torch", "torch", "model runtime (install the CUDA build first)"),
    Dependency("ultralytics", "ultralytics", "YOLO inference"),
)


# --------------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser.

    Returns:
        The configured :class:`argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(
        prog=PROGRAM,
        description=(
            "wildfire-watch ground station: one inference pass on this laptop, "
            f"drawn as an overlay on every tablet. A {PRODUCT_DESCRIPTOR}."
        ),
        epilog=(
            "Detections are computed here and sent to the tablets as JSON; they are never "
            "burned into the video. Nothing this program reports is a statement about the "
            "scene -- see docs/SAFETY.md."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-c", "--config", metavar="FILE",
        help="YAML config file. Without one, built-in defaults are used.",
    )
    parser.add_argument(
        "--set", metavar="KEY=VALUE", action="append", default=[], dest="overrides",
        help="Override one config value, e.g. --set stream.port=9443. Repeatable.",
    )
    parser.add_argument(
        "-v", "--verbose", action="count", default=0,
        help="More logging. Repeat for debug-level output.",
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true",
        help="Warnings and errors only.",
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    # ---- run ----------------------------------------------------------
    run = sub.add_parser(
        "run",
        help="run the station: ingest, inference, WebRTC and the PWA",
        description=(
            "Open the video source, run the model, and serve the overlay to the tablets "
            "over HTTPS on the LAN. Runs until interrupted, or until a file source ends."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    run.add_argument("--source", metavar="URI", help="Override source.uri.")
    run.add_argument(
        "--source-type", choices=("file", "rtsp", "rtmp", "hdmi", "synthetic"),
        help="Override source.type.",
    )
    run.add_argument("--weights", metavar="PATH", help="Override inference.weights.")
    run.add_argument("--device", metavar="DEV", help="Override inference.device (auto, cuda, cpu).")
    run.add_argument("--bind", metavar="ADDR", help="Override stream.bind.")
    run.add_argument("--port", type=int, metavar="N", help="Override stream.port.")
    run.add_argument("--app-dir", metavar="DIR", help="Directory holding the PWA (default: ./app).")
    run.add_argument(
        "--stub", action="store_true",
        help=(
            "Run the stub runner instead of the model: NO MODEL RUNS and the boxes are "
            "generated. For bench-testing the transport and the overlay only."
        ),
    )
    run.add_argument(
        "--no-stream", action="store_true",
        help="Do not serve WebRTC. The pipeline still runs and still writes the incident log.",
    )
    run.add_argument(
        "--no-log", action="store_true",
        help="Disable the incident log. Loses the record this system exists to accumulate.",
    )
    run.add_argument(
        "--insecure", action="store_true",
        help=(
            "Serve plain HTTP. Tablets then get no WebRTC and no service worker; "
            "development only."
        ),
    )
    run.add_argument(
        "--no-certs", action="store_true",
        help="Do not issue a certificate; use stream.cert / stream.key exactly as configured.",
    )
    run.add_argument(
        "--allow-origin", metavar="ORIGIN",
        help="CORS origin for the signalling routes. Only for a separate PWA dev server.",
    )

    # ---- certs --------------------------------------------------------
    certs = sub.add_parser(
        "certs",
        help="issue or inspect the station's self-signed TLS certificate",
        description=(
            "Browsers gate WebRTC and service workers behind a secure context, so the "
            "station needs a certificate even on a LAN with no CA and no DNS. The "
            "certificate covers this machine's IP addresses -- iOS rejects one that does "
            "not -- and is reused until it expires or the station changes network."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    certs.add_argument("--cert", metavar="PATH", help="Override stream.cert.")
    certs.add_argument("--key", metavar="PATH", help="Override stream.key.")
    certs.add_argument(
        "--host", metavar="NAME", action="append", default=[],
        help="Extra DNS name for the SAN. Repeatable.",
    )
    certs.add_argument(
        "--ip", metavar="ADDR", action="append", default=[],
        help="Extra IP address for the SAN. Repeatable.",
    )
    certs.add_argument("--days", type=int, metavar="N", help="Validity in days (default 365).")
    certs.add_argument(
        "--force", action="store_true",
        help="Reissue even if the current certificate is fine. Every tablet must then re-trust it.",
    )
    certs.add_argument("--show", action="store_true", help="Only describe the existing certificate.")

    # ---- check --------------------------------------------------------
    check = sub.add_parser(
        "check",
        help="validate the config and report what this laptop can actually do",
        description=(
            "Run this on the bench, not during an incident. Validates the configuration, "
            "reports which optional dependencies are importable, and says which parts of "
            "the station can run here.\n\n"
            "Exit status: 0 ready to run, 1 the configuration is invalid, "
            "2 the configuration is fine but something needed to run is missing."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    check.add_argument("--app-dir", metavar="DIR", help="Directory holding the PWA.")

    # ---- replay -------------------------------------------------------
    replay = sub.add_parser(
        "replay",
        help="re-serve a recorded incident through the real pipeline and PWA",
        description=(
            "Replays the video and the recorded detections of a past incident through the "
            "same pipeline, transport and app used live, for training and after-action "
            "review. The recorded payloads are republished as they were logged; nothing is "
            "re-inferred and no new incident log is written."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    replay.add_argument("incident_dir", metavar="INCIDENT_DIR", help="An incidents/<id>/ directory.")
    replay.add_argument(
        "--segment", type=int, default=0, metavar="N",
        help="Which recorded video segment to replay (default: the first).",
    )
    replay.add_argument("--loop", action="store_true", help="Replay the segment forever.")
    replay.add_argument("--bind", metavar="ADDR", help="Override stream.bind.")
    replay.add_argument("--port", type=int, metavar="N", help="Override stream.port.")
    replay.add_argument("--app-dir", metavar="DIR", help="Directory holding the PWA.")
    replay.add_argument("--insecure", action="store_true", help="Serve plain HTTP (development).")
    replay.add_argument("--no-certs", action="store_true", help="Do not issue a certificate.")
    return parser


# ----------------------------------------------------------------- entry point


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line.

    Args:
        argv: Arguments without the program name. Defaults to ``sys.argv[1:]``.

    Returns:
        A process exit status: 0 success, 1 a configuration or usage problem,
        2 a dependency or environment problem, 130 interrupted.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args)

    if not args.command:
        parser.print_help()
        return 1

    try:
        if args.command == "check":
            return _cmd_check(args)
        if args.command == "certs":
            return _cmd_certs(args)
        if args.command == "run":
            return asyncio.run(_cmd_run(args))
        if args.command == "replay":
            return asyncio.run(_cmd_replay(args))
    except KeyboardInterrupt:  # pragma: no cover - interactive
        print("\ninterrupted", file=sys.stderr)
        return 130
    except _CliError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return exc.status
    parser.print_help()
    return 1


class _CliError(RuntimeError):
    """An error already phrased for an operator, with the exit status to use."""

    def __init__(self, message: str, status: int = 1) -> None:
        super().__init__(message)
        self.status = status


# ------------------------------------------------------------------ run


async def _cmd_run(args: argparse.Namespace) -> int:
    """Run the station until interrupted."""
    cfg = _load(args, _run_overrides(args))

    runner = _build_run_runner(cfg, stub=args.stub)
    streamer = None
    if not args.no_stream:
        streamer = _build_streamer(cfg, offer_no_stream=True)

    if not args.insecure and not args.no_certs:
        _issue_certificate(cfg)

    from station.pipeline import Pipeline  # noqa: PLC0415
    from station.serve.http import StationServer  # noqa: PLC0415

    pipeline = Pipeline(cfg, runner=runner, streamer=streamer)
    server = StationServer(
        cfg,
        streamer=streamer,
        pipeline=pipeline,
        app_dir=args.app_dir,
        allow_origin=args.allow_origin,
        insecure=args.insecure,
    )
    return await _serve(cfg, pipeline, server, mode="live", stub=args.stub)


def _run_overrides(args: argparse.Namespace) -> dict[str, Any]:
    """Config overrides implied by the ``run`` flags."""
    out: dict[str, Any] = {}
    if args.source is not None:
        out["source.uri"] = args.source
    if args.source_type is not None:
        out["source.type"] = args.source_type
    if args.weights is not None:
        out["inference.weights"] = args.weights
    if args.device is not None:
        out["inference.device"] = args.device
    if args.bind is not None:
        out["stream.bind"] = args.bind
    if args.port is not None:
        out["stream.port"] = args.port
    if args.no_log:
        out["incident_log.enabled"] = False
    return out


def _build_run_runner(cfg: Config, *, stub: bool) -> Any:
    """Build the model runner for ``run``.

    The stub is only ever selected explicitly. A station that silently fell back
    to generated boxes when the weights were missing would be publishing
    detections that look exactly like a real model's, which is a worse failure
    than refusing to start.
    """
    if stub:
        from station.inference.stub import StubModelRunner  # noqa: PLC0415

        print(
            "\n*** --stub: NO MODEL IS RUNNING. Every box below is generated. Do not use\n"
            "*** this mode for anything an operator will act on.\n",
            file=sys.stderr,
        )
        return StubModelRunner(cfg.inference, detect_blobs=True)

    from station.pipeline import build_runner  # noqa: PLC0415

    weights = Path(cfg.inference.weights).expanduser()
    if not weights.is_file():
        raise _CliError(
            f"weights file not found: {weights}\n"
            f"  (inference.weights = {cfg.inference.weights!r})\n"
            "Put the trained .pt there, or run with --stub to exercise the transport with "
            "generated boxes and no model.",
            status=2,
        )
    if not _installed("ultralytics"):
        raise _CliError(
            "ultralytics is not installed, so no model can run.\n"
            "  pip install ultralytics\n"
            f"Run '{PROGRAM} check' for the full picture, or --stub to test without a model.",
            status=2,
        )
    return build_runner(cfg)


def _build_streamer(cfg: Config, *, offer_no_stream: bool = False) -> Any:
    """Build the WebRTC streamer, refusing to start quietly without it.

    Args:
        cfg: The station configuration.
        offer_no_stream: Mention ``--no-stream`` in the error. Only ``run`` has
            that flag -- a replay with nothing to stream to has no purpose.

    Returns:
        A :class:`station.stream.webrtc.WebRtcStreamer`.

    Raises:
        _CliError: If the transport's dependencies are not installed. Failing
            here rather than starting a station that serves no video is
            deliberate: an operator who is told the tablets are connected and
            gets a blank screen has no way to tell that apart from a quiet feed.
    """
    missing = [name for name in ("aiortc", "av") if not _installed(name)]
    if missing:
        hint = (
            "Pass --no-stream to run the pipeline and the incident log without serving "
            "tablets, or run "
            if offer_no_stream
            else "Run "
        )
        raise _CliError(
            f"cannot serve WebRTC: {', '.join(missing)} not installed.\n"
            f"  pip install {' '.join(missing)}\n"
            f"{hint}'{PROGRAM} check' for the full picture.",
            status=2,
        )
    from station.stream.webrtc import WebRtcStreamer  # noqa: PLC0415

    return WebRtcStreamer(cfg.stream, source=cfg.source.uri or cfg.source.type)


# ------------------------------------------------------------------ replay


async def _cmd_replay(args: argparse.Namespace) -> int:
    """Re-serve a recorded incident."""
    from station.pipeline import (  # noqa: PLC0415
        Pipeline,
        ReplayRunner,
        incident_video_segments,
        load_incident_detections,
        replay_source_config,
    )
    from station.serve.http import StationServer  # noqa: PLC0415

    incident = Path(args.incident_dir).expanduser()
    if not incident.is_dir():
        raise _CliError(f"not a directory: {incident}")

    try:
        recorded = load_incident_detections(incident)
    except FileNotFoundError as exc:
        raise _CliError(str(exc)) from exc
    if not recorded:
        raise _CliError(f"{incident}/detections.jsonl contains no readable frame results")

    segments = incident_video_segments(incident)
    if not segments:
        raise _CliError(
            f"no recorded video in {incident}/video/. A replay needs the footage the boxes "
            "were computed from -- an overlay with no video underneath it is exactly the "
            "thing this system must not show."
        )
    if not 0 <= args.segment < len(segments):
        raise _CliError(
            f"--segment {args.segment} is out of range; this incident has {len(segments)} "
            f"segment(s), numbered 0..{len(segments) - 1}"
        )
    segment = segments[args.segment]

    overrides: dict[str, Any] = {}
    if args.bind is not None:
        overrides["stream.bind"] = args.bind
    if args.port is not None:
        overrides["stream.port"] = args.port
    cfg = _load(args, overrides)
    cfg.source = replay_source_config(segment, loop=args.loop)
    # Deliberately NOT written back to cfg.station_name: that name becomes an
    # mDNS DNS SAN, so changing it makes ensure_cert() decide the existing
    # certificate no longer covers this station and silently issue a new key.
    # Every tablet that was told to trust the station would break on one replay,
    # and again on the next live run.
    replay_label = f"{cfg.station_name} [REPLAY {incident.name}]"
    # A replay must never write an incident log. The payloads are a copy of an
    # existing record, and a second directory holding them -- stamped with
    # today's date and this laptop's model -- would be a forgery of evidence.
    cfg.incident_log.enabled = False

    streamer = _build_streamer(cfg)
    if not args.insecure and not args.no_certs:
        _issue_certificate(cfg)

    pipeline = Pipeline(
        cfg,
        runner=ReplayRunner(recorded),
        streamer=streamer,
        # The recorded detections have already been through the N-of-M filter.
        # Filtering them a second time would re-apply the persistence
        # requirement and silently swallow the opening frames of every track.
        temporal=None,
    )
    server = StationServer(
        cfg,
        streamer=streamer,
        pipeline=pipeline,
        app_dir=args.app_dir,
        insecure=args.insecure,
        extra_config={"replay": {"incident": incident.name, "segment": segment.name}},
    )
    print(
        f"\n*** REPLAY of {incident.name}: {len(recorded)} recorded results over "
        f"{segment.name}.\n*** This is not a live feed.\n",
        file=sys.stderr,
    )
    return await _serve(cfg, pipeline, server, mode=f"replay {incident.name}", stub=False,
                        display_name=replay_label)


# --------------------------------------------------------------- shared serve


async def _serve(cfg: Config, pipeline: Any, server: Any, *, mode: str, stub: bool,
                 display_name: str | None = None) -> int:
    """Start the listener and the pipeline, then wait for a reason to stop.

    Args:
        cfg: The station configuration.
        pipeline: The :class:`station.pipeline.Pipeline` to drive.
        server: The :class:`station.serve.http.StationServer` to serve on.
        mode: ``"live"`` or a replay description, for the banner.
        stub: Whether the stub runner is in use, for the banner.

    Returns:
        The process exit status.
    """
    from station.serve.http import ServeError  # noqa: PLC0415

    stop = asyncio.Event()
    _install_signal_handlers(stop)

    try:
        info = await server.start()
    except ServeError as exc:
        raise _CliError(str(exc), status=2) from exc

    try:
        await pipeline.start()
    except Exception as exc:
        # stop() before the listener, so a half-started pipeline still closes
        # the incident log directory it may already have created.
        await pipeline.stop()
        await server.stop()
        raise _CliError(f"pipeline could not start: {exc}", status=2) from exc

    print(_banner(cfg, pipeline, info, mode=mode, stub=stub, display_name=display_name))
    sys.stdout.flush()

    finished = asyncio.create_task(pipeline.wait(), name="wildfire-cli-wait")
    interrupted = asyncio.create_task(stop.wait(), name="wildfire-cli-stop")
    try:
        await asyncio.wait({finished, interrupted}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for task in (finished, interrupted):
            task.cancel()
        # Order matters: stop taking frames before dropping the peers, so the
        # tablets get the final ``stopped`` heartbeat rather than a dead
        # connection they have to infer something from.
        await pipeline.stop()
        await server.stop()

    print(_closing_summary(pipeline))
    if pipeline.error is not None:
        print(f"error: the pipeline stopped on: {pipeline.error}", file=sys.stderr)
        return 2
    return 0


def _install_signal_handlers(stop: asyncio.Event) -> None:
    """Make Ctrl-C and SIGTERM shut the station down cleanly.

    A clean shutdown is what flushes the incident log's metadata and closes the
    video segment; killed mid-write the record is still readable but the
    manifest is not final.
    """
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except (NotImplementedError, RuntimeError, ValueError):  # pragma: no cover - non-POSIX
            # Windows, or a non-main thread. KeyboardInterrupt still unwinds
            # through main() and the finally block still runs.
            pass


def _banner(cfg: Config, pipeline: Any, info: Any, *, mode: str, stub: bool,
            display_name: str | None = None) -> str:
    """The block printed once the station is up.

    Reports what the station is doing and where to reach it. Contains no claim
    about the scene, and ends with the sentence that says so.
    """
    model = pipeline.model_info
    model_line = "none"
    if model is not None:
        sha = f" sha {model.weights_sha}" if model.weights_sha else ""
        model_line = f"{model.name} v{model.version}{sha}"
        if stub:
            model_line += "   *** GENERATED, NO MODEL IS RUNNING ***"

    lines = [
        "",
        "wildfire-watch ground station",
        f"  {PRODUCT_DESCRIPTOR} -- detections are drawn over video, never burned into it",
        "",
        f"  station    {display_name or cfg.station_name}",
        f"  mode       {mode}",
        f"  source     {cfg.source.uri or cfg.source.type}  ({cfg.source.type})",
        f"  model      {model_line}",
        f"  sampling   up to {cfg.inference.max_fps:g} inferences/s, "
        f"confirmed {cfg.temporal.n}-of-{cfg.temporal.m}",
        f"  overlay    boxes stop being drawn after {cfg.stream.max_overlay_age_s:g}s of media time",
        f"  record     {pipeline.incident_dir or 'disabled'}",
        "",
        f"  {info.describe()}",
    ]
    lines.extend(_tablet_entry(info))
    lines.append("")
    lines.extend(f"  {line}" for line in SAFETY_LINE.splitlines())
    lines.append("")
    lines.append("  Ctrl-C to stop.")
    lines.append("")
    return "\n".join(lines)


def _tablet_entry(info: Any) -> list[str]:
    """The lines telling the operator exactly what to open on the tablet.

    The self-test comes first, deliberately. On a LAN with a self-signed
    certificate the first connection is the one that fails, and it fails in
    ways that look like the app being broken rather than the certificate not
    being trusted. selftest.html says which it is, in about thirty seconds.

    A QR code is printed when the ``qrcode`` package is available, because the
    alternative is typing an IP address and a port into a tablet keyboard while
    people wait. It is optional on purpose: a missing convenience must never
    stop the station serving.
    """
    urls = list(getattr(info, "urls", ()) or [])
    # Prefer an address a tablet can actually reach: loopback is useless to
    # anyone not sitting at this laptop.
    lan = [u for u in urls if "localhost" not in u and "127.0.0.1" not in u and "[::1]" not in u]
    best = (lan or urls or [None])[0]
    if best is None:
        return []

    selftest = best.rstrip("/") + "/selftest.html"
    lines = [
        "",
        "  ON THE TABLET, OPEN THIS FIRST:",
        f"    {selftest}",
        "      checks whether this device can run the overlay, and whether the",
        "      certificate was actually trusted. Then open the app itself:",
        f"    {best}",
    ]

    try:
        import io  # noqa: PLC0415

        import qrcode  # noqa: PLC0415

        q = qrcode.QRCode(border=1, box_size=1)
        q.add_data(selftest)
        q.make(fit=True)
        buf = io.StringIO()
        q.print_ascii(out=buf, invert=True)
        lines.append("")
        lines.extend("    " + row for row in buf.getvalue().rstrip("\n").splitlines())
        lines.append("    scan for the self-test  (pip install qrcode to change this code)")
    except Exception:  # noqa: BLE001 - a convenience, never a reason to fail
        lines.append("")
        lines.append("    (pip install qrcode to print a scannable code here)")
    return lines


def _closing_summary(pipeline: Any) -> str:
    """What the run did, printed on the way out.

    Counters only: frames handled, frames the model could not keep up with,
    where the record went. Nothing about what was in them.
    """
    m = pipeline.metrics
    lines = [
        "",
        "run finished",
        f"  uptime            {m.uptime_s:.1f}s",
        f"  frames from source {m.frames_read}",
        f"  inferences         {m.frames_inferred}",
        f"  frames dropped     {m.frames_dropped}   (inference could not keep up)",
        f"  frames sampled out {m.frames_gated}   (max_fps gate; never examined, never logged)",
        f"  source restarts    {m.source_restarts}",
    ]
    if pipeline.incident_dir is not None:
        lines.append(f"  record             {pipeline.incident_dir}")
    lines.append("")
    return "\n".join(lines)


# ------------------------------------------------------------------ certs


def _cmd_certs(args: argparse.Namespace) -> int:
    """Issue or describe the station's certificate."""
    from station.serve.certs import (  # noqa: PLC0415
        DEFAULT_VALIDITY_DAYS,
        CertError,
        ensure_cert,
        local_hostnames,
        local_ip_addresses,
        read_cert_info,
    )

    cfg = _load(args, {})
    cert_path = Path(args.cert or cfg.stream.cert).expanduser()
    key_path = Path(args.key or cfg.stream.key).expanduser()

    if args.show:
        try:
            info = read_cert_info(cert_path, key_path)
        except CertError as exc:
            raise _CliError(str(exc), status=2) from exc
        print(info.describe())
        if info.expired:
            print("\nthis certificate is outside its validity window; reissue it")
            return 2
        return 0

    hosts = list(local_hostnames(cfg.station_name)) + list(args.host)
    ips = list(local_ip_addresses()) + list(args.ip)
    try:
        info = ensure_cert(
            cert_path,
            key_path,
            station_name=cfg.station_name,
            dns_names=hosts,
            ip_addresses=ips,
            validity_days=args.days or DEFAULT_VALIDITY_DAYS,
            force=args.force,
        )
    except CertError as exc:
        raise _CliError(str(exc), status=2) from exc

    print("issued a new certificate" if info.created else "existing certificate is still good")
    print(info.describe())
    if info.created:
        print(
            "\nEvery tablet must be told to trust this certificate before it can connect.\n"
            "  Android: open the station URL and accept, or install the .crt as a user CA.\n"
            "  iOS/iPadOS: download the .crt, install the profile, then enable it under\n"
            "    Settings > General > About > Certificate Trust Settings.\n"
            "Compare the sha256 above with the one the tablet shows before trusting it."
        )
    return 0


def _issue_certificate(cfg: Config) -> None:
    """Make sure a usable certificate exists before the listener binds."""
    from station.serve.certs import CertError, ensure_cert  # noqa: PLC0415

    try:
        info = ensure_cert(cfg.stream.cert, cfg.stream.key, station_name=cfg.station_name)
    except CertError as exc:
        raise _CliError(
            f"{exc}\n\nRun '{PROGRAM} certs' to issue one, or --insecure to serve plain HTTP "
            "(tablets then get no WebRTC and no service worker).",
            status=2,
        ) from exc
    if info.created:
        print(
            f"issued a new certificate {info.cert_path}\n"
            f"  sha256 {info.fingerprint_sha256}\n"
            "  every tablet has to trust it before it can connect\n",
            file=sys.stderr,
        )


# ------------------------------------------------------------------ check


def _cmd_check(args: argparse.Namespace) -> int:
    """Validate the config and report what this laptop can do."""
    problems: list[str] = []

    print(f"wildfire-watch {PROGRAM} check")
    print(f"  python {sys.version.split()[0]} on {sys.platform}")
    print()

    # --- configuration ---------------------------------------------------
    source = args.config or "(built-in defaults; no --config given)"
    try:
        cfg = _load(args, {})
    except _CliError as exc:
        print(f"configuration : {source}")
        print(f"  INVALID: {exc}")
        return 1
    try:
        validate(cfg)
    except (ValueError, TypeError) as exc:  # pragma: no cover - load_config validates already
        print(f"configuration : {source}")
        print(f"  INVALID: {exc}")
        return 1

    print(f"configuration : {source}")
    print(f"  station       {cfg.station_name}")
    print(f"  source        {cfg.source.type}  {cfg.source.uri or '(no uri set)'}")
    print(
        f"  inference     {cfg.inference.weights}  imgsz {cfg.inference.imgsz}  "
        f"conf {cfg.inference.conf_threshold:g}  max {cfg.inference.max_fps:g}/s  "
        f"device {cfg.inference.device}"
    )
    print(
        f"  temporal      {cfg.temporal.n}-of-{cfg.temporal.m}, iou {cfg.temporal.iou_match:g}, "
        f"max_age {cfg.temporal.max_age}"
    )
    print(
        f"  stream        {cfg.stream.bind}:{cfg.stream.port}  "
        f"overlay expires at {cfg.stream.max_overlay_age_s:g}s, "
        f"stall after {cfg.stream.stall_after_s:g}s"
    )
    print(
        f"  incident log  {cfg.incident_log.dir}  "
        f"{'enabled' if cfg.incident_log.enabled else 'DISABLED'}"
        f"{', video on' if cfg.incident_log.record_video else ', video off'}"
    )
    print()

    # --- dependencies ----------------------------------------------------
    print("optional dependencies")
    present: dict[str, bool] = {}
    for dep in OPTIONAL_DEPENDENCIES:
        ok = _installed(dep.module)
        present[dep.module] = ok
        version = _version(dep.distribution) if ok else ""
        mark = "present" if ok else "MISSING"
        detail = version if ok else dep.install_hint
        print(f"  {dep.module:<14} {mark:<8} {dep.purpose:<48} {detail}")
    print()

    # --- files -----------------------------------------------------------
    print("files")
    weights = Path(cfg.inference.weights).expanduser()
    have_weights = weights.is_file()
    print(f"  weights       {weights}  {'ok' if have_weights else 'MISSING'}")
    if not have_weights:
        problems.append("the model weights are missing")

    cert = Path(cfg.stream.cert).expanduser()
    key = Path(cfg.stream.key).expanduser()
    if cert.is_file() and key.is_file():
        print(f"  certificate   {cert}", end="")
        if present.get("cryptography"):
            try:
                from station.serve.certs import read_cert_info  # noqa: PLC0415

                info = read_cert_info(cert, key)
                state = "EXPIRED" if info.expired else f"{info.days_remaining:.0f} days left"
                print(f"  {state}, {len(info.ip_addresses)} IP SAN entries")
                if info.expired:
                    problems.append("the TLS certificate has expired")
                if not info.ip_addresses:
                    problems.append("the TLS certificate has no IP SAN entries; iOS will reject it")
            except Exception as exc:
                print(f"  UNREADABLE ({exc})")
                problems.append("the TLS certificate cannot be read")
        else:
            print("  (install cryptography to inspect it)")
    else:
        print(f"  certificate   {cert}  MISSING -- run '{PROGRAM} certs'")
        problems.append("there is no TLS certificate")

    from station.serve.http import default_app_dir  # noqa: PLC0415

    app_dir = Path(args.app_dir).expanduser() if args.app_dir else default_app_dir()
    index = app_dir / "index.html"
    if index.is_file():
        count = sum(1 for _ in app_dir.rglob("*") if _.is_file())
        print(f"  pwa           {app_dir}  ok ({count} files)")
    elif app_dir.is_dir():
        print(f"  pwa           {app_dir}  no index.html")
        problems.append("the PWA has no index.html, so tablets have nothing to load")
    else:
        print(f"  pwa           {app_dir}  MISSING")
        problems.append("the PWA directory does not exist")

    incidents = Path(cfg.incident_log.dir).expanduser()
    if not cfg.incident_log.enabled:
        print(f"  incidents     {incidents}  logging is DISABLED in the config")
        problems.append("incident logging is disabled, so this run leaves no record")
    elif incidents.is_dir():
        print(f"  incidents     {incidents}  ok, {'writable' if os.access(incidents, os.W_OK) else 'NOT WRITABLE'}")
        if not os.access(incidents, os.W_OK):
            problems.append("the incident log directory is not writable")
    else:
        print(f"  incidents     {incidents}  will be created on first run")
    print()

    # --- capabilities ----------------------------------------------------
    print("what this laptop can do")
    capabilities = (
        ("serve the app over HTTPS", ["aiohttp", "cryptography"], []),
        ("stream video and detections to tablets", ["aiortc", "av"], []),
        ("decode the video source", ["av"], ["cv2"]),
        ("run the model", ["ultralytics", "torch"], []),
        ("record incident video", ["av"], ["cv2"]),
        ("write the detections log", [], []),
        ("run without a model (--stub)", [], []),
    )
    for label, required, alternatives in capabilities:
        missing = [name for name in required if not present.get(name)]
        if missing and alternatives and any(present.get(name) for name in alternatives):
            print(f"  {label:<42} yes  (via {', '.join(n for n in alternatives if present.get(n))})")
            continue
        if missing:
            print(f"  {label:<42} no   (needs {', '.join(missing)})")
        else:
            print(f"  {label:<42} yes")
    print()

    blocking = [name for name in ("aiohttp", "aiortc", "av") if not present.get(name)]
    if blocking:
        problems.append(f"cannot serve tablets: {', '.join(blocking)} not installed")
    if not present.get("ultralytics") or not present.get("torch"):
        problems.append("cannot run the model: ultralytics/torch not installed")

    if problems:
        print("not ready to run here:")
        for problem in problems:
            print(f"  - {problem}")
        print()
        print(
            "This is a report on this laptop, not on any incident. It says what the station\n"
            "can and cannot do here; it says nothing about any scene."
        )
        return 2

    print("ready to run.")
    print()
    print(
        "This is a report on this laptop, not on any incident. It says what the station\n"
        "can and cannot do here; it says nothing about any scene."
    )
    return 0


# ------------------------------------------------------------------ helpers


def _load(args: argparse.Namespace, extra: dict[str, Any]) -> Config:
    """Load the configuration, applying ``--set`` and per-command overrides.

    Args:
        args: Parsed arguments; ``config`` and ``overrides`` are read.
        extra: Overrides implied by the subcommand's own flags. Applied after
            ``--set``, so an explicit flag wins.

    Returns:
        The validated :class:`~station.core.config.Config`.

    Raises:
        _CliError: If the file is missing or the configuration is invalid.
    """
    overrides: dict[str, Any] = {}
    for item in getattr(args, "overrides", []) or []:
        key, sep, value = item.partition("=")
        if not sep:
            raise _CliError(f"--set expects KEY=VALUE, got {item!r}")
        overrides[key.strip()] = _parse_value(value)
    overrides.update(extra)

    path = getattr(args, "config", None)
    if path is not None and not Path(path).expanduser().is_file():
        raise _CliError(f"config file not found: {path}")
    if path is not None and not _installed("yaml"):
        raise _CliError(
            "PyYAML is not installed, so a config file cannot be read.\n  pip install PyYAML",
            status=2,
        )
    try:
        cfg = load_config(Path(path).expanduser() if path else None, overrides=overrides or None)
    except (ValueError, TypeError, OSError) as exc:
        raise _CliError(f"invalid configuration: {exc}") from exc
    _repair_station_name(cfg)
    return cfg


def _repair_station_name(cfg: Config) -> None:
    """Work around a defaulting bug in the committed config loader.

    ``load_config`` reads its fallback as ``Config.station_name``, but ``Config``
    is a ``slots=True`` dataclass, so that attribute is the *slot descriptor*
    rather than the default string. A config file that omits ``station_name``
    therefore yields a Config whose station name renders as
    ``<member 'station_name' of 'Config' objects>`` -- which would then appear
    in the certificate's common name, in ``/config.json``, in the banner and in
    every incident's ``meta.json``.

    ``station/core/config.py`` is the committed contract and is not ours to
    edit, so the value is repaired here, at the one place the CLI loads it.
    """
    if not isinstance(cfg.station_name, str):
        cfg.station_name = str(Config.__dataclass_fields__["station_name"].default)


def _parse_value(raw: str) -> Any:
    """Coerce a ``--set`` value to the type the config field expects.

    Config sections are typed dataclasses that are constructed directly from
    this mapping, so ``--set stream.port=9443`` has to become an ``int`` here or
    the port arrives as a string and fails at bind time with a confusing error.
    """
    text = raw.strip()
    lowered = text.lower()
    if lowered in ("true", "yes", "on"):
        return True
    if lowered in ("false", "no", "off"):
        return False
    if lowered in ("null", "none", ""):
        return None
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        return [_parse_value(part) for part in inner.split(",")] if inner else []
    return text


def _installed(module: str) -> bool:
    """Whether a module can be imported, without importing it.

    ``find_spec`` walks the finders rather than executing the module, which is
    what keeps ``check`` from loading torch (and taking ten seconds, and
    touching the GPU) just to say whether it is there.
    """
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError, AttributeError):
        # A half-removed package can leave a finder that raises. Missing is the
        # right answer, and it is what the operator needs to act on either way.
        return False


def _version(distribution: str) -> str:
    """Installed version of a distribution, or an empty string."""
    try:
        from importlib.metadata import PackageNotFoundError, version  # noqa: PLC0415

        return version(distribution)
    except PackageNotFoundError:
        return ""
    except Exception:  # pragma: no cover - broken metadata
        return ""


def _configure_logging(args: argparse.Namespace) -> None:
    """Set up logging from ``-v`` / ``-q``."""
    if args.quiet:
        level = logging.WARNING
    elif args.verbose >= 2:
        level = logging.DEBUG
    elif args.verbose == 1:
        level = logging.INFO
    else:
        level = logging.INFO if args.command in ("run", "replay") else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    # aiortc and aioice are extremely chatty at DEBUG and would bury the
    # station's own lines during exactly the incident someone is debugging.
    for noisy in ("aioice", "aiortc", "aiohttp.access", "libav"):
        logging.getLogger(noisy).setLevel(max(level, logging.WARNING))
