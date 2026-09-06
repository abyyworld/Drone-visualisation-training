"""The HTTPS listener: serves the PWA, mounts signalling, answers health.

One origin, one port, one certificate. The tablets load the PWA from
``https://<station>:8443/`` and post their SDP offers to
``https://<station>:8443/webrtc/offer`` -- same scheme, same host, same port.
That is deliberate and it removes three whole categories of field failure: no
CORS preflight, no mixed-content block, and no second certificate for an
operator to install on twelve tablets while standing in a car park.

What is served:

===========================  ====================================================
``/``                        ``app/index.html``
``/<anything>``              the corresponding file under ``app/``
``/config.json``             overlay parameters the PWA reads at boot
``/healthz``                 pipeline liveness, for the operator and for a
                             process supervisor
``/webrtc/*``                mounted from :mod:`station.stream.signaling`
===========================  ====================================================

Two details that look cosmetic and are not:

**MIME types.** ``navigator.serviceWorker.register()`` rejects a script served
as ``text/plain`` -- silently, from the browser's point of view -- and a station
whose service worker never registers is a PWA that cannot be opened again once
the tablet loses the network. Types are therefore looked up in an explicit
table rather than left to ``mimetypes``, whose answers depend on
``/etc/mime.types`` and have historically differed between laptops.

**Cache headers.** Everything is served ``Cache-Control: no-cache``, which does
*not* mean "do not cache": it means "cache it, but revalidate before using it".
Combined with the entity tag that ``FileResponse`` derives from the file's mtime
and size, a tablet that has already seen the current build spends one
conditional request per file and reuses everything, while a tablet reconnecting
after the station has been updated gets the new build instead of yesterday's.
``no-store`` -- the header people reach for instinctively -- would forbid the
service worker's own caches too, and offline capability is the reason the PWA
exists.

Nothing served from here describes the scene. ``/healthz`` reports whether the
station is looking, not what it found; see ``station/core/safety.py``.

``aiohttp`` is imported inside the functions that need it, so this module -- and
therefore ``python -m station check`` -- works on a laptop that does not have
it installed.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from station.core.config import Config
from station.core.safety import PRODUCT_DESCRIPTOR
from station.core.types import WIRE_VERSION, PipelineState

__all__ = [
    "DEFAULT_APP_DIR",
    "MIME_TYPES",
    "ServeError",
    "ServerInfo",
    "StationServer",
    "build_app",
    "client_config_payload",
    "health_payload",
    "guess_content_type",
    "station_urls",
    "default_app_dir",
]

log = logging.getLogger(__name__)

#: The PWA lives beside the ``station`` package in the repository. Overridable
#: so a packaged install can point somewhere else.
DEFAULT_APP_DIR = Path(__file__).resolve().parents[2] / "app"

#: Explicit, because a service worker served as the wrong type does not
#: register and the browser reports nothing useful about why.
MIME_TYPES: dict[str, str] = {
    ".html": "text/html; charset=utf-8",
    ".htm": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".map": "application/json; charset=utf-8",
    # The registered type for a web app manifest. Chrome accepts application/json
    # too; Safari has been fussier about it, and the tablets include iPads.
    ".webmanifest": "application/manifest+json; charset=utf-8",
    ".txt": "text/plain; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".ico": "image/x-icon",
    ".woff": "font/woff",
    ".woff2": "font/woff2",
    ".ttf": "font/ttf",
    ".otf": "font/otf",
    ".wasm": "application/wasm",
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".m4s": "video/iso.segment",
    ".bin": "application/octet-stream",
}

#: Filenames that must never be revalidated against a stale copy, whatever a
#: proxy in between thinks. A stale service worker outlives every other kind of
#: stale file, because it is the thing that decides what the others are.
_ALWAYS_REVALIDATE = frozenset({"sw.js", "service-worker.js", "index.html", "manifest.webmanifest"})

#: Files never served, whatever the URL says.
_HIDDEN_PREFIXES = (".",)


class ServeError(RuntimeError):
    """The HTTP layer could not be built or started."""


def default_app_dir() -> Path:
    """Where the PWA is expected to live.

    Returns:
        ``$WILDFIRE_APP_DIR`` when set, otherwise the ``app/`` directory beside
        the ``station`` package.
    """
    override = os.environ.get("WILDFIRE_APP_DIR")
    return Path(override).expanduser() if override else DEFAULT_APP_DIR


def guess_content_type(path: Path | str) -> str:
    """Content type for a file the station serves.

    Args:
        path: The file being served; only its suffix is used.

    Returns:
        An explicit type from :data:`MIME_TYPES`, or
        ``application/octet-stream`` for anything unrecognised. Never
        ``text/plain``: guessing text for an unknown file is how a JavaScript
        module ends up refused by the browser.
    """
    return MIME_TYPES.get(Path(path).suffix.lower(), "application/octet-stream")


# ------------------------------------------------------------------ payloads


def client_config_payload(
    cfg: Config,
    *,
    streamer: Any | None = None,
    signaling_prefix: str = "/webrtc",
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The parameters the PWA reads at boot, from ``/config.json``.

    Everything here is configured once, on the station, and shipped to the
    tablets. ``max_overlay_age_s`` in particular is a safety parameter enforced
    on the tablet: sending it rather than hard-coding it in the JavaScript is
    what makes it impossible for a station and a tablet to disagree about when
    boxes stop being drawn.

    Args:
        cfg: The station configuration.
        streamer: The WebRTC streamer, if one is running. Its
            ``client_parameters()`` are merged in so the data-channel name and
            ICE servers come from the object that will actually serve the peer.
        signaling_prefix: Where the signalling routes are mounted, so the PWA
            does not hard-code it.
        extra: Additional fields to merge in.

    Returns:
        A JSON-ready mapping. It describes the station and the overlay; it says
        nothing about the scene, and there is no field here that could be read
        as reassurance.
    """
    payload: dict[str, Any] = {
        "station_name": cfg.station_name,
        "wire_version": WIRE_VERSION,
        # Repeated from station/core/safety.py so the tablet can render the
        # descriptor without inventing its own wording.
        "descriptor": PRODUCT_DESCRIPTOR,
        "max_overlay_age_s": cfg.stream.max_overlay_age_s,
        "overlay_buffer_s": cfg.stream.overlay_buffer_s,
        "status_interval_s": cfg.stream.status_interval_s,
        "stall_after_s": cfg.stream.stall_after_s,
        "ice_servers": list(cfg.stream.ice_servers),
        "signaling": {
            "offer": f"{signaling_prefix.rstrip('/')}/offer",
            "close": f"{signaling_prefix.rstrip('/')}/close",
            "config": f"{signaling_prefix.rstrip('/')}/config",
        },
    }
    if streamer is not None:
        try:
            payload.update(streamer.client_parameters())
        except Exception:  # pragma: no cover - defensive
            log.debug("streamer.client_parameters() raised", exc_info=True)
    if extra:
        payload.update(extra)
    return payload


def health_payload(
    cfg: Config,
    *,
    pipeline: Any | None = None,
    streamer: Any | None = None,
) -> tuple[dict[str, Any], int]:
    """Liveness of the station, and the HTTP status that goes with it.

    This endpoint answers exactly one question: *is the station still looking?*
    It is not, and must never become, a report on the scene. There is no
    reassuring summary here, no detection count and no green light -- an empty
    detection list from a model that cannot see thin smoke is identical to one
    from an empty field, so a status endpoint that summarised detections would
    be manufacturing reassurance out of a null result.

    Args:
        cfg: The station configuration.
        pipeline: The running :class:`station.pipeline.Pipeline`, if any.
        streamer: The WebRTC streamer, if any.

    Returns:
        ``(payload, http_status)``. The status is ``503`` when the pipeline has
        stalled or stopped, so that a process supervisor restarts the station
        rather than leaving a laptop that answers HTTP but is no longer running
        a model.
    """
    payload: dict[str, Any] = {
        "station_name": cfg.station_name,
        "wire_version": WIRE_VERSION,
        "descriptor": PRODUCT_DESCRIPTOR,
        "serving": True,
        "pipeline": None,
    }
    status_code = 200

    if pipeline is not None:
        status = pipeline.status()
        payload["pipeline"] = {
            "state": status.state,
            "source": status.source,
            "source_fps": status.source_fps,
            "inference_fps": status.inference_fps,
            "last_inference_pts": status.last_inference_pts,
            "last_inference_wall_time": status.last_inference_wall_time,
            "dropped_frames": status.dropped_frames,
            "uptime_s": status.uptime_s,
            "note": status.note,
            "model": status.model.to_wire() if status.model is not None else None,
            "metrics": pipeline.metrics.as_dict(),
            "incident_dir": str(pipeline.incident_dir) if pipeline.incident_dir else None,
        }
        if status.state in (PipelineState.STALLED, PipelineState.STOPPED):
            status_code = 503

    if streamer is not None:
        try:
            payload["peers"] = len(streamer.peers)
            payload["stream"] = {
                "frames_published": streamer.stats.frames_published,
                "detections_published": streamer.stats.detections_published,
                "statuses_published": streamer.stats.statuses_published,
            }
        except Exception:  # pragma: no cover - defensive
            log.debug("streamer stats unavailable", exc_info=True)
    return payload, status_code


def station_urls(cfg: Config, *, scheme: str = "https", extra_hosts: Sequence[str] = ()) -> tuple[str, ...]:
    """URLs an operator can type into a tablet to reach this station.

    Args:
        cfg: The station configuration; supplies the port and the name.
        scheme: ``https`` in any real deployment.
        extra_hosts: Additional hosts to list.

    Returns:
        One URL per reachable address, IPv6 bracketed, loopback last because it
        is the one address that will not work from a tablet.
    """
    from station.serve.certs import local_ip_addresses, mdns_name  # noqa: PLC0415

    port = int(cfg.stream.port)
    hosts: list[str] = []

    def add(host: str) -> None:
        if host and host not in hosts:
            hosts.append(host)

    for addr in local_ip_addresses(include_loopback=False):
        add(f"[{addr}]" if ":" in addr else addr)
    add(mdns_name(cfg.station_name))
    for host in extra_hosts:
        add(host)
    add("localhost")
    return tuple(f"{scheme}://{host}:{port}/" for host in hosts)


# ----------------------------------------------------------------- the app


def build_app(
    cfg: Config,
    *,
    streamer: Any | None = None,
    pipeline: Any | None = None,
    app_dir: str | os.PathLike[str] | None = None,
    signaling_prefix: str = "/webrtc",
    allow_origin: str | None = None,
    manage_streamer: bool = False,
    extra_config: Mapping[str, Any] | None = None,
) -> Any:
    """Build the aiohttp application the station serves.

    Args:
        cfg: The station configuration.
        streamer: A :class:`station.stream.webrtc.WebRtcStreamer`. When given,
            the signalling routes are mounted. When omitted the station still
            serves the PWA and ``/healthz``, which is what makes it possible to
            diagnose a laptop that cannot run aiortc.
        pipeline: The running pipeline, for ``/healthz``.
        app_dir: Directory holding the PWA. Defaults to :func:`default_app_dir`.
        signaling_prefix: Mount point for the signalling routes.
        allow_origin: ``Access-Control-Allow-Origin`` for the signalling
            routes. Leave ``None`` in production -- the PWA is same-origin, and
            not enabling CORS is one fewer way for a stray browser tab on the
            incident WiFi to open peers on the station.
        manage_streamer: Start and close the streamer with the application.
        extra_config: Extra fields merged into ``/config.json``.

    Returns:
        An ``aiohttp.web.Application``.

    Raises:
        ServeError: If aiohttp is not installed.
    """
    web = _aiohttp_web()
    root = Path(app_dir).expanduser().resolve() if app_dir is not None else default_app_dir().resolve()
    if not root.is_dir():
        # Not fatal. A station whose PWA is missing should still answer
        # /healthz and the signalling routes, because that is exactly the state
        # in which someone needs to diagnose it.
        log.error(
            "PWA directory %s does not exist; the station will serve /healthz and signalling "
            "but tablets will get 404s. Check the deployment or set WILDFIRE_APP_DIR.",
            root,
        )

    app = web.Application(client_max_size=_max_body_bytes())
    app["station_config"] = cfg
    app["station_app_dir"] = root
    app["station_streamer"] = streamer
    app["station_pipeline"] = pipeline

    async def config_json(request: Any) -> Any:
        """Overlay parameters, read by the PWA at boot."""
        response = web.json_response(
            client_config_payload(
                cfg, streamer=streamer, signaling_prefix=signaling_prefix, extra=extra_config
            )
        )
        # Revalidated, not stored stale: max_overlay_age_s is a safety
        # parameter and a tablet must not run on last week's value.
        response.headers["Cache-Control"] = "no-cache"
        return response

    async def healthz(request: Any) -> Any:
        """Pipeline liveness. Never a statement about the scene."""
        payload, status = health_payload(cfg, pipeline=pipeline, streamer=streamer)
        response = web.json_response(payload, status=status)
        response.headers["Cache-Control"] = "no-store"
        return response

    app.router.add_get("/config.json", config_json)
    app.router.add_get("/healthz", healthz)

    if streamer is not None:
        from station.stream.signaling import add_routes  # noqa: PLC0415

        add_routes(app, streamer, prefix=signaling_prefix, allow_origin=allow_origin)
        if manage_streamer:
            async def _start_streamer(_app: Any) -> None:
                await streamer.start()

            async def _close_streamer(_app: Any) -> None:
                await streamer.close()

            app.on_startup.append(_start_streamer)
            app.on_cleanup.append(_close_streamer)
    else:
        log.warning(
            "no WebRTC streamer attached; the station serves the app and /healthz but no video"
        )

    static = _make_static_handler(web, root)
    # Registered last, and as a catch-all, so the explicit routes above and the
    # signalling prefix win over any file that happens to share their name.
    app.router.add_get("/", static)
    app.router.add_get("/{path:.*}", static)
    return app


def _make_static_handler(web: Any, root: Path) -> Any:
    """Build the handler that serves files out of the PWA directory."""

    async def handler(request: Any) -> Any:
        rel = request.match_info.get("path", "")
        target = _resolve_static(root, rel)
        if target is None:
            raise web.HTTPNotFound(text=f"not found: /{rel}")

        headers = {
            "Content-Type": guess_content_type(target),
            "Cache-Control": _cache_control(target),
            # We set every content type explicitly above; telling the browser
            # not to second-guess them is what stops a .js served from a
            # misconfigured path being sniffed as something inert.
            "X-Content-Type-Options": "nosniff",
        }
        if target.name in ("sw.js", "service-worker.js"):
            # Lets a service worker file served from anywhere control the whole
            # origin. Without it a worker at /js/sw.js can only control /js/.
            headers["Service-Worker-Allowed"] = "/"
        # FileResponse handles conditional requests and ranges, and derives the
        # entity tag from the file's mtime and size -- which is what makes
        # ``no-cache`` cheap rather than a full re-download every time.
        return web.FileResponse(target, headers=headers)

    return handler


def _resolve_static(root: Path, rel: str) -> Path | None:
    """Map a URL path to a file inside ``root``, or ``None``.

    Rejects anything that escapes the directory. The check is done on the
    *resolved* path rather than on the text of the URL, so a symlink inside
    ``app/`` pointing at ``/etc`` is caught as well as a ``..`` in the request.

    Args:
        root: The resolved PWA directory.
        rel: The URL path, without a leading slash.

    Returns:
        The file to serve, or ``None`` when there is nothing to serve.
    """
    rel = rel.strip("/")
    if not rel:
        index = root / "index.html"
        return index if index.is_file() else None

    parts = [p for p in rel.split("/") if p not in ("", ".")]
    if any(p == ".." for p in parts):
        return None
    if any(p.startswith(_HIDDEN_PREFIXES) for p in parts):
        # Refuses /.git/config and friends without needing to know what they
        # are. Nothing the PWA needs begins with a dot.
        return None

    candidate = (root / Path(*parts)).resolve() if parts else root
    try:
        candidate.relative_to(root)
    except ValueError:
        return None

    if candidate.is_file():
        return candidate
    if candidate.is_dir():
        index = candidate / "index.html"
        return index if index.is_file() else None

    # Client-side routes have no file extension; serving the app shell for them
    # is what makes a deep link work after a reload. A missing *asset* -- which
    # has an extension -- stays a 404, because answering an absent .js file
    # with HTML produces a console error nobody can interpret.
    if not candidate.suffix:
        index = root / "index.html"
        return index if index.is_file() else None
    return None


def _cache_control(path: Path) -> str:
    """Cache policy for one served file.

    ``no-cache`` everywhere: store it, but check with the station before using
    it. That is what stops a tablet showing yesterday's overlay code after the
    station has been updated, while still letting the browser and the service
    worker keep their copies for use offline. ``no-store`` would give up
    offline capability, and a long ``max-age`` would give up updates.
    """
    if path.name in _ALWAYS_REVALIDATE:
        return "no-cache, must-revalidate"
    return "no-cache"


def _max_body_bytes() -> int:
    """Request body ceiling, taken from the signalling layer's own limit."""
    try:
        from station.stream.signaling import MAX_BODY_BYTES  # noqa: PLC0415

        return int(MAX_BODY_BYTES)
    except Exception:  # pragma: no cover - signalling module unavailable
        return 64 * 1024


# --------------------------------------------------------------- the server


@dataclass(slots=True)
class ServerInfo:
    """Where the station ended up listening."""

    scheme: str
    host: str
    port: int
    app_dir: Path
    urls: tuple[str, ...]

    def describe(self) -> str:
        """Operator-facing summary for the CLI banner."""
        lines = [f"listening on {self.scheme}://{self.host}:{self.port}  (serving {self.app_dir})"]
        lines.extend(f"  tablets: {url}" for url in self.urls)
        return "\n".join(lines)


class StationServer:
    """Runs the station's HTTPS listener for the life of a deployment.

    Example:
        >>> import asyncio
        >>> from station.core.config import Config
        >>> async def main():
        ...     async with StationServer(Config()) as server:   # doctest: +SKIP
        ...         print(server.info.describe())
        ...         await asyncio.sleep(3600)
    """

    def __init__(
        self,
        cfg: Config,
        *,
        streamer: Any | None = None,
        pipeline: Any | None = None,
        app_dir: str | os.PathLike[str] | None = None,
        signaling_prefix: str = "/webrtc",
        allow_origin: str | None = None,
        manage_streamer: bool = True,
        ssl_context: Any = None,
        insecure: bool = False,
        extra_config: Mapping[str, Any] | None = None,
    ) -> None:
        """Prepare the listener. Nothing binds until :meth:`start`.

        Args:
            cfg: The station configuration; supplies bind address, port and the
                certificate paths.
            streamer: The WebRTC streamer to mount signalling for.
            pipeline: The running pipeline, for ``/healthz``.
            app_dir: Directory holding the PWA.
            signaling_prefix: Mount point for the signalling routes.
            allow_origin: CORS origin for signalling; ``None`` in production.
            manage_streamer: Start and stop the streamer with the listener.
            ssl_context: A prepared ``ssl.SSLContext``. Built from
                ``cfg.stream.cert``/``.key`` when omitted.
            insecure: Serve plain HTTP. **Development only.** Tablets get no
                WebRTC and no service worker on a plain-HTTP origin, so this is
                only ever useful behind a reverse proxy that terminates TLS, or
                on ``localhost``.
            extra_config: Extra fields merged into ``/config.json``.
        """
        self.cfg = cfg
        self.streamer = streamer
        self.pipeline = pipeline
        self.app_dir = Path(app_dir).expanduser() if app_dir is not None else default_app_dir()
        self.signaling_prefix = signaling_prefix
        self.allow_origin = allow_origin
        self.manage_streamer = manage_streamer
        self.insecure = bool(insecure)
        self._ssl_context = ssl_context
        self._extra_config = dict(extra_config or {})
        self._runner: Any = None
        self._site: Any = None
        self._info: ServerInfo | None = None

    @property
    def info(self) -> ServerInfo:
        """Where the listener is bound.

        Raises:
            ServeError: If the server has not been started.
        """
        if self._info is None:
            raise ServeError("server has not been started")
        return self._info

    async def start(self) -> ServerInfo:
        """Bind the listener and start serving.

        Returns:
            A :class:`ServerInfo` describing the bound address and the URLs an
            operator can hand to the tablets.

        Raises:
            ServeError: If aiohttp is missing or the port cannot be bound.
            station.serve.certs.CertError: If TLS material is missing or
                unusable and ``insecure`` was not requested.
        """
        web = _aiohttp_web()
        if self._runner is not None:
            raise ServeError("server already started")

        context = None
        if not self.insecure:
            context = self._ssl_context
            if context is None:
                from station.serve.certs import ssl_context  # noqa: PLC0415

                context = ssl_context(self.cfg.stream.cert, self.cfg.stream.key)
        else:
            log.warning(
                "serving plain HTTP: tablets will get neither WebRTC nor a service worker on "
                "this origin. Development only."
            )

        app = build_app(
            self.cfg,
            streamer=self.streamer,
            pipeline=self.pipeline,
            app_dir=self.app_dir,
            signaling_prefix=self.signaling_prefix,
            allow_origin=self.allow_origin,
            manage_streamer=self.manage_streamer,
            extra_config=self._extra_config,
        )
        runner = web.AppRunner(app, access_log=None)
        await runner.setup()
        site = web.TCPSite(
            runner,
            host=self.cfg.stream.bind,
            port=int(self.cfg.stream.port),
            ssl_context=context,
            # A station restarted after a crash must be able to rebind at once
            # rather than waiting out TIME_WAIT with a dozen tablets retrying.
            reuse_address=True,
        )
        try:
            await site.start()
        except OSError as exc:
            await runner.cleanup()
            raise ServeError(
                f"cannot bind {self.cfg.stream.bind}:{self.cfg.stream.port}: {exc}\n"
                "Another station may already be running, or the port needs privileges. "
                "Change stream.port in the config."
            ) from exc

        self._runner, self._site = runner, site
        scheme = "http" if self.insecure else "https"
        self._info = ServerInfo(
            scheme=scheme,
            host=self.cfg.stream.bind,
            port=int(self.cfg.stream.port),
            app_dir=self.app_dir,
            urls=station_urls(self.cfg, scheme=scheme),
        )
        log.info("%s", self._info.describe())
        return self._info

    async def stop(self) -> None:
        """Stop serving and release the port. Idempotent."""
        runner, self._runner, self._site = self._runner, None, None
        if runner is None:
            return
        await runner.cleanup()
        log.info("http listener stopped")

    async def __aenter__(self) -> "StationServer":
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.stop()


def _aiohttp_web() -> Any:
    """Import ``aiohttp.web`` with an actionable error.

    Returns:
        The ``aiohttp.web`` module.

    Raises:
        ServeError: If aiohttp is not installed.
    """
    try:
        from aiohttp import web  # noqa: PLC0415 -- lazy by design; see module docstring.
    except ImportError as exc:
        raise ServeError(
            "aiohttp is not installed, so the station cannot serve the app or the signalling "
            "endpoint.\n"
            "  pip install aiohttp\n"
            "Run 'python -m station check' to see which of the optional dependencies this "
            "laptop is missing."
        ) from exc
    return web
