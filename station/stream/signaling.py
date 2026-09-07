"""Minimal HTTP signalling for the WebRTC path.

WebRTC needs exactly one thing from an out-of-band channel: get the tablet's
SDP offer to the station and the station's answer back. That is one POST. There
is no room and no need here for a signalling server, a websocket protocol or a
room abstraction -- the station and the tablets are on the same LAN, which may
have no internet at all, and every moving part is a moving part that can fail on
a fire ground at 3am.

Non-trickle by design. aiortc gathers its ICE candidates before
``setLocalDescription`` returns, and with host candidates only (the default --
see ``StreamConfig.ice_servers``) that takes a few milliseconds on a LAN. So the
answer is complete when it is sent, and the tablet needs no candidate exchange,
no second endpoint and no reconnect logic for one.

Designed to be **mounted**, not run: ``station/serve`` owns the HTTPS listener
that serves the PWA, and these routes go on the same origin so the tablet needs
no CORS, no second certificate and no second port to trust.
:func:`create_app` and :func:`run_standalone` exist for bench testing the
transport on its own.

aiohttp is imported inside the functions that need it, so this module imports
cleanly on a machine that has none of the streaming stack installed.

Endpoints (relative to ``prefix``, default ``/webrtc``):

===========================  ======================================================
``POST /offer``              body ``{"sdp": ..., "type": "offer"}`` ->
                             ``{"sdp", "type", "peer_id", "wire_version", ...}``
``POST /close``              body ``{"peer_id": ...}``; also accepted as a
                             ``navigator.sendBeacon`` body on unload
``GET  /config``             the overlay parameters, without opening a peer
``GET  /peers``              per-peer diagnostics for the operator
===========================  ======================================================

Nothing served from here reports on the contents of the scene. ``/peers`` and
``/config`` describe the station: whether it is streaming, to whom, and how the
overlay is synchronised. See ``station/core/safety.py``.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from station.stream.webrtc import StreamDependencyError, StreamError, WebRtcStreamer

__all__ = [
    "SignalingError",
    "DEFAULT_PREFIX",
    "MAX_BODY_BYTES",
    "add_routes",
    "create_app",
    "client_config",
    "run_standalone",
]

log = logging.getLogger(__name__)

#: Mount point. ``station/serve`` may override it; the tablet reads it from the
#: page it was served, so it is not hard-coded on both sides.
DEFAULT_PREFIX = "/webrtc"

#: An SDP offer is a few kilobytes. Anything larger is a mistake or an attack,
#: and reading it into memory before finding out is the mistake worth avoiding
#: on a laptop that is also running a GPU model.
MAX_BODY_BYTES = 64 * 1024


class SignalingError(RuntimeError):
    """Raised when the signalling layer cannot be set up."""


def client_config(streamer: WebRtcStreamer) -> dict[str, Any]:
    """Parameters the tablet needs before, or without, opening a peer.

    Args:
        streamer: The streamer whose configuration to describe.

    Returns:
        The overlay parameters and wire version, as JSON-ready data.
    """
    return {"station": True, **streamer.client_parameters()}


def add_routes(
    app: Any,
    streamer: WebRtcStreamer,
    *,
    prefix: str = DEFAULT_PREFIX,
    allow_origin: str | None = None,
) -> Any:
    """Mount the signalling routes on an existing aiohttp application.

    Args:
        app: An ``aiohttp.web.Application``.
        streamer: The streamer that will serve the peers.
        prefix: Path prefix for the routes, without a trailing slash.
        allow_origin: Value for ``Access-Control-Allow-Origin``. Leave ``None``
            in production: the PWA is served from this same origin, so CORS is
            not needed and not enabling it is one fewer way for a stray browser
            tab on the incident WiFi to open peers on the station. Set it (for
            example to ``"*"``) only when developing the PWA off a separate dev
            server.

    Returns:
        The application, so calls can be chained.

    Raises:
        SignalingError: If aiohttp is not installed.
    """
    web = _aiohttp_web()
    prefix = prefix.rstrip("/")

    def _json(payload: dict[str, Any], status: int = 200) -> Any:
        response = web.json_response(payload, status=status)
        # Signalling answers are single-use and peer-specific. A cached offer
        # response would hand a second tablet the first tablet's peer_id.
        response.headers["Cache-Control"] = "no-store"
        if allow_origin:
            response.headers["Access-Control-Allow-Origin"] = allow_origin
        return response

    async def _body(request: Any) -> dict[str, Any]:
        """Read and parse a bounded JSON body.

        Raises:
            SignalingError: If the body is too large or is not a JSON object.
        """
        raw = await request.content.read(MAX_BODY_BYTES + 1)
        if len(raw) > MAX_BODY_BYTES:
            raise SignalingError(f"request body exceeds {MAX_BODY_BYTES} bytes")
        try:
            payload = json.loads(raw.decode("utf-8") or "{}")
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SignalingError(f"body is not valid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise SignalingError("body must be a JSON object")
        return payload

    async def offer(request: Any) -> Any:
        """Answer one tablet's offer."""
        try:
            payload = await _body(request)
        except SignalingError as exc:
            return _json({"error": str(exc)}, status=400)

        sdp = payload.get("sdp")
        offer_type = payload.get("type", "offer")
        if not isinstance(sdp, str) or not isinstance(offer_type, str):
            return _json({"error": "expected string fields 'sdp' and 'type'"}, status=400)

        try:
            answer = await streamer.handle_offer(sdp, offer_type, peer_id=payload.get("peer_id"))
        except StreamDependencyError as exc:
            # 501, not 500: the station is running fine, it just cannot speak
            # WebRTC on this install. The tablet shows the operator a message
            # that names the missing package instead of "server error".
            log.error("cannot serve WebRTC: %s", exc)
            return _json({"error": str(exc)}, status=501)
        except StreamError as exc:
            log.warning("rejected an offer: %s", exc)
            return _json({"error": str(exc)}, status=409)
        except Exception as exc:  # pragma: no cover - unexpected aiortc failure
            log.exception("offer handling failed")
            return _json({"error": f"negotiation failed: {exc}"}, status=500)
        return _json(answer)

    async def close_peer(request: Any) -> Any:
        """Drop a peer the tablet says it is finished with.

        Tablets should call this from ``pagehide`` via ``navigator.sendBeacon``.
        It is an optimisation, not a requirement -- the connection watchdog and
        the ICE failure handler in :mod:`station.stream.webrtc` clean up peers
        that never say goodbye, which in the field is most of them.
        """
        try:
            payload = await _body(request)
        except SignalingError as exc:
            return _json({"error": str(exc)}, status=400)
        peer_id = payload.get("peer_id")
        if not isinstance(peer_id, str) or not peer_id:
            return _json({"error": "expected a string 'peer_id'"}, status=400)
        closed = await streamer.close_peer(peer_id)
        return _json({"closed": closed, "peer_id": peer_id})

    async def config(request: Any) -> Any:
        """Overlay parameters, fetchable before a peer exists."""
        return _json(client_config(streamer))

    async def peers(request: Any) -> Any:
        """Per-peer diagnostics for the operator at the station laptop."""
        return _json(
            {
                "peers": streamer.describe_peers(),
                "stats": {
                    "peers": len(streamer.peers),
                    "frames_published": streamer.stats.frames_published,
                    "detections_published": streamer.stats.detections_published,
                    "statuses_published": streamer.stats.statuses_published,
                    "peers_accepted": streamer.stats.peers_accepted,
                    "peers_rejected": streamer.stats.peers_rejected,
                    "peers_closed": streamer.stats.peers_closed,
                    "repaired_timestamps": streamer.stats.repaired_timestamps,
                },
            }
        )

    async def preflight(request: Any) -> Any:
        """CORS preflight, only meaningful when ``allow_origin`` is set."""
        response = web.Response(status=204)
        if allow_origin:
            response.headers["Access-Control-Allow-Origin"] = allow_origin
            response.headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return response

    app.router.add_post(f"{prefix}/offer", offer)
    app.router.add_post(f"{prefix}/close", close_peer)
    app.router.add_get(f"{prefix}/config", config)
    app.router.add_get(f"{prefix}/peers", peers)
    if allow_origin:
        for path in ("offer", "close", "config", "peers"):
            app.router.add_options(f"{prefix}/{path}", preflight)

    log.info("signalling mounted at %s/offer", prefix)
    return app


def create_app(
    streamer: WebRtcStreamer,
    *,
    prefix: str = DEFAULT_PREFIX,
    allow_origin: str | None = None,
    start_streamer: bool = True,
) -> Any:
    """Build a standalone aiohttp application serving only the signalling.

    For bench-testing the transport. In production ``station/serve`` builds the
    application (it also serves the PWA) and calls :func:`add_routes` on it.

    Args:
        streamer: The streamer to serve.
        prefix: Path prefix for the routes.
        allow_origin: See :func:`add_routes`.
        start_streamer: Start and stop the streamer with the application. The
            streamer must be started on the loop that will run it, which is why
            this is wired to the app's lifecycle rather than done by the caller.

    Returns:
        An ``aiohttp.web.Application``.

    Raises:
        SignalingError: If aiohttp is not installed.
    """
    web = _aiohttp_web()
    app = web.Application(client_max_size=MAX_BODY_BYTES)
    add_routes(app, streamer, prefix=prefix, allow_origin=allow_origin)

    if start_streamer:
        async def _start(_app: Any) -> None:
            await streamer.start()

        app.on_startup.append(_start)

    async def _cleanup(_app: Any) -> None:
        # Unconditional: even a streamer someone else started must have its
        # peers torn down when the listener goes away, or the peers outlive the
        # thing that was feeding them and sit there holding encoders.
        await streamer.close()

    app.on_cleanup.append(_cleanup)
    return app


def run_standalone(
    streamer: WebRtcStreamer,
    *,
    host: str = "0.0.0.0",
    port: int = 8443,
    prefix: str = DEFAULT_PREFIX,
    allow_origin: str | None = None,
    ssl_context: Any = None,
) -> None:
    """Run the signalling application until interrupted. Blocking.

    Args:
        streamer: The streamer to serve.
        host: Bind address.
        port: Bind port.
        prefix: Path prefix for the routes.
        allow_origin: See :func:`add_routes`.
        ssl_context: An ``ssl.SSLContext``. Strongly recommended: browsers
            treat a plain-HTTP origin as insecure, and a PWA served over HTTPS
            cannot POST to an HTTP signalling endpoint at all (mixed content).
            ``StreamConfig.cert`` / ``.key`` are where the station's certificate
            lives.

    Raises:
        SignalingError: If aiohttp is not installed.
    """
    web = _aiohttp_web()
    app = create_app(streamer, prefix=prefix, allow_origin=allow_origin)
    scheme = "https" if ssl_context is not None else "http"
    log.info("signalling listening on %s://%s:%d%s/offer", scheme, host, port, prefix.rstrip("/"))
    web.run_app(app, host=host, port=port, ssl_context=ssl_context, print=None)


def _aiohttp_web() -> Any:
    """Import ``aiohttp.web`` with an actionable error.

    Returns:
        The ``aiohttp.web`` module.

    Raises:
        SignalingError: If aiohttp is not installed.
    """
    try:
        from aiohttp import web  # noqa: PLC0415 -- lazy by design; see module docstring.
    except ImportError as exc:
        raise SignalingError(
            "aiohttp is not installed, so the station cannot serve signalling.\n"
            "  pip install aiohttp\n"
            "Only the HTTP offer/answer exchange needs it; the WebRTC transport itself "
            "is aiortc (station.stream.webrtc)."
        ) from exc
    return web
