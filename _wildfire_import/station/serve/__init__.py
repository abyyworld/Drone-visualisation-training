"""What the ground station puts on the LAN: HTTPS, the PWA, and its own cert.

The tablets get everything from one HTTPS origin -- the app, the overlay
parameters, the signalling endpoint -- and the certificate that origin is
served under is generated here, because on an incident LAN there is no CA, no
DNS and frequently no internet.

* :mod:`station.serve.http` -- the aiohttp listener. Serves ``app/`` with types
  and cache headers a PWA can actually be installed and updated from, mounts
  ``station.stream.signaling``, and answers ``/config.json`` and ``/healthz``.
* :mod:`station.serve.certs` -- issues and reuses the station's self-signed
  certificate, with the LAN IPs and a ``.local`` name in the SAN. Not optional:
  browsers gate WebRTC and service workers behind a secure context.

Neither aiohttp nor cryptography is imported at module scope, so this package
imports on a laptop that has neither and ``python -m station check`` can report
which one is missing.

Nothing served from here reports on the scene. ``/healthz`` reports whether the
station is still looking; see ``station/core/safety.py``.
"""

from __future__ import annotations

from station.serve.certs import (
    DEFAULT_VALIDITY_DAYS,
    RENEW_WITHIN_DAYS,
    CertError,
    CertInfo,
    ensure_cert,
    generate_cert,
    local_hostnames,
    local_ip_addresses,
    mdns_name,
    read_cert_info,
    ssl_context,
)
from station.serve.http import (
    DEFAULT_APP_DIR,
    MIME_TYPES,
    ServeError,
    ServerInfo,
    StationServer,
    build_app,
    client_config_payload,
    default_app_dir,
    guess_content_type,
    health_payload,
    station_urls,
)

__all__ = [
    # certs
    "DEFAULT_VALIDITY_DAYS",
    "RENEW_WITHIN_DAYS",
    "CertError",
    "CertInfo",
    "ensure_cert",
    "generate_cert",
    "local_hostnames",
    "local_ip_addresses",
    "mdns_name",
    "read_cert_info",
    "ssl_context",
    # http
    "DEFAULT_APP_DIR",
    "MIME_TYPES",
    "ServeError",
    "ServerInfo",
    "StationServer",
    "build_app",
    "client_config_payload",
    "default_app_dir",
    "guess_content_type",
    "health_payload",
    "station_urls",
]
