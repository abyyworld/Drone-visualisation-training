"""The other way to get video to the tablets: run MediaMTX.

MediaMTX (https://github.com/bluenviron/mediamtx) is a single static Go binary
that takes RTSP in and serves WebRTC (WHEP) out, with no dependencies and no
Python in the media path. For the *video* half of wildfire-watch it is very
likely all that is needed, and this module exists because pretending otherwise
would be dishonest engineering.

Which one to use
----------------
**MediaMTX** when:

* the video path is what matters and it must not fall over -- it is a mature,
  widely deployed relay, it will out-perform and out-survive an aiortc pipeline
  written for this project, and it costs one config file;
* you need RTSP/RTMP/SRT ingest variants, several publishers, or a re-publish
  point for a recorder;
* the station laptop is CPU-bound and you want packets forwarded rather than
  decoded and re-encoded per viewer -- MediaMTX passes the drone's existing
  H.264 through untouched, which is a large saving with several tablets;
* aiortc will not install (no wheels for the platform, no build tools).

**aiortc** (:mod:`station.stream.webrtc`) when the *overlay* matters, which on
this project is most of the time:

* the ``detections`` data channel is per-peer and lives inside the same peer
  connection as the video. MediaMTX's WHEP endpoint carries media only, so a
  MediaMTX deployment needs a second, separate channel to the tablets
  (websocket from ``station/serve``) and then has to synchronise two transports
  that share no clock;
* ``rtp_ts`` -- tier 1 of the overlay synchronisation algorithm in
  ``docs/CONTRACT.md`` -- needs the RTP timestamp the browser will actually see.
  MediaMTX re-stamps as it relays and does not report what it wrote, so that
  tier is simply unavailable: payloads carry ``rtp_ts: null`` and the tablet
  falls back to the pts-offset estimator. That fallback is documented, labelled
  on screen and good to a few tens of milliseconds -- but tier 1 is exact, and
  on a scene moving at 15 m/s the difference is metres on the ground.

**Recommendation for this system: aiortc.** The overlay is the product. Video
without boxes is what a plain RTSP viewer already gives, and the accuracy of the
box position is the whole reason for the three-tier algorithm. Keep MediaMTX as
the fallback that is *already configured* (``station/stream/mediamtx.yml``) so
that a station whose aiortc path fails at an incident is one command away from
plain low-latency video, which is a safe state: the operator is watching the
video regardless. Running both at once is reasonable and cheap -- MediaMTX on
8889 as the always-works video path, aiortc on the station port as the one with
the overlay.

Everything here is stdlib: ``subprocess`` and ``urllib``. No new dependency for
a fallback whose entire point is that it works when things are not going well.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

__all__ = [
    "MediaMtxError",
    "MediaMtxNotFoundError",
    "MediaMtxStatus",
    "MediaMtx",
    "DEFAULT_API_ADDRESS",
    "DEFAULT_WEBRTC_PORT",
    "DEFAULT_RTSP_PORT",
    "DEFAULT_PATH_NAME",
    "SOURCE_ENV_VAR",
    "find_binary",
    "default_config_path",
    "whep_url",
]

log = logging.getLogger(__name__)

#: Must match ``apiAddress`` in ``mediamtx.yml``. Bound to loopback there: the
#: API can reconfigure paths at runtime and has no authentication by default,
#: and the incident WiFi is not a network to expose that on.
DEFAULT_API_ADDRESS = "127.0.0.1:9997"

#: Must match ``webrtcAddress`` / ``rtspAddress`` in ``mediamtx.yml``.
DEFAULT_WEBRTC_PORT = 8889
DEFAULT_RTSP_PORT = 8554

#: The path name configured in ``mediamtx.yml`` for the drone feed.
DEFAULT_PATH_NAME = "drone"

#: ``mediamtx.yml`` refers to the drone's RTSP URL through this environment
#: variable, which MediaMTX expands itself. Templating the YAML from Python
#: would mean a generated file on disk that no longer matches the one in the
#: repository -- and the operator debugging at 3am would be reading the wrong
#: one.
SOURCE_ENV_VAR = "WILDFIRE_RTSP_SOURCE"

#: Candidate binary names/locations, in the order they are tried.
_BINARY_CANDIDATES = (
    "mediamtx",
    "/usr/local/bin/mediamtx",
    "/opt/mediamtx/mediamtx",
    "./mediamtx",
)


class MediaMtxError(RuntimeError):
    """MediaMTX could not be started, or would not become healthy."""


class MediaMtxNotFoundError(MediaMtxError):
    """The MediaMTX binary is not on this machine."""


@dataclass(frozen=True, slots=True)
class MediaMtxStatus:
    """One health probe of a running MediaMTX.

    Attributes:
        running: The child process is alive.
        api_reachable: The HTTP API answered.
        path_ready: The configured path has a publisher and is servable. False
            means no video is flowing -- which is a fact about the relay, not
            about the scene.
        readers: How many viewers are attached to the path.
        bytes_received: Bytes ingested on the path since it went ready.
        detail: Human-readable summary for the operator log.
    """

    running: bool
    api_reachable: bool
    path_ready: bool
    readers: int = 0
    bytes_received: int = 0
    detail: str = ""

    @property
    def healthy(self) -> bool:
        """Whether the relay is up *and* actually carrying the feed."""
        return self.running and self.api_reachable and self.path_ready


def find_binary(explicit: str | os.PathLike[str] | None = None) -> str:
    """Locate the MediaMTX binary.

    Args:
        explicit: A path to use instead of searching. Checked for existence so
            a typo in the config fails here rather than as a confusing
            ``FileNotFoundError`` from ``subprocess``.

    Returns:
        An absolute path to the binary.

    Raises:
        MediaMtxNotFoundError: If nothing usable was found.
    """
    if explicit is not None:
        candidate = Path(explicit).expanduser()
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate.resolve())
        raise MediaMtxNotFoundError(
            f"mediamtx binary not found or not executable at {candidate}"
        )
    for name in _BINARY_CANDIDATES:
        found = shutil.which(name)
        if found:
            return str(Path(found).resolve())
        path = Path(name).expanduser()
        if path.is_file() and os.access(path, os.X_OK):
            return str(path.resolve())
    raise MediaMtxNotFoundError(
        "mediamtx binary not found. It is a single static binary with no dependencies:\n"
        "  download the release for this platform from "
        "https://github.com/bluenviron/mediamtx/releases,\n"
        "  put it on PATH (or pass binary=... explicitly), and keep a copy in the "
        "station's kit -- the fire ground has no internet."
    )


def default_config_path() -> Path:
    """Path to the ``mediamtx.yml`` shipped alongside this module."""
    return Path(__file__).with_name("mediamtx.yml")


def whep_url(
    host: str,
    *,
    path: str = DEFAULT_PATH_NAME,
    port: int = DEFAULT_WEBRTC_PORT,
    tls: bool = False,
) -> str:
    """Build the WHEP URL a tablet plays.

    Args:
        host: Station address as the tablet reaches it.
        path: MediaMTX path name.
        port: MediaMTX WebRTC port.
        tls: Whether MediaMTX is serving WebRTC over HTTPS.

    Returns:
        The WHEP endpoint URL.

    Note:
        A PWA served over HTTPS cannot fetch an ``http://`` WHEP endpoint --
        browsers block it as mixed content, silently as far as the operator is
        concerned. Either enable TLS in ``mediamtx.yml`` or reverse-proxy this
        endpoint through ``station/serve``, which already holds the certificate.
    """
    scheme = "https" if tls else "http"
    return f"{scheme}://{host}:{port}/{path.strip('/')}/whep"


class MediaMtx:
    """Launches and health-checks a MediaMTX process.

    Example:
        >>> relay = MediaMtx(source="rtsp://192.168.144.25:8554/main.264")  # doctest: +SKIP
        >>> relay.start()                                                   # doctest: +SKIP
        >>> relay.wait_until_ready(timeout_s=15)                            # doctest: +SKIP
        >>> relay.stop()                                                    # doctest: +SKIP

    Or as a context manager, which is the form that cannot leak the child::

        with MediaMtx(source=uri) as relay:
            relay.wait_until_ready()
            ...
    """

    def __init__(
        self,
        *,
        source: str | None = None,
        binary: str | os.PathLike[str] | None = None,
        config: str | os.PathLike[str] | None = None,
        path_name: str = DEFAULT_PATH_NAME,
        api_address: str = DEFAULT_API_ADDRESS,
        env: Mapping[str, str] | None = None,
        log_output: bool = True,
    ) -> None:
        """Prepare a relay. Starts nothing.

        Args:
            source: The drone's RTSP URL, exported to MediaMTX as
                ``$WILDFIRE_RTSP_SOURCE``. Omit only if the config does not
                reference it (for a publish-to-us topology).
            binary: Explicit binary path; searched for when omitted.
            config: Explicit config path; the shipped ``mediamtx.yml`` when
                omitted.
            path_name: The path in the config to health-check.
            api_address: ``host:port`` of MediaMTX's HTTP API, matching
                ``apiAddress`` in the config.
            env: Extra environment for the child process.
            log_output: Forward MediaMTX's stdout/stderr into this module's
                logger. Worth having: its startup errors name the exact config
                key it rejected, and a config key that drifted between MediaMTX
                versions is the single most likely reason this will not start.

        Raises:
            MediaMtxNotFoundError: If the binary cannot be located.
            MediaMtxError: If the config file does not exist.
        """
        self.binary = find_binary(binary)
        self.config = Path(config) if config is not None else default_config_path()
        if not self.config.is_file():
            raise MediaMtxError(f"mediamtx config not found: {self.config}")
        self.source = source
        self.path_name = path_name
        self.api_address = api_address
        self.log_output = log_output
        self._env_extra = dict(env or {})
        self._process: subprocess.Popen[str] | None = None
        self._log_thread: Any = None
        #: Tail of the child's output, kept so a startup failure can be
        #: reported with the line that caused it rather than "exit code 1".
        self._recent_output: list[str] = []

    # ------------------------------------------------------------ lifecycle

    @property
    def running(self) -> bool:
        """Whether the child process is alive."""
        return self._process is not None and self._process.poll() is None

    @property
    def pid(self) -> int | None:
        """PID of the child, or ``None`` when not running."""
        return self._process.pid if self._process is not None else None

    def start(self) -> None:
        """Launch MediaMTX. Idempotent while it is running.

        Raises:
            MediaMtxError: If the process cannot be spawned, or exits
                immediately -- which on this binary means a rejected config
                key, and the message carries its own output so the operator does
                not have to go looking for it.
        """
        if self.running:
            return

        env = dict(os.environ)
        if self.source:
            env[SOURCE_ENV_VAR] = self.source
        env.update(self._env_extra)

        log.info("starting mediamtx: %s %s (source=%s)", self.binary, self.config, self.source or "-")
        try:
            self._process = subprocess.Popen(  # noqa: S603 -- path validated by find_binary
                [self.binary, str(self.config)],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                # Own process group, so a Ctrl-C in the station's terminal does
                # not race us to the child: we stop it deliberately in stop().
                start_new_session=True,
            )
        except OSError as exc:
            raise MediaMtxError(f"could not start {self.binary}: {exc}") from exc

        if self.log_output:
            self._start_log_pump()

        # A rejected config key kills MediaMTX in well under a second. Catching
        # it here turns "the tablets show nothing" into a startup error naming
        # the key.
        time.sleep(0.3)
        if not self.running:
            output = self._drain_output()
            raise MediaMtxError(
                f"mediamtx exited immediately (code {self._process.returncode if self._process else '?'}).\n"
                f"{output or 'no output captured'}\n"
                "A rejected key usually means this config targets a different MediaMTX "
                "version; check the header comment in mediamtx.yml."
            )

    def stop(self, timeout_s: float = 5.0) -> None:
        """Stop MediaMTX, escalating to SIGKILL. Safe to call twice."""
        process = self._process
        if process is None:
            return
        if process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                log.warning("mediamtx did not exit in %.0fs; killing it", timeout_s)
                try:
                    process.kill()
                    process.wait(timeout=timeout_s)
                except Exception:  # pragma: no cover
                    log.debug("mediamtx kill failed", exc_info=True)
            except (OSError, ValueError):  # pragma: no cover
                log.debug("mediamtx terminate failed", exc_info=True)
        self._process = None
        log.info("mediamtx stopped")

    def __enter__(self) -> "MediaMtx":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.stop()

    # -------------------------------------------------------------- health

    def status(self, timeout_s: float = 2.0) -> MediaMtxStatus:
        """Probe the relay once.

        Args:
            timeout_s: HTTP timeout for the API call.

        Returns:
            A :class:`MediaMtxStatus`. Never raises: this is called on a timer
            by whatever is supervising the station, and a health check that can
            throw is a health check that takes the supervisor down with it.
        """
        if not self.running:
            return MediaMtxStatus(False, False, False, detail="mediamtx is not running")
        try:
            payload = self._api_get("/v3/paths/list", timeout_s=timeout_s)
        except MediaMtxError as exc:
            return MediaMtxStatus(True, False, False, detail=str(exc))

        items: Iterable[Mapping[str, Any]] = payload.get("items") or ()
        for item in items:
            if item.get("name") != self.path_name:
                continue
            ready = bool(item.get("ready"))
            readers = len(item.get("readers") or ())
            received = int(item.get("bytesReceived") or 0)
            return MediaMtxStatus(
                running=True,
                api_reachable=True,
                path_ready=ready,
                readers=readers,
                bytes_received=received,
                detail=(
                    f"path {self.path_name!r} ready, {readers} reader(s), {received} bytes in"
                    if ready
                    else f"path {self.path_name!r} exists but has no publisher yet"
                ),
            )
        return MediaMtxStatus(
            running=True,
            api_reachable=True,
            path_ready=False,
            detail=f"path {self.path_name!r} is not configured in {self.config}",
        )

    def wait_until_ready(self, timeout_s: float = 20.0, poll_s: float = 0.5) -> MediaMtxStatus:
        """Block until the path is carrying video, or give up.

        Args:
            timeout_s: How long to wait in total.
            poll_s: Interval between probes.

        Returns:
            The healthy :class:`MediaMtxStatus`.

        Raises:
            MediaMtxError: On timeout, or if the process dies while waiting.
                The message carries the last probe detail, because "not ready"
                after 20 s of a drone RTSP feed usually means a wrong URL, a
                wrong transport (UDP where the link needs TCP) or a firewall,
                and the detail is what tells them apart.
        """
        deadline = time.monotonic() + timeout_s
        last = MediaMtxStatus(False, False, False, detail="not probed")
        while time.monotonic() < deadline:
            last = self.status()
            if last.healthy:
                log.info("mediamtx ready: %s", last.detail)
                return last
            if not last.running:
                raise MediaMtxError(f"mediamtx exited while starting up: {self._drain_output() or last.detail}")
            time.sleep(poll_s)
        raise MediaMtxError(
            f"mediamtx did not become ready within {timeout_s:.0f}s: {last.detail}. "
            f"Check that {SOURCE_ENV_VAR}={self.source!r} is reachable and that the path's "
            "rtspTransport matches the link (tcp for a lossy radio downlink)."
        )

    def paths(self, timeout_s: float = 2.0) -> list[dict[str, Any]]:
        """Every path MediaMTX knows about, as reported by its API.

        Raises:
            MediaMtxError: If the API is unreachable.
        """
        payload = self._api_get("/v3/paths/list", timeout_s=timeout_s)
        return list(payload.get("items") or ())

    def whep_url(self, host: str, *, tls: bool = False, port: int = DEFAULT_WEBRTC_PORT) -> str:
        """WHEP URL for this relay's path. See :func:`whep_url`."""
        return whep_url(host, path=self.path_name, port=port, tls=tls)

    # ----------------------------------------------------------- internals

    def _api_get(self, route: str, timeout_s: float = 2.0) -> dict[str, Any]:
        """GET one JSON document from the MediaMTX API."""
        url = f"http://{self.api_address}{route}"
        try:
            with urllib.request.urlopen(url, timeout=timeout_s) as response:  # noqa: S310 -- loopback, fixed scheme
                body = response.read()
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            raise MediaMtxError(f"mediamtx API unreachable at {url}: {exc}") from exc
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise MediaMtxError(f"mediamtx API returned non-JSON from {url}: {exc}") from exc
        if not isinstance(payload, dict):
            raise MediaMtxError(f"mediamtx API returned {type(payload).__name__} from {url}, expected an object")
        return payload

    def _start_log_pump(self) -> None:
        """Forward the child's output into the station log, in a thread."""
        import threading  # noqa: PLC0415 -- only needed when log_output is on

        process = self._process
        if process is None or process.stdout is None:
            return

        def pump() -> None:
            try:
                for line in process.stdout:  # type: ignore[union-attr]
                    self._recent_output.append(line.rstrip())
                    del self._recent_output[:-40]
                    log.info("mediamtx: %s", line.rstrip())
            except Exception:  # pragma: no cover - the pipe closes on exit
                pass

        self._log_thread = threading.Thread(target=pump, name="mediamtx-log", daemon=True)
        self._log_thread.start()

    def _drain_output(self) -> str:
        """Best-effort recent output from the child, for an error message."""
        if self._recent_output:
            return "\n".join(self._recent_output[-20:])
        process = self._process
        if process is not None and process.stdout is not None and not self.log_output:
            try:
                return process.stdout.read() or ""
            except Exception:  # pragma: no cover
                return ""
        return ""
