"""Self-signed TLS for the incident LAN.

This module is load-bearing, not hygiene. Browsers gate three things this
system cannot work without behind a *secure context*:

* ``getUserMedia`` / ``RTCPeerConnection`` -- no WebRTC on a plain-HTTP origin
  in any current mobile browser;
* ``navigator.serviceWorker.register`` -- no service worker, so no offline PWA
  on a fire ground with no cell coverage;
* mixed content -- an HTTPS page cannot POST an SDP offer to an HTTP endpoint.

``http://localhost`` is exempt from all three, which is why the station laptop
itself can be developed against without a certificate and a tablet cannot. On
the LAN there is no CA, frequently no internet and often no DNS, so the station
issues its own certificate and the tablets are told to trust it once.

Two details are the difference between a certificate that works on a tablet and
one that is rejected before the operator sees anything:

1. **IP addresses must be in the SAN as ``iPAddress`` entries.** Tablets reach
   the station by typing an IP, because there is no DNS on the incident WiFi.
   Safari on iOS rejects a certificate with no matching SAN entry outright --
   it does not offer the "proceed anyway" escape hatch that desktop Chrome
   does, and it has ignored the legacy Common Name field since iOS 13.
2. **Validity must be short.** Apple refuses server certificates whose validity
   exceeds 398 days (issued on or after 2020-09-01), regardless of who signed
   them. :data:`DEFAULT_VALIDITY_DAYS` sits inside that limit.

The certificate is generated with ``CA:TRUE`` and ``keyCertSign`` so that it is
a well-formed trust anchor when an operator installs it on a tablet: iOS will
only enable full trust (Settings -> General -> About -> Certificate Trust
Settings) for a certificate that is a valid root, and a self-signed leaf with
``CA:FALSE`` never gets that switch. It is then served directly as the leaf,
which every browser TLS stack accepts.

``cryptography`` is imported inside the functions that need it, so
``station.serve.certs`` imports on a machine that does not have it and the
``check`` subcommand can report its absence rather than crashing on it.

Nothing here reports on the contents of the scene; it reports on a file on
disk.
"""

from __future__ import annotations

import ipaddress
import logging
import os
import re
import socket
import ssl
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

__all__ = [
    "DEFAULT_VALIDITY_DAYS",
    "RENEW_WITHIN_DAYS",
    "CertError",
    "CertInfo",
    "ensure_cert",
    "generate_cert",
    "read_cert_info",
    "local_ip_addresses",
    "local_hostnames",
    "mdns_name",
    "ssl_context",
]

log = logging.getLogger(__name__)

#: Inside Apple's 398-day ceiling with a month of margin, so a station that
#: sits in a cupboard between deployments does not hand out a certificate that
#: expires the week it is next needed.
DEFAULT_VALIDITY_DAYS = 365

#: Regenerate rather than reuse when this little life remains. A certificate
#: that expires mid-incident is a fleet of tablets that stop connecting at
#: once, and nobody is going to debug TLS at 3am.
RENEW_WITHIN_DAYS = 30

#: Backdate ``notBefore``. Ground stations run on isolated LANs with no NTP,
#: and a laptop whose clock is twenty minutes slow would otherwise reject the
#: certificate it just generated itself.
_CLOCK_SKEW = timedelta(hours=2)

#: TEST-NET-1 (RFC 5737). Connecting a UDP socket to it sends no packets and
#: needs no route to exist beyond the default one -- it just asks the kernel
#: which local address it *would* use, which is the address tablets will reach.
_ROUTE_PROBE = ("192.0.2.1", 9)

_SLUG_RE = re.compile(r"[^a-z0-9-]+")


class CertError(RuntimeError):
    """The certificate could not be generated, read or loaded."""


@dataclass(frozen=True, slots=True)
class CertInfo:
    """What is actually in the certificate on disk."""

    cert_path: Path
    key_path: Path
    common_name: str
    not_before: datetime
    not_after: datetime
    dns_names: tuple[str, ...]
    ip_addresses: tuple[str, ...]
    #: Colon-separated uppercase SHA-256 of the DER form. This is the string an
    #: operator compares against the one the tablet shows when it asks whether
    #: to trust the certificate, so it must be printable and stable.
    fingerprint_sha256: str
    #: True when the call that returned this object wrote a new certificate.
    created: bool = False

    @property
    def days_remaining(self) -> float:
        """Days until expiry; negative once expired."""
        return (self.not_after - datetime.now(timezone.utc)).total_seconds() / 86400.0

    @property
    def expired(self) -> bool:
        """Whether the certificate is outside its validity window right now."""
        now = datetime.now(timezone.utc)
        return not (self.not_before <= now <= self.not_after)

    def covers(self, dns_names: Iterable[str], ip_addresses: Iterable[str]) -> bool:
        """Whether every requested name and address is in the SAN.

        Args:
            dns_names: DNS names the station wants to be reachable as.
            ip_addresses: IP addresses the station wants to be reachable at.

        Returns:
            ``True`` when nothing is missing. A missing address is what makes a
            tablet refuse the connection, so this is the check that decides
            whether an existing certificate can be reused.
        """
        have_dns = {n.lower() for n in self.dns_names}
        have_ip = set(self.ip_addresses)
        return all(n.lower() in have_dns for n in dns_names) and all(a in have_ip for a in ip_addresses)

    def describe(self) -> str:
        """Multi-line, operator-facing summary for the CLI."""
        names = ", ".join(self.dns_names) or "(none)"
        addrs = ", ".join(self.ip_addresses) or "(none)"
        return (
            f"certificate : {self.cert_path}\n"
            f"private key : {self.key_path}\n"
            f"common name : {self.common_name}\n"
            f"valid       : {self.not_before:%Y-%m-%d %H:%M} .. {self.not_after:%Y-%m-%d %H:%M} UTC "
            f"({self.days_remaining:.0f} days remaining)\n"
            f"dns names   : {names}\n"
            f"ip addresses: {addrs}\n"
            f"sha256      : {self.fingerprint_sha256}"
        )


# ---------------------------------------------------------------- discovery


def local_ip_addresses(*, include_loopback: bool = True) -> tuple[str, ...]:
    """Every local IP address a tablet might reach this station on.

    Three independent probes are unioned, because no single one is reliable on
    every laptop: the route probe misses secondary interfaces, the hostname
    lookup misses machines whose hostname does not resolve, and the interface
    scan is Linux-only. A missing address here becomes a certificate a tablet
    rejects, so over-collecting is the right error to make.

    Args:
        include_loopback: Include ``127.0.0.1`` and ``::1``. Wanted in the
            certificate so the operator can open the PWA on the station laptop
            itself over the same HTTPS origin the tablets use.

    Returns:
        Addresses as strings, route-probe result first (it is the one the
        operator will read off the banner and type into a tablet), then the
        rest in discovery order, deduplicated.
    """
    found: list[str] = []

    def add(raw: str) -> None:
        try:
            addr = ipaddress.ip_address(raw.split("%", 1)[0])
        except ValueError:
            return
        if addr.is_link_local or addr.is_multicast or addr.is_unspecified:
            # Link-local (169.254/16, fe80::) addresses are not what a tablet
            # is handed by the incident WiFi's DHCP; putting them in the SAN
            # only makes the certificate noisier to audit.
            return
        if addr.is_loopback and not include_loopback:
            return
        text = str(addr)
        if text not in found:
            found.append(text)

    primary = primary_ip_address()
    if primary:
        add(primary)

    try:
        hostname = socket.gethostname()
        for info in socket.getaddrinfo(hostname, None, proto=socket.IPPROTO_TCP):
            add(info[4][0])
    except OSError:
        log.debug("hostname address lookup failed", exc_info=True)

    for addr in _interface_ipv4_addresses():
        add(addr)

    if include_loopback:
        add("127.0.0.1")
        add("::1")
    return tuple(found)


def primary_ip_address() -> str | None:
    """The local address the kernel would use to reach the outside world.

    Returns:
        The address as a string, or ``None`` when the machine has no route at
        all -- which is normal on a genuinely isolated station, and is why the
        callers treat it as one probe among several rather than the answer.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # No packet leaves the machine: connect() on a UDP socket only fixes
        # the local endpoint.
        sock.connect(_ROUTE_PROBE)
        return str(sock.getsockname()[0])
    except OSError:
        return None
    finally:
        sock.close()


def _interface_ipv4_addresses() -> tuple[str, ...]:
    """IPv4 addresses of every up interface, via ``SIOCGIFADDR``. Linux only.

    Returns:
        Addresses found, or an empty tuple on any platform or kernel where the
        ioctl is unavailable. Never raises: this is the optional third probe.
    """
    try:
        import array
        import fcntl
        import struct
    except ImportError:  # pragma: no cover - non-POSIX
        return ()
    if not hasattr(socket, "if_nameindex"):  # pragma: no cover - non-POSIX
        return ()

    siocgifaddr = 0x8915
    out: list[str] = []
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    except OSError:  # pragma: no cover
        return ()
    try:
        for _index, name in socket.if_nameindex():
            try:
                # ioctl() with a mutable buffer rewrites it in place and
                # returns the syscall result, so the answer is read back out of
                # ``buf`` rather than from the return value.
                buf = array.array("B", name.encode("utf-8")[:15].ljust(256, b"\0"))
                fcntl.ioctl(sock.fileno(), siocgifaddr, buf, True)
                # struct ifreq: 16 bytes of interface name, then a sockaddr_in
                # whose 4-byte IPv4 address sits at offset 20.
                out.append(socket.inet_ntoa(struct.unpack_from("4s", buf.tobytes(), 20)[0]))
            except OSError:
                continue  # interface is down or has no IPv4 address
    finally:
        sock.close()
    return tuple(out)


def mdns_name(station_name: str) -> str:
    """A ``.local`` hostname derived from the station's configured name.

    Args:
        station_name: ``Config.station_name``.

    Returns:
        A lowercase RFC-1123-safe label with a ``.local`` suffix, e.g.
        ``"wildfire-watch-station.local"``.

    Note:
        The station does not run an mDNS responder. The name is put in the
        certificate so that it works wherever one already exists (avahi on
        Linux, Bonjour on macOS and iOS, which covers the iPad half of the
        fleet). Where it does not resolve, the IP SANs are what the tablets
        use, which is why both are always included.
    """
    slug = _SLUG_RE.sub("-", station_name.strip().lower()).strip("-")
    slug = re.sub(r"-{2,}", "-", slug) or "wildfire-watch"
    return f"{slug[:63]}.local"


def local_hostnames(station_name: str | None = None) -> tuple[str, ...]:
    """DNS names to put in the certificate.

    Args:
        station_name: ``Config.station_name``, used to derive a ``.local`` name.

    Returns:
        ``localhost``, the machine's hostname, that hostname with a ``.local``
        suffix, and the station's ``.local`` name, deduplicated in that order.
    """
    names: list[str] = ["localhost"]

    def add(name: str) -> None:
        name = name.strip().rstrip(".").lower()
        if name and name not in names:
            names.append(name)

    try:
        host = socket.gethostname()
    except OSError:  # pragma: no cover
        host = ""
    if host:
        add(host)
        if "." not in host:
            add(f"{host}.local")
    if station_name:
        add(mdns_name(station_name))
    return tuple(names)


# ------------------------------------------------------------------- issuing


def ensure_cert(
    cert_path: str | os.PathLike[str],
    key_path: str | os.PathLike[str],
    *,
    station_name: str = "wildfire-watch station",
    dns_names: Sequence[str] | None = None,
    ip_addresses: Sequence[str] | None = None,
    validity_days: int = DEFAULT_VALIDITY_DAYS,
    renew_within_days: float = RENEW_WITHIN_DAYS,
    force: bool = False,
) -> CertInfo:
    """Return a usable certificate, generating one only when needed.

    Idempotent by design: the station calls this on every start, and a fleet of
    tablets that has already been taught to trust the station's certificate must
    not be invalidated just because the station was restarted. A new certificate
    is written only when there is not one, when it is expired or expiring, when
    the key does not match it, or when the station has acquired an address the
    existing certificate does not cover.

    Args:
        cert_path: Where the PEM certificate lives.
        key_path: Where the PEM private key lives. Written ``0600``.
        station_name: Used for the common name and the ``.local`` SAN.
        dns_names: DNS names to cover. Defaults to :func:`local_hostnames`.
        ip_addresses: IP addresses to cover. Defaults to
            :func:`local_ip_addresses`.
        validity_days: Lifetime of a newly generated certificate.
        renew_within_days: Regenerate when less than this much life remains.
        force: Regenerate unconditionally. Every tablet must then be re-taught
            to trust the station, so this is never the automatic path.

    Returns:
        A :class:`CertInfo` whose :attr:`~CertInfo.created` says whether this
        call wrote a new certificate.

    Raises:
        CertError: If ``cryptography`` is missing, or the files cannot be
            written.
    """
    cert_file = Path(cert_path).expanduser()
    key_file = Path(key_path).expanduser()
    wanted_dns = tuple(dns_names) if dns_names is not None else local_hostnames(station_name)
    wanted_ips = tuple(ip_addresses) if ip_addresses is not None else local_ip_addresses()

    if not force:
        reuse = _reusable(cert_file, key_file, wanted_dns, wanted_ips, renew_within_days)
        if reuse is not None:
            log.info(
                "reusing certificate %s (%.0f days remaining, %d SAN entries)",
                cert_file, reuse.days_remaining, len(reuse.dns_names) + len(reuse.ip_addresses),
            )
            return reuse

    return generate_cert(
        cert_file,
        key_file,
        station_name=station_name,
        dns_names=wanted_dns,
        ip_addresses=wanted_ips,
        validity_days=validity_days,
    )


def _reusable(
    cert_file: Path,
    key_file: Path,
    dns_names: Sequence[str],
    ip_addresses: Sequence[str],
    renew_within_days: float,
) -> CertInfo | None:
    """Decide whether the certificate on disk can be kept.

    Returns:
        The existing :class:`CertInfo` when it is valid, matches its key and
        covers every requested name, otherwise ``None`` with the reason logged.
        Any read failure is a ``None``, not an exception: an unreadable or
        corrupt certificate must lead to a new one, never to a station that
        refuses to start.
    """
    if not cert_file.is_file() or not key_file.is_file():
        return None
    try:
        info = read_cert_info(cert_file, key_file)
    except CertError as exc:
        log.warning("existing certificate %s is unusable (%s); generating a new one", cert_file, exc)
        return None
    if info.expired:
        log.warning("certificate %s is outside its validity window; regenerating", cert_file)
        return None
    if info.days_remaining < renew_within_days:
        log.warning(
            "certificate %s expires in %.0f days; regenerating now rather than mid-incident",
            cert_file, info.days_remaining,
        )
        return None
    if not info.covers(dns_names, ip_addresses):
        missing_dns = [n for n in dns_names if n.lower() not in {d.lower() for d in info.dns_names}]
        missing_ip = [a for a in ip_addresses if a not in set(info.ip_addresses)]
        log.warning(
            "certificate %s does not cover %s; the station has moved network. Regenerating -- "
            "tablets will need to trust the new certificate.",
            cert_file, ", ".join(missing_dns + missing_ip),
        )
        return None
    if not _key_matches(cert_file, key_file):
        log.warning("private key %s does not match certificate %s; regenerating", key_file, cert_file)
        return None
    return info


def generate_cert(
    cert_path: str | os.PathLike[str],
    key_path: str | os.PathLike[str],
    *,
    station_name: str = "wildfire-watch station",
    dns_names: Sequence[str] = (),
    ip_addresses: Sequence[str] = (),
    validity_days: int = DEFAULT_VALIDITY_DAYS,
) -> CertInfo:
    """Write a new self-signed certificate and key. Always overwrites.

    Args:
        cert_path: Destination for the PEM certificate.
        key_path: Destination for the PEM private key.
        station_name: Common name, and source of the ``.local`` SAN.
        dns_names: DNS names for the SAN.
        ip_addresses: IP addresses for the SAN. **Must not be empty in
            practice** -- see the module docstring.
        validity_days: Lifetime. Kept under Apple's 398-day ceiling.

    Returns:
        A :class:`CertInfo` for what was written, with ``created=True``.

    Raises:
        CertError: If ``cryptography`` is missing, no SAN entry could be built,
            or the files cannot be written.
    """
    x509, hashes, ec, serialization, NameOID, ExtendedKeyUsageOID = _crypto()

    cert_file = Path(cert_path).expanduser()
    key_file = Path(key_path).expanduser()

    san: list[Any] = []
    seen_dns: set[str] = set()
    for name in dns_names:
        key = name.strip().rstrip(".").lower()
        if key and key not in seen_dns:
            seen_dns.add(key)
            san.append(x509.DNSName(key))
    seen_ip: set[str] = set()
    for raw in ip_addresses:
        try:
            addr = ipaddress.ip_address(str(raw).split("%", 1)[0])
        except ValueError:
            log.warning("skipping unparseable SAN address %r", raw)
            continue
        if str(addr) in seen_ip:
            continue
        seen_ip.add(str(addr))
        san.append(x509.IPAddress(addr))
    if not san:
        raise CertError(
            "refusing to generate a certificate with an empty subjectAltName: a certificate "
            "with no SAN entry is rejected outright by iOS, so it would fail on the tablets "
            "rather than on this laptop. Pass --host / --ip, or check that this machine has "
            "a network address."
        )
    if not seen_ip:
        # Not fatal (an all-DNS deployment is legitimate) but almost always a
        # mistake on an incident LAN, where tablets are given an IP to type.
        log.warning(
            "generating a certificate with no IP SAN entries; tablets reaching this station "
            "by IP address will reject it"
        )

    # P-256: universally supported, and the handshake is meaningfully cheaper
    # than RSA-2048 on a laptop that is also running a GPU model for a dozen
    # simultaneously reconnecting tablets.
    key = ec.generate_private_key(ec.SECP256R1())
    now = datetime.now(timezone.utc)
    subject = x509.Name(
        [
            x509.NameAttribute(NameOID.COMMON_NAME, station_name[:64] or "wildfire-watch station"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "wildfire-watch"),
        ]
    )
    builder = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)  # self-signed: subject == issuer
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .add_extension(x509.SubjectAlternativeName(san), critical=False)
        # CA:TRUE so an operator can install this on a tablet as a trust
        # anchor; iOS only offers the "enable full trust" switch for a
        # well-formed root. It is then served directly as the leaf.
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=False,  # meaningless for ECDSA; ECDHE is used
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=True,  # required for a trust anchor
                crl_sign=False,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        # serverAuth is mandatory on Apple platforms; a certificate without it
        # is rejected even when the user has explicitly trusted it.
        .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False)
    )
    builder = _with_validity(builder, now - _CLOCK_SKEW, now + timedelta(days=int(validity_days)))
    certificate = builder.sign(private_key=key, algorithm=hashes.SHA256())

    cert_pem = certificate.public_bytes(serialization.Encoding.PEM)
    key_pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        # Unencrypted: the station must come up unattended after a power cut,
        # and a passphrase nobody is present to type is a station that does not
        # start. The file mode below is the control that matters.
        encryption_algorithm=serialization.NoEncryption(),
    )

    try:
        cert_file.parent.mkdir(parents=True, exist_ok=True)
        key_file.parent.mkdir(parents=True, exist_ok=True)
        _write_private(key_file, key_pem)
        cert_file.write_bytes(cert_pem)
    except OSError as exc:
        raise CertError(f"cannot write certificate to {cert_file} / {key_file}: {exc}") from exc

    info = _info_from_certificate(certificate, cert_file, key_file, created=True)
    log.warning(
        "generated a NEW self-signed certificate %s (sha256 %s); every tablet must be told to "
        "trust it before it can connect",
        cert_file, info.fingerprint_sha256,
    )
    return info


def read_cert_info(
    cert_path: str | os.PathLike[str],
    key_path: str | os.PathLike[str] | None = None,
) -> CertInfo:
    """Parse a certificate on disk.

    Args:
        cert_path: PEM certificate to read.
        key_path: Recorded on the result for display; not read here.

    Returns:
        The parsed :class:`CertInfo`, with ``created=False``.

    Raises:
        CertError: If ``cryptography`` is missing, the file cannot be read, or
            it is not a PEM certificate.
    """
    x509, _hashes, _ec, _serialization, _name_oid, _eku_oid = _crypto()
    cert_file = Path(cert_path).expanduser()
    try:
        data = cert_file.read_bytes()
    except OSError as exc:
        raise CertError(f"cannot read certificate {cert_file}: {exc}") from exc
    try:
        certificate = x509.load_pem_x509_certificate(data)
    except Exception as exc:
        raise CertError(f"{cert_file} is not a readable PEM certificate: {exc}") from exc
    return _info_from_certificate(
        certificate,
        cert_file,
        Path(key_path).expanduser() if key_path is not None else cert_file.with_suffix(".key"),
        created=False,
    )


def ssl_context(
    cert_path: str | os.PathLike[str],
    key_path: str | os.PathLike[str],
) -> ssl.SSLContext:
    """Build the server-side ``SSLContext`` the HTTPS listener runs on.

    Args:
        cert_path: PEM certificate.
        key_path: PEM private key.

    Returns:
        A TLS server context with the certificate loaded.

    Raises:
        CertError: If the files are missing or do not form a usable pair. The
            message names both paths, because the person reading it is at a
            laptop in a field.
    """
    cert_file = Path(cert_path).expanduser()
    key_file = Path(key_path).expanduser()
    missing = [str(p) for p in (cert_file, key_file) if not p.is_file()]
    if missing:
        raise CertError(
            f"missing TLS material: {', '.join(missing)}\n"
            "Run 'python -m station certs' to generate a self-signed certificate for this LAN. "
            "Without HTTPS the tablets get no WebRTC and no service worker."
        )
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    # TLS 1.2 floor: 1.0/1.1 are refused by current iOS and Android anyway, and
    # nothing on an incident LAN is older than the tablets.
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    try:
        context.load_cert_chain(certfile=str(cert_file), keyfile=str(key_file))
    except (ssl.SSLError, OSError) as exc:
        raise CertError(
            f"cannot load {cert_file} with key {key_file}: {exc}\n"
            "If the pair has got out of step, 'python -m station certs --force' will reissue "
            "both (every tablet then has to trust the new certificate)."
        ) from exc
    return context


# ------------------------------------------------------------------ internals


def _crypto() -> tuple[Any, Any, Any, Any, Any, Any]:
    """Import ``cryptography`` with an actionable error.

    Returns:
        ``(x509, hashes, ec, serialization, NameOID, ExtendedKeyUsageOID)``.

    Raises:
        CertError: If the package is not installed.
    """
    try:
        from cryptography import x509  # noqa: PLC0415 -- lazy by design.
        from cryptography.hazmat.primitives import hashes, serialization  # noqa: PLC0415
        from cryptography.hazmat.primitives.asymmetric import ec  # noqa: PLC0415
        from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID  # noqa: PLC0415
    except ImportError as exc:
        raise CertError(
            "cryptography is not installed, so the station cannot issue its own certificate.\n"
            "  pip install cryptography\n"
            "Alternatively supply an existing certificate and key at stream.cert / stream.key."
        ) from exc
    return x509, hashes, ec, serialization, NameOID, ExtendedKeyUsageOID


def _with_validity(builder: Any, not_before: datetime, not_after: datetime) -> Any:
    """Set the validity window across cryptography versions.

    ``not_valid_before``/``not_valid_after`` were renamed to ``*_utc`` in
    cryptography 42; both spellings exist for a while and the field laptop's
    version is not something this code gets to choose.
    """
    if hasattr(builder, "not_valid_before_utc"):
        return builder.not_valid_before_utc(not_before).not_valid_after_utc(not_after)
    return builder.not_valid_before(not_before).not_valid_after(not_after)


def _validity_window(certificate: Any) -> tuple[datetime, datetime]:
    """Read the validity window as timezone-aware UTC, across versions."""
    if hasattr(certificate, "not_valid_before_utc"):
        return certificate.not_valid_before_utc, certificate.not_valid_after_utc
    # Pre-42 returns naive datetimes that are documented to be UTC.
    return (
        certificate.not_valid_before.replace(tzinfo=timezone.utc),
        certificate.not_valid_after.replace(tzinfo=timezone.utc),
    )


def _info_from_certificate(certificate: Any, cert_file: Path, key_file: Path, *, created: bool) -> CertInfo:
    """Build a :class:`CertInfo` from a parsed certificate."""
    x509, hashes, _ec, _serialization, NameOID, _eku_oid = _crypto()

    dns: tuple[str, ...] = ()
    ips: tuple[str, ...] = ()
    try:
        san = certificate.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
        dns = tuple(san.get_values_for_type(x509.DNSName))
        ips = tuple(str(a) for a in san.get_values_for_type(x509.IPAddress))
    except x509.ExtensionNotFound:
        # A certificate with no SAN is unusable on iOS; surfaced as empty
        # tuples so ``covers()`` fails and it gets reissued.
        log.warning("certificate %s has no subjectAltName extension", cert_file)

    try:
        common_name = certificate.subject.get_attributes_for_oid(NameOID.COMMON_NAME)[0].value
    except (IndexError, ValueError):
        common_name = ""

    digest = certificate.fingerprint(hashes.SHA256())
    not_before, not_after = _validity_window(certificate)
    return CertInfo(
        cert_path=cert_file,
        key_path=key_file,
        common_name=str(common_name),
        not_before=not_before,
        not_after=not_after,
        dns_names=dns,
        ip_addresses=ips,
        fingerprint_sha256=":".join(f"{b:02X}" for b in digest),
        created=created,
    )


def _key_matches(cert_file: Path, key_file: Path) -> bool:
    """Whether the private key on disk belongs to the certificate on disk.

    A mismatched pair is a station whose TLS handshake fails for every tablet
    at once, and the failure is opaque from the tablet end, so it is worth one
    load at startup to catch it here.
    """
    try:
        ssl_context(cert_file, key_file)
    except CertError:
        return False
    return True


def _write_private(path: Path, data: bytes) -> None:
    """Write a private key, never leaving it world-readable even briefly.

    The file is created ``0600`` by ``os.open`` rather than chmod-ed after the
    fact: between ``write_bytes`` and ``chmod`` there is a window in which the
    key is readable by every process on the laptop.
    """
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        handle = os.fdopen(fd, "wb")
    except BaseException:
        os.close(fd)
        raise
    with handle as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    # Re-assert the mode: an existing file keeps its old permissions through
    # O_CREAT, so a key written once as 0644 would stay 0644 for ever.
    os.chmod(path, 0o600)
