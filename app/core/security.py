"""URL safety: SSRF defence and domain policy.

This module decides whether the service will fetch a URL at all. It runs
before any network request and again on every redirect hop, because a URL
that passes validation can still redirect to `127.0.0.1` — validating only the
original URL is the commonest way an SSRF guard is defeated.

Three checks, in order:

1. **Scheme** — only `http` and `https`. `file://`, `data:`, `gopher://` and
   friends are rejected outright, as is a URL carrying inline credentials.
2. **Port** — only 80 and 443. An internal service on an odd port is already
   blocked by check 3, but there is no research reason to reach one.
3. **Address** — the hostname is resolved, and **every** returned address is
   checked against private, loopback, link-local, multicast, reserved and
   unspecified ranges. All addresses must be public: a hostname that resolves
   to both a public and a private address is rejected, because which one gets
   connected to is not ours to choose.

Check 3 resolves DNS itself rather than trusting the hostname, which is what
makes `localtest.me`-style names and cloud metadata endpoints
(`169.254.169.254`, covered by link-local) fail.
"""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlsplit

import structlog

from app.core.errors import UnsafeURLError

logger = structlog.get_logger(__name__)

ALLOWED_SCHEMES = frozenset({"http", "https"})
ALLOWED_PORTS = frozenset({80, 443})
_DEFAULT_PORTS = {"http": 80, "https": 443}

# Hostnames that must never be resolved, regardless of what DNS would say.
# Belt-and-braces: the address check below would catch these anyway, but
# failing early gives a clearer error and avoids a pointless lookup.
BLOCKED_HOSTNAMES = frozenset(
    {
        "localhost",
        "localhost.localdomain",
        "ip6-localhost",
        "ip6-loopback",
        "metadata",
        "metadata.google.internal",
        "instance-data",
    }
)


class AddressResolver:
    """Hostname → IP addresses. A seam so tests can resolve deterministically.

    Real DNS in a test suite is slow, flaky, and in the case of SSRF testing
    actively dangerous — asserting that a guard blocks a private address should
    not depend on what a public resolver happens to return today.
    """

    def resolve(self, hostname: str) -> list[str]:
        try:
            infos = socket.getaddrinfo(hostname, None, proto=socket.IPPROTO_TCP)
        except socket.gaierror as exc:
            raise UnsafeURLError(
                f"Hostname {hostname!r} could not be resolved.",
                details={"hostname": hostname},
            ) from exc
        # getaddrinfo returns duplicates per socket type; dedupe, keep order.
        seen: dict[str, None] = {}
        for info in infos:
            seen.setdefault(str(info[4][0]), None)
        return list(seen)


def _classify(address: str) -> str | None:
    """Return the reason an address is unsafe, or None if it is public.

    IPv4-mapped IPv6 (`::ffff:127.0.0.1`) is unwrapped first: without that,
    a mapped loopback address looks like an ordinary global IPv6 address and
    passes every other check. It is a standard SSRF bypass.
    """
    try:
        ip = ipaddress.ip_address(address)
    except ValueError:
        return "not a valid IP address"

    mapped = getattr(ip, "ipv4_mapped", None)
    if mapped is not None:
        ip = mapped

    # 6to4 and Teredo embed an IPv4 address that could be private.
    sixtofour = getattr(ip, "sixtofour", None)
    if sixtofour is not None:
        ip = sixtofour
    teredo = getattr(ip, "teredo", None)
    if teredo is not None:
        ip = teredo[1]

    if ip.is_loopback:
        return "loopback address"
    if ip.is_private:
        return "private address"
    if ip.is_link_local:
        # Covers 169.254.0.0/16, i.e. cloud instance metadata.
        return "link-local address"
    if ip.is_multicast:
        return "multicast address"
    if ip.is_reserved:
        return "reserved address"
    if ip.is_unspecified:
        return "unspecified address"
    # Catch-all. The specific checks above exist for their clearer messages,
    # but they are an enumeration and enumerations go stale: 100.64.0.0/10
    # (CGNAT, RFC 6598) reports is_private=False and is_reserved=False, so it
    # passed every check above while being unroutable shared address space.
    # `is_global` is the authoritative test. See failure-analysis F-005.
    if not ip.is_global:
        return "non-global address"
    return None


def _host_matches(host: str, pattern: str) -> bool:
    """Match a hostname against a domain pattern, including subdomains.

    `example.com` matches `example.com` and `a.example.com`, but not
    `notexample.com` — a plain `endswith` would wrongly match the last one.
    """
    host = host.lower().rstrip(".")
    pattern = pattern.lower().lstrip("*.").rstrip(".")
    return host == pattern or host.endswith(f".{pattern}")


def validate_url(
    url: str,
    *,
    allowed_domains: tuple[str, ...] = (),
    blocked_domains: tuple[str, ...] = (),
    resolver: AddressResolver | None = None,
) -> str:
    """Validate a URL for outbound fetching. Returns it, or raises.

    `allowed_domains` empty means "any public host". `blocked_domains` always
    applies and is checked first, so a denylist cannot be overridden by an
    allowlist entry.
    """
    parts = urlsplit(url.strip())

    if parts.scheme.lower() not in ALLOWED_SCHEMES:
        raise UnsafeURLError(
            f"Scheme {parts.scheme or '(none)'!r} is not permitted; "
            "only http and https are fetched.",
            details={"scheme": parts.scheme, "url": url[:200]},
        )

    # Inline credentials mean the URL is carrying someone's secret into a log
    # or a referer header. Refuse rather than strip and proceed silently.
    if parts.username or parts.password:
        raise UnsafeURLError(
            "URL contains inline credentials.", details={"url": parts.hostname or ""}
        )

    hostname = (parts.hostname or "").lower().rstrip(".")
    if not hostname:
        raise UnsafeURLError("URL has no host.", details={"url": url[:200]})

    if hostname in BLOCKED_HOSTNAMES:
        raise UnsafeURLError(
            f"Hostname {hostname!r} is not permitted.", details={"hostname": hostname}
        )

    try:
        port = parts.port or _DEFAULT_PORTS[parts.scheme.lower()]
    except ValueError as exc:
        # urlsplit raises on a non-numeric or out-of-range port.
        raise UnsafeURLError("URL has an invalid port.", details={"url": url[:200]}) from exc
    if port not in ALLOWED_PORTS:
        raise UnsafeURLError(
            f"Port {port} is not permitted; only 80 and 443 are fetched.",
            details={"port": port, "hostname": hostname},
        )

    for pattern in blocked_domains:
        if pattern and _host_matches(hostname, pattern):
            raise UnsafeURLError(
                f"Host {hostname!r} is on the blocked-domain list.",
                details={"hostname": hostname},
            )

    if allowed_domains and not any(p and _host_matches(hostname, p) for p in allowed_domains):
        raise UnsafeURLError(
            f"Host {hostname!r} is not on the allowed-domain list.",
            details={"hostname": hostname},
        )

    # A literal IP in the URL still goes through _classify; resolving it is a
    # no-op that getaddrinfo handles, so there is no separate branch.
    addresses = (resolver or AddressResolver()).resolve(hostname)
    if not addresses:  # pragma: no cover - getaddrinfo either raises or returns
        raise UnsafeURLError(
            f"Hostname {hostname!r} resolved to no addresses.",
            details={"hostname": hostname},
        )

    for address in addresses:
        reason = _classify(address)
        if reason is not None:
            # Deliberately does not say which address: the caller supplied the
            # URL, and echoing the resolved internal address back is itself a
            # small information leak about the network this runs in.
            logger.warning("ssrf_blocked", hostname=hostname, reason=reason, address=address)
            raise UnsafeURLError(
                f"Host {hostname!r} resolves to a non-public address ({reason}).",
                details={"hostname": hostname, "reason": reason},
            )

    return url.strip()
