"""SSRF defence.

M2's exit criterion lives here. DNS is injected rather than real: asserting
that a guard blocks `127.0.0.1` must not depend on what a public resolver
returns today, and a test suite that performs real lookups to prove a security
property is both slow and unreliable.
"""

from __future__ import annotations

import pytest

from app.core.errors import UnsafeURLError
from app.core.security import AddressResolver, _host_matches, validate_url


class FakeResolver(AddressResolver):
    """Resolves hostnames from a fixed map, so tests are deterministic."""

    def __init__(self, mapping: dict[str, list[str]] | None = None) -> None:
        self._mapping = mapping or {}

    def resolve(self, hostname: str) -> list[str]:
        if hostname in self._mapping:
            return self._mapping[hostname]
        # Unknown hosts resolve to a public address, so a test that is not
        # about DNS does not have to configure it.
        return ["93.184.216.34"]


def _validate(url: str, **kw: object) -> str:
    kw.setdefault("resolver", FakeResolver())
    return validate_url(url, **kw)  # type: ignore[arg-type]


class TestSchemes:
    @pytest.mark.parametrize(
        "url",
        [
            "file:///etc/passwd",
            "file://localhost/etc/shadow",
            "data:text/html,<script>alert(1)</script>",
            "gopher://example.com:70/",
            "javascript:alert(1)",
            "ftp://example.com/x",
            "dict://example.com:2628/",
            "ldap://example.com/",
            "jar:http://example.com!/",
            "\\\\server\\share",
        ],
    )
    def test_non_http_schemes_are_blocked(self, url: str) -> None:
        with pytest.raises(UnsafeURLError):
            _validate(url)

    def test_file_scheme_names_the_scheme_in_the_error(self) -> None:
        """Explicitly required by M2's exit criterion."""
        with pytest.raises(UnsafeURLError, match="not permitted"):
            _validate("file:///etc/passwd")

    @pytest.mark.parametrize("url", ["http://example.com/a", "https://example.com/a"])
    def test_http_and_https_are_allowed(self, url: str) -> None:
        assert _validate(url) == url


class TestPrivateAndSpecialAddresses:
    @pytest.mark.parametrize(
        ("host", "address"),
        [
            ("loopback.test", "127.0.0.1"),
            ("loopback2.test", "127.1.2.3"),
            ("private10.test", "10.0.0.5"),
            ("private172.test", "172.16.4.9"),
            ("private192.test", "192.168.1.1"),
            ("carrier.test", "100.64.0.1"),
            ("linklocal.test", "169.254.0.5"),
            ("metadata.test", "169.254.169.254"),
            ("multicast.test", "224.0.0.1"),
            ("unspecified.test", "0.0.0.0"),
            ("reserved.test", "240.0.0.1"),
            ("v6loopback.test", "::1"),
            ("v6private.test", "fc00::1"),
            ("v6linklocal.test", "fe80::1"),
            # Ranges that the enumerated flags miss but is_global catches.
            # 100.64.0.0/10 reports is_private=False and is_reserved=False
            # (see failure-analysis F-005).
            ("cgnat.test", "100.64.0.1"),
            ("cgnat2.test", "100.127.255.254"),
            ("testnet1.test", "192.0.2.1"),
            ("testnet2.test", "198.51.100.1"),
            ("testnet3.test", "203.0.113.1"),
            ("benchmark.test", "198.18.0.1"),
            ("broadcast.test", "255.255.255.255"),
        ],
    )
    def test_non_public_addresses_are_blocked(self, host: str, address: str) -> None:
        resolver = FakeResolver({host: [address]})
        with pytest.raises(UnsafeURLError, match="non-public address"):
            validate_url(f"http://{host}/x", resolver=resolver)

    def test_ipv4_mapped_ipv6_loopback_is_blocked(self) -> None:
        """A standard bypass: ::ffff:127.0.0.1 looks like a global IPv6 address.

        Without unwrapping the mapped address, every other check passes.
        """
        resolver = FakeResolver({"sneaky.test": ["::ffff:127.0.0.1"]})
        with pytest.raises(UnsafeURLError, match="loopback"):
            validate_url("http://sneaky.test/x", resolver=resolver)

    def test_ipv4_mapped_private_is_blocked(self) -> None:
        resolver = FakeResolver({"sneaky2.test": ["::ffff:10.0.0.1"]})
        with pytest.raises(UnsafeURLError, match="private"):
            validate_url("http://sneaky2.test/x", resolver=resolver)

    def test_literal_private_ip_in_url_is_blocked(self) -> None:
        resolver = FakeResolver({"127.0.0.1": ["127.0.0.1"]})
        with pytest.raises(UnsafeURLError, match="non-public"):
            validate_url("http://127.0.0.1/admin", resolver=resolver)

    def test_decimal_encoded_loopback_is_blocked(self) -> None:
        """2130706433 == 127.0.0.1; getaddrinfo accepts the decimal form."""
        resolver = FakeResolver({"2130706433": ["127.0.0.1"]})
        with pytest.raises(UnsafeURLError, match="non-public"):
            validate_url("http://2130706433/", resolver=resolver)

    def test_a_host_resolving_to_both_public_and_private_is_rejected(self) -> None:
        """Which address gets connected to is not ours to choose."""
        resolver = FakeResolver({"mixed.test": ["93.184.216.34", "10.0.0.1"]})
        with pytest.raises(UnsafeURLError, match="non-public"):
            validate_url("http://mixed.test/x", resolver=resolver)

    def test_public_address_passes(self) -> None:
        resolver = FakeResolver({"ok.test": ["93.184.216.34"]})
        assert validate_url("https://ok.test/x", resolver=resolver)

    def test_error_does_not_echo_the_internal_address(self) -> None:
        """Echoing the resolved address back leaks network topology."""
        resolver = FakeResolver({"leaky.test": ["10.11.12.13"]})
        with pytest.raises(UnsafeURLError) as info:
            validate_url("http://leaky.test/x", resolver=resolver)
        assert "10.11.12.13" not in str(info.value)
        assert "10.11.12.13" not in str(info.value.details)


class TestBlockedHostnames:
    @pytest.mark.parametrize(
        "host", ["localhost", "LOCALHOST", "metadata.google.internal", "instance-data"]
    )
    def test_known_internal_names_are_blocked_before_dns(self, host: str) -> None:
        # Resolver would return a public address; the name is refused anyway.
        with pytest.raises(UnsafeURLError, match="not permitted"):
            validate_url(f"http://{host}/x", resolver=FakeResolver())

    def test_trailing_dot_does_not_bypass(self) -> None:
        with pytest.raises(UnsafeURLError):
            validate_url("http://localhost./x", resolver=FakeResolver())


class TestCredentialsAndPorts:
    def test_inline_credentials_are_refused(self) -> None:
        with pytest.raises(UnsafeURLError, match="inline credentials"):
            _validate("http://user:pass@example.com/x")

    def test_error_for_credentials_does_not_contain_the_password(self) -> None:
        with pytest.raises(UnsafeURLError) as info:
            _validate("http://user:hunter2@example.com/x")
        assert "hunter2" not in str(info.value)
        assert "hunter2" not in str(info.value.details)

    @pytest.mark.parametrize("port", [22, 23, 25, 3306, 5432, 6379, 8080, 9200, 11211])
    def test_non_web_ports_are_refused(self, port: int) -> None:
        with pytest.raises(UnsafeURLError, match="Port"):
            _validate(f"http://example.com:{port}/x")

    @pytest.mark.parametrize("url", ["http://example.com:80/x", "https://example.com:443/x"])
    def test_explicit_standard_ports_allowed(self, url: str) -> None:
        assert _validate(url) == url

    def test_invalid_port_is_refused(self) -> None:
        with pytest.raises(UnsafeURLError):
            _validate("http://example.com:99999/x")


class TestMalformed:
    @pytest.mark.parametrize("url", ["", "   ", "http://", "https://", "not-a-url", "/relative"])
    def test_unusable_urls_are_refused(self, url: str) -> None:
        with pytest.raises(UnsafeURLError):
            _validate(url)

    def test_unresolvable_hostname_is_refused(self) -> None:
        class Failing(AddressResolver):
            def resolve(self, hostname: str) -> list[str]:
                raise UnsafeURLError("nope", details={"hostname": hostname})

        with pytest.raises(UnsafeURLError):
            validate_url("http://nx.test/x", resolver=Failing())


class TestDomainPolicy:
    def test_blocked_domain_refused_including_subdomains(self) -> None:
        for host in ("evil.test", "a.evil.test", "deep.a.evil.test"):
            with pytest.raises(UnsafeURLError, match="blocked-domain"):
                _validate(f"http://{host}/x", blocked_domains=("evil.test",))

    def test_similar_domain_is_not_wrongly_blocked(self) -> None:
        """A plain endswith would wrongly match `notevil.test`."""
        assert _validate("http://notevil.test/x", blocked_domains=("evil.test",))

    def test_allowlist_permits_only_listed_hosts_and_subdomains(self) -> None:
        allowed = ("gov.uk", "fca.org.uk")
        assert _validate("https://www.gov.uk/a", allowed_domains=allowed)
        assert _validate("https://data.fca.org.uk/b", allowed_domains=allowed)
        with pytest.raises(UnsafeURLError, match="not on the allowed-domain list"):
            _validate("https://example.com/c", allowed_domains=allowed)

    def test_denylist_wins_over_allowlist(self) -> None:
        """A denylist that an allowlist could override would be useless."""
        with pytest.raises(UnsafeURLError, match="blocked-domain"):
            _validate(
                "https://bad.gov.uk/x",
                allowed_domains=("gov.uk",),
                blocked_domains=("bad.gov.uk",),
            )

    def test_empty_allowlist_means_any_public_host(self) -> None:
        assert _validate("https://anything.test/x", allowed_domains=())


class TestHostMatching:
    @pytest.mark.parametrize(
        ("host", "pattern", "expected"),
        [
            ("example.com", "example.com", True),
            ("a.example.com", "example.com", True),
            ("example.com.", "example.com", True),
            ("notexample.com", "example.com", False),
            ("example.com.evil.test", "example.com", False),
            ("EXAMPLE.COM", "example.com", True),
            ("a.example.com", "*.example.com", True),
        ],
    )
    def test_matching_rules(self, host: str, pattern: str, expected: bool) -> None:
        assert _host_matches(host, pattern) is expected


class TestNonGlobalCatchAll:
    """The enumerated checks are an enumeration, so is_global backs them up."""

    def test_cgnat_is_blocked_despite_not_being_private_or_reserved(self) -> None:
        import ipaddress

        addr = ipaddress.ip_address("100.64.0.1")
        # Documents precisely why the catch-all is needed, so a future
        # simplification does not delete it as redundant.
        assert addr.is_private is False
        assert addr.is_reserved is False
        assert addr.is_global is False

        with pytest.raises(UnsafeURLError, match="non-public address"):
            validate_url(
                "http://cgnat.test/x", resolver=FakeResolver({"cgnat.test": ["100.64.0.1"]})
            )

    def test_a_genuinely_global_address_still_passes(self) -> None:
        resolver = FakeResolver({"real.test": ["93.184.216.34"]})
        assert validate_url("https://real.test/x", resolver=resolver)
