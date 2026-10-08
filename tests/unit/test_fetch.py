"""Fetching, extraction and normalisation.

HTTP is mocked with respx: no test reaches the network. The redirect tests are
security tests in substance — a guard that validates only the submitted URL is
defeated by a public URL that redirects to loopback.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from app.core.config import Settings
from app.core.errors import UnsafeURLError, UpstreamError
from app.schemas.source import content_hash
from app.tools.fetch import (
    MAX_CONTENT_BYTES,
    SEC_USER_AGENT_NAME,
    SourceFetcher,
    extract_text,
    normalise_text,
)
from tests.security.test_ssrf import FakeResolver

HTML = """<!doctype html>
<html><head><title>UK Pet Insurance Review</title></head>
<body>
<nav>Home | About | Contact</nav>
<article>
<h1>Market Overview</h1>
<p>Gross written premium reached GBP 1.6 billion in the period under review.</p>
<p>Three insurers accounted for the majority of policies in force.</p>
</article>
<footer>Copyright 2026</footer>
</body></html>
"""


def _settings(**kw: object) -> Settings:
    return Settings(environment="test", _env_file=None, **kw)  # type: ignore[arg-type,call-arg]


def _fetcher(settings: Settings | None = None) -> SourceFetcher:
    return SourceFetcher(settings or _settings(), resolver=FakeResolver())


class TestNormalisation:
    def test_is_idempotent(self) -> None:
        """The property every stored offset depends on.

        If normalising twice changed the string, offsets recorded against the
        first form would be wrong against the second, by an amount nobody
        could reconstruct.
        """
        raw = "A\r\nB​\n\n\n\nC   \nD﻿  \t\n\n  "
        once = normalise_text(raw)
        assert normalise_text(once) == once
        assert normalise_text(normalise_text(once)) == once

    def test_crlf_becomes_lf(self) -> None:
        assert "\r" not in normalise_text("a\r\nb\rc")

    def test_runs_of_blank_lines_collapse_to_one(self) -> None:
        assert normalise_text("a\n\n\n\n\nb") == "a\n\nb"

    def test_invisible_characters_are_removed(self) -> None:
        """Zero-width characters would break exact citation matching invisibly."""
        for ch in ("​", "‎", "﻿", "⁠", "‮"):
            assert ch not in normalise_text(f"a{ch}b")
        assert normalise_text("a​b") == "ab"

    def test_trailing_whitespace_per_line_is_stripped(self) -> None:
        assert normalise_text("a   \nb\t\nc") == "a\nb\nc"

    def test_leading_and_trailing_whitespace_stripped(self) -> None:
        assert normalise_text("  \n hello \n  ") == "hello"

    def test_nfkc_applied(self) -> None:
        # Full-width characters normalise to ASCII under NFKC.
        assert normalise_text("ＡＢＣ") == "ABC"

    def test_empty_input(self) -> None:
        assert normalise_text("") == ""
        assert normalise_text("   \n  ") == ""


class TestExtraction:
    def test_extracts_body_and_drops_chrome(self) -> None:
        text, title = extract_text(HTML, url="https://example.com/a")
        assert "Gross written premium reached GBP 1.6 billion" in text
        # trafilatura prefers the document's main heading over the <title>
        # tag. That is the better choice — a <title> usually carries site
        # branding ("… | Example Insurance Ltd") that is noise in a citation.
        assert title == "Market Overview"
        # Navigation and footer boilerplate should not survive.
        assert "Home | About | Contact" not in text
        assert "Copyright 2026" not in text

    def test_extracted_text_is_already_normalised(self) -> None:
        text, _ = extract_text(HTML, url="https://example.com/a")
        assert normalise_text(text) == text

    def test_falls_back_rather_than_discarding_an_unextractable_page(self) -> None:
        text, _ = extract_text(
            "<html><body><div>bare text</div></body></html>",
            url="https://example.com/a",
        )
        assert "bare text" in text


class TestFetchSuccess:
    @respx.mock
    async def test_returns_canonical_text_with_matching_hash(self) -> None:
        respx.get("https://example.com/a").mock(
            return_value=httpx.Response(
                200, html=HTML, headers={"content-type": "text/html; charset=utf-8"}
            )
        )
        source = await _fetcher().fetch("https://example.com/a")

        assert source.status_code == 200
        assert "Gross written premium" in source.text
        assert source.content_hash == content_hash(source.text)
        assert source.verify_hash()
        assert source.domain == "example.com"
        assert source.redirect_chain == []
        assert source.byte_length > 0

    @respx.mock
    async def test_plain_text_source(self) -> None:
        respx.get("https://example.com/t.txt").mock(
            return_value=httpx.Response(
                200, text="Line one.\n\nLine two.", headers={"content-type": "text/plain"}
            )
        )
        source = await _fetcher().fetch("https://example.com/t.txt")
        assert "Line one." in source.text


class TestRedirects:
    @respx.mock
    async def test_follows_a_redirect_and_records_the_chain(self) -> None:
        respx.get("https://example.com/old").mock(
            return_value=httpx.Response(302, headers={"location": "https://example.com/new"})
        )
        respx.get("https://example.com/new").mock(
            return_value=httpx.Response(200, html=HTML, headers={"content-type": "text/html"})
        )
        source = await _fetcher().fetch("https://example.com/old")
        assert str(source.final_url) == "https://example.com/new"
        assert source.redirect_chain == ["https://example.com/old"]

    @respx.mock
    async def test_redirect_to_loopback_is_blocked(self) -> None:
        """The bypass a one-shot guard misses: public URL, private destination."""
        respx.get("https://example.com/open").mock(
            return_value=httpx.Response(302, headers={"location": "http://127.0.0.1/admin"})
        )
        fetcher = SourceFetcher(_settings(), resolver=FakeResolver({"127.0.0.1": ["127.0.0.1"]}))
        with pytest.raises(UnsafeURLError, match="non-public"):
            await fetcher.fetch("https://example.com/open")

    @respx.mock
    async def test_redirect_to_file_scheme_is_blocked(self) -> None:
        respx.get("https://example.com/open2").mock(
            return_value=httpx.Response(302, headers={"location": "file:///etc/passwd"})
        )
        with pytest.raises(UnsafeURLError):
            await _fetcher().fetch("https://example.com/open2")

    @respx.mock
    async def test_relative_redirect_is_resolved_then_validated(self) -> None:
        respx.get("https://example.com/a").mock(
            return_value=httpx.Response(301, headers={"location": "/b"})
        )
        respx.get("https://example.com/b").mock(
            return_value=httpx.Response(200, html=HTML, headers={"content-type": "text/html"})
        )
        source = await _fetcher().fetch("https://example.com/a")
        assert str(source.final_url) == "https://example.com/b"

    @respx.mock
    async def test_redirect_loop_terminates(self) -> None:
        respx.get("https://example.com/loop").mock(
            return_value=httpx.Response(302, headers={"location": "https://example.com/loop"})
        )
        with pytest.raises(UpstreamError, match="redirects"):
            await _fetcher().fetch("https://example.com/loop")

    @respx.mock
    async def test_redirect_without_location_is_an_error(self) -> None:
        respx.get("https://example.com/bad").mock(return_value=httpx.Response(302))
        with pytest.raises(UpstreamError, match="Location"):
            await _fetcher().fetch("https://example.com/bad")


class TestFetchFailures:
    @respx.mock
    @pytest.mark.parametrize("status", [400, 403, 404, 410, 500, 503])
    async def test_error_statuses_are_reported(self, status: int) -> None:
        respx.get("https://example.com/e").mock(return_value=httpx.Response(status))
        with pytest.raises(UpstreamError, match=f"HTTP {status}"):
            await _fetcher().fetch("https://example.com/e")

    @respx.mock
    @pytest.mark.parametrize(
        "content_type",
        ["application/pdf", "image/png", "application/zip", "video/mp4", "application/json"],
    )
    async def test_non_textual_content_is_refused(self, content_type: str) -> None:
        respx.get("https://example.com/f").mock(
            return_value=httpx.Response(
                200, content=b"binary", headers={"content-type": content_type}
            )
        )
        with pytest.raises(UpstreamError, match="Unsupported content type"):
            await _fetcher().fetch("https://example.com/f")

    @respx.mock
    async def test_oversized_body_is_refused(self) -> None:
        """An unbounded read is a trivial memory-exhaustion vector."""
        respx.get("https://example.com/big").mock(
            return_value=httpx.Response(
                200,
                content=b"x" * (MAX_CONTENT_BYTES + 1),
                headers={"content-type": "text/html"},
            )
        )
        with pytest.raises(UpstreamError, match="maximum fetch size"):
            await _fetcher().fetch("https://example.com/big")

    @respx.mock
    async def test_timeout_is_reported_as_upstream(self) -> None:
        respx.get("https://example.com/slow").mock(side_effect=httpx.ConnectTimeout("timed out"))
        with pytest.raises(UpstreamError, match="timed out"):
            await _fetcher().fetch("https://example.com/slow")

    @respx.mock
    async def test_connection_error_is_reported_as_upstream(self) -> None:
        respx.get("https://example.com/down").mock(side_effect=httpx.ConnectError("refused"))
        with pytest.raises(UpstreamError, match="fetch failed"):
            await _fetcher().fetch("https://example.com/down")

    @respx.mock
    async def test_empty_page_yields_no_text_error(self) -> None:
        respx.get("https://example.com/empty").mock(
            return_value=httpx.Response(200, text="", headers={"content-type": "text/html"})
        )
        with pytest.raises(UpstreamError, match="no extractable text"):
            await _fetcher().fetch("https://example.com/empty")


class TestGuardAppliesBeforeAnyRequest:
    async def test_file_url_never_reaches_the_transport(self) -> None:
        """Validation precedes the request, so a blocked URL costs nothing."""
        with respx.mock:
            route = respx.get("https://example.com/").mock(return_value=httpx.Response(200))
            with pytest.raises(UnsafeURLError):
                await _fetcher().fetch("file:///etc/passwd")
            assert route.call_count == 0

    async def test_blocked_domain_is_refused(self) -> None:
        fetcher = SourceFetcher(
            _settings(blocked_source_domains=("spam.test",)), resolver=FakeResolver()
        )
        with pytest.raises(UnsafeURLError, match="blocked-domain"):
            await fetcher.fetch("https://spam.test/x")

    async def test_allowlist_excludes_everything_else(self) -> None:
        fetcher = SourceFetcher(
            _settings(allowed_source_domains=("gov.uk",)), resolver=FakeResolver()
        )
        with pytest.raises(UnsafeURLError, match="allowed-domain"):
            await fetcher.fetch("https://example.com/x")


# ==========================================================================
# SEC EDGAR access — sec.gov 403s automated clients that do not declare a
# contact address. Reproduced in production by run_67498d61c5794183, where
# two NVIDIA 10-K filings failed with HTTP 403 / access_restricted.
# ==========================================================================

SEC_URL = "https://www.sec.gov/Archives/edgar/data/1045810/nvda-20250126.htm"
CONTACT = "research@example.com"


def _sec_fetcher(contact: str | None = CONTACT) -> SourceFetcher:
    return _fetcher(_settings(sec_contact_email=contact))


class TestSecUserAgent:
    @respx.mock
    async def test_a_sec_request_declares_the_contact_address(self) -> None:
        route = respx.get(SEC_URL).mock(
            return_value=httpx.Response(200, html=HTML, headers={"content-type": "text/html"})
        )
        await _sec_fetcher().fetch(SEC_URL)

        headers = route.calls.last.request.headers
        assert headers["user-agent"] == f"{SEC_USER_AGENT_NAME} {CONTACT}"
        # Required alongside the User-Agent by sec.gov/os/webmaster-faq.
        assert headers["accept-encoding"] == "gzip, deflate"

    @respx.mock
    async def test_the_contact_address_comes_from_configuration(self) -> None:
        """Not hard-coded: a different contact must reach the wire."""
        route = respx.get(SEC_URL).mock(
            return_value=httpx.Response(200, html=HTML, headers={"content-type": "text/html"})
        )
        await _sec_fetcher("someone.else@example.org").fetch(SEC_URL)
        assert "someone.else@example.org" in route.calls.last.request.headers["user-agent"]

    @respx.mock
    async def test_a_sec_subdomain_is_covered(self) -> None:
        url = "https://data.sec.gov/api/xbrl/companyfacts/CIK0001045810.json"
        route = respx.get(url).mock(
            return_value=httpx.Response(200, text="plain", headers={"content-type": "text/plain"})
        )
        await _sec_fetcher().fetch(url)
        assert CONTACT in route.calls.last.request.headers["user-agent"]


class TestNonSecRequestsAreUnchanged:
    @respx.mock
    async def test_an_ordinary_source_does_not_receive_the_contact(self) -> None:
        """The contact address is not broadcast to the open web."""
        url = "https://www.abi.org.uk/data/pet-insurance"
        route = respx.get(url).mock(
            return_value=httpx.Response(200, html=HTML, headers={"content-type": "text/html"})
        )
        await _sec_fetcher().fetch(url)

        headers = route.calls.last.request.headers
        assert CONTACT not in headers["user-agent"]
        assert "@" not in headers["user-agent"]
        # Exactly the client default, with nothing appended for this host.
        assert headers["user-agent"] == (
            f"{SEC_USER_AGENT_NAME} (+https://github.com/ChaturvediShivam)"
        )

    @respx.mock
    async def test_a_lookalike_host_does_not_receive_the_contact(self) -> None:
        """`sec.gov.example.com` and `notsec.gov` are not sec.gov."""
        for url in (
            "https://sec.gov.example.com/filing.htm",
            "https://notsec.gov/filing.htm",
        ):
            route = respx.get(url).mock(
                return_value=httpx.Response(200, html=HTML, headers={"content-type": "text/html"})
            )
            await _sec_fetcher().fetch(url)
            assert CONTACT not in route.calls.last.request.headers["user-agent"], url

    @respx.mock
    async def test_the_header_follows_the_host_across_redirects(self) -> None:
        """Recomputed per hop: added on entering sec.gov, dropped on leaving."""
        off_sec = "https://www.example.org/mirror.htm"
        sec_route = respx.get(SEC_URL).mock(
            return_value=httpx.Response(302, headers={"location": off_sec})
        )
        off_route = respx.get(off_sec).mock(
            return_value=httpx.Response(200, html=HTML, headers={"content-type": "text/html"})
        )
        await _sec_fetcher().fetch(SEC_URL)

        assert CONTACT in sec_route.calls.last.request.headers["user-agent"]
        assert CONTACT not in off_route.calls.last.request.headers["user-agent"]


class TestSecAccessIsNotWeakened:
    @respx.mock
    async def test_an_unconfigured_contact_sends_no_contact_header(self) -> None:
        route = respx.get(SEC_URL).mock(
            return_value=httpx.Response(200, html=HTML, headers={"content-type": "text/html"})
        )
        await _sec_fetcher(None).fetch(SEC_URL)
        assert "@" not in route.calls.last.request.headers["user-agent"]

    @respx.mock
    async def test_a_403_is_still_a_failure(self) -> None:
        """The fix is to be allowed in, never to pretend a refusal succeeded."""
        respx.get(SEC_URL).mock(return_value=httpx.Response(403))
        with pytest.raises(UpstreamError) as exc:
            await _sec_fetcher().fetch(SEC_URL)
        assert "403" in str(exc.value)

    @respx.mock
    async def test_a_403_is_still_a_failure_when_unconfigured(self) -> None:
        respx.get(SEC_URL).mock(return_value=httpx.Response(403))
        with pytest.raises(UpstreamError):
            await _sec_fetcher(None).fetch(SEC_URL)

    @respx.mock
    async def test_timeout_classification_is_unchanged_for_sec(self) -> None:
        respx.get(SEC_URL).mock(side_effect=httpx.ConnectTimeout("slow"))
        with pytest.raises(UpstreamError) as exc:
            await _sec_fetcher().fetch(SEC_URL)
        assert "timed out" in str(exc.value)


class TestSecContactValidation:
    def test_a_malformed_contact_is_rejected_at_startup(self) -> None:
        with pytest.raises(ValueError, match="single email address"):
            _settings(sec_contact_email="not-an-email")

    def test_a_contact_with_whitespace_is_rejected(self) -> None:
        """Whitespace would split the User-Agent and break the format."""
        with pytest.raises(ValueError, match="single email address"):
            _settings(sec_contact_email="first last@example.com")

    def test_blank_is_treated_as_unset(self) -> None:
        assert _settings(sec_contact_email="   ").sec_contact_email is None

    def test_a_valid_contact_is_stripped(self) -> None:
        assert _settings(sec_contact_email="  a@b.com ").sec_contact_email == "a@b.com"
