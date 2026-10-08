"""Source fetching, extraction, sanitisation and normalisation.

Fetching happens here rather than through Anthropic's server-side `web_fetch`
tool, and the reason is architectural rather than incidental: deterministic
citation verification (ADR-002) requires this service to hold the exact text
that citations index into, byte for byte. A server-side fetch returns a
document to the model; it does not hand us a canonical string we can slice at
character offsets later. Discovery still uses the server-side search tool
(ADR-006) — see the amendment recorded in that ADR.

Doing the fetch locally is also what makes the SSRF guard meaningful: there is
an outbound request of ours to refuse.

Normalisation is applied **once**, here, and the result is canonical. It must
be idempotent — if normalising twice changed the string, every stored offset
would be wrong by an amount nobody could reconstruct. `tests/unit/test_fetch.py`
asserts that property directly.
"""

from __future__ import annotations

import re
import unicodedata

import httpx
import structlog
import trafilatura

from app.core.config import Settings
from app.core.errors import UnsafeURLError, UpstreamError
from app.core.security import AddressResolver, _host_matches, validate_url
from app.schemas.source import FetchedSource, content_hash

logger = structlog.get_logger(__name__)

# Cap the download. A research agent has no reason to pull a 200 MB file, and
# an unbounded read is a trivial memory-exhaustion vector.
MAX_CONTENT_BYTES = 5 * 1024 * 1024
MAX_REDIRECTS = 5

# sec.gov requires automated clients to declare a contact address in the
# User-Agent, and returns 403 otherwise. The documented format is
# "Company Name contact@domain" (sec.gov/os/webmaster-faq), plus
# `Accept-Encoding: gzip, deflate`. `Host` is also required and httpx sets it.
SEC_DOMAIN = "sec.gov"
SEC_USER_AGENT_NAME = "ResearchIntelligenceAgent/0.1"

# Content types worth extracting text from. Anything else is refused before
# the body is read.
TEXTUAL_CONTENT_TYPES = (
    "text/html",
    "application/xhtml+xml",
    "text/plain",
    "application/xml",
    "text/xml",
)

_MULTI_NEWLINE = re.compile(r"\n{3,}")
_TRAILING_WS = re.compile(r"[ \t]+\n")
# Zero-width and bidi-control characters. These are invisible, survive copy
# and paste, and would make an exact citation match fail for no visible
# reason — so they are removed before the text becomes canonical.
_INVISIBLE = re.compile(r"[​-‏‪-‮⁠-⁤﻿]")


def normalise_text(text: str) -> str:
    """Produce the canonical form of extracted text.

    Idempotent by construction: every step is a fixed-point transformation.
    NFKC runs first so that the regexes below see a stable representation.
    """
    out = unicodedata.normalize("NFKC", text)
    out = _INVISIBLE.sub("", out)
    out = out.replace("\r\n", "\n").replace("\r", "\n")
    out = _TRAILING_WS.sub("\n", out)
    out = _MULTI_NEWLINE.sub("\n\n", out)
    return out.strip()


def extract_text(raw: str, *, url: str) -> tuple[str, str]:
    """Extract readable text and a title from raw HTML.

    Returns `(text, title)`. Falls back to treating the input as plain text
    when extraction yields nothing, which happens on pages that are almost
    entirely script or markup.
    """
    extracted = trafilatura.extract(
        raw,
        url=url,
        include_comments=False,
        include_tables=True,
        favor_precision=True,
    )
    title = ""
    metadata = trafilatura.extract_metadata(raw)
    if metadata is not None and metadata.title:
        title = str(metadata.title)[:500]

    if not extracted:
        # No boilerplate removal possible; strip tags crudely rather than
        # discard the source. Marked in the log so a pattern is visible.
        logger.info("extraction_fallback", url=url)
        extracted = re.sub(r"<[^>]+>", " ", raw)

    return normalise_text(extracted), normalise_text(title)


def _is_textual(content_type: str) -> bool:
    base = content_type.split(";", 1)[0].strip().lower()
    return any(base == t or base.endswith("+xml") for t in TEXTUAL_CONTENT_TYPES)


class SourceFetcher:
    """Fetches a URL safely and returns canonical text.

    Every redirect hop is re-validated. Validating only the submitted URL is
    the standard way an SSRF guard is bypassed: a public URL that 302s to
    `http://127.0.0.1:6379/` defeats a one-shot check.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        client: httpx.AsyncClient | None = None,
        resolver: AddressResolver | None = None,
    ) -> None:
        self._settings = settings
        self._client = client
        self._resolver = resolver
        self._owns_client = client is None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self._settings.fetch_timeout_seconds,
                # Redirects are followed manually so each hop is validated.
                follow_redirects=False,
                headers={
                    # Identify honestly. A research agent that disguises itself
                    # as a browser is a different kind of tool.
                    "User-Agent": (
                        "ResearchIntelligenceAgent/0.1 (+https://github.com/ChaturvediShivam)"
                    ),
                    "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9",
                },
            )
        return self._client

    def _request_headers(self, url: str) -> dict[str, str]:
        """Per-request header overrides for this hop. Usually empty.

        Set on the request rather than on the client on purpose: the client is
        shared across every source on the open web, and a contact address
        installed as a default header would be disclosed to all of them. This
        sends it only to the host that requires it.

        Recomputed per redirect hop by the caller, so a sec.gov -> sec.gov
        redirect keeps the header and a redirect off sec.gov drops it.
        """
        host = httpx.URL(url).host
        if not _host_matches(host, SEC_DOMAIN):
            return {}

        contact = self._settings.sec_contact_email
        if not contact:
            # Not an error: the fetch proceeds and SEC will refuse it. Logged
            # because a 403 from sec.gov is otherwise indistinguishable from
            # a genuine access restriction.
            logger.warning(
                "sec_contact_not_configured",
                host=host,
                remedy="set SEC_CONTACT_EMAIL to a contactable address",
            )
            return {}

        return {
            # Format required by sec.gov/os/webmaster-faq.
            "User-Agent": f"{SEC_USER_AGENT_NAME} {contact}",
            "Accept-Encoding": "gzip, deflate",
        }

    def _validate(self, url: str) -> str:
        return validate_url(
            url,
            allowed_domains=self._settings.allowed_source_domains,
            blocked_domains=self._settings.blocked_source_domains,
            resolver=self._resolver,
        )

    async def fetch(self, url: str) -> FetchedSource:
        """Fetch one URL and return its canonical text, or raise."""
        client = await self._get_client()
        current = self._validate(url)
        chain: list[str] = []

        for _ in range(MAX_REDIRECTS + 1):
            try:
                response = await client.get(current, headers=self._request_headers(current))
            except httpx.TimeoutException as exc:
                raise UpstreamError(
                    "Source fetch timed out.", details={"url": current[:200]}
                ) from exc
            except httpx.HTTPError as exc:
                raise UpstreamError(
                    "Source fetch failed.",
                    details={"url": current[:200], "error": type(exc).__name__},
                ) from exc

            if response.is_redirect:
                location = response.headers.get("location")
                if not location:
                    raise UpstreamError(
                        "Redirect without a Location header.",
                        details={"url": current[:200]},
                    )
                # Resolve relative redirects, then re-run the full guard.
                nxt = str(httpx.URL(current).join(location))
                chain.append(current)
                current = self._validate(nxt)
                continue

            return self._to_source(url, current, response, chain)

        raise UpstreamError(f"Exceeded {MAX_REDIRECTS} redirects.", details={"url": url[:200]})

    def _to_source(
        self,
        original_url: str,
        final_url: str,
        response: httpx.Response,
        chain: list[str],
    ) -> FetchedSource:
        if response.status_code >= 400:
            raise UpstreamError(
                f"Source returned HTTP {response.status_code}.",
                details={"url": final_url[:200], "status": response.status_code},
            )

        content_type = response.headers.get("content-type", "")
        if not _is_textual(content_type):
            raise UpstreamError(
                f"Unsupported content type {content_type!r}; only textual sources are processed.",
                details={"url": final_url[:200], "content_type": content_type[:100]},
            )

        body = response.content
        if len(body) > MAX_CONTENT_BYTES:
            raise UpstreamError(
                "Source exceeds the maximum fetch size.",
                details={"url": final_url[:200], "bytes": len(body)},
            )

        raw = body.decode(response.encoding or "utf-8", errors="replace")
        text, title = extract_text(raw, url=final_url)
        if not text:
            raise UpstreamError(
                "Source yielded no extractable text.", details={"url": final_url[:200]}
            )

        logger.info(
            "source_fetched",
            url=final_url,
            status=response.status_code,
            bytes=len(body),
            text_chars=len(text),
            redirects=len(chain),
        )
        return FetchedSource(
            url=original_url,
            final_url=final_url,
            title=title,
            text=text,
            content_hash=content_hash(text),
            content_type=content_type[:100],
            status_code=response.status_code,
            byte_length=len(body),
            redirect_chain=chain,
        )

    async def aclose(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()


__all__ = [
    "MAX_CONTENT_BYTES",
    "MAX_REDIRECTS",
    "SourceFetcher",
    "UnsafeURLError",
    "extract_text",
    "normalise_text",
]
