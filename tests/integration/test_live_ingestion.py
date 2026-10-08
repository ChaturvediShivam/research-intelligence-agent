"""Live verification of stage 3 (PROCESS) and source discovery.

Marked `live`: real network, and the search test makes a billable API call.
These satisfy M2's exit criterion — "real fetches" — which mocked HTTP cannot.
The SSRF half of that criterion is deliberately **not** live: asserting that a
guard blocks `127.0.0.1` must not depend on a public resolver, so those tests
inject DNS and live in `tests/security/test_ssrf.py`.

Run with:  uv run pytest -m live
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from app.core.config import Settings
from app.core.errors import UnsafeURLError
from app.retrieval.chunking import verify_chunks
from app.schemas.source import SourceCandidate, content_hash
from app.tools.fetch import SourceFetcher, normalise_text
from app.tools.search import AnthropicSearchProvider

pytestmark = pytest.mark.live


def _key_available() -> bool:
    try:
        Settings(environment="local").require_anthropic_key()
    except Exception:
        return False
    return True


async def _drain() -> None:
    """Let the loop run pending callbacks before pytest-asyncio closes it.

    The transports used by httpx and the Anthropic SDK can schedule their
    final close callback on the loop. If the loop is torn down in the same
    tick, that callback raises "Event loop is closed" as an unraisable
    exception, which pytest reports non-deterministically — it failed one run
    and passed the next. Yielding twice gives those callbacks a tick to run.
    See docs/failure-analysis.md F-006.
    """
    await asyncio.sleep(0)
    await asyncio.sleep(0)


@pytest.fixture
async def fetcher() -> AsyncIterator[SourceFetcher]:
    """A fetcher that is always closed, and whose teardown is drained."""
    f = SourceFetcher(Settings(environment="local"))
    try:
        yield f
    finally:
        await f.aclose()
        await _drain()


@pytest.fixture
async def provider() -> AsyncIterator[AnthropicSearchProvider]:
    p = AnthropicSearchProvider(Settings(environment="local"))
    try:
        yield p
    finally:
        await p.aclose()
        await _drain()


# A stable, text-heavy, robots-friendly page that exists to be fetched.
REAL_URL = "https://example.com/"
# A larger real document, for chunking behaviour on realistic prose.
REAL_ARTICLE = "https://www.gutenberg.org/cache/epub/74/pg74.txt"


class TestRealFetch:
    async def test_fetches_a_real_page_end_to_end(self, fetcher: SourceFetcher) -> None:
        """Fetch → extract → normalise → hash, against the live internet."""
        source = await fetcher.fetch(REAL_URL)

        assert source.status_code == 200
        assert source.text.strip()
        # Canonical text is already normalised: normalising again is a no-op.
        assert normalise_text(source.text) == source.text
        # The hash commits to exactly the bytes citations will be checked against.
        assert source.content_hash == content_hash(source.text)
        assert source.verify_hash()
        assert source.byte_length > 0

        print(
            f"\nLIVE FETCH {REAL_URL} -> {source.status_code} · "
            f"{source.byte_length} bytes · {len(source.text)} chars · "
            f"title={source.title[:60]!r}"
        )

    async def test_real_document_chunks_with_verified_offsets(self, fetcher: SourceFetcher) -> None:
        """The offset invariant on a real document, not a crafted fixture."""
        settings = Settings(environment="local")
        source = await fetcher.fetch(REAL_ARTICLE)

        from app.retrieval.chunking import chunk_source

        chunks = chunk_source(
            source,
            chunk_tokens=settings.chunk_tokens,
            overlap_tokens=settings.chunk_overlap_tokens,
        )

        assert len(chunks) > 10, "a book-length text should produce many chunks"
        bad = verify_chunks(source, chunks)
        assert bad == [], f"offset drift on real content at chunks {bad}"
        # Every chunk must be a genuine substring of the source.
        for chunk in chunks:
            assert source.text[chunk.start_char : chunk.end_char] == chunk.text

        print(
            f"\nLIVE CHUNK {REAL_ARTICLE.rsplit('/', 1)[-1]} · "
            f"{len(source.text)} chars -> {len(chunks)} chunks · "
            f"all {len(chunks)} offsets verified"
        )

    async def test_ssrf_guard_blocks_a_real_loopback_fetch(self, fetcher: SourceFetcher) -> None:
        """With real DNS, a loopback URL must still be refused.

        The offline suite proves the classification logic; this proves the
        guard is actually wired into the fetch path with a real resolver.
        """
        for url in ("http://127.0.0.1/", "http://localhost/", "file:///etc/passwd"):
            with pytest.raises(UnsafeURLError):
                await fetcher.fetch(url)
        print("\nLIVE SSRF: loopback, localhost and file:// all refused")


class TestRealSearch:
    @pytest.mark.skipif(not _key_available(), reason="ANTHROPIC_API_KEY not resolvable")
    async def test_discovers_real_sources(self, provider: AnthropicSearchProvider) -> None:
        """Server-side web search returns usable candidates. Billable."""
        candidates = await provider.search(
            "UK pet insurance market size gross written premium",
            max_results=6,
            sub_question_id="SQ1",
        )

        assert candidates, "search returned no candidates"
        assert all(isinstance(c, SourceCandidate) for c in candidates)
        assert all(str(c.url).startswith("http") for c in candidates)
        # Deduplicated by url.
        assert len({str(c.url) for c in candidates}) == len(candidates)
        assert all(c.sub_question_id == "SQ1" for c in candidates)

        print(f"\nLIVE SEARCH: {len(candidates)} candidates")
        for c in candidates:
            print(f"  {c.domain:35} {str(c.url)[:80]}")

    @pytest.mark.skipif(not _key_available(), reason="ANTHROPIC_API_KEY not resolvable")
    async def test_discovered_sources_can_actually_be_fetched(
        self, provider: AnthropicSearchProvider, fetcher: SourceFetcher
    ) -> None:
        """The two halves joined: discovery feeds ingestion.

        Not every discovered URL will fetch — paywalls, bot walls and 403s are
        normal on the open web. The assertion is that *some* do, because a
        discovery step whose results are all unfetchable is useless.
        """
        settings = Settings(environment="local")
        candidates = await provider.search(
            "UK Financial Conduct Authority general insurance statistics",
            max_results=5,
        )
        assert candidates

        fetched, failed = [], []
        for candidate in candidates:
            try:
                fetched.append(await fetcher.fetch(str(candidate.url)))
            except Exception as exc:
                failed.append((candidate.domain, type(exc).__name__))

        print(f"\nLIVE DISCOVER+FETCH: {len(fetched)} fetched, {len(failed)} failed")
        for source in fetched:
            print(f"  OK   {source.domain:30} {len(source.text)} chars")
        for domain, error in failed:
            print(f"  FAIL {domain:30} {error}")

        assert fetched, f"no discovered source could be fetched; failures: {failed}"
        # And the invariant holds on whatever really came back.
        from app.retrieval.chunking import chunk_source

        for source in fetched:
            chunks = chunk_source(source, chunk_tokens=settings.chunk_tokens)
            assert verify_chunks(source, chunks) == []
