"""Stage 3 (PROCESS): fetch, chunk, and survive partial failure."""

from __future__ import annotations

import httpx
import respx

from app.core.config import Settings
from app.pipeline.process import OffsetIntegrityError, run_process_stage
from app.schemas.runs import Stage
from app.schemas.source import SourceCandidate
from app.tools.fetch import SourceFetcher
from tests.security.test_ssrf import FakeResolver

HTML = (
    "<html><head><title>T</title></head><body><article><h1>Overview</h1>"
    + "<p>A substantive paragraph about the market under review.</p>" * 20
    + "</article></body></html>"
)


def _settings(**kw: object) -> Settings:
    return Settings(environment="test", _env_file=None, **kw)  # type: ignore[arg-type,call-arg]


def _candidates(*urls: str) -> list[SourceCandidate]:
    return [SourceCandidate(url=u, title="t") for u in urls]


class TestHappyPath:
    @respx.mock
    async def test_fetches_and_chunks_every_source(self) -> None:
        for path in ("a", "b"):
            respx.get(f"https://example.com/{path}").mock(
                return_value=httpx.Response(200, html=HTML, headers={"content-type": "text/html"})
            )
        settings = _settings()
        result, metric = await run_process_stage(
            _candidates("https://example.com/a", "https://example.com/b"),
            fetcher=SourceFetcher(settings, resolver=FakeResolver()),
            settings=settings,
        )
        assert len(result.sources) == 2
        assert result.chunks
        assert result.failures == []
        assert metric.stage is Stage.PROCESS
        # No model call in this stage, so no cost should be attributed to it.
        assert metric.model is None
        assert metric.calls == 0
        assert metric.cost_usd == 0.0

    @respx.mock
    async def test_chunks_are_grouped_by_source(self) -> None:
        for path in ("a", "b"):
            respx.get(f"https://example.com/{path}").mock(
                return_value=httpx.Response(200, html=HTML, headers={"content-type": "text/html"})
            )
        settings = _settings()
        result, _ = await run_process_stage(
            _candidates("https://example.com/a", "https://example.com/b"),
            fetcher=SourceFetcher(settings, resolver=FakeResolver()),
            settings=settings,
        )
        grouped = result.chunks_by_source
        assert len(grouped) == 2
        assert sum(len(v) for v in grouped.values()) == len(result.chunks)

    @respx.mock
    async def test_offsets_verify_against_their_source(self) -> None:
        respx.get("https://example.com/a").mock(
            return_value=httpx.Response(200, html=HTML, headers={"content-type": "text/html"})
        )
        settings = _settings()
        result, _ = await run_process_stage(
            _candidates("https://example.com/a"),
            fetcher=SourceFetcher(settings, resolver=FakeResolver()),
            settings=settings,
        )
        source = result.sources[0]
        for chunk in result.chunks:
            assert chunk.verify_against(source.text)


class TestPartialFailure:
    @respx.mock
    async def test_one_bad_source_does_not_fail_the_stage(self) -> None:
        """Five of eight sources is a usable result with a recorded gap."""
        respx.get("https://example.com/ok").mock(
            return_value=httpx.Response(200, html=HTML, headers={"content-type": "text/html"})
        )
        respx.get("https://example.com/gone").mock(return_value=httpx.Response(404))
        settings = _settings()
        result, _ = await run_process_stage(
            _candidates("https://example.com/ok", "https://example.com/gone"),
            fetcher=SourceFetcher(settings, resolver=FakeResolver()),
            settings=settings,
        )
        assert len(result.sources) == 1
        assert len(result.failures) == 1
        assert result.failures[0].url == "https://example.com/gone"
        assert result.failures[0].code == "upstream_error"

    @respx.mock
    async def test_blocked_url_is_recorded_as_a_failure_not_an_exception(self) -> None:
        respx.get("https://example.com/ok").mock(
            return_value=httpx.Response(200, html=HTML, headers={"content-type": "text/html"})
        )
        settings = _settings(blocked_source_domains=("spam.test",))
        result, _ = await run_process_stage(
            _candidates("https://example.com/ok", "https://spam.test/x"),
            fetcher=SourceFetcher(settings, resolver=FakeResolver()),
            settings=settings,
        )
        assert len(result.sources) == 1
        assert [f.code for f in result.failures] == ["unsafe_url"]

    @respx.mock
    async def test_all_sources_failing_yields_an_empty_but_valid_result(self) -> None:
        respx.get("https://example.com/a").mock(return_value=httpx.Response(500))
        settings = _settings()
        result, metric = await run_process_stage(
            _candidates("https://example.com/a"),
            fetcher=SourceFetcher(settings, resolver=FakeResolver()),
            settings=settings,
        )
        assert result.sources == []
        assert result.chunks == []
        assert len(result.failures) == 1
        assert metric.duration_ms >= 0

    async def test_no_candidates_is_not_an_error(self) -> None:
        settings = _settings()
        result, _ = await run_process_stage(
            [], fetcher=SourceFetcher(settings, resolver=FakeResolver()), settings=settings
        )
        assert result.sources == []


class TestSourceCeiling:
    @respx.mock
    async def test_candidates_beyond_the_ceiling_are_not_fetched(self) -> None:
        routes = []
        for i in range(6):
            routes.append(
                respx.get(f"https://example.com/{i}").mock(
                    return_value=httpx.Response(
                        200, html=HTML, headers={"content-type": "text/html"}
                    )
                )
            )
        settings = _settings(max_sources_per_run=2)
        result, _ = await run_process_stage(
            _candidates(*[f"https://example.com/{i}" for i in range(6)]),
            fetcher=SourceFetcher(settings, resolver=FakeResolver()),
            settings=settings,
        )
        assert len(result.sources) == 2
        # The ceiling is a spend control: the extra URLs must not be requested.
        assert sum(r.call_count for r in routes) == 2


class TestOffsetIntegrityGuard:
    def test_offset_integrity_error_is_an_app_error_with_a_code(self) -> None:
        """A chunk whose offsets drift must fail loudly, not degrade quietly."""
        err = OffsetIntegrityError("bad", details={"chunk_indexes": [3]})
        assert err.code == "offset_integrity_error"
        assert err.details["chunk_indexes"] == [3]
