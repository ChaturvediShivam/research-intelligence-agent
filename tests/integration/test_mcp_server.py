"""MCP boundary tests: discovery, invocation, and what must survive crossing.

Every test here drives a **real `fastmcp.Client`** against the real server
over FastMCP's in-memory transport, so the MCP protocol itself is exercised —
`initialize`, `tools/list`, `tools/call` — rather than the handlers being
called directly. Only the outbound transports are faked: HTTP via respx, the
model via `ScriptedLLM`, search via `FakeSourceProvider`. No API calls.

The distinction that matters: these are not "does the handler work" tests
(M0-M7 already cover that). They test the boundary — that an external client
can discover the tools, that the schema it discovers is the schema that
validates its call, that failures arrive structured, and that provenance,
citations and UNKNOWN survive serialisation out of the process.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest
import respx
from fastmcp import Client

from app.core.config import Settings
from app.llm.client import LLMClient
from app.mcp.server import build_server
from app.schemas.source import SourceCandidate
from app.tools.fetch import SourceFetcher
from app.tools.registry import TOOLS, TOOLS_BY_NAME, ToolContext, get_tool
from tests.fixtures.fake_pipeline import FakeSourceProvider, ScriptedLLM
from tests.security.test_ssrf import FakeResolver

HTML = """
<html><head><title>FCA value measures</title></head><body>
<h1>General insurance value measures</h1>
<p>The Financial Conduct Authority publishes general insurance value measures
data covering claims frequencies, claims acceptance rates and average payouts
for home, motor and travel insurance products sold in the United Kingdom.</p>
<p>Claims acceptance rates for home emergency products averaged 78 per cent
across the reporting period, the lowest of any product line reported.</p>
</body></html>
"""

URL = "https://www.fca.org.uk/value-measures"


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        anthropic_api_key="sk-ant-test",
        sqlite_path=str(tmp_path / "mcp.db"),
        max_sources_per_run=3,
    )


CANDIDATES = [SourceCandidate(url=URL, title="FCA value measures")]


def make_context(settings: Settings, **overrides: Any) -> ToolContext:
    """A context whose transports are fakes but whose handlers are real.

    Only the three outbound edges are stood in: search, HTTP and the model.
    Everything the handlers then call is production code.
    """
    context = ToolContext(settings=settings)
    context._provider = overrides.get("provider", FakeSourceProvider(list(CANDIDATES)))
    # FakeResolver lets a hostname resolve offline, but it answers for a
    # literal IP too — so an SSRF test that used it would pass for the wrong
    # reason. Those tests ask for the real resolver instead.
    context._fetcher = SourceFetcher(
        settings,
        **({} if overrides.get("real_resolver") else {"resolver": FakeResolver()}),
    )
    context._llm = LLMClient(
        settings,
        client=overrides.get("llm", ScriptedLLM(sub_question_ids=["SQ1"], discovery_turns=1)),
    )
    return context


def server_for(settings: Settings, **overrides: Any):
    return build_server(settings, make_context(settings, **overrides))


# ==========================================================================
# 6 · Discovery
# ==========================================================================


class TestDiscovery:
    async def test_server_starts_and_a_client_can_connect(self, settings: Settings) -> None:
        """A client completes the MCP handshake and gets a usable session."""
        async with Client(server_for(settings)) as client:
            assert client.is_connected()
            assert await client.list_tools()

    async def test_all_four_approved_tools_are_discoverable(self, settings: Settings) -> None:
        """Architecture §8 names exactly these four."""
        async with Client(server_for(settings)) as client:
            names = {t.name for t in await client.list_tools()}
        assert names == {
            "search_sources",
            "fetch_and_index",
            "retrieve_evidence",
            "run_research",
        }

    async def test_tool_names_match_the_registry_exactly(self, settings: Settings) -> None:
        """Names are the client's API; the registry is their only source."""
        async with Client(server_for(settings)) as client:
            names = {t.name for t in await client.list_tools()}
        assert names == set(TOOLS_BY_NAME)

    async def test_every_tool_has_a_description(self, settings: Settings) -> None:
        async with Client(server_for(settings)) as client:
            for tool in await client.list_tools():
                assert tool.description
                assert len(tool.description) > 40, tool.name

    async def test_advertised_schema_is_the_schema_that_validates(self, settings: Settings) -> None:
        """A discovered schema that differs from the validating one is a lie."""
        async with Client(server_for(settings)) as client:
            for tool in await client.list_tools():
                expected = get_tool(tool.name).input_model.model_json_schema()
                assert tool.input_schema["properties"].keys() == expected["properties"].keys()
                assert tool.input_schema.get("required") == expected.get("required")

    async def test_schemas_reject_unknown_arguments(self, settings: Settings) -> None:
        async with Client(server_for(settings)) as client:
            for tool in await client.list_tools():
                assert tool.input_schema.get("additionalProperties") is False, tool.name

    async def test_billable_tools_are_marked_for_the_client(self, settings: Settings) -> None:
        """A client should be able to see which calls cost money."""
        async with Client(server_for(settings)) as client:
            meta = {t.name: (t.meta or {}) for t in await client.list_tools()}
        assert meta["run_research"].get("billable") is True
        assert meta["fetch_and_index"].get("billable") is False

    async def test_server_advertises_usage_instructions(self, settings: Settings) -> None:
        """Instructions tell a client how to treat returned source text."""
        from app.mcp.server import INSTRUCTIONS

        assert "untrusted" in INSTRUCTIONS
        assert "UNKNOWN" in INSTRUCTIONS


# ==========================================================================
# 7 · Invocation
# ==========================================================================


class TestInvocation:
    async def test_search_sources_routes_to_the_provider(self, settings: Settings) -> None:
        provider = FakeSourceProvider(list(CANDIDATES))
        async with Client(server_for(settings, provider=provider)) as client:
            result = await client.call_tool("search_sources", {"query": "insurance value measures"})
        assert result.data["status"] == "ok"
        assert result.data["count"] == 1
        assert result.data["candidates"][0]["url"] == URL
        # It delegated rather than reimplementing search.
        assert provider.queries

    @respx.mock
    async def test_fetch_and_index_returns_provenance(self, settings: Settings) -> None:
        respx.get(URL).mock(return_value=httpx.Response(200, html=HTML))
        async with Client(server_for(settings)) as client:
            result = await client.call_tool("fetch_and_index", {"url": URL})

        data = result.data
        assert data["status"] == "ok"
        assert data["chunks"] >= 1
        # Provenance that must not be dropped at the boundary.
        assert data["source_id"]
        assert len(data["content_hash"]) == 64
        assert data["credibility"] == "primary"
        assert data["domain"] == "fca.org.uk"
        assert data["status_code"] == 200

    @respx.mock
    async def test_retrieve_evidence_returns_verifiable_offsets(self, settings: Settings) -> None:
        """Offsets are the thing that makes a later citation checkable."""
        respx.get(URL).mock(return_value=httpx.Response(200, html=HTML))
        async with Client(server_for(settings)) as client:
            result = await client.call_tool(
                "retrieve_evidence",
                {"url": URL, "query": "claims acceptance rates", "k": 3},
            )

        data = result.data
        assert data["status"] == "ok"
        assert data["passages"]
        for passage in data["passages"]:
            assert passage["end_char"] > passage["start_char"]
            assert passage["source_id"] == data["source_id"]
            assert passage["retrievers"]

    @respx.mock
    async def test_k_bounds_the_number_of_passages(self, settings: Settings) -> None:
        respx.get(URL).mock(return_value=httpx.Response(200, html=HTML))
        async with Client(server_for(settings)) as client:
            result = await client.call_tool(
                "retrieve_evidence", {"url": URL, "query": "claims", "k": 1}
            )
        assert len(result.data["passages"]) <= 1


# ==========================================================================
# 4 · Input validation — deterministic, structured, and free
# ==========================================================================


class TestInputValidation:
    async def test_missing_required_argument_is_a_structured_error(
        self, settings: Settings
    ) -> None:
        async with Client(server_for(settings)) as client:
            result = await client.call_tool("search_sources", {})
        assert result.data["status"] == "error"
        assert result.data["code"] == "invalid_input"
        assert result.data["details"]["errors"]

    async def test_invalid_value_names_the_offending_field(self, settings: Settings) -> None:
        async with Client(server_for(settings)) as client:
            result = await client.call_tool("search_sources", {"query": "x"})
        fields = {e["field"] for e in result.data["details"]["errors"]}
        assert "query" in fields

    async def test_out_of_range_value_is_rejected(self, settings: Settings) -> None:
        async with Client(server_for(settings)) as client:
            result = await client.call_tool(
                "search_sources", {"query": "a real query", "max_results": 500}
            )
        assert result.data["code"] == "invalid_input"

    async def test_unknown_argument_is_rejected(self, settings: Settings) -> None:
        async with Client(server_for(settings)) as client:
            result = await client.call_tool(
                "search_sources", {"query": "a real query", "surprise": True}
            )
        assert result.data["code"] == "invalid_input"

    async def test_invalid_input_never_reaches_the_provider(self, settings: Settings) -> None:
        """An invalid call must not cost money. search_sources is billable."""
        provider = FakeSourceProvider(list(CANDIDATES))
        async with Client(server_for(settings, provider=provider)) as client:
            await client.call_tool("search_sources", {"query": "x"})
        assert not provider.queries

    async def test_wrong_type_is_rejected_rather_than_coerced_into_a_run(
        self, settings: Settings
    ) -> None:
        async with Client(server_for(settings)) as client:
            result = await client.call_tool("run_research", {"question": 42})
        assert result.data["code"] == "invalid_input"

    async def test_unknown_tool_name_fails_without_reaching_a_handler(
        self, settings: Settings
    ) -> None:
        from fastmcp.exceptions import ToolError

        async with Client(server_for(settings)) as client:
            with pytest.raises(ToolError):
                await client.call_tool("delete_everything", {})


# ==========================================================================
# 7 · Failure propagation
# ==========================================================================


class TestFailurePropagation:
    @respx.mock
    async def test_upstream_failure_is_structured_not_an_exception(
        self, settings: Settings
    ) -> None:
        respx.get(URL).mock(return_value=httpx.Response(503))
        async with Client(server_for(settings)) as client:
            result = await client.call_tool("fetch_and_index", {"url": URL})
        assert result.data["status"] == "error"
        assert result.data["code"]
        assert "message" in result.data

    async def test_blocked_url_is_refused_at_the_mcp_boundary(self, settings: Settings) -> None:
        """The SSRF guard applies to MCP callers because the fetcher applies it."""
        async with Client(server_for(settings, real_resolver=True)) as client:
            result = await client.call_tool(
                "fetch_and_index", {"url": "http://169.254.169.254/latest/meta-data/"}
            )
        assert result.data["status"] == "error"
        assert result.data["code"] == "unsafe_url"

    async def test_file_scheme_is_refused(self, settings: Settings) -> None:
        async with Client(server_for(settings, real_resolver=True)) as client:
            result = await client.call_tool("fetch_and_index", {"url": "file:///etc/passwd"})
        assert result.data["status"] == "error"
        assert result.data["code"] == "unsafe_url"

    async def test_an_unexpected_exception_becomes_a_structured_error(
        self, settings: Settings
    ) -> None:
        """The boundary must not leak a traceback to an external client."""

        class Exploding:
            async def search(self, *a: Any, **k: Any) -> list[Any]:
                raise RuntimeError("boom, with internal detail")

        async with Client(server_for(settings, provider=Exploding())) as client:
            result = await client.call_tool("search_sources", {"query": "a real query"})
        assert result.data["status"] == "error"
        assert result.data["code"] == "internal_error"
        assert "RuntimeError" in result.data["message"]


# ==========================================================================
# 3, 7 · What must survive the boundary
# ==========================================================================


class TestSemanticsSurviveTheBoundary:
    @respx.mock
    async def test_run_research_returns_the_full_report(self, settings: Settings) -> None:
        respx.get(URL).mock(return_value=httpx.Response(200, html=HTML))
        async with Client(server_for(settings)) as client:
            result = await client.call_tool(
                "run_research",
                {"question": "What are UK insurance claims acceptance rates?"},
            )

        data = result.data
        report = data["report"]
        assert data["status"] in {"completed", "partial"}
        assert report["run_id"]
        # M6 structure intact, not flattened to prose.
        assert "sub_questions" in report
        assert "information_gaps" in report
        assert "source_coverage" in report
        assert "limitations" in report

    @respx.mock
    async def test_citations_cross_the_boundary_with_their_offsets(
        self, settings: Settings
    ) -> None:
        """A citation without offsets cannot be verified by the recipient."""
        respx.get(URL).mock(return_value=httpx.Response(200, html=HTML))
        async with Client(server_for(settings)) as client:
            result = await client.call_tool(
                "run_research",
                {"question": "What are UK insurance claims acceptance rates?"},
            )

        report = result.data["report"]
        citations = [
            citation
            for assessment in report["sub_questions"]
            for claim in assessment["supporting_claims"]
            for citation in claim["citations"]
        ]
        assert citations, "a successful run should carry citations"
        for citation in citations:
            assert citation["source_id"]
            assert citation["cited_text"]
            assert citation["end_char"] > citation["start_char"]

    @respx.mock
    async def test_unknown_status_survives(self, settings: Settings) -> None:
        """A run that finds nothing must report UNKNOWN, not an empty success."""
        respx.get(URL).mock(return_value=httpx.Response(404))
        async with Client(server_for(settings)) as client:
            result = await client.call_tool(
                "run_research", {"question": "A question nothing can answer here?"}
            )

        report = result.data["report"]
        assert result.data["claims_supported"] == 0
        statuses = {s["status"] for s in report["sub_questions"]}
        assert statuses <= {"unknown", "partial"}
        assert report["information_gaps"], "a gap must be stated, not implied"

    @respx.mock
    async def test_information_gap_carries_its_cause(self, settings: Settings) -> None:
        respx.get(URL).mock(return_value=httpx.Response(404))
        async with Client(server_for(settings)) as client:
            result = await client.call_tool(
                "run_research", {"question": "A question nothing can answer here?"}
            )
        for gap in result.data["report"]["information_gaps"]:
            assert gap["cause"]
            assert gap["why_insufficient"]

    @respx.mock
    async def test_failed_sources_are_reported_not_hidden(self, settings: Settings) -> None:
        respx.get(URL).mock(return_value=httpx.Response(403))
        async with Client(server_for(settings)) as client:
            result = await client.call_tool(
                "run_research", {"question": "A question whose source is paywalled?"}
            )
        assert result.data["sources_failed"] >= 1
        assert result.data["report"]["source_coverage"]["failed"] >= 1

    @respx.mock
    async def test_measured_cost_crosses_the_boundary(self, settings: Settings) -> None:
        respx.get(URL).mock(return_value=httpx.Response(200, html=HTML))
        async with Client(server_for(settings)) as client:
            result = await client.call_tool(
                "run_research", {"question": "What are UK claims acceptance rates?"}
            )
        assert "total_cost_usd" in result.data
        assert result.data["total_duration_ms"] >= 0


# ==========================================================================
# 8 · No pipeline bypass
# ==========================================================================


class TestNoPipelineBypass:
    async def test_no_tool_exposes_a_stage_directly(self) -> None:
        """MCP must not offer synthesis without the verification before it.

        A `synthesise` or `extract` tool would let a caller assemble prose
        from unverified input — the exact path the pipeline's ordering exists
        to prevent.
        """
        forbidden = {"synthesise", "synthesize", "extract", "validate", "assess", "plan"}
        assert forbidden.isdisjoint(set(TOOLS_BY_NAME))

    @respx.mock
    async def test_run_research_goes_through_the_orchestrator(
        self, settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The regression test for a parallel execution path.

        If someone later reimplements the pipeline inside the MCP layer, the
        orchestrator stops being called and this fails.
        """
        import app.pipeline.orchestrator as orchestrator_module

        calls: list[str] = []
        original = orchestrator_module.ResearchOrchestrator.run

        async def tracked(self: Any, request: Any) -> Any:
            calls.append(request.question)
            return await original(self, request)

        monkeypatch.setattr(orchestrator_module.ResearchOrchestrator, "run", tracked)
        respx.get(URL).mock(return_value=httpx.Response(200, html=HTML))

        async with Client(server_for(settings)) as client:
            await client.call_tool(
                "run_research", {"question": "What are UK claims acceptance rates?"}
            )

        assert calls == ["What are UK claims acceptance rates?"]

    @respx.mock
    async def test_every_stage_ran(self, settings: Settings) -> None:
        """Not just the orchestrator object — the full stage sequence."""
        respx.get(URL).mock(return_value=httpx.Response(200, html=HTML))
        async with Client(server_for(settings)) as client:
            result = await client.call_tool(
                "run_research", {"question": "What are UK claims acceptance rates?"}
            )

        stages = result.data["stages"]
        from app.schemas.runs import Stage

        for stage in (
            Stage.PLAN,
            Stage.DISCOVER,
            Stage.PROCESS,
            Stage.RETRIEVE,
            Stage.EXTRACT,
            Stage.SYNTHESISE,
            Stage.VALIDATE,
            Stage.ASSESS,
        ):
            assert stage.value in stages, f"{stage.value} did not run via MCP"
            assert stages[stage.value] == "passed"

    async def test_handlers_are_the_registry_handlers(self) -> None:
        """The MCP layer holds no handler of its own."""
        import app.mcp.server as server_module

        for spec in TOOLS:
            assert spec.handler.__module__ == "app.tools.registry"
        # And the server module defines no business logic of its own.
        assert not hasattr(server_module, "search")
        assert not hasattr(server_module, "fetch")

    async def test_max_sources_is_honoured_as_a_cost_ceiling(self, settings: Settings) -> None:
        """An MCP caller cannot raise the ceiling past the schema's bound."""
        async with Client(server_for(settings)) as client:
            result = await client.call_tool(
                "run_research",
                {"question": "A legitimate research question here?", "max_sources": 999},
            )
        assert result.data["code"] == "invalid_input"


class TestOneSearchImplementation:
    """§6's real guarantee: one execution path, whatever the wrapper."""

    def test_both_callers_use_the_same_tool_name(self) -> None:
        """Stage 2's tool and the MCP tool must not drift apart in name."""
        import inspect

        import app.pipeline.discover as discover

        source = inspect.getsource(discover.run_discover_stage)
        assert "async def search_sources(" in source
        assert "search_sources" in TOOLS_BY_NAME

    def test_the_registry_does_not_reimplement_search(self) -> None:
        """The handler must call the provider, not an HTTP client of its own."""
        import inspect

        from app.tools import registry

        source = inspect.getsource(registry.handle_search_sources)
        assert "context.provider.search" in source
        assert "httpx" not in source
