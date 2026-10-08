"""The shared tool registry (architecture §6).

§6 names this module the source of truth for tool definitions. What lives
here: the name, description, input schema and **handler** for each capability.
What does not live here: any business logic. Every handler delegates into the
module that already owns the work and was verified there — `app.tools.search`,
`app.tools.fetch`, `app.retrieval`, and for `run_research`, the orchestrator
itself. A second implementation path is the specific failure this registry
exists to prevent.

One honest qualification on §6's diagram. The **execution path** is shared:
stage 2's `search_sources` and this module's both call
`SourceProvider.search`, so there is exactly one search implementation. The
**tool definition** is not literally shared, because stage 2's version is a
closure that accumulates into a `DiscoveryResult`, dedupes URLs across
sub-questions and takes a `sub_question_id` the external tool has no use for.
Collapsing the two would mean either giving MCP callers a parameter that is
meaningless to them or giving the agent a stateless tool it cannot accumulate
with. The names are pinned together by a test instead, so they cannot drift.

`run_research` routes through `ResearchOrchestrator`, not through the stages
directly. An MCP caller therefore gets the same plan → discover → process →
retrieve → extract → validate → synthesise → verify → assess sequence, the
same citation verifier, and the same UNKNOWN semantics as any other caller.
There is no path by which MCP reaches a stage without the stages before it.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import structlog
from pydantic import BaseModel, Field

from app.core.config import Settings
from app.llm.client import LLMClient
from app.retrieval.embeddings import Embedder, FastEmbedEmbedder
from app.tools.fetch import SourceFetcher
from app.tools.search import AnthropicSearchProvider, SourceProvider

logger = structlog.get_logger(__name__)

REGISTRY_VERSION = "registry_v1"


# --------------------------------------------------------------------------
# Execution context
# --------------------------------------------------------------------------


@dataclass(slots=True)
class ToolContext:
    """Dependencies a handler needs, with real transports built on demand.

    Components are injectable so tests exercise the real handlers against
    fake transports — the handler under test is the production one, which is
    the whole point. Construction is lazy so a caller that only lists tools
    never builds an HTTP client or loads an embedding model.
    """

    settings: Settings
    _llm: LLMClient | None = None
    _provider: SourceProvider | None = None
    _fetcher: SourceFetcher | None = None
    _embedder: Embedder | None = None

    @property
    def llm(self) -> LLMClient:
        if self._llm is None:
            self._llm = LLMClient(self.settings)
        return self._llm

    @property
    def provider(self) -> SourceProvider:
        if self._provider is None:
            self._provider = AnthropicSearchProvider(self.settings)
        return self._provider

    @property
    def fetcher(self) -> SourceFetcher:
        if self._fetcher is None:
            self._fetcher = SourceFetcher(self.settings)
        return self._fetcher

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = FastEmbedEmbedder()
        return self._embedder

    async def aclose(self) -> None:
        """Close whatever was actually built."""
        if self._llm is not None:
            await self._llm.aclose()
        if self._fetcher is not None:
            await self._fetcher.aclose()
        provider = self._provider
        closer = getattr(provider, "aclose", None)
        if closer is not None:
            await closer()


# --------------------------------------------------------------------------
# Input schemas — strict, so an invalid call fails before any work happens
# --------------------------------------------------------------------------


class SearchSourcesInput(BaseModel):
    """Input for `search_sources`."""

    model_config = {"extra": "forbid"}

    query: str = Field(
        min_length=3,
        max_length=500,
        description="The search query. Use the terminology the kind of document you want would use.",
    )
    max_results: int = Field(
        default=6,
        ge=1,
        le=20,
        description="Maximum number of candidate sources to return.",
    )


class FetchAndIndexInput(BaseModel):
    """Input for `fetch_and_index`."""

    model_config = {"extra": "forbid"}

    url: str = Field(
        min_length=8,
        max_length=2000,
        description="HTTP(S) URL to fetch. Refused if it resolves to a non-public address.",
    )


class RetrieveEvidenceInput(BaseModel):
    """Input for `retrieve_evidence`."""

    model_config = {"extra": "forbid"}

    url: str = Field(
        min_length=8, max_length=2000, description="Source URL to fetch, index and search."
    )
    query: str = Field(
        min_length=3, max_length=500, description="What to retrieve from that source."
    )
    k: int = Field(default=5, ge=1, le=25, description="Number of passages to return.")


class RunResearchInput(BaseModel):
    """Input for `run_research`."""

    model_config = {"extra": "forbid"}

    question: str = Field(
        min_length=12,
        max_length=2000,
        description="The research question. Runs the full pipeline; this is the expensive tool.",
    )
    max_sources: int = Field(
        default=5,
        ge=1,
        le=12,
        description="Ceiling on sources fetched, bounding cost and latency.",
    )


# --------------------------------------------------------------------------
# Output schemas — structured, reusing existing vocabulary
# --------------------------------------------------------------------------


class CandidateOut(BaseModel):
    url: str
    title: str
    domain: str


class SearchSourcesOutput(BaseModel):
    status: str = "ok"
    query: str
    count: int
    candidates: list[CandidateOut] = Field(default_factory=list)


class FetchAndIndexOutput(BaseModel):
    status: str = "ok"
    url: str
    final_url: str
    domain: str
    title: str
    # Provenance that must survive the MCP boundary.
    source_id: str
    content_hash: str
    credibility: str
    status_code: int
    text_chars: int
    chunks: int
    redirect_chain: list[str] = Field(default_factory=list)


class PassageOut(BaseModel):
    """A retrieved passage with its offsets into the canonical source text."""

    rank: int
    text: str
    start_char: int
    end_char: int
    section: str | None = None
    source_id: str
    source_url: str
    # Which retrievers contributed it, as the hybrid retriever reports.
    retrievers: list[str] = Field(default_factory=list)


class RetrieveEvidenceOutput(BaseModel):
    status: str = "ok"
    url: str
    query: str
    source_id: str
    content_hash: str
    chunks_indexed: int
    passages: list[PassageOut] = Field(default_factory=list)


class RunResearchOutput(BaseModel):
    """The full report plus the run's measured facts.

    Carries the M6 report verbatim rather than a flattened summary: UNKNOWN
    status, information gaps, confidence and citations all have to survive the
    boundary, and re-serialising them by hand is how they would quietly stop
    doing so.
    """

    status: str
    run_id: str
    question: str
    report: dict[str, Any]
    # Measured, from the run's own trace.
    sources_fetched: int
    sources_failed: int
    evidence_items: int
    citations_verified: int
    citations_rejected: int
    claims_supported: int
    claims_unknown: int
    total_cost_usd: float
    total_duration_ms: int
    stages: dict[str, str] = Field(default_factory=dict)
    error: str | None = None


# --------------------------------------------------------------------------
# Handlers — each delegates, none implements
# --------------------------------------------------------------------------


async def handle_search_sources(
    payload: SearchSourcesInput, context: ToolContext
) -> SearchSourcesOutput:
    """Delegate to the existing `SourceProvider` (ADR-006).

    Goes through the provider abstraction, so the domain allow/deny policy it
    enforces applies to an MCP caller exactly as it does to stage 2.
    """
    candidates = await context.provider.search(payload.query, max_results=payload.max_results)
    return SearchSourcesOutput(
        query=payload.query,
        count=len(candidates),
        candidates=[
            CandidateOut(url=str(c.url), title=c.title, domain=c.domain) for c in candidates
        ],
    )


async def handle_fetch_and_index(
    payload: FetchAndIndexInput, context: ToolContext
) -> FetchAndIndexOutput:
    """Delegate to the existing fetcher and chunker.

    The SSRF guard runs because the fetcher runs; there is no MCP-specific
    fetch path to forget it.
    """
    from app.pipeline.validate import classify_credibility
    from app.retrieval.chunking import chunk_source, verify_chunks
    from app.schemas.evidence import source_id_for

    source = await context.fetcher.fetch(payload.url)
    chunks = chunk_source(
        source,
        chunk_tokens=context.settings.chunk_tokens,
        overlap_tokens=context.settings.chunk_overlap_tokens,
    )
    bad = verify_chunks(source, chunks)
    if bad:  # pragma: no cover - would mean an offset bug upstream
        from app.pipeline.process import OffsetIntegrityError

        raise OffsetIntegrityError(
            "Chunk offsets do not match source text.",
            details={"url": str(source.final_url), "chunk_indexes": bad},
        )

    return FetchAndIndexOutput(
        url=payload.url,
        final_url=str(source.final_url),
        domain=source.domain,
        title=source.title,
        source_id=source_id_for(str(source.final_url)),
        content_hash=source.content_hash,
        credibility=classify_credibility(source.domain).value,
        status_code=source.status_code,
        text_chars=len(source.text),
        chunks=len(chunks),
        redirect_chain=list(source.redirect_chain),
    )


async def handle_retrieve_evidence(
    payload: RetrieveEvidenceInput, context: ToolContext
) -> RetrieveEvidenceOutput:
    """Fetch, index and retrieve, using the existing retrieval layer.

    Stateless by design: each call builds its own ephemeral index and discards
    it, which is what ADR-004 already says the corpus is. Holding indexes
    between MCP calls would mean inventing a session store, which nothing has
    asked for.
    """
    from app.pipeline.retrieve import build_index
    from app.retrieval.chunking import chunk_source
    from app.retrieval.hybrid import HybridRetriever
    from app.schemas.evidence import source_id_for

    source = await context.fetcher.fetch(payload.url)
    chunks = chunk_source(
        source,
        chunk_tokens=context.settings.chunk_tokens,
        overlap_tokens=context.settings.chunk_overlap_tokens,
    )

    store = build_index(chunks, context.embedder)
    try:
        scored = HybridRetriever(store, context.embedder).retrieve(payload.query, k=payload.k)
        source_id = source_id_for(str(source.final_url))
        return RetrieveEvidenceOutput(
            url=payload.url,
            query=payload.query,
            source_id=source_id,
            content_hash=source.content_hash,
            chunks_indexed=len(chunks),
            passages=[
                PassageOut(
                    rank=s.rank,
                    text=s.chunk.text,
                    start_char=s.chunk.start_char,
                    end_char=s.chunk.end_char,
                    section=s.chunk.section,
                    source_id=source_id,
                    source_url=str(s.chunk.source_url),
                    retrievers=list(s.retrievers),
                )
                for s in scored
            ],
        )
    finally:
        store.close()


async def handle_run_research(payload: RunResearchInput, context: ToolContext) -> RunResearchOutput:
    """Delegate to `ResearchOrchestrator`. No stage is reached directly.

    This is the no-bypass guarantee in code: MCP constructs the same
    orchestrator any other caller uses and hands it a `ResearchRequest`. The
    report returned is the M6 report verbatim, so UNKNOWN status, information
    gaps, derived confidence and verified citations cross the boundary intact.
    """
    from app.pipeline.orchestrator import ResearchOrchestrator
    from app.schemas.research import ResearchRequest

    settings = context.settings.model_copy(update={"max_sources_per_run": payload.max_sources})
    orchestrator = ResearchOrchestrator(
        client=context.llm,
        provider=context.provider,
        fetcher=context.fetcher,
        embedder=context.embedder,
        settings=settings,
    )
    result = await orchestrator.run(
        ResearchRequest(question=payload.question, max_sources=payload.max_sources)
    )

    # Read the typed fields rather than `summary()`, whose values are `object`
    # and would need casting back out at the boundary.
    report = result.report
    coverage = report.source_coverage if report else None
    validation = result.claim_validation

    return RunResearchOutput(
        status=result.status.value,
        run_id=result.run_id,
        question=payload.question,
        report=report.model_dump(mode="json") if report else {},
        sources_fetched=coverage.fetched if coverage else 0,
        sources_failed=coverage.failed if coverage else 0,
        evidence_items=len(result.extraction.items) if result.extraction else 0,
        citations_verified=validation.verified_count if validation else 0,
        citations_rejected=validation.rejected_count if validation else 0,
        claims_supported=result.verified_claims,
        claims_unknown=len(validation.unknown_claims) if validation else 0,
        total_cost_usd=result.total_cost_usd,
        total_duration_ms=result.trace.total_duration_ms,
        stages={outcome.stage.value: outcome.status.value for outcome in result.stages},
        error=result.error,
    )


# --------------------------------------------------------------------------
# The registry
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """One capability: its name, description, input model and handler."""

    name: str
    title: str
    description: str
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    handler: Callable[[Any, ToolContext], Awaitable[BaseModel]]
    # Whether invoking it can spend money. Surfaced so an MCP client can see
    # which tools are cheap to explore and which are not.
    billable: bool


TOOLS: tuple[ToolSpec, ...] = (
    ToolSpec(
        name="search_sources",
        title="Search for candidate sources",
        description=(
            "Discover candidate sources for a research query. Returns URLs, "
            "titles and domains; it does not fetch or read them. Billable: "
            "runs a model-backed web search."
        ),
        input_model=SearchSourcesInput,
        output_model=SearchSourcesOutput,
        handler=handle_search_sources,
        billable=True,
    ),
    ToolSpec(
        name="fetch_and_index",
        title="Fetch, sanitise and chunk a source",
        description=(
            "Fetch one URL, extract its readable text, normalise it and chunk "
            "it. Returns provenance — source id, content hash, credibility "
            "tier, redirect chain — and the chunk count. URLs resolving to "
            "non-public addresses are refused. Not billable."
        ),
        input_model=FetchAndIndexInput,
        output_model=FetchAndIndexOutput,
        handler=handle_fetch_and_index,
        billable=False,
    ),
    ToolSpec(
        name="retrieve_evidence",
        title="Retrieve passages from a source",
        description=(
            "Fetch and index one URL, then retrieve the passages most "
            "relevant to a query using hybrid dense plus BM25 retrieval. "
            "Each passage carries character offsets into the canonical source "
            "text, so a citation built from it can be verified. Not billable."
        ),
        input_model=RetrieveEvidenceInput,
        output_model=RetrieveEvidenceOutput,
        handler=handle_retrieve_evidence,
        billable=False,
    ),
    ToolSpec(
        name="run_research",
        title="Run the full research pipeline",
        description=(
            "Run the complete pipeline for a research question: plan, "
            "discover sources, fetch, retrieve, extract evidence, verify "
            "citations, synthesise and assess. Returns a structured report "
            "with per-sub-question ANSWERED/PARTIAL/UNKNOWN status, verified "
            "citations, derived confidence, information gaps and measured "
            "cost. Billable and slow: expect multiple model calls."
        ),
        input_model=RunResearchInput,
        output_model=RunResearchOutput,
        handler=handle_run_research,
        billable=True,
    ),
)

TOOLS_BY_NAME: dict[str, ToolSpec] = {spec.name: spec for spec in TOOLS}


def get_tool(name: str) -> ToolSpec:
    """Look up a tool, or raise with the available names."""
    try:
        return TOOLS_BY_NAME[name]
    except KeyError as exc:
        raise KeyError(f"No tool named {name!r}. Available: {sorted(TOOLS_BY_NAME)}") from exc
