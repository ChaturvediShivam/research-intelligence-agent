"""Runs golden cases through the real pipeline, offline.

The transports are the only things stood in for: the search provider returns
the case's declared candidates, and HTTP serves the golden corpus from
`retrieval_v1`. Everything measured — fetching, chunking, embedding, hybrid
retrieval, quote location, offset arithmetic, citation verification, report
assembly — is the real implementation.

The LLM transport is `ScriptedLLM`, which is content-aware: it quotes real
sentences from the chunks it is handed and cites real offsets. That is what
lets citation correctness be a genuine measurement rather than a tautology:
the evaluator re-slices the stored source itself and either agrees or does
not.

This lives under `tests/` deliberately. `app/` must not depend on respx or on
test fixtures, so the harness wires them and `app.evaluation` only ever sees
a finished `RunResult`.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import respx

from app.core.config import Settings
from app.evaluation.golden import GoldenCase
from app.llm.client import LLMClient
from app.pipeline.orchestrator import ResearchOrchestrator, RunResult
from app.retrieval.embeddings import FastEmbedEmbedder
from app.schemas.research import ResearchPlan, ResearchRequest, SourceType, SubQuestion
from app.schemas.source import SourceCandidate
from app.tools.fetch import SourceFetcher
from tests.fixtures.fake_pipeline import FakeSourceProvider, ScriptedLLM
from tests.security.test_ssrf import FakeResolver

CORPUS_DIR = Path("evals/datasets/retrieval_v1")


def load_corpus() -> dict[str, dict[str, str]]:
    """The golden corpus, keyed by doc_id."""
    path = CORPUS_DIR / "corpus.jsonl"
    return {
        doc["doc_id"]: doc
        for doc in (
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }


def doc_url_map(corpus: dict[str, dict[str, str]]) -> dict[str, str]:
    """doc_id -> the URL the fixture serves it at."""
    return {doc_id: doc["url"] for doc_id, doc in corpus.items()}


def _html(doc: dict[str, str]) -> str:
    """Wrap a corpus document as a page, so real extraction runs on it."""
    paragraphs = "".join(
        f"<p>{part.strip()}</p>" for part in doc["text"].split("\n\n") if part.strip()
    )
    return (
        f"<!doctype html><html><head><title>{doc['title']}</title></head>"
        f"<body><nav>Home | About</nav><article><h1>{doc['title']}</h1>"
        f"{paragraphs}</article><footer>Copyright</footer></body></html>"
    )


def _plan_for(case: GoldenCase) -> ResearchPlan:
    """A single-sub-question plan matching the case's question.

    One sub-question, because the golden case annotates relevance at the level
    of the whole question. A synthesised multi-part decomposition would make
    the retrieval judgments ambiguous about which part they applied to.
    """
    return ResearchPlan(
        restated_question=case.question,
        sub_questions=[
            SubQuestion(
                id="SQ1",
                question=case.question,
                rationale="The question as annotated in the golden set.",
                rank=1,
                expected_source_types=[SourceType.OFFICIAL_STATISTICS],
                answerable_if="A source in the corpus states the figure or fact.",
            )
        ],
    )


def _mount(case: GoldenCase, corpus: dict[str, dict[str, str]]) -> None:
    """Serve the case's documents, applying its source condition."""
    docs = [corpus[doc_id] for doc_id in case.available_docs]

    for index, doc in enumerate(docs):
        url = doc["url"]
        if case.source_condition == "all_fetch_fail":
            respx.get(url).mock(return_value=httpx.Response(404))
        elif case.source_condition == "access_restricted":
            respx.get(url).mock(return_value=httpx.Response(403))
        elif case.source_condition == "one_fetch_fail" and index == 0:
            respx.get(url).mock(return_value=httpx.Response(404))
        else:
            respx.get(url).mock(
                return_value=httpx.Response(
                    200, html=_html(doc), headers={"content-type": "text/html"}
                )
            )


async def run_case(
    case: GoldenCase,
    corpus: dict[str, dict[str, str]],
    embedder: FastEmbedEmbedder,
) -> RunResult:
    """Execute one golden case through the real pipeline."""
    settings = Settings(
        environment="test",
        _env_file=None,  # type: ignore[call-arg]
        max_sources_per_run=max(len(case.available_docs), 1),
        retrieval_top_k=5,
        chunk_tokens=120,
        chunk_overlap_tokens=16,
    )

    candidates = [
        SourceCandidate(url=corpus[doc_id]["url"], title=corpus[doc_id]["title"])
        for doc_id in case.available_docs
    ]

    llm = ScriptedLLM(plan=_plan_for(case), sub_question_ids=["SQ1"], discovery_turns=1)
    orchestrator = ResearchOrchestrator(
        client=LLMClient(settings, client=llm),
        provider=FakeSourceProvider(candidates),
        fetcher=SourceFetcher(settings, resolver=FakeResolver()),
        embedder=embedder,
        settings=settings,
    )

    with respx.mock:
        _mount(case, corpus)
        return await orchestrator.run(ResearchRequest(question=case.question))
