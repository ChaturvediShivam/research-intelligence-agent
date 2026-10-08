"""Run the approved evaluation (architecture §9) and write the artifact.

    uv run python scripts/run_evaluation.py                 # metrics 1-5, free
    uv run python scripts/run_evaluation.py --judge         # adds 6-7, billable
    uv run python scripts/run_evaluation.py --label after_improvement
    uv run python scripts/run_evaluation.py --compare evals/results/<baseline>.json

Metrics 1-5 are deterministic and cost nothing. Metrics 6-7 need the judge
the architecture specifies (Haiku), so they are opt-in behind --judge and are
the only thing in this script that spends credits.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from app.core.config import Settings
from app.evaluation.golden import load_golden_set
from app.evaluation.judge import JUDGE_RUBRIC_VERSION, judge_case
from app.evaluation.research_eval import (
    EVALUATION_VERSION,
    RETRIEVAL_K,
    EvaluationReport,
    score_case,
)
from app.llm.client import LLMClient
from app.retrieval.embeddings import FastEmbedEmbedder
from tests.fixtures.eval_harness import doc_url_map, load_corpus, run_case

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "evals" / "datasets" / "golden_v1"
RESULTS = ROOT / "evals" / "results"


def git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="baseline")
    parser.add_argument(
        "--judge",
        action="store_true",
        help="score metrics 6-7 with the LLM judge. BILLABLE.",
    )
    parser.add_argument(
        "--judge-cases",
        type=int,
        default=6,
        help="how many cases to judge (metrics 6-7 only); keeps spend bounded",
    )
    parser.add_argument("--compare", type=Path, default=None)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    corpus = load_corpus()
    golden = load_golden_set(GOLDEN, corpus_doc_ids=set(corpus))
    urls = doc_url_map(corpus)
    embedder = FastEmbedEmbedder()

    print(
        f"golden set: {golden.version}  cases: {len(golden.cases)} "
        f"(train {len(golden.train)}, holdout {len(golden.holdout)})"
    )
    print(f"evaluation: {EVALUATION_VERSION}  sha: {git_sha()}  k={RETRIEVAL_K}")

    report = EvaluationReport(
        evaluation_version=EVALUATION_VERSION,
        golden_set_version=golden.version,
        git_sha=git_sha(),
        retrieval_k=RETRIEVAL_K,
        label=args.label,
    )

    runs = {}
    for case in golden.cases:
        result = await run_case(case, corpus, embedder)
        runs[case.case_id] = result
        report.cases.append(score_case(case, result, urls))

    judge_cost = 0.0
    judge_calls = 0
    if args.judge:
        settings = Settings(environment="local")
        client = LLMClient(settings)
        # Judge a bounded, deterministic slice: the first N cases by id that
        # actually produced a report. Fixed selection so the metric is
        # comparable between runs rather than sampled.
        judgeable = [
            c
            for c in sorted(report.cases, key=lambda c: c.case_id)
            if runs[c.case_id].report is not None
        ][: args.judge_cases]
        print(f"judging {len(judgeable)} case(s) with {settings.extraction_model}")
        try:
            for scored in judgeable:
                run = runs[scored.case_id]
                if run.report is None:  # pragma: no cover - filtered above
                    continue
                verdict = await judge_case(
                    scored.case_id,
                    scored.question,
                    run.report,
                    client=client,
                    settings=settings,
                )
                scored.answer_relevance = verdict.answer_relevance
                scored.report_quality = verdict.report_quality
                judge_cost += verdict.cost_usd
                judge_calls += verdict.calls
        finally:
            await client.aclose()
            await asyncio.sleep(0)
            await asyncio.sleep(0)

    print()
    for line in report.summary_lines():
        print(line)
    print()
    print("per-case:")
    for case in report.cases:
        recall = "  n/a" if case.recall_at_k is None else f"{case.recall_at_k:.2f}"
        coverage = "  n/a" if case.source_coverage is None else f"{case.source_coverage:.2f}"
        citation = (
            "  n/a" if case.citation_correctness is None else f"{case.citation_correctness:.2f}"
        )
        unsupported = (
            "  n/a" if case.unsupported_claim_rate is None else f"{case.unsupported_claim_rate:.2f}"
        )
        flag = "" if case.unknown_expectation_met else "  <-- expectation missed"
        print(
            f"  {case.case_id} {case.split:<7} {case.run_status:<9} "
            f"recall={recall} cov={coverage} cite={citation} "
            f"unsup={unsupported} tools={case.tool_sequence_correct}{flag}"
        )

    if args.judge:
        print(f"\njudge: {judge_calls} calls, ${judge_cost:.6f}")

    if args.compare is not None:
        baseline_payload = json.loads(args.compare.read_text(encoding="utf-8"))
        print(
            f"\ncompared against {args.compare.name} "
            f"({baseline_payload.get('label')}, sha {baseline_payload.get('git_sha')})"
        )
        print(f"\n  {'metric':<26} {'before':>8} {'after':>8}  verdict")
        print("  " + "-" * 60)
        before = baseline_payload["metrics"]
        for metric in report.metrics():
            was = before.get(metric.name, {}).get("value")
            now = metric.value
            if was is None and now is None:
                verdict = "not measured"
            elif was is None:
                verdict = "newly measured"
            elif now is None:
                verdict = "no longer measured"
            elif abs(now - was) < 1e-9:
                verdict = "unchanged"
            else:
                lower_better = metric.name == "unsupported_claim_rate"
                better = (now < was) if lower_better else (now > was)
                verdict = "improved" if better else "regressed"
            print(
                f"  {metric.name:<26} "
                f"{'n/a' if was is None else f'{was:.3f}':>8} "
                f"{'n/a' if now is None else f'{now:.3f}':>8}  {verdict}"
            )

    if not args.no_write:
        RESULTS.mkdir(parents=True, exist_ok=True)
        payload = report.to_dict()
        payload["generated_at"] = datetime.now(UTC).isoformat()
        payload["judge"] = {
            "enabled": args.judge,
            "rubric_version": JUDGE_RUBRIC_VERSION if args.judge else None,
            "calls": judge_calls,
            "cost_usd": round(judge_cost, 6),
        }
        payload["reproducibility"] = {
            "deterministic_metrics": [m.name for m in report.metrics() if m.deterministic],
            "nondeterministic_metrics": [m.name for m in report.metrics() if not m.deterministic],
            "command": "uv run python scripts/run_evaluation.py"
            + (" --judge" if args.judge else ""),
            "embedding_model": embedder.model_name,
        }
        out = RESULTS / f"evaluation_{args.label}_{git_sha()}.json"
        out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        print(f"\nwritten: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    asyncio.run(main())
