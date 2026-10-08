"""Run the retrieval evaluation and write a timestamped result.

    uv run python scripts/run_retrieval_eval.py
    uv run python scripts/run_retrieval_eval.py --k 5 --compare

`--compare` also runs dense-only and lexical-only, which is how the hybrid
decision is tested rather than assumed.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from app.evaluation.runner import EvalReport, run_retrieval_eval
from app.retrieval.embeddings import FastEmbedEmbedder

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "evals" / "datasets" / "retrieval_v1"
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=[1, 3, 5, 10],
        help="cutoffs to report; must be smaller than the corpus to mean anything",
    )
    parser.add_argument("--chunk-tokens", type=int, default=512)
    parser.add_argument("--overlap-tokens", type=int, default=64)
    parser.add_argument(
        "--retrievers",
        nargs="+",
        default=["hybrid", "dense", "lexical"],
        help="hybrid is the shipped configuration; the others are the comparison",
    )
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    embedder = FastEmbedEmbedder()
    reports: list[EvalReport] = []

    for retriever in args.retrievers:
        for k in args.k:
            reports.append(
                run_retrieval_eval(
                    DATASET,
                    embedder,
                    k=k,
                    chunk_tokens=args.chunk_tokens,
                    overlap_tokens=args.overlap_tokens,
                    embedding_model_name=embedder.model_name,
                    retriever_name=retriever,
                    dense_only=retriever == "dense",
                    lexical_only=retriever == "lexical",
                )
            )

    first = reports[0]
    print(f"\ndataset: {DATASET.name}   sha: {git_sha()}   model: {embedder.model_name}")
    print(
        f"documents: {first.documents}   chunks: {first.chunks}   "
        f"queries: {first.queries}   chunk_tokens: {first.chunk_tokens}"
    )
    if max(args.k) >= first.documents:
        print(
            f"\n  NOTE: k={max(args.k)} is >= the corpus size ({first.documents}); "
            "recall at that cutoff is trivially 1.0 and carries no information."
        )

    print("\n  retriever   k   recall@k   P@k      MRR     nDCG@k")
    print("  " + "-" * 52)
    for report in reports:
        print(
            f"  {report.retriever:10} {report.k:<3} "
            f"{report.recall_at_k:8.3f}  {report.precision_at_k:6.3f}  "
            f"{report.mrr:6.3f}  {report.ndcg_at_k:7.3f}"
        )

    headline = next(r for r in reports if r.retriever == "hybrid" and r.k == min(args.k))
    print(f"\nper-query (hybrid, k={headline.k}):")
    for result in headline.per_query:
        flag = "" if result.reciprocal_rank == 1.0 else "   <-- top hit not relevant"
        print(
            f"  {result.query_id}  RR={result.reciprocal_rank:.3f}  "
            f"nDCG={result.ndcg_at_k:.3f}  recall={result.recall_at_k:.3f}"
            f"  [{','.join(result.top_hit_retrievers)}]{flag}"
        )

    if not args.no_write:
        RESULTS.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        out = RESULTS / f"retrieval_{stamp}_{git_sha()}.json"
        out.write_text(
            json.dumps(
                {
                    "generated_at": datetime.now(UTC).isoformat(),
                    "git_sha": git_sha(),
                    "reports": [json.loads(r.to_json()) for r in reports],
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        print(f"\nwritten: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
