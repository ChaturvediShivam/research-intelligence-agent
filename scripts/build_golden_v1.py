"""Build evals/datasets/golden_v1/cases.jsonl.

Kept in the repository so the golden set is traceable to its construction
rather than appearing as an unexplained artifact. Run once; the output is
committed and is the source of truth from then on.

Cases G01-G10 reuse the ten queries and judgments already annotated in
`retrieval_v1`, which were written before retrieval was tuned. Cases G11-G20
are failure-mode cases derived from documented system behaviour: each one
targets a guarantee the pipeline claims to provide, with the expected outcome
stated in advance.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RETRIEVAL = ROOT / "evals" / "datasets" / "retrieval_v1"
OUT = ROOT / "evals" / "datasets" / "golden_v1"

# Train/holdout split is fixed here, by case id, so it cannot drift between
# runs or be chosen after seeing a score.
HOLDOUT = {"G08", "G09", "G10", "G17", "G18", "G19", "G20"}


def load_retrieval_queries() -> list[dict]:
    path = RETRIEVAL / "queries.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_corpus_ids() -> set[str]:
    path = RETRIEVAL / "corpus.jsonl"
    return {json.loads(line)["doc_id"] for line in path.read_text().splitlines() if line.strip()}


# Failure-mode cases. Each names the guarantee it tests and the outcome
# expected in advance of any run.
FAILURE_CASES = [
    {
        "case_id": "G11",
        "question": "What is the registered office address of a company absent from the corpus?",
        "intent": "No document answers this; the system must report UNKNOWN rather than infer.",
        "available_docs": ["abi-pet-market", "fca-value-measures"],
        "judgments": {},
        "expect_unknown": True,
        "expected_tools": ["search_sources"],
        "source_condition": "healthy",
    },
    {
        "case_id": "G12",
        "question": "What were UK pet insurance gross written premium figures?",
        "intent": "Every discovered source fails to fetch; must yield a gap, not an answer.",
        "available_docs": ["abi-pet-market"],
        "judgments": {"abi-pet-market": 2},
        "expect_unknown": True,
        "expected_tools": ["search_sources"],
        "source_condition": "all_fetch_fail",
    },
    {
        "case_id": "G13",
        "question": "What does the paywalled market report say about concentration?",
        "intent": "A 403 source must be classified access_restricted, not merely broken.",
        "available_docs": ["abi-pet-market"],
        "judgments": {"abi-pet-market": 2},
        "expect_unknown": True,
        "expected_tools": ["search_sources"],
        "source_condition": "access_restricted",
    },
    {
        "case_id": "G14",
        "question": "What are the Solvency Capital Requirement rules for UK insurers?",
        "intent": "One source fails while another succeeds; partial coverage must still answer.",
        "available_docs": ["pra-solvency", "fca-value-measures"],
        "judgments": {"pra-solvency": 2},
        "expect_unknown": False,
        "expected_tools": ["search_sources"],
        "source_condition": "one_fetch_fail",
    },
    {
        "case_id": "G15",
        "question": "What is the Insurance Premium Tax higher rate?",
        "intent": "A single primary source answering cleanly; citation must verify.",
        "available_docs": ["hmrc-ipt"],
        "judgments": {"hmrc-ipt": 2},
        "expect_unknown": False,
        "expected_tools": ["search_sources"],
        "source_condition": "healthy",
    },
    {
        "case_id": "G16",
        "question": "How much compensation applies if an insurer fails?",
        "intent": "Paraphrase question over a source that never uses the query's words.",
        "available_docs": ["fscs-compensation", "insurtech-funding"],
        "judgments": {"fscs-compensation": 2},
        "expect_unknown": False,
        "expected_tools": ["search_sources"],
        "source_condition": "healthy",
    },
    {
        "case_id": "G17",
        "question": "What drives pet insurance claims costs upward?",
        "intent": "Holdout: multi-source question, mixed credibility.",
        "available_docs": ["pet-claims-trends", "abi-pet-market", "vet-workforce"],
        "judgments": {"pet-claims-trends": 2, "abi-pet-market": 1},
        "expect_unknown": False,
        "expected_tools": ["search_sources"],
        "source_condition": "healthy",
    },
    {
        "case_id": "G18",
        "question": "When does the CMA make a market investigation reference?",
        "intent": "Holdout: procedural question with one authoritative source.",
        "available_docs": ["cma-market-investigation", "fca-consumer-duty"],
        "judgments": {"cma-market-investigation": 2},
        "expect_unknown": False,
        "expected_tools": ["search_sources"],
        "source_condition": "healthy",
    },
    {
        "case_id": "G19",
        "question": "What is the average rainfall in the Atacama Desert?",
        "intent": "Holdout: wholly out-of-domain. The corpus cannot answer it and must not try.",
        "available_docs": ["abi-pet-market", "motor-claims-inflation"],
        "judgments": {},
        "expect_unknown": True,
        "expected_tools": ["search_sources"],
        "source_condition": "healthy",
    },
    {
        "case_id": "G20",
        "question": "What was motor insurance claims cost inflation driven by?",
        "intent": "Holdout: a distractor-heavy corpus; weak discovery must not fabricate.",
        "available_docs": [
            "motor-claims-inflation",
            "motor-telematics",
            "reinsurance-market",
            "gap-insurance",
        ],
        "judgments": {"motor-claims-inflation": 2},
        "expect_unknown": False,
        "expected_tools": ["search_sources"],
        "source_condition": "healthy",
    },
]


# The discovery stage asks its search tool for at most 8 results per query
# (app/pipeline/discover.py). A case that makes all 20 corpus documents
# available therefore has only its alphabetically-first 8 discovered, so
# retrieval recall would measure document naming rather than retrieval. Each
# case instead gets its judged documents plus deterministic distractors, sized
# to fit inside that cap. See docs/failure-analysis.md F-012.
DISCOVERY_CAP = 8

# Hard negatives from retrieval_v1, in fixed order, used as distractors.
DISTRACTOR_POOL = [
    "travel-insurance-claims",
    "home-emergency-cover",
    "fca-consumer-duty",
    "pet-ownership-demographics",
    "vet-workforce",
    "motor-telematics",
    "reinsurance-market",
    "employers-liability",
    "gap-insurance",
    "annuity-rates",
]


def available_for(judged: list[str], corpus_ids: set[str]) -> list[str]:
    """Judged documents plus distractors, within the discovery cap.

    Deterministic: the distractor order is fixed, so the same golden set is
    produced on every build and a metric cannot move because of shuffling.
    """
    chosen = sorted(judged)
    for candidate in DISTRACTOR_POOL:
        if len(chosen) >= DISCOVERY_CAP:
            break
        if candidate in corpus_ids and candidate not in chosen:
            chosen.append(candidate)
    return sorted(chosen)


def main() -> None:
    corpus_ids = load_corpus_ids()
    cases: list[dict] = []

    # G01-G10 from the pre-annotated retrieval queries.
    for index, query in enumerate(load_retrieval_queries(), start=1):
        case_id = f"G{index:02d}"
        judged = list(query["judgments"])
        cases.append(
            {
                "case_id": case_id,
                "split": "holdout" if case_id in HOLDOUT else "train",
                "question": query["query"],
                "intent": (
                    "Reused from retrieval_v1; judgments were annotated before retrieval was tuned."
                ),
                # Judged documents plus fixed distractors, inside the
                # discovery cap, so retrieval is measured rather than
                # alphabetical position (F-012).
                "available_docs": available_for(judged, corpus_ids),
                "judgments": query["judgments"],
                "expect_unknown": not judged,
                "expected_tools": ["search_sources"],
                "source_condition": "healthy",
                "origin": f"retrieval_v1:{query['query_id']}",
            }
        )

    for case in FAILURE_CASES:
        unknown = set(case["available_docs"]) - corpus_ids
        if unknown:
            raise SystemExit(f"{case['case_id']} names documents absent from corpus: {unknown}")
        bad = set(case["judgments"]) - set(case["available_docs"])
        if bad:
            raise SystemExit(
                f"{case['case_id']} judges documents it does not make available: {bad}"
            )
        cases.append(
            {
                **case,
                "split": "holdout" if case["case_id"] in HOLDOUT else "train",
                "origin": "derived from documented system behaviour",
            }
        )

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "cases.jsonl"
    path.write_text(
        "\n".join(json.dumps(c, sort_keys=True) for c in cases) + "\n", encoding="utf-8"
    )

    train = sum(1 for c in cases if c["split"] == "train")
    print(f"wrote {len(cases)} cases to {path.relative_to(ROOT)}")
    print(f"  train: {train}   holdout: {len(cases) - train}")
    print(f"  expect_unknown: {sum(1 for c in cases if c['expect_unknown'])}")
    print(f"  source conditions: {sorted({c['source_condition'] for c in cases})}")


if __name__ == "__main__":
    main()
