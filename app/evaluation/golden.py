"""The golden set: loading and validation.

`evals/datasets/golden_v1/cases.jsonl` is the source of truth. It is
committed, versioned by directory name, and built by
`scripts/build_golden_v1.py` from the already-annotated `retrieval_v1`
corpus plus failure-mode cases derived from documented system behaviour.

The loader validates rather than trusts. A case judging a document it does
not make available, or naming a document absent from the corpus, is an
annotation error that would silently depress a metric forever — so it is
rejected at load time rather than scored.

Nothing here mutates a case. `GoldenCase` is frozen, which is what stops an
evaluator from quietly editing an expected outcome to match what the system
produced.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

GOLDEN_SET_VERSION = "golden_v1"

# How a case's sources are made to behave, so failure handling is evaluated
# rather than only the happy path.
SourceCondition = Literal["healthy", "one_fetch_fail", "all_fetch_fail", "access_restricted"]


class GoldenCase(BaseModel):
    """One evaluation case with its expected outcome stated in advance."""

    model_config = ConfigDict(frozen=True)

    case_id: str = Field(min_length=2, max_length=16)
    split: Literal["train", "holdout"]
    question: str = Field(min_length=12)
    intent: str = Field(min_length=8)
    # Documents the fixture corpus makes discoverable for this case.
    available_docs: tuple[str, ...]
    # Graded relevance, as in retrieval_v1: 2 answers it, 1 is context, 0/absent
    # is irrelevant.
    judgments: dict[str, int] = Field(default_factory=dict)
    # Whether the case is expected to yield no supported answer at all.
    expect_unknown: bool
    expected_tools: tuple[str, ...]
    source_condition: SourceCondition
    origin: str = ""

    @field_validator("judgments")
    @classmethod
    def _grades_in_scale(cls, value: dict[str, int]) -> dict[str, int]:
        for doc_id, grade in value.items():
            if grade not in (0, 1, 2):
                raise ValueError(f"grade for {doc_id} must be 0, 1 or 2, got {grade}")
        return value

    @property
    def relevant_docs(self) -> set[str]:
        """Documents graded relevant. Empty for a deliberately unanswerable case."""
        return {doc for doc, grade in self.judgments.items() if grade >= 1}

    @property
    def is_answerable(self) -> bool:
        return bool(self.relevant_docs)


class GoldenSet(BaseModel):
    """A loaded, validated golden set."""

    model_config = ConfigDict(frozen=True)

    version: str
    cases: tuple[GoldenCase, ...]

    @property
    def train(self) -> tuple[GoldenCase, ...]:
        return tuple(c for c in self.cases if c.split == "train")

    @property
    def holdout(self) -> tuple[GoldenCase, ...]:
        """Scored every round and never tuned against."""
        return tuple(c for c in self.cases if c.split == "holdout")

    def by_id(self, case_id: str) -> GoldenCase:
        for case in self.cases:
            if case.case_id == case_id:
                return case
        raise KeyError(case_id)


class GoldenSetError(ValueError):
    """Raised when the golden set is internally inconsistent."""


def load_golden_set(directory: Path, *, corpus_doc_ids: set[str] | None = None) -> GoldenSet:
    """Load and validate a golden set directory."""
    path = directory / "cases.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Golden set is missing {path}")

    cases: list[GoldenCase] = []
    seen: set[str] = set()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            case = GoldenCase.model_validate_json(line)
        except Exception as exc:
            raise GoldenSetError(f"{path}:{number} is not a valid case: {exc}") from exc

        if case.case_id in seen:
            raise GoldenSetError(f"duplicate case_id {case.case_id!r}")
        seen.add(case.case_id)

        # A case cannot judge a document it does not make available: the
        # metric would be measuring an absence the case itself created.
        orphan = set(case.judgments) - set(case.available_docs)
        if orphan:
            raise GoldenSetError(
                f"{case.case_id} judges documents it does not make available: {sorted(orphan)}"
            )
        if corpus_doc_ids is not None:
            missing = set(case.available_docs) - corpus_doc_ids
            if missing:
                raise GoldenSetError(
                    f"{case.case_id} names documents absent from the corpus: {sorted(missing)}"
                )
        # Annotation consistency, conditioned on whether the sources can be
        # read at all. A case may legitimately judge a document relevant AND
        # expect UNKNOWN when that document is deliberately unreachable —
        # that is precisely what a paywall or fetch-failure case tests.
        if case.source_condition == "healthy":
            if case.expect_unknown and case.relevant_docs:
                raise GoldenSetError(
                    f"{case.case_id} is healthy and judges "
                    f"{sorted(case.relevant_docs)} relevant, so it cannot expect UNKNOWN"
                )
            if not case.expect_unknown and not case.relevant_docs:
                raise GoldenSetError(
                    f"{case.case_id} expects an answer but judges nothing relevant"
                )
        elif case.source_condition in {"all_fetch_fail", "access_restricted"} and (
            not case.expect_unknown
        ):
            # No source can be read, so an answer is impossible by construction.
            raise GoldenSetError(
                f"{case.case_id} makes every source unreadable "
                f"({case.source_condition}) but does not expect UNKNOWN"
            )
        cases.append(case)

    if not cases:
        raise GoldenSetError(f"{path} contains no cases")
    return GoldenSet(version=directory.name, cases=tuple(cases))
