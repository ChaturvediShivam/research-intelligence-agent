"""Stage 7 — VALIDATE. Deterministic citation verification.

**No model call appears anywhere in this module.** The quality property this
system is most responsible for cannot depend on the component whose output is
under audit (ADR-002). Everything here is string and set arithmetic over text
this service fetched and stored itself.

Three tiers of check, all deterministic, all blocking:

1. **Quote integrity.** The cited source must be one in the corpus, the
   offsets must lie inside its text, and re-slicing must reproduce the quote.
   A citation naming a source that was never fetched, or quoting words that
   are not at those offsets, is rejected. This is what catches a fabrication.

2. **Figure grounding.** Every specific number in a claim must appear in at
   least one of its own cited quotes. A claim may be fluent, correctly
   formatted, and cite a real passage that simply does not contain the figure
   being asserted — which is the most dangerous failure mode in research
   output, because it looks exactly like a correct one.

3. **Provenance and corroboration.** Credibility is assigned by rule from the
   domain; corroboration counts *independent* sources. Confidence is then
   derived from those, never asserted.

What this module does **not** claim: that a verified quote semantically
entails its claim. Proving entailment is not available deterministically. The
checks here are necessary conditions — a claim passing them has its specifics
physically present in a real, named, re-sliceable passage. That is a far
stronger guarantee than a model's assurance, and it is stated as what it is.
"""

from __future__ import annotations

import re
import time
import unicodedata
from dataclasses import dataclass, field

import structlog

from app.schemas.evidence import (
    Citation,
    Claim,
    ClaimStatus,
    Confidence,
    CredibilityTier,
    EvidenceGrade,
    EvidenceItem,
    FailureCode,
    Quote,
    SourceRef,
)
from app.schemas.runs import Stage, StageMetric
from app.schemas.source import content_hash

logger = structlog.get_logger(__name__)

# --------------------------------------------------------------------------
# Normalisation for quote comparison
# --------------------------------------------------------------------------
# Deliberately narrow. Every rule here is a way a quote can differ from its
# source without the meaning changing: a smart quote, a non-breaking space, a
# line break where the page had one. Anything broader would start hiding real
# mismatches, which is the opposite of the point.

_QUOTE_CHARS = {
    "‘": "'",
    "’": "'",
    "‚": "'",
    "‛": "'",
    "“": '"',
    "”": '"',
    "„": '"',
    "‟": '"',
    "′": "'",
    "″": '"',
    "«": '"',
    "»": '"',
}
_DASH_CHARS = {
    "‐": "-",
    "‑": "-",
    "‒": "-",
    "–": "-",
    "—": "-",
    "―": "-",
    "−": "-",
}
_WHITESPACE = re.compile(r"\s+")


def normalise_for_comparison(text: str) -> str:
    """Canonical form for comparing a quote with its source slice."""
    out = unicodedata.normalize("NFKC", text)
    for source_char, replacement in {**_QUOTE_CHARS, **_DASH_CHARS}.items():
        out = out.replace(source_char, replacement)
    out = _WHITESPACE.sub(" ", out)
    return out.strip().lower()


# --------------------------------------------------------------------------
# Credibility, by rule
# --------------------------------------------------------------------------

_PRIMARY_SUFFIXES = (
    ".gov.uk",
    ".gov",
    ".mil",
    "europa.eu",
    ".int",
    "legislation.gov.uk",
    "ons.gov.uk",
)
_PRIMARY_DOMAINS = frozenset(
    {
        "fca.org.uk",
        "bankofengland.co.uk",
        "pra.bankofengland.co.uk",
        "sec.gov",
        "federalreserve.gov",
        "bis.org",
        "imf.org",
        "oecd.org",
        "worldbank.org",
        "eiopa.europa.eu",
        "esma.europa.eu",
        "companieshouse.gov.uk",
        "find-and-update.company-information.service.gov.uk",
    }
)
_SECONDARY_SUFFIXES = (".ac.uk", ".edu", ".edu.au")
_SECONDARY_DOMAINS = frozenset(
    {
        "reuters.com",
        "ft.com",
        "bbc.co.uk",
        "bbc.com",
        "economist.com",
        "bloomberg.com",
        "wsj.com",
        "nature.com",
        "science.org",
        "abi.org.uk",
        "biba.org.uk",
        "lloyds.com",
        "swissre.com",
        "munichre.com",
        "actuaries.org.uk",
        "nao.org.uk",
        "ifs.org.uk",
    }
)


def _matches_suffix(host: str, suffixes: tuple[str, ...]) -> bool:
    """Whether a host sits under one of these suffixes, apex included.

    `host.endswith(".gov.uk")` is false for the apex `gov.uk` itself, because
    the apex has no leading dot — so a bare government domain was being
    classified UNVETTED, downgrading the most credible sources in the corpus.
    Both forms are checked. See docs/failure-analysis.md F-009.
    """
    for suffix in suffixes:
        bare = suffix.lstrip(".")
        if host == bare or host.endswith(f".{bare}"):
            return True
    return False


def classify_credibility(domain: str) -> CredibilityTier:
    """Assign a provenance tier from the domain, by rule.

    Unknown domains are UNVETTED rather than assumed credible: the default
    must be the cautious one, or an unrecognised blog inherits the benefit of
    the doubt.
    """
    host = domain.lower().removeprefix("www.").rstrip(".")
    if host in _PRIMARY_DOMAINS or _matches_suffix(host, _PRIMARY_SUFFIXES):
        return CredibilityTier.PRIMARY
    if host in _SECONDARY_DOMAINS or _matches_suffix(host, _SECONDARY_SUFFIXES):
        return CredibilityTier.ESTABLISHED_SECONDARY
    return CredibilityTier.UNVETTED


# --------------------------------------------------------------------------
# Figure grounding
# --------------------------------------------------------------------------

# A "specific" figure: two or more digits, or any digit with a percent sign,
# currency symbol or decimal point. Bare single digits are excluded because
# prose freely alternates between "3" and "three", and flagging that would
# produce false rejections far more often than real catches.
_FIGURE = re.compile(
    r"(?<![\w.])"
    r"(?:[£$€]\s?\d[\d,.]*\s?(?:bn|billion|m|million|k|thousand|tn|trillion)?"
    r"|\d[\d,.]*\s?%"
    r"|\d[\d,.]*\s?(?:bn|billion|m|million|tn|trillion)"
    r"|\d{2,}(?:[.,]\d+)*"
    r"|\d+\.\d+)"
    r"(?![\w])",
    re.IGNORECASE,
)

_NUMBER_WORDS = {
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
    "10": "ten",
    "11": "eleven",
    "12": "twelve",
    "20": "twenty",
    "30": "thirty",
    "40": "forty",
    "50": "fifty",
    "100": "hundred",
}


def _digits_only(text: str) -> str:
    return re.sub(r"[^\d]", "", text)


def extract_figures(text: str) -> list[str]:
    """Specific numeric assertions in a piece of text."""
    return [match.group(0).strip() for match in _FIGURE.finditer(text)]


def figure_is_grounded(figure: str, haystack: str) -> bool:
    """Whether a figure appears in the supporting text.

    Matches on the digit sequence rather than the formatted string, so
    "1,600" in a claim is grounded by "1600" in the source. Falls back to the
    spelled-out form for small round numbers.
    """
    digits = _digits_only(figure)
    if not digits:
        return False
    haystack_lower = haystack.lower()
    if digits in _digits_only(haystack):
        return True
    word = _NUMBER_WORDS.get(digits)
    return bool(word and word in haystack_lower)


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CitationVerdict:
    """The outcome of verifying one citation."""

    citation: Citation
    ok: bool
    failure: FailureCode | None = None
    detail: str = ""
    # Set when normalisation was needed to make the match, so a reviewer can
    # see that the comparison was not byte-exact.
    normalised: bool = False


@dataclass(slots=True)
class ValidationResult:
    """Everything stage 7 determined."""

    claims: list[Claim] = field(default_factory=list)
    verdicts: list[CitationVerdict] = field(default_factory=list)

    @property
    def verified_count(self) -> int:
        return sum(1 for v in self.verdicts if v.ok)

    @property
    def rejected_count(self) -> int:
        return sum(1 for v in self.verdicts if not v.ok)

    @property
    def failure_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for verdict in self.verdicts:
            if verdict.failure is not None:
                counts[verdict.failure.value] = counts.get(verdict.failure.value, 0) + 1
        return counts

    @property
    def supported_claims(self) -> list[Claim]:
        return [c for c in self.claims if c.is_supported]

    @property
    def unknown_claims(self) -> list[Claim]:
        return [c for c in self.claims if not c.is_supported]


# --------------------------------------------------------------------------
# The verifier
# --------------------------------------------------------------------------


class CitationVerifier:
    """Verifies citations against the exact source text they name.

    `sources` maps `source_id` to the `SourceRef`, and `texts` maps
    `source_id` to the canonical text. Both come from stage 3, which stored
    them; nothing here re-fetches, because a citation must be checked against
    the bytes it was produced from.
    """

    def __init__(
        self,
        sources: dict[str, SourceRef],
        texts: dict[str, str],
        *,
        check_content_hash: bool = True,
    ) -> None:
        self._sources = sources
        self._texts = texts
        self._check_content_hash = check_content_hash

    # -- quote level ------------------------------------------------------

    def verify_quote(self, quote: Quote) -> CitationVerdict:
        """Verify one quote. The check that catches a fabricated citation."""
        citation = Citation(
            source_id=quote.source_id,
            start_char=quote.start_char,
            end_char=quote.end_char,
            cited_text=quote.text,
        )

        text = self._texts.get(quote.source_id)
        if text is None:
            # The citation names a source that was never fetched. A model that
            # invents a plausible URL lands here.
            return CitationVerdict(
                citation=citation,
                ok=False,
                failure=FailureCode.UNKNOWN_SOURCE,
                detail=f"No stored source with id {quote.source_id!r}.",
            )

        if self._check_content_hash:
            reference = self._sources.get(quote.source_id)
            if reference is not None and content_hash(text) != reference.content_hash:
                return CitationVerdict(
                    citation=citation,
                    ok=False,
                    failure=FailureCode.CONTENT_CHANGED,
                    detail="Stored text no longer matches the recorded content hash.",
                )

        if quote.end_char <= quote.start_char:  # pragma: no cover - schema guards
            return CitationVerdict(
                citation=citation,
                ok=False,
                failure=FailureCode.OFFSET_INVALID,
                detail="end_char is not after start_char.",
            )

        if quote.end_char > len(text):
            return CitationVerdict(
                citation=citation,
                ok=False,
                failure=FailureCode.OFFSET_OUT_OF_RANGE,
                detail=(
                    f"Offsets [{quote.start_char}:{quote.end_char}] exceed the "
                    f"source length of {len(text)}."
                ),
            )

        actual = text[quote.start_char : quote.end_char]
        if actual == quote.text:
            return CitationVerdict(citation=citation, ok=True)

        normalised_actual = normalise_for_comparison(actual)
        normalised_quote = normalise_for_comparison(quote.text)
        if normalised_actual == normalised_quote:
            # Matched only after normalisation: logged, because a reviewer
            # should be able to see that the comparison was not byte-exact.
            logger.info(
                "citation_normalised",
                source_id=quote.source_id,
                start=quote.start_char,
                end=quote.end_char,
            )
            return CitationVerdict(citation=citation, ok=True, normalised=True)

        # The quote does not sit at those offsets. Distinguish "the text is
        # real but somewhere else" from "the text is not in this corpus at
        # all" — the first is a misattribution, the second a fabrication, and
        # a reader needs to know which.
        for other_id, other_text in self._texts.items():
            if other_id == quote.source_id:
                continue
            if normalised_quote and normalised_quote in normalise_for_comparison(other_text):
                return CitationVerdict(
                    citation=citation,
                    ok=False,
                    failure=FailureCode.WRONG_SOURCE,
                    detail=(
                        f"Quoted text is not at those offsets in "
                        f"{quote.source_id}, but does appear in {other_id}."
                    ),
                )

        return CitationVerdict(
            citation=citation,
            ok=False,
            failure=FailureCode.QUOTE_MISMATCH,
            detail=(
                "Quoted text is not present at those offsets, and does not "
                "appear in any stored source."
            ),
        )

    def verify_citation(self, citation: Citation) -> CitationVerdict:
        verdict = self.verify_quote(citation.as_quote())
        # Preserve the caller's citation object, including evidence_id.
        return CitationVerdict(
            citation=citation,
            ok=verdict.ok,
            failure=verdict.failure,
            detail=verdict.detail,
            normalised=verdict.normalised,
        )

    def verify_evidence(self, item: EvidenceItem) -> CitationVerdict:
        """Verify an extracted evidence item's quote."""
        return self.verify_quote(item.quote)

    # -- claim level ------------------------------------------------------

    def _independent_sources(self, source_ids: set[str]) -> int:
        """Count independent sources among those cited.

        Independence is approximated by distinct registrable domain *and*
        distinct publisher where known. Two outlets syndicating one wire story
        can still register as two — a documented limitation of this heuristic
        (ADR-010), and a named candidate for improvement.
        """
        seen: set[tuple[str, str]] = set()
        for source_id in source_ids:
            reference = self._sources.get(source_id)
            if reference is None:
                continue
            publisher = (reference.publisher or reference.domain).lower()
            seen.add((reference.domain.lower(), publisher))
        # Collapse entries sharing a publisher across different domains.
        publishers = {publisher for _, publisher in seen}
        return len(publishers)

    def _derive_confidence(
        self,
        *,
        verified: int,
        corroboration: int,
        best_credibility: CredibilityTier | None,
        best_grade: EvidenceGrade | None,
    ) -> tuple[Confidence, str]:
        """Derive confidence from verified facts, with its basis in words."""
        if verified == 0:
            return Confidence.UNKNOWN, "No citation survived verification."

        tier = best_credibility or CredibilityTier.UNVETTED
        grade_note = f", strongest evidence {best_grade.value}" if best_grade else ""

        if corroboration >= 2 and tier is CredibilityTier.PRIMARY:
            return (
                Confidence.HIGH,
                f"{corroboration} independent sources, at least one primary{grade_note}.",
            )
        if corroboration >= 2:
            return (
                Confidence.MODERATE,
                f"{corroboration} independent sources, none primary{grade_note}.",
            )
        if tier is CredibilityTier.PRIMARY:
            return (
                Confidence.MODERATE,
                f"Single primary source, uncorroborated{grade_note}.",
            )
        return (
            Confidence.LOW,
            f"Single {tier.value.replace('_', ' ')} source, uncorroborated{grade_note}.",
        )

    def verify_claim(
        self, claim: Claim, *, evidence: dict[str, EvidenceItem] | None = None
    ) -> tuple[Claim, list[CitationVerdict]]:
        """Verify a claim's citations and derive its status and confidence."""
        evidence = evidence or {}
        failures: list[FailureCode] = []
        verdicts: list[CitationVerdict] = []

        if not claim.citations:
            # An uncited factual claim is UNKNOWN, not deleted: a removed
            # claim is invisible, a marked one is auditable.
            updated = claim.model_copy(
                update={
                    "status": ClaimStatus.UNKNOWN,
                    "confidence": Confidence.UNKNOWN,
                    "corroboration": 0,
                    "verified_citations": 0,
                    "failures": [FailureCode.NO_CITATION],
                    "confidence_basis": "Claim carries no citation.",
                }
            )
            return updated, []

        for citation in claim.citations:
            verdict = self.verify_citation(citation)
            verdicts.append(verdict)
            if not verdict.ok and verdict.failure is not None:
                failures.append(verdict.failure)

        verified = [v for v in verdicts if v.ok]
        verified_source_ids = {v.citation.source_id for v in verified}

        # Figure grounding, over the quotes that actually verified. Using an
        # unverified quote here would let a fabricated passage ground a figure.
        grounding_text = " ".join(v.citation.cited_text for v in verified)
        ungrounded = [
            figure
            for figure in extract_figures(claim.text)
            if not figure_is_grounded(figure, grounding_text)
        ]
        if ungrounded:
            failures.append(FailureCode.UNGROUNDED_FIGURE)

        corroboration = self._independent_sources(verified_source_ids)

        best_credibility: CredibilityTier | None = None
        for source_id in verified_source_ids:
            reference = self._sources.get(source_id)
            if reference is None:
                continue
            if best_credibility is None or reference.credibility.rank > best_credibility.rank:
                best_credibility = reference.credibility

        best_grade: EvidenceGrade | None = None
        for citation in claim.citations:
            item = evidence.get(citation.evidence_id or "")
            if item is None:
                continue
            if best_grade is None or item.grade.rank > best_grade.rank:
                best_grade = item.grade

        # A claim is supported only if at least one citation verified AND no
        # figure it asserts is ungrounded. A partly-verified claim asserting an
        # unsupported number is not "mostly right"; the number is the claim.
        supported = bool(verified) and not ungrounded

        confidence, basis = self._derive_confidence(
            verified=len(verified),
            corroboration=corroboration,
            best_credibility=best_credibility,
            best_grade=best_grade,
        )
        if ungrounded:
            confidence = Confidence.UNKNOWN
            basis = f"Claim asserts {ungrounded!r}, which appears in no verified citation."

        updated = claim.model_copy(
            update={
                "status": ClaimStatus.SUPPORTED if supported else ClaimStatus.UNKNOWN,
                "confidence": confidence,
                "corroboration": corroboration,
                "verified_citations": len(verified),
                "failures": failures,
                "confidence_basis": basis,
            }
        )
        return updated, verdicts


def run_validate_stage(
    claims: list[Claim],
    *,
    sources: dict[str, SourceRef],
    texts: dict[str, str],
    evidence: dict[str, EvidenceItem] | None = None,
) -> tuple[ValidationResult, StageMetric]:
    """Stage 7. No model call, so no cost is attributed to it."""
    started = time.perf_counter()
    verifier = CitationVerifier(sources, texts)

    result = ValidationResult()
    for claim in claims:
        verified_claim, verdicts = verifier.verify_claim(claim, evidence=evidence)
        result.claims.append(verified_claim)
        result.verdicts.extend(verdicts)

    metric = StageMetric(
        stage=Stage.VALIDATE,
        model=None,
        duration_ms=int((time.perf_counter() - started) * 1000),
        calls=0,
    )
    logger.info(
        "stage_complete",
        stage=Stage.VALIDATE.value,
        claims=len(result.claims),
        supported=len(result.supported_claims),
        unknown=len(result.unknown_claims),
        citations_verified=result.verified_count,
        citations_rejected=result.rejected_count,
        failures=result.failure_counts,
        duration_ms=metric.duration_ms,
    )
    return result, metric
