"""Deterministic citation fixtures.

Two real source documents and a case for every way a citation can be wrong.
Nothing here is random or model-generated: each case is constructed so exactly
one failure mode fires, which is what lets the tests assert on a specific
`FailureCode` rather than on a boolean.

The six required shapes, and the code each must produce:

| Case | Expected |
|---|---|
| valid claim + valid evidence | verifies |
| valid claim + wrong evidence (real quote, missing figure) | `UNGROUNDED_FIGURE` |
| fabricated citation (quote exists nowhere) | `QUOTE_MISMATCH` |
| citation pointing to the wrong source | `WRONG_SOURCE` |
| citation with out-of-range offsets | `OFFSET_OUT_OF_RANGE` |
| unsupported claim with no citation | `NO_CITATION` |
"""

from __future__ import annotations

from app.schemas.evidence import (
    Citation,
    Claim,
    CredibilityTier,
    EvidenceGrade,
    EvidenceItem,
    Quote,
    SourceRef,
    source_id_for,
)
from app.schemas.source import content_hash

# --------------------------------------------------------------------------
# Sources
# --------------------------------------------------------------------------

FCA_URL = "https://www.fca.org.uk/data/value-measures"
ABI_URL = "https://www.abi.org.uk/data/pet-insurance"
BLOG_URL = "https://someblog.example.com/opinion"

FCA_TEXT = (
    "The Financial Conduct Authority publishes general insurance value "
    "measures data twice yearly.\n\n"
    "Claims acceptance rates for home emergency cover averaged 61% across "
    "reporting firms in the period.\n\n"
    "Add-on products showed materially lower acceptance than core cover."
)

ABI_TEXT = (
    "Gross written premium for UK pet insurance reached 1,650 million pounds "
    "in the reporting year.\n\n"
    "Lifetime policies accounted for the majority of that premium.\n\n"
    "Average premiums rose faster than general inflation."
)

BLOG_TEXT = (
    "In my view the pet insurance market is obviously going to double.\n\n"
    "Everyone I know is buying a policy."
)

FCA_ID = source_id_for(FCA_URL)
ABI_ID = source_id_for(ABI_URL)
BLOG_ID = source_id_for(BLOG_URL)


def sources() -> dict[str, SourceRef]:
    """The corpus the verifier knows about."""
    return {
        FCA_ID: SourceRef(
            source_id=FCA_ID,
            url=FCA_URL,
            title="General insurance value measures",
            domain="fca.org.uk",
            content_hash=content_hash(FCA_TEXT),
            credibility=CredibilityTier.PRIMARY,
            publisher="fca.org.uk",
        ),
        ABI_ID: SourceRef(
            source_id=ABI_ID,
            url=ABI_URL,
            title="Pet insurance data",
            domain="abi.org.uk",
            content_hash=content_hash(ABI_TEXT),
            credibility=CredibilityTier.ESTABLISHED_SECONDARY,
            publisher="abi.org.uk",
        ),
        BLOG_ID: SourceRef(
            source_id=BLOG_ID,
            url=BLOG_URL,
            title="Opinion",
            domain="someblog.example.com",
            content_hash=content_hash(BLOG_TEXT),
            credibility=CredibilityTier.UNVETTED,
            publisher="someblog.example.com",
        ),
    }


def texts() -> dict[str, str]:
    return {FCA_ID: FCA_TEXT, ABI_ID: ABI_TEXT, BLOG_ID: BLOG_TEXT}


def _span(text: str, phrase: str) -> tuple[int, int]:
    """Exact offsets of a phrase, asserting it is actually present.

    A fixture whose "valid" case is quietly invalid would make the whole
    suite meaningless, so this fails loudly at import rather than producing a
    case that tests nothing.
    """
    index = text.find(phrase)
    if index == -1:
        raise AssertionError(f"fixture phrase not present in source: {phrase!r}")
    return index, index + len(phrase)


# --------------------------------------------------------------------------
# Case 1 — valid claim, valid evidence
# --------------------------------------------------------------------------

VALID_PHRASE = "Claims acceptance rates for home emergency cover averaged 61%"
_vs, _ve = _span(FCA_TEXT, VALID_PHRASE)

VALID_CITATION = Citation(
    source_id=FCA_ID,
    start_char=_vs,
    end_char=_ve,
    cited_text=VALID_PHRASE,
    evidence_id="E_valid",
)

VALID_EVIDENCE = EvidenceItem(
    evidence_id="E_valid",
    sub_question_id="SQ1",
    statement="Home emergency claims acceptance averaged 61% in the period.",
    quote=Quote(source_id=FCA_ID, start_char=_vs, end_char=_ve, text=VALID_PHRASE),
    grade=EvidenceGrade.BEHAVIOR,
    chunk_index=0,
)

VALID_CLAIM = Claim(
    claim_id="C1",
    sub_question_id="SQ1",
    text="Home emergency cover showed a 61% claims acceptance rate.",
    citations=[VALID_CITATION],
)


# --------------------------------------------------------------------------
# Case 2 — valid claim, WRONG evidence: the quote is real and correctly
# located, but does not contain the figure the claim asserts.
# --------------------------------------------------------------------------

UNRELATED_PHRASE = "Add-on products showed materially lower acceptance than core cover."
_us, _ue = _span(FCA_TEXT, UNRELATED_PHRASE)

WRONG_EVIDENCE_CITATION = Citation(
    source_id=FCA_ID, start_char=_us, end_char=_ue, cited_text=UNRELATED_PHRASE
)

CLAIM_WITH_WRONG_EVIDENCE = Claim(
    claim_id="C2",
    sub_question_id="SQ1",
    # 61% appears in the source, but NOT in the passage this claim cites.
    text="Home emergency cover showed a 61% claims acceptance rate.",
    citations=[WRONG_EVIDENCE_CITATION],
)


# --------------------------------------------------------------------------
# Case 3 — fabricated citation: wording that appears in no stored source.
# --------------------------------------------------------------------------

FABRICATED_CITATION = Citation(
    source_id=FCA_ID,
    start_char=_vs,
    end_char=_vs + 72,
    cited_text=("Claims acceptance rates for home emergency cover averaged 94% across firms"),
)

CLAIM_WITH_FABRICATED_CITATION = Claim(
    claim_id="C3",
    sub_question_id="SQ1",
    text="Home emergency cover showed a 94% claims acceptance rate.",
    citations=[FABRICATED_CITATION],
)


# --------------------------------------------------------------------------
# Case 4 — right quote, wrong source: ABI's words attributed to the FCA.
# --------------------------------------------------------------------------

ABI_PHRASE = "Lifetime policies accounted for the majority of that premium."
_as, _ae = _span(ABI_TEXT, ABI_PHRASE)

WRONG_SOURCE_CITATION = Citation(
    source_id=FCA_ID,  # the text is real, but it is in ABI_TEXT
    start_char=_as,
    end_char=_ae,
    cited_text=ABI_PHRASE,
)

CLAIM_WITH_WRONG_SOURCE = Claim(
    claim_id="C4",
    sub_question_id="SQ2",
    text="Lifetime policies make up most of the premium.",
    citations=[WRONG_SOURCE_CITATION],
)


# --------------------------------------------------------------------------
# Case 5 — offsets beyond the end of the source.
# --------------------------------------------------------------------------

OUT_OF_RANGE_CITATION = Citation(
    source_id=FCA_ID,
    start_char=len(FCA_TEXT) + 500,
    end_char=len(FCA_TEXT) + 600,
    cited_text=VALID_PHRASE,
)

CLAIM_WITH_OUT_OF_RANGE_OFFSETS = Claim(
    claim_id="C5",
    sub_question_id="SQ1",
    text="Acceptance rates were reported.",
    citations=[OUT_OF_RANGE_CITATION],
)


# --------------------------------------------------------------------------
# Case 6 — a claim with no citation at all.
# --------------------------------------------------------------------------

UNSUPPORTED_CLAIM = Claim(
    claim_id="C6",
    sub_question_id="SQ1",
    text="The pet insurance market will double within three years.",
    citations=[],
)


# --------------------------------------------------------------------------
# Case 7 — a citation naming a source that was never fetched.
# --------------------------------------------------------------------------

INVENTED_URL = "https://www.invented-regulator.example/report-2026"
INVENTED_ID = source_id_for(INVENTED_URL)

UNKNOWN_SOURCE_CITATION = Citation(
    source_id=INVENTED_ID,
    start_char=0,
    end_char=40,
    cited_text="The regulator confirmed the figure in full.",
)

CLAIM_CITING_UNKNOWN_SOURCE = Claim(
    claim_id="C7",
    sub_question_id="SQ1",
    text="The regulator confirmed the figure.",
    citations=[UNKNOWN_SOURCE_CITATION],
)


# --------------------------------------------------------------------------
# Corroboration case — the same fact in two independent sources.
# --------------------------------------------------------------------------

PREMIUM_PHRASE = "Gross written premium for UK pet insurance reached 1,650 million pounds"
_ps, _pe = _span(ABI_TEXT, PREMIUM_PHRASE)

CORROBORATED_CLAIM = Claim(
    claim_id="C8",
    sub_question_id="SQ2",
    text="UK pet insurance gross written premium reached 1,650 million pounds.",
    citations=[
        Citation(source_id=ABI_ID, start_char=_ps, end_char=_pe, cited_text=PREMIUM_PHRASE),
        Citation(source_id=FCA_ID, start_char=_vs, end_char=_ve, cited_text=VALID_PHRASE),
    ],
)

ALL_CASES = [
    VALID_CLAIM,
    CLAIM_WITH_WRONG_EVIDENCE,
    CLAIM_WITH_FABRICATED_CITATION,
    CLAIM_WITH_WRONG_SOURCE,
    CLAIM_WITH_OUT_OF_RANGE_OFFSETS,
    UNSUPPORTED_CLAIM,
    CLAIM_CITING_UNKNOWN_SOURCE,
]
