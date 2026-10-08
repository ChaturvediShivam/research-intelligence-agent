"""Citation verification — M4's exit criterion.

This file contains the test the whole project is accountable to: a
deliberately fabricated citation must be caught by the automated verifier,
with the correct failure code, and the claim depending on it must be marked
UNKNOWN rather than reported as supported.

Placed under `tests/security/` deliberately. A verifier that can be talked
into accepting an invented source is not a quality bug, it is the failure of
the system's central guarantee.
"""

from __future__ import annotations

import pytest

from app.pipeline.validate import (
    CitationVerifier,
    classify_credibility,
    extract_figures,
    figure_is_grounded,
    normalise_for_comparison,
    run_validate_stage,
)
from app.schemas.evidence import (
    Citation,
    Claim,
    ClaimStatus,
    Confidence,
    CredibilityTier,
    FailureCode,
    Quote,
)
from tests.fixtures import citation_cases as cases


@pytest.fixture
def verifier() -> CitationVerifier:
    return CitationVerifier(cases.sources(), cases.texts())


# ==========================================================================
# THE EXIT CRITERION
# ==========================================================================


class TestExitCriterionFabricatedCitationIsCaught:
    """M4 exit criterion: fabricate a citation, prove the verifier catches it."""

    def test_fabricated_quote_is_rejected(self, verifier: CitationVerifier) -> None:
        """A quote whose wording appears in no stored source must be rejected.

        The fabrication is realistic: correct source, plausible offsets, and
        wording that differs from the real passage only in the figure — 94%
        where the source says 61%. Nothing about its shape is suspicious.
        """
        verdict = verifier.verify_citation(cases.FABRICATED_CITATION)

        assert verdict.ok is False
        assert verdict.failure is FailureCode.QUOTE_MISMATCH
        assert "not present at those offsets" in verdict.detail

    def test_the_claim_resting_on_it_is_marked_unknown_not_supported(
        self, verifier: CitationVerifier
    ) -> None:
        """The consequence that matters: it must not reach a reader as fact."""
        claim, verdicts = verifier.verify_claim(cases.CLAIM_WITH_FABRICATED_CITATION)

        assert claim.status is ClaimStatus.UNKNOWN
        assert claim.is_supported is False
        assert claim.confidence is Confidence.UNKNOWN
        assert claim.verified_citations == 0
        assert FailureCode.QUOTE_MISMATCH in claim.failures
        assert all(not v.ok for v in verdicts)

    def test_the_stage_reports_it(self) -> None:
        """End to end through the stage, as the pipeline would run it."""
        result, metric = run_validate_stage(
            [cases.CLAIM_WITH_FABRICATED_CITATION],
            sources=cases.sources(),
            texts=cases.texts(),
        )

        assert result.verified_count == 0
        assert result.rejected_count == 1
        assert result.failure_counts == {FailureCode.QUOTE_MISMATCH.value: 1}
        assert len(result.unknown_claims) == 1
        assert result.supported_claims == []
        # Stage 7 makes no model call, so it must cost nothing.
        assert metric.model is None
        assert metric.cost_usd == 0.0

    def test_a_valid_citation_still_passes(self, verifier: CitationVerifier) -> None:
        """The verifier must discriminate, not reject everything.

        Without this, a verifier that returns False unconditionally would
        pass every test above.
        """
        verdict = verifier.verify_citation(cases.VALID_CITATION)
        assert verdict.ok is True
        assert verdict.failure is None

        claim, _ = verifier.verify_claim(cases.VALID_CLAIM)
        assert claim.status is ClaimStatus.SUPPORTED
        assert claim.verified_citations == 1

    def test_fabricated_and_valid_are_distinguished_in_one_batch(self) -> None:
        """Mixed input: the real claim survives, the fabricated one does not."""
        result, _ = run_validate_stage(
            [cases.VALID_CLAIM, cases.CLAIM_WITH_FABRICATED_CITATION],
            sources=cases.sources(),
            texts=cases.texts(),
        )
        supported = {c.claim_id for c in result.supported_claims}
        unknown = {c.claim_id for c in result.unknown_claims}
        assert supported == {"C1"}
        assert unknown == {"C3"}


# ==========================================================================
# Every required failure mode
# ==========================================================================


class TestRequiredFailureModes:
    def test_valid_claim_with_valid_evidence_verifies(self, verifier: CitationVerifier) -> None:
        assert verifier.verify_evidence(cases.VALID_EVIDENCE).ok is True

    def test_valid_claim_with_wrong_evidence_is_ungrounded(
        self, verifier: CitationVerifier
    ) -> None:
        """The subtlest case: the citation verifies, the figure is absent.

        The quote is real, correctly located and correctly attributed. It just
        does not contain the 61% the claim asserts. A verifier checking only
        quote integrity would pass this, and it is the most dangerous output
        the system can produce — fluent, cited, and wrong.
        """
        claim, verdicts = verifier.verify_claim(cases.CLAIM_WITH_WRONG_EVIDENCE)

        assert all(v.ok for v in verdicts), "the citation itself is genuine"
        assert claim.status is ClaimStatus.UNKNOWN
        assert FailureCode.UNGROUNDED_FIGURE in claim.failures
        assert "61%" in claim.confidence_basis

    def test_citation_pointing_at_the_wrong_source(self, verifier: CitationVerifier) -> None:
        """Real wording, attributed to a source that does not contain it."""
        verdict = verifier.verify_citation(cases.WRONG_SOURCE_CITATION)
        assert verdict.ok is False
        assert verdict.failure is FailureCode.WRONG_SOURCE
        # The detail must name where it actually came from, or a reviewer
        # cannot tell misattribution from fabrication.
        assert cases.ABI_ID in verdict.detail

    def test_out_of_range_offsets(self, verifier: CitationVerifier) -> None:
        verdict = verifier.verify_citation(cases.OUT_OF_RANGE_CITATION)
        assert verdict.ok is False
        assert verdict.failure is FailureCode.OFFSET_OUT_OF_RANGE
        assert "exceed the source length" in verdict.detail

    def test_unsupported_claim_with_no_citation(self, verifier: CitationVerifier) -> None:
        claim, verdicts = verifier.verify_claim(cases.UNSUPPORTED_CLAIM)
        assert verdicts == []
        assert claim.status is ClaimStatus.UNKNOWN
        assert claim.failures == [FailureCode.NO_CITATION]
        assert claim.confidence is Confidence.UNKNOWN
        assert "no citation" in claim.confidence_basis.lower()

    def test_citation_naming_a_source_never_fetched(self, verifier: CitationVerifier) -> None:
        """A model inventing a plausible regulator URL lands here."""
        verdict = verifier.verify_citation(cases.UNKNOWN_SOURCE_CITATION)
        assert verdict.ok is False
        assert verdict.failure is FailureCode.UNKNOWN_SOURCE
        assert cases.INVENTED_ID in verdict.detail

    def test_content_changed_under_the_citation(self) -> None:
        """A citation is only meaningful against the bytes it was made from."""
        tampered = dict(cases.texts())
        tampered[cases.FCA_ID] = tampered[cases.FCA_ID].replace("61%", "99%")
        verifier = CitationVerifier(cases.sources(), tampered)

        verdict = verifier.verify_citation(cases.VALID_CITATION)
        assert verdict.ok is False
        assert verdict.failure is FailureCode.CONTENT_CHANGED

    def test_every_case_in_the_fixture_resolves_to_its_expected_outcome(self) -> None:
        """One table, so a regression anywhere shows up here."""
        expected = {
            "C1": None,
            "C2": FailureCode.UNGROUNDED_FIGURE,
            "C3": FailureCode.QUOTE_MISMATCH,
            "C4": FailureCode.WRONG_SOURCE,
            "C5": FailureCode.OFFSET_OUT_OF_RANGE,
            "C6": FailureCode.NO_CITATION,
            "C7": FailureCode.UNKNOWN_SOURCE,
        }
        result, _ = run_validate_stage(
            cases.ALL_CASES, sources=cases.sources(), texts=cases.texts()
        )
        for claim in result.claims:
            want = expected[claim.claim_id]
            if want is None:
                assert claim.failures == [], claim.claim_id
                assert claim.is_supported, claim.claim_id
            else:
                assert want in claim.failures, f"{claim.claim_id}: {claim.failures}"
                assert not claim.is_supported, claim.claim_id


# ==========================================================================
# The verifier must not be fooled by near-misses
# ==========================================================================


class TestVerifierIsNotTooPermissive:
    def test_a_single_changed_digit_is_caught(self, verifier: CitationVerifier) -> None:
        """Normalisation must not tolerate a changed number."""
        citation = Citation(
            source_id=cases.FCA_ID,
            start_char=cases.VALID_CITATION.start_char,
            end_char=cases.VALID_CITATION.end_char,
            cited_text=cases.VALID_PHRASE.replace("61%", "62%"),
        )
        assert verifier.verify_citation(citation).ok is False

    def test_a_negation_inserted_into_the_quote_is_caught(self, verifier: CitationVerifier) -> None:
        citation = Citation(
            source_id=cases.FCA_ID,
            start_char=cases.VALID_CITATION.start_char,
            end_char=cases.VALID_CITATION.end_char,
            cited_text=cases.VALID_PHRASE.replace("averaged", "never averaged"),
        )
        assert verifier.verify_citation(citation).ok is False

    def test_offsets_shifted_by_one_are_caught(self, verifier: CitationVerifier) -> None:
        citation = Citation(
            source_id=cases.FCA_ID,
            start_char=cases.VALID_CITATION.start_char + 1,
            end_char=cases.VALID_CITATION.end_char + 1,
            cited_text=cases.VALID_PHRASE,
        )
        assert verifier.verify_citation(citation).ok is False

    def test_a_quote_that_is_merely_a_substring_elsewhere_is_caught(
        self, verifier: CitationVerifier
    ) -> None:
        """Existing somewhere in the corpus is not the same as being cited."""
        citation = Citation(
            source_id=cases.FCA_ID,
            start_char=0,
            end_char=20,
            cited_text="Average premiums rose faster than general inflation.",
        )
        verdict = verifier.verify_citation(citation)
        assert verdict.ok is False
        assert verdict.failure is FailureCode.WRONG_SOURCE


class TestNormalisationIsNarrow:
    """Normalisation exists for typographic variance only."""

    def test_smart_quotes_and_dashes_are_tolerated(self, verifier: CitationVerifier) -> None:
        phrase = cases.VALID_PHRASE
        citation = Citation(
            source_id=cases.FCA_ID,
            start_char=cases.VALID_CITATION.start_char,
            end_char=cases.VALID_CITATION.end_char,
            cited_text=phrase.replace("'", "’"),
        )
        assert verifier.verify_citation(citation).ok is True

    def test_whitespace_differences_are_tolerated_and_flagged(
        self, verifier: CitationVerifier
    ) -> None:
        citation = Citation(
            source_id=cases.FCA_ID,
            start_char=cases.VALID_CITATION.start_char,
            end_char=cases.VALID_CITATION.end_char,
            cited_text=cases.VALID_PHRASE.replace(" ", "  "),
        )
        verdict = verifier.verify_citation(citation)
        assert verdict.ok is True
        # Flagged, so a reviewer can see the match was not byte-exact.
        assert verdict.normalised is True

    def test_byte_exact_match_is_not_flagged_as_normalised(
        self, verifier: CitationVerifier
    ) -> None:
        assert verifier.verify_citation(cases.VALID_CITATION).normalised is False

    @pytest.mark.parametrize(
        ("a", "b", "equal"),
        [
            ("Hello  world", "Hello world", True),
            ("it’s", "it's", True),
            ("a—b", "a-b", True),
            ("61%", "62%", False),
            ("rose", "fell", False),
            ("", "", True),
        ],
    )
    def test_comparison_rules(self, a: str, b: str, equal: bool) -> None:
        assert (normalise_for_comparison(a) == normalise_for_comparison(b)) is equal


# ==========================================================================
# Figure grounding
# ==========================================================================


class TestFigureGrounding:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("acceptance was 61%", ["61%"]),
            ("premium of £1,650 million", ["£1,650 million"]),
            ("rose 12.5% in 2024", ["12.5%", "2024"]),
            ("the top 3 insurers", []),
            ("no numbers here", []),
            ("$4.2bn of funding", ["$4.2bn"]),
        ],
    )
    def test_extracts_specific_figures_only(self, text: str, expected: list[str]) -> None:
        """Bare single digits are excluded: prose alternates '3' and 'three',
        and flagging that would produce false rejections constantly."""
        assert extract_figures(text) == expected

    @pytest.mark.parametrize(
        ("figure", "haystack", "grounded"),
        [
            ("61%", "averaged 61% across firms", True),
            ("1,650", "reached 1650 million", True),
            ("1650", "reached 1,650 million", True),
            ("61%", "averaged 62% across firms", False),
            ("12", "twelve firms reported", True),
            ("2024", "in 2025 the figure", False),
        ],
    )
    def test_grounding_matches_on_digits_not_formatting(
        self, figure: str, haystack: str, grounded: bool
    ) -> None:
        assert figure_is_grounded(figure, haystack) is grounded

    def test_a_claim_whose_figures_are_all_present_is_supported(
        self, verifier: CitationVerifier
    ) -> None:
        claim, _ = verifier.verify_claim(cases.VALID_CLAIM)
        assert claim.is_supported
        assert FailureCode.UNGROUNDED_FIGURE not in claim.failures

    def test_an_ungrounded_figure_cannot_be_rescued_by_a_second_claim(
        self, verifier: CitationVerifier
    ) -> None:
        """Grounding uses only this claim's own verified quotes."""
        claim, _ = verifier.verify_claim(cases.CLAIM_WITH_WRONG_EVIDENCE)
        assert not claim.is_supported

    def test_grounding_ignores_unverified_quotes(self) -> None:
        """A fabricated quote must not be allowed to ground a figure."""
        claim = Claim(
            claim_id="CX",
            text="Acceptance was 94%.",
            citations=[
                Citation(
                    source_id=cases.FCA_ID,
                    start_char=0,
                    end_char=30,
                    cited_text="acceptance was 94% exactly here",
                )
            ],
        )
        verifier = CitationVerifier(cases.sources(), cases.texts())
        verified, _ = verifier.verify_claim(claim)
        assert not verified.is_supported
        assert verified.verified_citations == 0


# ==========================================================================
# Credibility, corroboration, derived confidence
# ==========================================================================


class TestCredibilityClassification:
    @pytest.mark.parametrize(
        ("domain", "tier"),
        [
            ("fca.org.uk", CredibilityTier.PRIMARY),
            ("www.gov.uk", CredibilityTier.PRIMARY),
            ("ons.gov.uk", CredibilityTier.PRIMARY),
            ("sec.gov", CredibilityTier.PRIMARY),
            ("eiopa.europa.eu", CredibilityTier.PRIMARY),
            ("reuters.com", CredibilityTier.ESTABLISHED_SECONDARY),
            ("abi.org.uk", CredibilityTier.ESTABLISHED_SECONDARY),
            ("cam.ac.uk", CredibilityTier.ESTABLISHED_SECONDARY),
            ("mit.edu", CredibilityTier.ESTABLISHED_SECONDARY),
            ("someblog.example.com", CredibilityTier.UNVETTED),
            ("reddit.com", CredibilityTier.UNVETTED),
            ("medium.com", CredibilityTier.UNVETTED),
        ],
    )
    def test_tier_by_rule(self, domain: str, tier: CredibilityTier) -> None:
        assert classify_credibility(domain) is tier

    def test_unknown_domains_default_to_unvetted(self) -> None:
        """The cautious default: an unrecognised blog gets no benefit of doubt."""
        assert classify_credibility("whatever.xyz") is CredibilityTier.UNVETTED

    def test_www_prefix_and_trailing_dot_are_ignored(self) -> None:
        assert classify_credibility("www.fca.org.uk.") is CredibilityTier.PRIMARY

    def test_lookalike_domain_is_not_promoted(self) -> None:
        """`fca.org.uk.evil.test` must not inherit primary credibility."""
        assert classify_credibility("fca.org.uk.evil.test") is CredibilityTier.UNVETTED

    @pytest.mark.parametrize("host", ["gov.uk", "www.gov.uk", "europa.eu", "ac.uk"])
    def test_apex_domain_of_a_tier_suffix_is_classified(self, host: str) -> None:
        """F-009: endswith('.gov.uk') is false for the apex 'gov.uk' itself.

        The apex was falling through to UNVETTED, which downgraded the most
        credible sources in a corpus.
        """
        assert classify_credibility(host) is not CredibilityTier.UNVETTED

    def test_a_substring_match_does_not_promote(self) -> None:
        """'notgov.uk' must not match the '.gov.uk' suffix."""
        assert classify_credibility("notgov.uk") is CredibilityTier.UNVETTED
        assert classify_credibility("fakeac.uk") is CredibilityTier.UNVETTED


class TestDerivedConfidence:
    def test_single_primary_source_is_moderate(self, verifier: CitationVerifier) -> None:
        claim, _ = verifier.verify_claim(cases.VALID_CLAIM)
        assert claim.confidence is Confidence.MODERATE
        assert claim.corroboration == 1
        assert "primary" in claim.confidence_basis.lower()

    def test_two_independent_sources_with_a_primary_is_high(
        self, verifier: CitationVerifier
    ) -> None:
        claim, _ = verifier.verify_claim(cases.CORROBORATED_CLAIM)
        assert claim.verified_citations == 2
        assert claim.corroboration == 2
        assert claim.confidence is Confidence.HIGH
        assert "2 independent sources" in claim.confidence_basis

    def test_confidence_is_never_a_percentage(self, verifier: CitationVerifier) -> None:
        """Ordinal by design: a numeric score would be fake precision."""
        claim, _ = verifier.verify_claim(cases.VALID_CLAIM)
        assert claim.confidence in set(Confidence)
        assert "%" not in claim.confidence_basis.replace("61%", "")

    def test_basis_is_always_populated_for_a_verified_claim(
        self, verifier: CitationVerifier
    ) -> None:
        """A reader must be able to argue with the rule, not just the verdict."""
        for case in cases.ALL_CASES:
            claim, _ = verifier.verify_claim(case)
            assert claim.confidence_basis.strip(), claim.claim_id

    def test_unvetted_single_source_is_low(self) -> None:
        phrase = "Everyone I know is buying a policy."
        text = cases.texts()[cases.BLOG_ID]
        start = text.find(phrase)
        claim = Claim(
            claim_id="CB",
            text="Buying is widespread among acquaintances.",
            citations=[
                Citation(
                    source_id=cases.BLOG_ID,
                    start_char=start,
                    end_char=start + len(phrase),
                    cited_text=phrase,
                )
            ],
        )
        verified, _ = CitationVerifier(cases.sources(), cases.texts()).verify_claim(claim)
        assert verified.is_supported
        assert verified.confidence is Confidence.LOW
        assert "unvetted" in verified.confidence_basis.lower()


class TestQuoteSchema:
    def test_end_must_follow_start(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="greater than start_char"):
            Quote(source_id="s", start_char=10, end_char=10, text="x")

    def test_negative_offsets_rejected(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            Quote(source_id="s", start_char=-1, end_char=5, text="x")
