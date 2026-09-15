"""Tests for the grounding validator."""
from __future__ import annotations

import pytest

from app.rag.citations.citation_service import CitationService
from app.rag.generation.evidence_selector import EvidenceBundle, EvidenceItem
from app.rag.validation.grounding_validator import GroundingValidator


def _make_evidence(citation_id: str = "SOURCE 1", **kwargs) -> EvidenceItem:
    return EvidenceItem(
        citation_id=citation_id,
        chunk_id=kwargs.get("chunk_id", "c1"),
        document_id=kwargs.get("document_id", "doc1"),
        content=kwargs.get("content", "test"),
        title=kwargs.get("title", "Test"),
    )


class TestGroundingValidator:
    def _setup_validator_with_evidence(self, n_sources: int = 3):
        validator = GroundingValidator()
        citation_svc = CitationService()
        items = [_make_evidence(f"SOURCE {i}") for i in range(1, n_sources + 1)]
        bundle = EvidenceBundle(
            items=items, query="test", total_candidates=n_sources, filtered_count=n_sources,
        )
        citation_svc.load_evidence(bundle)
        return validator, citation_svc

    def test_valid_citations_pass(self):
        validator, citation_svc = self._setup_validator_with_evidence(3)
        result = validator.validate(
            "The answer is [SOURCE 1] and [SOURCE 2].",
            citation_svc, has_evidence=True,
        )
        assert result.is_grounded
        assert len(result.invalid_citations) == 0

    def test_fabricated_source_detected(self):
        validator, citation_svc = self._setup_validator_with_evidence(3)
        result = validator.validate(
            "Refer to [SOURCE 99] for this.",
            citation_svc, has_evidence=True,
        )
        assert not result.is_grounded
        assert "SOURCE 99" in result.invalid_citations

    def test_no_citations_with_evidence_warns(self):
        validator, citation_svc = self._setup_validator_with_evidence(3)
        result = validator.validate(
            "This is an answer without any citations.",
            citation_svc, has_evidence=True,
        )
        assert result.is_grounded  # Still grounded, just a warning
        assert len(result.warnings) > 0

    def test_empty_evidence_no_citations(self):
        validator = GroundingValidator()
        citation_svc = CitationService()
        result = validator.validate(
            "No sources available.",
            citation_svc, has_evidence=False,
        )
        assert result.is_grounded

    def test_citation_with_no_evidence_flagged(self):
        validator = GroundingValidator()
        citation_svc = CitationService()
        result = validator.validate(
            "According to [SOURCE 1].",
            citation_svc, has_evidence=False,
        )
        assert not result.is_grounded

    def test_mixed_valid_invalid(self):
        validator, citation_svc = self._setup_validator_with_evidence(2)
        result = validator.validate(
            "As per [SOURCE 1] and [SOURCE 5].",
            citation_svc, has_evidence=True,
        )
        assert not result.is_grounded
        assert "SOURCE 5" in result.invalid_citations
        assert "SOURCE 1" not in result.invalid_citations

    def test_validation_result_summary(self):
        validator, citation_svc = self._setup_validator_with_evidence(2)
        result = validator.validate(
            "[SOURCE 99] is cited.",
            citation_svc, has_evidence=True,
        )
        summary = result.summary
        assert "SOURCE 99" in summary
