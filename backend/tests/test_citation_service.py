"""Tests for the citation service."""
from __future__ import annotations

import pytest

from app.rag.citations.citation_service import CitationService
from app.rag.generation.evidence_selector import EvidenceBundle, EvidenceItem


def _make_evidence(citation_id: str = "SOURCE 1", **kwargs) -> EvidenceItem:
    return EvidenceItem(
        citation_id=citation_id,
        chunk_id=kwargs.get("chunk_id", "c1"),
        document_id=kwargs.get("document_id", "doc1"),
        content=kwargs.get("content", "test"),
        title=kwargs.get("title", "Test Doc"),
        section=kwargs.get("section", "Section 1"),
        page_start=kwargs.get("page_start", 1),
        page_end=kwargs.get("page_end", 1),
        source_url=kwargs.get("source_url", None),
    )


class TestCitationService:
    def test_load_evidence(self):
        svc = CitationService()
        bundle = EvidenceBundle(
            items=[_make_evidence("SOURCE 1"), _make_evidence("SOURCE 2")],
            query="test", total_candidates=2, filtered_count=2,
        )
        svc.load_evidence(bundle)
        assert svc.valid_citation_count == 2

    def test_extract_valid_citations(self):
        svc = CitationService()
        bundle = EvidenceBundle(
            items=[_make_evidence("SOURCE 1"), _make_evidence("SOURCE 2")],
            query="test", total_candidates=2, filtered_count=2,
        )
        svc.load_evidence(bundle)
        citations = svc.extract_citations("The answer is [SOURCE 1] and also [SOURCE 2].")
        assert len(citations) == 2
        assert citations[0].citation_id == "SOURCE 1"
        assert citations[1].citation_id == "SOURCE 2"

    def test_invalid_citation_ignored(self):
        svc = CitationService()
        bundle = EvidenceBundle(
            items=[_make_evidence("SOURCE 1")],
            query="test", total_candidates=1, filtered_count=1,
        )
        svc.load_evidence(bundle)
        citations = svc.extract_citations("Refer to [SOURCE 99] for details.")
        assert len(citations) == 0

    def test_duplicate_citations_deduplicated(self):
        svc = CitationService()
        bundle = EvidenceBundle(
            items=[_make_evidence("SOURCE 1")],
            query="test", total_candidates=1, filtered_count=1,
        )
        svc.load_evidence(bundle)
        citations = svc.extract_citations("[SOURCE 1] says [SOURCE 1] is correct.")
        assert len(citations) == 1

    def test_citation_metadata_from_evidence(self):
        svc = CitationService()
        item = _make_evidence(
            "SOURCE 1",
            title="IPC",
            section="Section 302",
            page_start=10,
            page_end=12,
            source_url="http://example.com",
        )
        bundle = EvidenceBundle(
            items=[item], query="test", total_candidates=1, filtered_count=1,
        )
        svc.load_evidence(bundle)
        citations = svc.extract_citations("As per [SOURCE 1].")
        assert citations[0].title == "IPC"
        assert citations[0].section == "Section 302"
        assert citations[0].page_start == 10
        assert citations[0].source_url == "http://example.com"

    def test_no_citations_in_output(self):
        svc = CitationService()
        bundle = EvidenceBundle(
            items=[_make_evidence("SOURCE 1")],
            query="test", total_candidates=1, filtered_count=1,
        )
        svc.load_evidence(bundle)
        citations = svc.extract_citations("This is an answer without any citations.")
        assert len(citations) == 0

    def test_get_citation_by_id(self):
        svc = CitationService()
        bundle = EvidenceBundle(
            items=[_make_evidence("SOURCE 1")],
            query="test", total_candidates=1, filtered_count=1,
        )
        svc.load_evidence(bundle)
        item = svc.get_citation_by_id("SOURCE 1")
        assert item is not None
        assert item.chunk_id == "c1"

    def test_get_citation_by_id_invalid(self):
        svc = CitationService()
        bundle = EvidenceBundle(
            items=[_make_evidence("SOURCE 1")],
            query="test", total_candidates=1, filtered_count=1,
        )
        svc.load_evidence(bundle)
        item = svc.get_citation_by_id("SOURCE 99")
        assert item is None
