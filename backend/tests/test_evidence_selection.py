"""Tests for the evidence selection component."""
from __future__ import annotations

import pytest

from app.rag.generation.evidence_selector import EvidenceBundle, EvidenceItem, EvidenceSelector
from app.rag.reranking.base import RerankResult


def _make_candidate(
    chunk_id: str = "c1",
    content: str = "test content",
    rerank_score: float = 0.5,
    **kwargs,
) -> RerankResult:
    return RerankResult(
        chunk_id=chunk_id,
        rerank_score=rerank_score,
        content=content,
        document_id=kwargs.get("document_id", "doc1"),
        title=kwargs.get("title", "Test Doc"),
        section=kwargs.get("section", None),
        page_start=kwargs.get("page_start", 1),
        page_end=kwargs.get("page_end", 1),
        source=kwargs.get("source", "test"),
        source_url=kwargs.get("source_url", None),
        vector_score=kwargs.get("vector_score", 0.8),
        keyword_score=kwargs.get("keyword_score", 5.0),
        rrf_score=kwargs.get("rrf_score", 0.03),
    )


class TestEvidenceSelector:
    def test_empty_candidates(self):
        selector = EvidenceSelector()
        bundle = selector.select([], "test query")
        assert bundle.items == []
        assert bundle.total_candidates == 0
        assert bundle.filtered_count == 0

    def test_selects_top_candidates(self):
        selector = EvidenceSelector(max_chunks=3)
        candidates = [
            _make_candidate(f"c{i}", rerank_score=0.9 - i * 0.1)
            for i in range(5)
        ]
        bundle = selector.select(candidates, "test")
        assert len(bundle.items) == 3
        assert bundle.items[0].rerank_score >= bundle.items[1].rerank_score

    def test_citation_ids_are_stable(self):
        selector = EvidenceSelector(max_chunks=5)
        candidates = [_make_candidate(f"c{i}") for i in range(3)]
        bundle = selector.select(candidates, "test")
        ids = [item.citation_id for item in bundle.items]
        assert ids == ["SOURCE 1", "SOURCE 2", "SOURCE 3"]

    def test_metadata_preserved(self):
        selector = EvidenceSelector(max_chunks=5)
        candidates = [
            _make_candidate(
                "c1",
                title="Indian Contract Act",
                section="Section 3",
                page_start=5,
                page_end=5,
                source="raw_file",
            )
        ]
        bundle = selector.select(candidates, "test")
        assert bundle.items[0].title == "Indian Contract Act"
        assert bundle.items[0].section == "Section 3"
        assert bundle.items[0].page_start == 5
        assert bundle.items[0].source == "raw_file"

    def test_scores_preserved(self):
        selector = EvidenceSelector(max_chunks=5)
        candidates = [
            _make_candidate("c1", vector_score=0.9, keyword_score=8.0, rrf_score=0.05, rerank_score=0.95)
        ]
        bundle = selector.select(candidates, "test")
        item = bundle.items[0]
        assert item.vector_score == 0.9
        assert item.keyword_score == 8.0
        assert item.rrf_score == 0.05
        assert item.rerank_score == 0.95

    def test_max_chunks_limit(self):
        selector = EvidenceSelector(max_chunks=2)
        candidates = [_make_candidate(f"c{i}") for i in range(10)]
        bundle = selector.select(candidates, "test")
        assert len(bundle.items) <= 2

    def test_total_candidates_counted(self):
        selector = EvidenceSelector(max_chunks=2)
        candidates = [_make_candidate(f"c{i}") for i in range(10)]
        bundle = selector.select(candidates, "test")
        assert bundle.total_candidates == 10

    def test_avg_rerank_score(self):
        selector = EvidenceSelector(max_chunks=5)
        candidates = [
            _make_candidate("c1", rerank_score=0.8),
            _make_candidate("c2", rerank_score=0.6),
        ]
        bundle = selector.select(candidates, "test")
        assert abs(bundle.avg_rerank_score - 0.7) < 1e-6
