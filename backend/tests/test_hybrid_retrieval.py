"""Tests for hybrid retrieval and RRF fusion."""
from __future__ import annotations

import pytest

from app.rag.retrieval.hybrid_retriever import reciprocal_rank_fusion
from app.rag.retrieval.schemas import FusedResult, KeywordResult, RetrievalResult


# ---------------------------------------------------------------------------
# RRF fusion tests
# ---------------------------------------------------------------------------

class TestReciprocalRankFusion:
    def _make_vresult(self, chunk_id: str, score: float = 0.5, rank: int = 1) -> RetrievalResult:
        return RetrievalResult(
            chunk_id=chunk_id,
            content=f"content of {chunk_id}",
            document_id="doc1",
            vector_score=score,
            vector_rank=rank,
        )

    def _make_kresult(self, chunk_id: str, score: float = 5.0, rank: int = 1) -> RetrievalResult:
        return RetrievalResult(
            chunk_id=chunk_id,
            content=f"content of {chunk_id}",
            document_id="doc1",
            keyword_score=score,
        )

    def test_single_list_passthrough(self):
        vresults = [
            self._make_vresult("c1", 0.9, 1),
            self._make_vresult("c2", 0.8, 2),
        ]
        fused = reciprocal_rank_fusion([vresults])
        assert len(fused) == 2
        assert fused[0].chunk_id == "c1"
        assert fused[0].rrf_score > fused[1].rrf_score

    def test_merge_duplicates(self):
        # Same chunk in both lists
        vresults = [self._make_vresult("c1", 0.9, 1)]
        kresults = [self._make_kresult("c1", 10.0)]
        fused = reciprocal_rank_fusion([vresults, kresults])
        assert len(fused) == 1
        # RRF score should be sum from both lists
        expected = 1.0 / (60 + 1) + 1.0 / (60 + 1)
        assert abs(fused[0].rrf_score - expected) < 1e-6

    def test_ranking_by_rrf(self):
        # c1 is rank 1 in list A, rank 2 in list B
        # c2 is rank 2 in list A, rank 1 in list B
        list_a = [
            self._make_vresult("c1", 0.9, 1),
            self._make_vresult("c2", 0.8, 2),
        ]
        list_b = [
            self._make_kresult("c2", 10.0),
            self._make_kresult("c1", 8.0),
        ]
        fused = reciprocal_rank_fusion([list_a, list_b])
        assert len(fused) == 2
        # Both should have same RRF score (symmetric)
        assert abs(fused[0].rrf_score - fused[1].rrf_score) < 1e-6

    def test_empty_lists(self):
        fused = reciprocal_rank_fusion([[], []])
        assert fused == []

    def test_one_empty_list(self):
        vresults = [self._make_vresult("c1", 0.9, 1)]
        fused = reciprocal_rank_fusion([vresults, []])
        assert len(fused) == 1
        assert fused[0].chunk_id == "c1"

    def test_rrf_k_constant(self):
        vresults = [self._make_vresult("c1", 0.9, 1)]
        # With k=1, first rank gets 1/2 = 0.5
        fused = reciprocal_rank_fusion([vresults], k=1)
        assert abs(fused[0].rrf_score - 0.5) < 1e-6

    def test_metadata_preserved(self):
        vresults = [
            RetrievalResult(
                chunk_id="c1",
                content="hello",
                document_id="doc1",
                title="Title",
                section="Section 1",
                vector_score=0.9,
                vector_rank=1,
            )
        ]
        fused = reciprocal_rank_fusion([vresults])
        assert fused[0].title == "Title"
        assert fused[0].section == "Section 1"
        assert fused[0].vector_score == 0.9
        assert fused[0].vector_rank == 1

    def test_scores_preserved(self):
        vresults = [self._make_vresult("c1", 0.9, 1)]
        kresults = [self._make_kresult("c1", 10.0)]
        fused = reciprocal_rank_fusion([vresults, kresults])
        assert fused[0].vector_score == 0.9
        assert fused[0].keyword_score == 10.0
        assert fused[0].vector_rank == 1

    def test_many_chunks_sorted_correctly(self):
        # Create 10 chunks with known ranking
        vresults = [
            self._make_vresult(f"c{i}", 0.9 - i * 0.05, i) for i in range(1, 11)
        ]
        fused = reciprocal_rank_fusion([vresults])
        assert len(fused) == 10
        # Verify sorted by RRF score descending
        for i in range(len(fused) - 1):
            assert fused[i].rrf_score >= fused[i + 1].rrf_score
