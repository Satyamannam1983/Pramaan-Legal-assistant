"""Tests for the reranking module."""
from __future__ import annotations

import pytest

from app.rag.reranking.base import RerankResult


class TestRerankResult:
    def test_creation(self):
        r = RerankResult(chunk_id="c1", rerank_score=0.9)
        assert r.chunk_id == "c1"
        assert r.rerank_score == 0.9
        assert r.content == ""

    def test_defaults(self):
        r = RerankResult(chunk_id="c1", rerank_score=0.5)
        assert r.vector_score == 0.0
        assert r.keyword_score == 0.0
        assert r.rrf_score == 0.0
        assert r.vector_rank is None
        assert r.keyword_rank is None


class TestCrossEncoderReranker:
    @pytest.mark.asyncio
    async def test_empty_candidates(self):
        """Reranker should handle empty candidate list gracefully."""
        from app.rag.reranking.cross_encoder import CrossEncoderReranker
        reranker = CrossEncoderReranker(model_name="dummy")
        results = await reranker.rerank("test query", [])
        assert results == []

    @pytest.mark.asyncio
    async def test_fallback_when_model_unavailable(self):
        """When model is unavailable, should return candidates with fallback scores."""
        from app.rag.reranking.cross_encoder import CrossEncoderReranker
        reranker = CrossEncoderReranker(model_name="non-existent-model-12345")
        candidates = [
            RerankResult(chunk_id="c1", rerank_score=0.0, rrf_score=0.5, content="test"),
            RerankResult(chunk_id="c2", rerank_score=0.0, rrf_score=0.3, content="test2"),
        ]
        results = await reranker.rerank("test", candidates, top_k=2)
        # Should return results (fallback assigns rrf_score as rerank_score)
        assert len(results) == 2
        # First result should have higher fallback score
        assert results[0].rerank_score >= results[1].rerank_score

    @pytest.mark.asyncio
    async def test_top_k_limits_output(self):
        from app.rag.reranking.cross_encoder import CrossEncoderReranker
        reranker = CrossEncoderReranker(model_name="non-existent-model-12345")
        candidates = [
            RerankResult(chunk_id=f"c{i}", rerank_score=0.0, rrf_score=1.0 - i * 0.1, content=f"text {i}")
            for i in range(10)
        ]
        results = await reranker.rerank("test", candidates, top_k=3)
        assert len(results) == 3
