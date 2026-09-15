"""Tests for fallback behaviour when components are unavailable."""
from __future__ import annotations

import pytest


class TestFallbacks:
    """Test that the system degrades gracefully when components fail."""

    @pytest.mark.asyncio
    async def test_retrieval_service_empty_query(self):
        """Empty query should return empty results, not crash."""
        from app.rag.services.retrieval_service import LegalRetrievalService
        service = LegalRetrievalService()
        results = await service.search("", top_k=5)
        assert results == []

    @pytest.mark.asyncio
    async def test_retrieval_service_debug_empty_query(self):
        """Debug endpoint should handle empty query."""
        from app.rag.services.retrieval_service import LegalRetrievalService
        service = LegalRetrievalService()
        debug = await service.search_debug("", top_k=5)
        assert debug.query == ""
        assert debug.total_time_ms == 0.0

    @pytest.mark.asyncio
    async def test_keyword_retriever_empty_query(self):
        """Keyword retriever should reject empty query."""
        from app.rag.retrieval.keyword_retriever import KeywordRetriever
        retriever = KeywordRetriever()
        with pytest.raises(ValueError):
            await retriever.retrieve("")

    @pytest.mark.asyncio
    async def test_hybrid_retriever_empty_query(self):
        """Hybrid retriever should reject empty query."""
        from app.rag.retrieval.hybrid_retriever import HybridRetriever
        retriever = HybridRetriever()
        with pytest.raises(ValueError):
            await retriever.retrieve("")

    def test_bm25_empty_index(self):
        """BM25 index with no documents should return empty."""
        from app.rag.retrieval.keyword_retriever import BM25Index
        index = BM25Index()
        assert index.is_empty
        results = index.search("anything")
        assert results == []

    @pytest.mark.asyncio
    async def test_reranker_empty_candidates(self):
        """Reranker should handle empty candidates."""
        from app.rag.reranking.cross_encoder import CrossEncoderReranker
        reranker = CrossEncoderReranker(model_name="dummy")
        results = await reranker.rerank("query", [])
        assert results == []
