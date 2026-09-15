"""Tests for the retrieval pipeline (vector → keyword → fusion → reranker)."""
from __future__ import annotations

import pytest

from app.rag.retrieval.schemas import FusedResult, KeywordResult, RetrievalResult


class TestRetrievalSchemas:
    def test_retrieval_result_defaults(self):
        r = RetrievalResult(chunk_id="c1", content="text", document_id="d1")
        assert r.vector_score == 0.0
        assert r.keyword_score == 0.0
        assert r.rrf_score == 0.0
        assert r.rerank_score == 0.0
        assert r.final_rank is None

    def test_fused_result_defaults(self):
        f = FusedResult(chunk_id="c1", content="text", document_id="d1")
        assert f.vector_score == 0.0
        assert f.keyword_score == 0.0
        assert f.rrf_score == 0.0

    def test_keyword_result_defaults(self):
        k = KeywordResult(chunk_id="c1", content="text", document_id="d1")
        assert k.keyword_score == 0.0


class TestPipelineConsistency:
    """Verify the pipeline data flow is consistent."""

    def test_vector_to_fused(self):
        """VectorRetriever output can become FusedResult input."""
        vr = RetrievalResult(
            chunk_id="c1", content="text", document_id="d1",
            vector_score=0.85, vector_rank=1,
        )
        # Simulate what hybrid_retriever does
        fr = FusedResult(
            chunk_id=vr.chunk_id,
            content=vr.content,
            document_id=vr.document_id,
            title=vr.title,
            section=vr.section,
            vector_score=vr.vector_score,
            vector_rank=vr.vector_rank,
            rrf_score=0.5,
        )
        assert fr.chunk_id == "c1"
        assert fr.vector_score == 0.85
        assert fr.rrf_score == 0.5

    def test_fused_to_rerank(self):
        """FusedResult can become RerankResult input."""
        fr = FusedResult(
            chunk_id="c1", content="text", document_id="d1",
            vector_score=0.85, keyword_score=5.0, rrf_score=0.03,
        )
        from app.rag.reranking.base import RerankResult
        rr = RerankResult(
            chunk_id=fr.chunk_id,
            rerank_score=0.0,
            content=fr.content,
            document_id=fr.document_id,
            vector_score=fr.vector_score,
            keyword_score=fr.keyword_score,
            rrf_score=fr.rrf_score,
        )
        assert rr.chunk_id == "c1"
        assert rr.rerank_score == 0.0  # Will be set by reranker
        assert rr.rrf_score == 0.03
