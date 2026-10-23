"""Focused tests for retrieval orchestration and optional reranking."""
from __future__ import annotations

import pytest

from app.rag.retrieval.schemas import FusedResult
from app.rag.services.retrieval_service import LegalRetrievalService


def _fused(chunk_id: str = "doc:0") -> FusedResult:
    return FusedResult(
        chunk_id=chunk_id,
        content="Section 302 addresses the offence.",
        document_id="doc",
        title="Penal Code",
        section="Section 302",
        source="official",
        source_url="https://example.test/act",
        rrf_score=0.02,
    )


@pytest.mark.asyncio
async def test_search_assembles_hybrid_result_with_provenance(monkeypatch):
    from app.config import settings

    class Hybrid:
        async def retrieve(self, query, top_k, document_id):
            assert query == "Section 302"
            assert document_id == "doc"
            return [_fused()]

    monkeypatch.setattr(settings, "reranker_provider", "")
    service = LegalRetrievalService()
    service._hybrid_retriever = Hybrid()

    results = await service.search(" Section 302 ", top_k=1, document_id="doc")

    assert len(results) == 1
    assert results[0].chunk_id == "doc:0"
    assert results[0].document_id == "doc"
    assert results[0].section == "Section 302"
    assert results[0].source_url == "https://example.test/act"
    assert results[0].final_rank == 1
    assert results[0].rrf_score == results[0].rerank_score == 0.02


@pytest.mark.asyncio
async def test_reranker_order_and_score_are_used(monkeypatch):
    from app.config import settings
    from app.rag.reranking.base import RerankResult

    class Hybrid:
        async def retrieve(self, query, top_k, document_id):
            return [_fused()]

    class Reranker:
        async def rerank(self, query, candidates, top_k):
            assert query == "Section 302"
            assert candidates[0].source_url == "https://example.test/act"
            assert top_k == 1
            result = candidates[0]
            return [RerankResult(
                chunk_id=result.chunk_id,
                content=result.content,
                document_id=result.document_id,
                section=result.section,
                source_url=result.source_url,
                rrf_score=result.rrf_score,
                rerank_score=0.91,
            )]

    monkeypatch.setattr(settings, "reranker_provider", "cross_encoder")
    monkeypatch.setattr(settings, "reranker_model", "configured-model")
    service = LegalRetrievalService()
    service._hybrid_retriever = Hybrid()
    service._reranker = Reranker()

    results = await service.search("Section 302", top_k=1)

    assert len(results) == 1
    assert results[0].rerank_score == 0.91
    assert results[0].rrf_score == 0.02
    assert results[0].source_url == "https://example.test/act"


def test_cross_encoder_is_created_only_with_provider_and_model(monkeypatch):
    import app.rag.reranking.cross_encoder as cross_encoder_module
    from app.config import settings

    class ConfiguredReranker:
        pass

    monkeypatch.setattr(
        cross_encoder_module,
        "CrossEncoderReranker",
        ConfiguredReranker,
    )
    service = LegalRetrievalService()

    monkeypatch.setattr(settings, "reranker_provider", "cross_encoder")
    monkeypatch.setattr(settings, "reranker_model", "configured-model")
    assert isinstance(service._get_reranker(), ConfiguredReranker)

    disabled = LegalRetrievalService()
    monkeypatch.setattr(settings, "reranker_provider", "")
    assert disabled._get_reranker() is None


@pytest.mark.asyncio
async def test_no_hybrid_results_returns_empty_without_fallback(monkeypatch):
    class Hybrid:
        async def retrieve(self, query, top_k, document_id):
            return []

    service = LegalRetrievalService()
    service._hybrid_retriever = Hybrid()

    assert await service.search("unmatched legal query") == []


@pytest.mark.asyncio
async def test_failed_hybrid_and_vector_retrieval_returns_empty_safely(
    monkeypatch, caplog
):
    import app.rag.services.retrieval_service as retrieval_module

    class Hybrid:
        async def retrieve(self, query, top_k, document_id):
            raise RuntimeError("do-not-log-source-text")

    class BrokenVector:
        async def retrieve(self, query, top_k, document_id):
            raise RuntimeError("do-not-log-source-text")

    monkeypatch.setattr(retrieval_module, "VectorRetriever", BrokenVector)
    service = LegalRetrievalService()
    service._hybrid_retriever = Hybrid()

    assert await service.search("query") == []
    assert "RuntimeError" in caplog.text
    assert "do-not-log-source-text" not in caplog.text


@pytest.mark.asyncio
async def test_nonpositive_top_k_is_rejected():
    with pytest.raises(ValueError, match="top_k"):
        await LegalRetrievalService().search("query", top_k=0)
