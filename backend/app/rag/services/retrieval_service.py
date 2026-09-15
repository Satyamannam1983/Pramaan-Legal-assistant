"""High-level retrieval service — coordinates hybrid retrieval + reranking.

This is the single entry point for search. The API layer should use this
service and never interact with retrievers or rerankers directly.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from app.config import settings
from app.rag.retrieval.hybrid_retriever import HybridRetriever
from app.rag.retrieval.schemas import FusedResult, RetrievalResult
from app.rag.retrieval.vector_retriever import VectorRetriever

logger = logging.getLogger(__name__)


@dataclass
class RetrievalDebugInfo:
    """Detailed retrieval stage information for debugging."""

    query: str
    vector_results: list[RetrievalResult] = field(default_factory=list)
    keyword_results: list[RetrievalResult] = field(default_factory=list)
    fused_results: list[FusedResult] = field(default_factory=list)
    reranked_results: list[FusedResult] = field(default_factory=list)
    final_results: list[FusedResult] = field(default_factory=list)
    vector_time_ms: float = 0.0
    keyword_time_ms: float = 0.0
    fusion_time_ms: float = 0.0
    rerank_time_ms: float = 0.0
    total_time_ms: float = 0.0
    reranker_available: bool = True


@dataclass
class SearchResultItem:
    """A single search result for the API response."""

    chunk_id: str
    content: str
    document_id: str
    title: str | None = None
    section: str | None = None
    subsection: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    document_type: str | None = None
    court: str | None = None
    date: str | None = None
    source: str | None = None
    source_url: str | None = None
    vector_score: float = 0.0
    keyword_score: float = 0.0
    rrf_score: float = 0.0
    rerank_score: float = 0.0
    final_rank: int | None = None


class LegalRetrievalService:
    """Single high-level service coordinating hybrid retrieval + reranking.

    Usage::

        service = LegalRetrievalService()
        results = await service.search("What is consideration?", top_k=5)
        debug = await service.search_debug("What is consideration?", top_k=5)
    """

    def __init__(self) -> None:
        self._hybrid_retriever: HybridRetriever | None = None
        self._reranker = None

    def _get_hybrid_retriever(self) -> HybridRetriever:
        if self._hybrid_retriever is None:
            self._hybrid_retriever = HybridRetriever()
        return self._hybrid_retriever

    def _get_reranker(self):
        if self._reranker is None:
            if settings.reranker_provider == "cross_encoder" and settings.reranker_model:
                from app.rag.reranking.cross_encoder import CrossEncoderReranker  # noqa: PLC0415
                self._reranker = CrossEncoderReranker()
            else:
                self._reranker = None
        return self._reranker

    async def search(
        self,
        query: str,
        top_k: int | None = None,
        document_id: str | None = None,
    ) -> list[SearchResultItem]:
        """Search using hybrid retrieval + optional reranking.

        Returns top_k results. Falls back gracefully if any stage fails.
        """
        k = top_k or settings.final_top_k
        query = query.strip()
        if not query:
            return []

        try:
            hybrid = self._get_hybrid_retriever()
            fused = await hybrid.retrieve(query, top_k=settings.rerank_top_k, document_id=document_id)
        except Exception as exc:
            logger.warning("Hybrid retrieval failed, trying vector-only: %s", exc)
            try:
                vector = VectorRetriever()
                v_results = await vector.retrieve(query, top_k=k, document_id=document_id)
                return [self._vector_to_search_item(r, i + 1) for i, r in enumerate(v_results)]
            except Exception as exc2:
                logger.warning("Vector-only retrieval also failed: %s", exc2)
                return []

        if not fused:
            return []

        # Apply reranking if available
        reranker = self._get_reranker()
        if reranker is not None:
            try:
                from app.rag.reranking.base import RerankResult  # noqa: PLC0415

                candidates = [
                    RerankResult(
                        chunk_id=f.chunk_id,
                        rerank_score=0.0,
                        content=f.content,
                        document_id=f.document_id,
                        title=f.title,
                        section=f.section,
                        subsection=f.subsection,
                        page_start=f.page_start,
                        page_end=f.page_end,
                        document_type=f.document_type,
                        court=f.court,
                        date=f.date,
                        source=f.source,
                        source_url=f.source_url,
                        vector_score=f.vector_score,
                        keyword_score=f.keyword_score,
                        rrf_score=f.rrf_score,
                        vector_rank=f.vector_rank,
                        keyword_rank=f.keyword_rank,
                    )
                    for f in fused
                ]
                reranked = await reranker.rerank(query, candidates, top_k=k)
                return [
                    SearchResultItem(
                        chunk_id=r.chunk_id,
                        content=r.content,
                        document_id=r.document_id,
                        title=r.title,
                        section=r.section,
                        subsection=r.subsection,
                        page_start=r.page_start,
                        page_end=r.page_end,
                        document_type=r.document_type,
                        court=r.court,
                        date=r.date,
                        source=r.source,
                        source_url=r.source_url,
                        vector_score=r.vector_score,
                        keyword_score=r.keyword_score,
                        rrf_score=r.rrf_score,
                        rerank_score=r.rerank_score,
                        final_rank=i + 1,
                    )
                    for i, r in enumerate(reranked)
                ]
            except Exception as exc:
                logger.warning("Reranking failed, using RRF scores: %s", exc)

        # Fallback: use RRF scores without reranking
        return [
            SearchResultItem(
                chunk_id=f.chunk_id,
                content=f.content,
                document_id=f.document_id,
                title=f.title,
                section=f.section,
                subsection=f.subsection,
                page_start=f.page_start,
                page_end=f.page_end,
                document_type=f.document_type,
                court=f.court,
                date=f.date,
                source=f.source,
                source_url=f.source_url,
                vector_score=f.vector_score,
                keyword_score=f.keyword_score,
                rrf_score=f.rrf_score,
                rerank_score=f.rrf_score,
                final_rank=i + 1,
            )
            for i, f in enumerate(fused[:k])
        ]

    async def search_debug(
        self,
        query: str,
        top_k: int | None = None,
    ) -> RetrievalDebugInfo:
        """Search with full debug information for each retrieval stage."""
        k = top_k or settings.final_top_k
        query = query.strip()
        if not query:
            return RetrievalDebugInfo(query=query)

        debug = RetrievalDebugInfo(query=query)
        total_start = time.monotonic()

        # Stage 1: Vector retrieval
        v_start = time.monotonic()
        try:
            vector = VectorRetriever()
            debug.vector_results = await vector.retrieve(
                query, top_k=settings.vector_top_k
            )
        except Exception as exc:
            logger.warning("Vector retrieval failed in debug: %s", exc)
        debug.vector_time_ms = (time.monotonic() - v_start) * 1000

        # Stage 2: Keyword retrieval
        kw_start = time.monotonic()
        try:
            hybrid = self._get_hybrid_retriever()
            # Access the keyword retriever directly
            from app.rag.retrieval.keyword_retriever import KeywordRetriever  # noqa: PLC0415
            from app.rag.retrieval.hybrid_retriever import _keyword_to_retrieval_result  # noqa: PLC0415

            kw_retriever = hybrid._keyword_retriever
            kw_results = await kw_retriever.retrieve(
                query, top_k=settings.keyword_top_k
            )
            debug.keyword_results = [_keyword_to_retrieval_result(kr) for kr in kw_results]
        except Exception as exc:
            logger.warning("Keyword retrieval failed in debug: %s", exc)
        debug.keyword_time_ms = (time.monotonic() - kw_start) * 1000

        # Stage 3: RRF fusion
        fusion_start = time.monotonic()
        try:
            from app.rag.retrieval.hybrid_retriever import reciprocal_rank_fusion  # noqa: PLC0415

            debug.fused_results = reciprocal_rank_fusion(
                [debug.vector_results, debug.keyword_results]
            )
        except Exception as exc:
            logger.warning("Fusion failed in debug: %s", exc)
        debug.fusion_time_ms = (time.monotonic() - fusion_start) * 1000

        # Stage 4: Reranking
        rerank_start = time.monotonic()
        reranker = self._get_reranker()
        if reranker is not None and debug.fused_results:
            try:
                from app.rag.reranking.base import RerankResult  # noqa: PLC0415

                candidates = [
                    RerankResult(
                        chunk_id=f.chunk_id,
                        rerank_score=0.0,
                        content=f.content,
                        document_id=f.document_id,
                        title=f.title,
                        section=f.section,
                        subsection=f.subsection,
                        page_start=f.page_start,
                        page_end=f.page_end,
                        document_type=f.document_type,
                        court=f.court,
                        date=f.date,
                        source=f.source,
                        source_url=f.source_url,
                        vector_score=f.vector_score,
                        keyword_score=f.keyword_score,
                        rrf_score=f.rrf_score,
                        vector_rank=f.vector_rank,
                        keyword_rank=f.keyword_rank,
                    )
                    for f in debug.fused_results
                ]
                debug.reranked_results = await reranker.rerank(
                    query, candidates, top_k=k
                )
            except Exception as exc:
                logger.warning("Reranking failed in debug: %s", exc)
                debug.reranker_available = False
        else:
            debug.reranker_available = False
        debug.rerank_time_ms = (time.monotonic() - rerank_start) * 1000

        # Final results
        if debug.reranked_results:
            debug.final_results = debug.reranked_results[:k]
        else:
            debug.final_results = debug.fused_results[:k]

        debug.total_time_ms = (time.monotonic() - total_start) * 1000
        return debug

    @staticmethod
    def _vector_to_search_item(r: RetrievalResult, rank: int) -> SearchResultItem:
        """Convert a RetrievalResult to a SearchResultItem."""
        return SearchResultItem(
            chunk_id=r.chunk_id,
            content=r.content,
            document_id=r.document_id,
            title=r.title,
            section=r.section,
            subsection=r.subsection,
            page_start=r.page_start,
            page_end=r.page_end,
            document_type=r.document_type,
            court=r.court,
            date=r.date,
            source=r.source,
            source_url=r.source_url,
            vector_score=r.vector_score,
            final_rank=rank,
        )
