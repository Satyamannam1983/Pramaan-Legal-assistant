"""Search service — wires LegalRetrievalService to the API (backward compat).

The primary search path now goes through LegalRetrievalService directly
in routes.py. This service is kept for any code that still uses the
old SearchService.search() interface.
"""
from __future__ import annotations

import logging

from app.schemas import SemanticSearchRequest, SemanticSearchResponse, SemanticSearchResult

logger = logging.getLogger(__name__)


class SearchService:
    """Semantic search using MongoDB vector search or local cosine similarity.

    Falls back to an empty result list when MongoDB or the embedding provider
    is not available so the API stays up.
    """

    def __init__(self) -> None:
        self._retriever = None

    def _get_retriever(self):
        if self._retriever is None:
            from app.rag.retrieval.vector_retriever import VectorRetriever  # noqa: PLC0415
            self._retriever = VectorRetriever()
        return self._retriever

    async def search(self, request: SemanticSearchRequest) -> SemanticSearchResponse:
        query = request.query.strip()
        if not query:
            return SemanticSearchResponse(query=query, results=[])

        try:
            retriever = self._get_retriever()
            results = await retriever.retrieve(query, top_k=request.top_k)
        except RuntimeError as exc:
            logger.warning("Search failed (DB or embedding unavailable): %s", exc)
            return SemanticSearchResponse(query=query, results=[])

        return SemanticSearchResponse(
            query=query,
            results=[
                SemanticSearchResult(
                    chunk_id=r.chunk_id,
                    content=r.content,
                    document_id=r.document_id,
                    title=r.title,
                    section=r.section,
                    page_start=r.page_start,
                    page_end=r.page_end,
                    document_type=r.document_type,
                    court=r.court,
                    date=r.date,
                    source=r.source,
                    source_url=r.source_url,
                    similarity_score=r.vector_score,
                )
                for r in results
            ],
        )
