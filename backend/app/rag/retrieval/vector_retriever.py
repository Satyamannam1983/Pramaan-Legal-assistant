"""Vector retriever — embeds query then performs cosine similarity search via MongoDB."""
from __future__ import annotations

import logging

from app.config import settings
from app.db.connection import check_connection
from app.db.repository import DocumentRepository
from app.rag.embeddings.base import BaseEmbeddingService
from app.rag.embeddings.factory import get_embedding_service
from app.rag.retrieval.schemas import RetrievalResult

logger = logging.getLogger(__name__)


class VectorRetriever:
    """Semantic retriever using MongoDB + NumPy cosine similarity.

    Pipeline
    --------
    query (str)
        ↓ embed_text()
        ↓ vector
        ↓ MongoDB similarity_search()
        ↓ list[RetrievalResult]
    """

    def __init__(
        self,
        embedding_service: BaseEmbeddingService | None = None,
        top_k: int | None = None,
    ) -> None:
        self._embedding_service = embedding_service or get_embedding_service()
        self._default_top_k = top_k if top_k is not None else settings.vector_top_k
        if self._default_top_k <= 0:
            raise ValueError("top_k must be positive.")

    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        document_id: str | None = None,
    ) -> list[RetrievalResult]:
        """Return top_k chunks most semantically similar to the query.

        Raises
        ------
        ValueError
            If the query is empty.
        RuntimeError
            If the database or embedding provider is unavailable.
        """
        query = query.strip()
        if not query:
            raise ValueError("Query must not be empty.")

        k = top_k if top_k is not None else self._default_top_k
        if k <= 0:
            raise ValueError("top_k must be positive.")

        # Avoid loading a multi-gigabyte local model when MongoDB is unavailable.
        if not await check_connection():
            raise RuntimeError("MongoDB is unavailable.")

        # Generate query embedding
        try:
            query_vector = self._embedding_service.embed_text(query)
        except Exception as exc:
            logger.warning("Query embedding failed (%s).", type(exc).__name__)
            raise RuntimeError("Embedding provider unavailable.") from None

        # Soft dimension check — warn but don't crash (model may differ from config)
        expected = settings.embedding_dimension
        if len(query_vector) != expected:
            logger.warning(
                "Embedding dimension mismatch: config=%d, model=%d. "
                "Update EMBEDDING_DIMENSION in .env.",
                expected,
                len(query_vector),
            )

        # Similarity search
        try:
            repo = DocumentRepository()
            rows = await repo.similarity_search(query_vector, top_k=k, document_id=document_id)
        except Exception as exc:
            logger.warning("Dense database search failed (%s).", type(exc).__name__)
            raise RuntimeError("Database unavailable or dense search failed.") from None

        results: list[RetrievalResult] = []
        for rank, row in enumerate(rows, start=1):
            results.append(
                RetrievalResult(
                    chunk_id=row.chunk_id,
                    content=row.content,
                    document_id=row.document_id,
                    title=row.title,
                    section=row.section,
                    subsection=row.subsection,
                    page_start=row.page_start,
                    page_end=row.page_end,
                    document_type=row.document_type,
                    court=row.court,
                    date=row.date,
                    source=row.source,
                    source_url=row.source_url,
                    vector_score=row.similarity_score,
                    vector_rank=rank,
                )
            )
        return results
