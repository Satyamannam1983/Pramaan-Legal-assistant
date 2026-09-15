"""Base protocol for retrieval."""
from __future__ import annotations

from typing import Protocol

from app.rag.retrieval.schemas import RetrievalResult


class BaseRetriever(Protocol):
    """Contract every retriever must satisfy."""

    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """Return top_k relevant chunks for the query, sorted by relevance."""
        ...
