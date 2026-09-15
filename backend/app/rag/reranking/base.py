"""Base protocol and result type for reranking."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class RerankResult:
    """A single reranked result with its relevance score."""

    chunk_id: str
    rerank_score: float
    content: str = ""
    document_id: str = ""
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
    vector_rank: int | None = None
    keyword_rank: int | None = None


class BaseReranker(Protocol):
    """Contract every reranker must satisfy."""

    async def rerank(
        self,
        query: str,
        candidates: list[RerankResult],
        top_k: int | None = None,
    ) -> list[RerankResult]:
        """Re-score candidates and return top_k sorted by relevance."""
        ...
