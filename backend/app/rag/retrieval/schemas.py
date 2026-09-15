"""Data types shared across retrieval and reranking stages."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RetrievalResult:
    """A single retrieved chunk with full provenance and multi-signal scores.

    Populated progressively as the chunk moves through retrieval stages:
    - After vector search: chunk_id, content, metadata, vector_score
    - After keyword search: keyword_score
    - After RRF fusion: rrf_score, vector_rank, keyword_rank
    - After reranking: rerank_score, final_rank
    """

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

    # Score fields — populated by different retrieval stages
    vector_score: float = 0.0
    keyword_score: float = 0.0
    rrf_score: float = 0.0
    rerank_score: float = 0.0

    # Rank fields
    vector_rank: int | None = None
    keyword_rank: int | None = None
    final_rank: int | None = None


@dataclass
class KeywordResult:
    """A single result from the keyword (BM25) retriever."""

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
    keyword_score: float = 0.0


@dataclass
class FusedResult:
    """A merged result after RRF fusion of vector and keyword results."""

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
    vector_rank: int | None = None
    keyword_rank: int | None = None
