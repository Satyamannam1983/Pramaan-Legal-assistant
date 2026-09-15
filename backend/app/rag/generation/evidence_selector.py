"""Evidence selection — filters and limits retrieval results before LLM generation."""
from __future__ import annotations

import logging
from dataclasses import dataclass

from app.config import settings
from app.rag.reranking.base import RerankResult

logger = logging.getLogger(__name__)


@dataclass
class EvidenceItem:
    """A single evidence chunk selected for LLM context."""

    citation_id: str  # e.g. "SOURCE 1"
    chunk_id: str
    document_id: str
    content: str
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
    rerank_score: float = 0.0
    vector_score: float = 0.0
    keyword_score: float = 0.0
    rrf_score: float = 0.0


@dataclass
class EvidenceBundle:
    """Selected evidence for LLM generation."""

    items: list[EvidenceItem]
    query: str
    total_candidates: int
    filtered_count: int
    avg_rerank_score: float = 0.0


class EvidenceSelector:
    """Selects and limits evidence chunks for LLM context.

    Applies:
    - Minimum rerank score threshold
    - Maximum evidence chunks
    - Maximum context tokens (approximate)
    """

    def __init__(
        self,
        max_chunks: int | None = None,
        max_context_tokens: int | None = None,
        min_rerank_score: float | None = None,
    ) -> None:
        self._max_chunks = max_chunks or settings.max_evidence_chunks
        self._max_context_tokens = max_context_tokens or settings.max_context_tokens
        self._min_rerank_score = min_rerank_score if min_rerank_score is not None else settings.min_rerank_score

    def select(
        self,
        candidates: list[RerankResult],
        query: str,
    ) -> EvidenceBundle:
        """Filter and select evidence from reranked candidates.

        Parameters
        ----------
        candidates : Reranked results from the retrieval pipeline.
        query : The original user query.

        Returns
        -------
        EvidenceBundle with selected evidence items.
        """
        total = len(candidates)

        if not candidates:
            return EvidenceBundle(
                items=[],
                query=query,
                total_candidates=0,
                filtered_count=0,
            )

        # Step 1: Filter by minimum rerank score
        filtered = [
            c for c in candidates
            if c.rerank_score >= self._min_rerank_score
        ]

        if not filtered:
            # If all filtered out, take top results anyway with a warning
            logger.warning(
                "All %d candidates below min_rerank_score=%.2f. "
                "Taking top results anyway.",
                total, self._min_rerank_score,
            )
            filtered = candidates[:self._max_chunks]

        # Step 2: Apply max chunks limit
        selected = filtered[:self._max_chunks]

        # Step 3: Approximate token limit (rough: 1 token ≈ 4 chars)
        approx_tokens = 0
        final: list[RerankResult] = []
        for item in selected:
            item_tokens = len(item.content) // 4
            if approx_tokens + item_tokens > self._max_context_tokens:
                logger.debug(
                    "Context token limit reached at %d/%d tokens after %d items",
                    approx_tokens, self._max_context_tokens, len(final),
                )
                break
            approx_tokens += item_tokens
            final.append(item)

        # Build evidence items with stable citation IDs
        evidence_items: list[EvidenceItem] = []
        for i, item in enumerate(final, start=1):
            evidence_items.append(
                EvidenceItem(
                    citation_id=f"SOURCE {i}",
                    chunk_id=item.chunk_id,
                    document_id=item.document_id,
                    content=item.content,
                    title=item.title,
                    section=item.section,
                    subsection=getattr(item, "subsection", None),
                    page_start=item.page_start,
                    page_end=item.page_end,
                    document_type=item.document_type,
                    court=item.court,
                    date=item.date,
                    source=item.source,
                    source_url=item.source_url,
                    rerank_score=item.rerank_score,
                    vector_score=item.vector_score,
                    keyword_score=item.keyword_score,
                    rrf_score=item.rrf_score,
                )
            )

        avg_score = (
            sum(e.rerank_score for e in evidence_items) / len(evidence_items)
            if evidence_items else 0.0
        )

        logger.info(
            "Evidence selection: %d candidates → %d filtered → %d selected "
            "(avg_rerank=%.3f, approx_tokens=%d)",
            total, len(filtered), len(evidence_items), avg_score, approx_tokens,
        )

        return EvidenceBundle(
            items=evidence_items,
            query=query,
            total_candidates=total,
            filtered_count=len(filtered),
            avg_rerank_score=avg_score,
        )
