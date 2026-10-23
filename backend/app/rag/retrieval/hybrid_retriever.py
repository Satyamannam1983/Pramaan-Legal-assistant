"""Hybrid retriever — combines vector and keyword search with RRF fusion."""
from __future__ import annotations

import logging
import math
import time

from app.config import settings
from app.rag.retrieval.keyword_retriever import KeywordRetriever
from app.rag.retrieval.schemas import FusedResult, KeywordResult, RetrievalResult
from app.rag.retrieval.vector_retriever import VectorRetriever

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    result_lists: list[list[RetrievalResult]],
    k: int | None = None,
    weights: list[float] | None = None,
) -> list[FusedResult]:
    """Merge multiple ranked lists using Reciprocal Rank Fusion (RRF).

    For each result:
        RRF score = SUM weight / (k + rank)

    where rank is 1-based position in each list.

    Parameters
    ----------
    result_lists : list of ranked result lists from different retrievers.
    k : RRF constant (default from config). Higher k reduces the impact of
        top-ranked results.
    weights : Per-list weights. Two-list calls default to configured dense and
        lexical weights; other list counts default to equal weights.

    Returns
    -------
    Fused results sorted by descending RRF score.
    """
    rrf_k = k if k is not None else settings.rrf_k
    if rrf_k <= 0:
        raise ValueError("RRF k must be a positive integer.")
    if weights is None:
        weights = (
            [settings.vector_weight, settings.keyword_weight]
            if len(result_lists) == 2
            else [1.0] * len(result_lists)
        )
    if len(weights) != len(result_lists):
        raise ValueError("RRF weights must match the number of result lists.")
    if (
        any(not math.isfinite(weight) or weight < 0 for weight in weights)
        or not any(weights)
    ):
        raise ValueError("RRF weights must be non-negative with at least one positive weight.")

    # Accumulate scores per source-document/chunk pair to avoid cross-document collisions.
    chunk_data: dict[tuple[str, str], dict] = {}

    for list_index, result_list in enumerate(result_lists):
        seen_in_list: set[tuple[str, str]] = set()
        for rank, result in enumerate(result_list, start=1):
            identity = (result.document_id, result.chunk_id)
            if identity in seen_in_list:
                continue
            seen_in_list.add(identity)
            key = identity
            if key not in chunk_data:
                chunk_data[key] = {
                    "result": result,
                    "rrf_score": 0.0,
                    "vector_rank": None,
                    "keyword_rank": None,
                    "vector_score": 0.0,
                    "keyword_score": 0.0,
                }
            entry = chunk_data[key]

            entry["rrf_score"] += weights[list_index] / (rrf_k + rank)

            if list_index == 0:
                entry["vector_rank"] = rank
                entry["vector_score"] = result.vector_score
            elif result.vector_score != 0:
                entry["vector_score"] = result.vector_score
            if list_index == 1:
                entry["keyword_rank"] = rank
                entry["keyword_score"] = result.keyword_score
            elif result.keyword_score != 0:
                entry["keyword_score"] = result.keyword_score

            # Fill sparse metadata from the other retrieval result without overwriting
            # provenance already supplied by the first ranked list.
            preferred = entry["result"]
            for field in (
                "content", "title", "section", "subsection", "page_start",
                "page_end", "document_type", "court", "date", "source", "source_url",
            ):
                if not getattr(preferred, field) and getattr(result, field):
                    setattr(preferred, field, getattr(result, field))

    # Stable tie-breaking keeps identical retrieval results ordered consistently.
    sorted_chunks = sorted(
        chunk_data.values(),
        key=lambda x: (
            -x["rrf_score"],
            x["result"].document_id,
            x["result"].chunk_id,
        ),
    )

    fused: list[FusedResult] = []
    for entry in sorted_chunks:
        r = entry["result"]
        fused.append(
            FusedResult(
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
                vector_score=entry["vector_score"],
                keyword_score=entry["keyword_score"],
                rrf_score=entry["rrf_score"],
                vector_rank=entry["vector_rank"],
                keyword_rank=entry["keyword_rank"],
            )
        )

    return fused


def _keyword_to_retrieval_result(kr: KeywordResult) -> RetrievalResult:
    """Convert a KeywordResult to a RetrievalResult for fusion."""
    return RetrievalResult(
        chunk_id=kr.chunk_id,
        content=kr.content,
        document_id=kr.document_id,
        title=kr.title,
        section=kr.section,
        subsection=kr.subsection,
        page_start=kr.page_start,
        page_end=kr.page_end,
        document_type=kr.document_type,
        court=kr.court,
        date=kr.date,
        source=kr.source,
        source_url=kr.source_url,
        keyword_score=kr.keyword_score,
    )


class HybridRetriever:
    """Combines vector semantic search and BM25 keyword search via RRF.

    Pipeline
    --------
    query
        ↓ Vector retrieval (top vector_top_k)
        ↓ Keyword retrieval (top keyword_top_k)
        ↓ Candidate merging via RRF
        ↓ Ranked fused results
    """

    def __init__(
        self,
        vector_retriever: VectorRetriever | None = None,
        keyword_retriever: KeywordRetriever | None = None,
        vector_top_k: int | None = None,
        keyword_top_k: int | None = None,
    ) -> None:
        self._vector_retriever = vector_retriever or VectorRetriever()
        self._keyword_retriever = keyword_retriever or KeywordRetriever()
        self._vector_top_k = vector_top_k if vector_top_k is not None else settings.vector_top_k
        self._keyword_top_k = keyword_top_k if keyword_top_k is not None else settings.keyword_top_k
        if self._vector_top_k <= 0 or self._keyword_top_k <= 0:
            raise ValueError("Retrieval candidate limits must be positive.")

    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        document_id: str | None = None,
    ) -> list[FusedResult]:
        """Retrieve using hybrid vector + BM25 with RRF fusion.

        Returns fused results sorted by RRF score. The top_k parameter
        controls the final output size after fusion.
        """
        query = query.strip()
        if not query:
            raise ValueError("Query must not be empty.")

        k = top_k if top_k is not None else settings.rerank_top_k
        if k <= 0:
            raise ValueError("top_k must be positive.")

        # Run both retrievers
        vector_results: list[RetrievalResult] = []
        keyword_results: list[RetrievalResult] = []
        vector_error: Exception | None = None
        keyword_error: Exception | None = None

        vector_start = time.monotonic()
        try:
            vector_results = await self._vector_retriever.retrieve(
                query, top_k=self._vector_top_k, document_id=document_id
            )
            vector_results.sort(
                key=lambda result: (
                    -result.vector_score,
                    result.document_id,
                    result.chunk_id,
                )
            )
        except Exception as exc:
            vector_error = exc
            logger.warning("Vector retrieval failed (%s).", type(exc).__name__)
        logger.info(
            "Dense retrieval completed results=%d latency_ms=%.1f",
            len(vector_results),
            (time.monotonic() - vector_start) * 1000,
        )

        keyword_start = time.monotonic()
        try:
            kw_results = await self._keyword_retriever.retrieve(
                query, top_k=self._keyword_top_k, document_id=document_id
            )
            kw_results.sort(
                key=lambda result: (
                    -result.keyword_score,
                    result.document_id,
                    result.chunk_id,
                )
            )
            keyword_results = [_keyword_to_retrieval_result(kr) for kr in kw_results]
        except Exception as exc:
            keyword_error = exc
            logger.warning("Keyword retrieval failed (%s).", type(exc).__name__)
        logger.info(
            "Lexical retrieval completed results=%d latency_ms=%.1f",
            len(keyword_results),
            (time.monotonic() - keyword_start) * 1000,
        )

        if not vector_results and not keyword_results:
            if vector_error is not None and keyword_error is not None:
                raise RuntimeError("Both dense and lexical retrieval stages failed.") from None
            logger.info("Dense and lexical retrieval completed with no results.")
            return []

        # Fuse via RRF
        fused = reciprocal_rank_fusion([vector_results, keyword_results])

        return fused[:k]

    async def invalidate_keyword_index(self) -> None:
        """Force keyword index rebuild on next query."""
        await self._keyword_retriever.invalidate()
