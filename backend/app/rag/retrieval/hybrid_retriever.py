"""Hybrid retriever — combines vector and keyword search with RRF fusion."""
from __future__ import annotations

import logging

from app.config import settings
from app.rag.retrieval.keyword_retriever import KeywordRetriever
from app.rag.retrieval.schemas import FusedResult, KeywordResult, RetrievalResult
from app.rag.retrieval.vector_retriever import VectorRetriever

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    result_lists: list[list[RetrievalResult]],
    k: int | None = None,
) -> list[FusedResult]:
    """Merge multiple ranked lists using Reciprocal Rank Fusion (RRF).

    For each result:
        RRF score = SUM 1 / (k + rank)

    where rank is 1-based position in each list.

    Parameters
    ----------
    result_lists : list of ranked result lists from different retrievers.
    k : RRF constant (default from config). Higher k reduces the impact of
        top-ranked results.

    Returns
    -------
    Fused results sorted by descending RRF score.
    """
    rrf_k = k if k is not None else settings.rrf_k

    # Accumulate scores per chunk_id
    chunk_data: dict[str, dict] = {}

    for result_list in result_lists:
        for rank, result in enumerate(result_list, start=1):
            cid = result.chunk_id
            if cid not in chunk_data:
                chunk_data[cid] = {
                    "result": result,
                    "rrf_score": 0.0,
                    "vector_rank": None,
                    "keyword_rank": None,
                    "vector_score": 0.0,
                    "keyword_score": 0.0,
                }

            chunk_data[cid]["rrf_score"] += 1.0 / (rrf_k + rank)

            # Track which source contributed this result
            if result.vector_score > 0 and result.vector_rank is not None:
                chunk_data[cid]["vector_rank"] = result.vector_rank
                chunk_data[cid]["vector_score"] = result.vector_score
            if result.keyword_score > 0:
                chunk_data[cid]["keyword_rank"] = rank
                chunk_data[cid]["keyword_score"] = result.keyword_score

    # Sort by RRF score descending
    sorted_chunks = sorted(
        chunk_data.values(), key=lambda x: x["rrf_score"], reverse=True
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
        self._vector_top_k = vector_top_k or settings.vector_top_k
        self._keyword_top_k = keyword_top_k or settings.keyword_top_k

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

        # Run both retrievers
        vector_results: list[RetrievalResult] = []
        keyword_results: list[RetrievalResult] = []

        try:
            vector_results = await self._vector_retriever.retrieve(
                query, top_k=self._vector_top_k, document_id=document_id
            )
        except Exception as exc:
            logger.warning("Vector retrieval failed: %s", exc)

        try:
            kw_results = await self._keyword_retriever.retrieve(
                query, top_k=self._keyword_top_k, document_id=document_id
            )
            keyword_results = [_keyword_to_retrieval_result(kr) for kr in kw_results]
        except Exception as exc:
            logger.warning("Keyword retrieval failed: %s", exc)

        if not vector_results and not keyword_results:
            logger.warning("Both vector and keyword retrieval returned no results.")
            return []

        # Fuse via RRF
        fused = reciprocal_rank_fusion([vector_results, keyword_results])

        return fused[:k]

    async def invalidate_keyword_index(self) -> None:
        """Force keyword index rebuild on next query."""
        await self._keyword_retriever.invalidate()
