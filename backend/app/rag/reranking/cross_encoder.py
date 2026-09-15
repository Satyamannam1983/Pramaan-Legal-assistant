"""Cross-encoder reranker — re-scores (query, document) pairs using a transformer model.

Uses a multilingual cross-encoder suitable for Indian legal text.
The model independently scores each (query, chunk) pair, producing more accurate
relevance judgments than the bi-encoder used for initial retrieval.
"""
from __future__ import annotations

import logging

from app.config import settings
from app.rag.reranking.base import RerankResult

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Cross-encoder reranker using sentence-transformers.

    Parameters
    ----------
    model_name : str | None
        HuggingFace model ID. Falls back to settings.reranker_model.
    device : str | None
        Device for inference ('cpu', 'cuda', etc.). None = auto.
    """

    _model_cache: dict[str, object] = {}

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
    ) -> None:
        self._model_name = model_name or settings.reranker_model
        self._device = device
        self._model = None

    def _load_model(self):
        """Lazy-load the cross-encoder model."""
        if self._model is not None:
            return self._model

        if not self._model_name:
            raise RuntimeError(
                "No reranker model configured. Set RERANKER_MODEL in .env."
            )

        # Check class-level cache
        if self._model_name in CrossEncoderReranker._model_cache:
            self._model = CrossEncoderReranker._model_cache[self._model_name]
            return self._model

        try:
            from sentence_transformers import CrossEncoder  # noqa: PLC0415

            self._model = CrossEncoder(
                self._model_name,
                device=self._device,
            )
            CrossEncoderReranker._model_cache[self._model_name] = self._model
            logger.info("Cross-encoder reranker loaded: %s", self._model_name)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load reranker model '{self._model_name}': {exc}"
            ) from exc

        return self._model

    async def rerank(
        self,
        query: str,
        candidates: list[RerankResult],
        top_k: int | None = None,
    ) -> list[RerankResult]:
        """Re-score (query, chunk) pairs using the cross-encoder.

        Parameters
        ----------
        query : The search query.
        candidates : Pre-scored candidates from hybrid retrieval.
        top_k : Number of results to return after reranking.

        Returns
        -------
        Reranked results sorted by descending rerank_score.
        """
        if not candidates:
            return []

        k = top_k if top_k is not None else settings.final_top_k

        try:
            model = self._load_model()
        except RuntimeError as exc:
            logger.warning("Reranker unavailable, returning candidates as-is: %s", exc)
            # Fallback: return candidates without reranking
            for i, c in enumerate(candidates[:k]):
                c.rerank_score = c.rrf_score  # Use RRF as fallback score
            return candidates[:k]

        # Build (query, document) pairs
        pairs = [(query, c.content) for c in candidates]

        try:
            scores = model.predict(pairs)
        except Exception as exc:
            logger.warning("Reranker prediction failed: %s", exc)
            for c in candidates[:k]:
                c.rerank_score = c.rrf_score
            return candidates[:k]

        # Assign scores and sort
        for candidate, score in zip(candidates, scores):
            candidate.rerank_score = float(score)

        reranked = sorted(candidates, key=lambda c: c.rerank_score, reverse=True)
        return reranked[:k]
