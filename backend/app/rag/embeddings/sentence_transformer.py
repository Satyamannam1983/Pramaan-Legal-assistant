"""Sentence-Transformers embedding service (multilingual-e5-large).

Model choice rationale
----------------------
`intfloat/multilingual-e5-large` is a state-of-the-art multilingual
dense retrieval model trained specifically for semantic similarity and
passage retrieval across 100+ languages. It achieves strong performance
on English legal text and covers Indian languages (Hindi, Marathi, Tamil,
Bengali, etc.) which will appear in future document sets.

The model produces 1024-dimensional embeddings. The dimension is derived
at runtime from the loaded model so changing the model only requires
updating EMBEDDING_MODEL in the environment.

For lower-resource environments, set:
  EMBEDDING_MODEL=paraphrase-multilingual-MiniLM-L12-v2  (384-dim, ~400MB)
  EMBEDDING_DIMENSION=384

The query prefix "query: " is prepended to queries and "passage: " to
documents as required by the multilingual-e5 family of models.
"""
from __future__ import annotations

import logging
from typing import ClassVar

from app.config import settings

logger = logging.getLogger(__name__)

# Lazy import so tests that don't use this provider don't require torch
_sentence_transformers = None


def _get_st():
    global _sentence_transformers  # noqa: PLW0603
    if _sentence_transformers is None:
        import sentence_transformers  # noqa: PLC0415
        _sentence_transformers = sentence_transformers
    return _sentence_transformers


class SentenceTransformerEmbeddingService:
    """Embedding service backed by sentence-transformers models."""

    # Class-level model cache so we don't reload across requests
    _model_cache: ClassVar[dict[str, object]] = {}

    def __init__(
        self,
        model_name: str | None = None,
        batch_size: int | None = None,
        query_prefix: str = "query: ",
        passage_prefix: str = "passage: ",
    ) -> None:
        self._model_name = model_name or settings.embedding_model
        self._batch_size = batch_size or settings.embedding_batch_size
        self._query_prefix = query_prefix
        self._passage_prefix = passage_prefix
        self._model = None  # lazy load

    def _load_model(self):
        if self._model is None:
            if self._model_name in self.__class__._model_cache:
                self._model = self.__class__._model_cache[self._model_name]
            else:
                logger.info("Loading embedding model: %s", self._model_name)
                st = _get_st()
                model = st.SentenceTransformer(self._model_name)
                self.__class__._model_cache[self._model_name] = model
                self._model = model
                logger.info(
                    "Embedding model loaded: %s (dim=%d)",
                    self._model_name,
                    self.embedding_dimension,
                )
        return self._model

    @property
    def embedding_dimension(self) -> int:
        """Return dimension from the actual model — not a hard-coded constant."""
        model = self._load_model()
        return model.get_sentence_embedding_dimension()  # type: ignore[attr-defined]

    def embed_text(self, text: str) -> list[float]:
        """Embed a single query string (adds query prefix)."""
        model = self._load_model()
        vector = model.encode(  # type: ignore[attr-defined]
            self._query_prefix + text,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vector.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of document passages in batches (adds passage prefix)."""
        if not texts:
            return []
        model = self._load_model()
        prefixed = [self._passage_prefix + t for t in texts]
        vectors = model.encode(  # type: ignore[attr-defined]
            prefixed,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [v.tolist() for v in vectors]
