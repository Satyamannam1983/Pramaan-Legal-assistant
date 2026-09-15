"""Factory that returns the correct embedding service from configuration."""
from __future__ import annotations

import logging

from app.config import settings
from app.rag.embeddings.base import BaseEmbeddingService

logger = logging.getLogger(__name__)

_instance: BaseEmbeddingService | None = None


def get_embedding_service(force_new: bool = False) -> BaseEmbeddingService:
    """Return a (singleton) embedding service based on EMBEDDING_PROVIDER."""
    global _instance  # noqa: PLW0603
    if _instance is not None and not force_new:
        return _instance

    provider = settings.embedding_provider.lower()

    if provider == "stub":
        from app.rag.embeddings.stub import StubEmbeddingService  # noqa: PLC0415
        _instance = StubEmbeddingService()
        logger.info("Embedding provider: stub (dimension=%d)", _instance.embedding_dimension)

    elif provider in ("sentence_transformer", "sentence-transformer", "st"):
        from app.rag.embeddings.sentence_transformer import (  # noqa: PLC0415
            SentenceTransformerEmbeddingService,
        )
        _instance = SentenceTransformerEmbeddingService()
        logger.info(
            "Embedding provider: sentence-transformers model=%s",
            settings.embedding_model,
        )

    else:
        raise ValueError(
            f"Unknown EMBEDDING_PROVIDER={provider!r}. "
            "Supported values: 'stub', 'sentence_transformer'."
        )

    return _instance
