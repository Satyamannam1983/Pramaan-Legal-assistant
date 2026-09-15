"""Stub embedding service — deterministic random unit vectors.

Used in tests and when EMBEDDING_PROVIDER=stub.
Vectors are seeded from the text hash so the same text always gets
the same vector (within a process), which makes stub-based retrieval
tests somewhat meaningful.
"""
from __future__ import annotations

import hashlib
import struct

from app.config import settings


class StubEmbeddingService:
    """Returns reproducible unit-length random vectors without loading a model."""

    def __init__(self, dimension: int | None = None) -> None:
        self._dimension = dimension or settings.embedding_dimension

    @property
    def embedding_dimension(self) -> int:
        return self._dimension

    def _vector_for(self, text: str) -> list[float]:
        # Deterministic seed from text hash
        digest = hashlib.sha256(text.encode()).digest()
        # Generate enough floats from the digest by cycling
        floats: list[float] = []
        idx = 0
        while len(floats) < self._dimension:
            chunk = digest[idx % len(digest) : idx % len(digest) + 4]
            if len(chunk) < 4:
                chunk = chunk + digest[: 4 - len(chunk)]
            (val,) = struct.unpack(">f", chunk)
            floats.append(val)
            idx += 3  # stride to avoid repeating pattern

        # L2 normalise so vectors behave like real cosine-normalised embeddings
        mag = sum(f * f for f in floats) ** 0.5
        if mag == 0:
            return [0.0] * self._dimension
        return [f / mag for f in floats[: self._dimension]]

    def embed_text(self, text: str) -> list[float]:
        return self._vector_for(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector_for(t) for t in texts]
