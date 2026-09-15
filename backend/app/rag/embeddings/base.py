"""Base protocol for embedding services."""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class BaseEmbeddingService(Protocol):
    """Contract every embedding provider must satisfy."""

    @property
    def embedding_dimension(self) -> int:
        """Return the number of dimensions in each embedding vector."""
        ...

    def embed_text(self, text: str) -> list[float]:
        """Embed a single string and return a vector."""
        ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of strings.  Uses batching internally."""
        ...
