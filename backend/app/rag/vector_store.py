from typing import Sequence

from app.schemas import SearchResult


class VectorStore:
    async def upsert(self, texts: Sequence[str], vectors: Sequence[Sequence[float]]) -> None:
        return None

    async def search(self, query_vector: Sequence[float], limit: int = 10) -> list[SearchResult]:
        return []
