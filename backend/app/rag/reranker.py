from collections.abc import Sequence
from app.schemas import SearchResult


class Reranker:
    async def rerank(self, query: str, results: Sequence[SearchResult]) -> list[SearchResult]:
        return list(results)
