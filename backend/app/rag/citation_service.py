from collections.abc import Sequence
from app.schemas import SearchResult


class CitationService:
    def format(self, results: Sequence[SearchResult]) -> list[SearchResult]:
        return list(results)
