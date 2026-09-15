from app.schemas import SearchResult
from .vector_store import VectorStore


class Retriever:
    def __init__(self, store: VectorStore | None = None):
        self.store = store or VectorStore()

    async def retrieve(self, query: str, limit: int = 10) -> list[SearchResult]:
        return []
