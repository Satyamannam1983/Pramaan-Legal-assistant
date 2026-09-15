from typing import Protocol, Sequence
from app.schemas import SearchRequest, SearchResult


class Retriever(Protocol):
    async def retrieve(self, request: SearchRequest) -> Sequence[SearchResult]: ...


class Reranker(Protocol):
    async def rerank(self, query: str, results: Sequence[SearchResult]) -> Sequence[SearchResult]: ...


class Generator(Protocol):
    async def generate(self, prompt: str, context: Sequence[SearchResult]) -> str: ...
