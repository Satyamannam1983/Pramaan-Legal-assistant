from .citation_service import CitationService
from .llm_service import LLMService
from .reranker import Reranker
from .retriever import Retriever


class RAGPipeline:
    """Orchestration boundary; deliberately returns no legal conclusions."""

    def __init__(self):
        self.retriever, self.reranker = Retriever(), Reranker()
        self.citations, self.llm = CitationService(), LLMService()

    async def answer(self, query: str) -> tuple[str, list]:
        results = await self.retriever.retrieve(query)
        results = await self.reranker.rerank(query, results)
        return await self.llm.generate(query, results), self.citations.format(results)
