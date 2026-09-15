from collections.abc import Sequence
from app.schemas import SearchResult


class LLMService:
    async def generate(self, prompt: str, context: Sequence[SearchResult]) -> str:
        return "Placeholder response. Configure a reviewed LLM and retrieval pipeline before use."
