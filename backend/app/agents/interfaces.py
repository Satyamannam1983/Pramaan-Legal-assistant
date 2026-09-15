from typing import Protocol
from app.schemas import ChatRequest, ChatResponse


class LegalResearchAgent(Protocol):
    async def run(self, request: ChatRequest) -> ChatResponse: ...
