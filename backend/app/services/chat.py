"""Chat service — delegates to LegalRAGService for grounded answers."""
from __future__ import annotations

import logging
from uuid import uuid4

from app.schemas import ChatRequest, ChatResponse

logger = logging.getLogger(__name__)


class ChatService:
    """Chat service using the grounded RAG pipeline."""

    def __init__(self, search=None) -> None:
        self._rag_service = None

    def _get_rag_service(self):
        if self._rag_service is None:
            from app.rag.services.rag_service import LegalRAGService  # noqa: PLC0415
            self._rag_service = LegalRAGService()
        return self._rag_service

    async def respond(self, request: ChatRequest) -> ChatResponse:
        """Generate a grounded legal answer with citations."""
        query = request.query.strip()
        if not query:
            return ChatResponse(
                answer="Please provide a valid legal question.",
                conversation_id=request.conversation_id or str(uuid4()),
            )

        try:
            rag = self._get_rag_service()
            result = await rag.answer(query, conversation_id=request.conversation_id)
            return ChatResponse(
                answer=result.answer,
                key_points=result.key_points,
                citations=[
                    {
                        "citation_id": c.citation_id,
                        "chunk_id": c.chunk_id,
                        "document_id": c.document_id,
                        "title": c.title,
                        "section": c.section,
                        "page_start": c.page_start,
                        "page_end": c.page_end,
                        "source_url": c.source_url,
                        "relevant_quote": c.relevant_quote,
                    }
                    for c in result.citations
                ],
                limitations=result.limitations,
                confidence=result.confidence,
                conversation_id=request.conversation_id or str(uuid4()),
            )
        except Exception as exc:
            logger.warning("Chat/RAG failed: %s", exc)
            return ChatResponse(
                answer=(
                    "I encountered an error while processing your question. "
                    "Please try again or use the search endpoint for direct retrieval."
                ),
                conversation_id=request.conversation_id or str(uuid4()),
                limitations=[f"Error: {exc}"],
            )
