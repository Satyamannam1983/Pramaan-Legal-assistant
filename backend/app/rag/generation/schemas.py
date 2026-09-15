"""Structured output schemas for legal RAG responses."""
from __future__ import annotations

from pydantic import BaseModel, Field


class CitationRef(BaseModel):
    """A citation referenced by the LLM in its answer."""

    citation_id: str = Field(description="Source identifier, e.g. 'SOURCE 1'")
    chunk_id: str = ""
    document_id: str = ""
    title: str | None = None
    section: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    source_url: str | None = None
    relevant_quote: str | None = None


class LegalAnswer(BaseModel):
    """Structured legal answer from the RAG pipeline."""

    answer: str = Field(description="The grounded legal answer")
    key_points: list[str] = Field(default_factory=list, description="Key conclusions from the evidence")
    citations: list[CitationRef] = Field(default_factory=list, description="Cited sources with metadata")
    limitations: list[str] = Field(default_factory=list, description="Limitations of the evidence or answer")
    confidence: str = Field(
        default="low",
        description="Confidence level: high, medium, or low",
    )


class ChatRequest(BaseModel):
    """Request for the chat/RAG endpoint."""

    query: str = Field(min_length=1, max_length=4000, description="Legal question")
    conversation_id: str | None = Field(default=None, description="Optional conversation ID")


class ChatResponse(BaseModel):
    """Response from the RAG endpoint."""

    answer: str
    key_points: list[str] = []
    citations: list[CitationRef] = []
    limitations: list[str] = []
    confidence: str = "low"
    conversation_id: str = ""
