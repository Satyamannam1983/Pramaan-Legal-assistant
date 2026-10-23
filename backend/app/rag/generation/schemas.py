"""Structured output schemas for legal RAG responses."""
from __future__ import annotations

from pydantic import BaseModel, Field


class CitationRef(BaseModel):
    """A citation referenced by the LLM in its answer."""

    citation_id: str = Field(
        pattern=r"^SOURCE [1-9]\d*$",
        description="Source identifier, e.g. 'SOURCE 1'",
    )
    chunk_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    title: str | None = None
    section: str | None = None
    subsection: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    document_type: str | None = None
    court: str | None = None
    date: str | None = None
    source: str | None = None
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
    grounding_valid: bool = Field(
        default=False,
        description="Whether the returned answer passed citation validation or is source-only fallback text",
    )
    abstained: bool = Field(
        default=False,
        description="Whether the system explicitly declined to answer due to missing or invalid evidence",
    )
    generation_mode: str = Field(
        default="not_run",
        description="Provider or fallback mode used to produce the response",
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
