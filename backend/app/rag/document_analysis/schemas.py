from __future__ import annotations

from pydantic import BaseModel, Field


class MetadataEvidence(BaseModel):
    field: str
    value: str | None = None
    source_chunk_id: str | None = None
    confidence: float = 0.0


class LegalDocumentMetadata(BaseModel):
    document_id: str
    title: str | None = None
    document_type: str | None = None
    court: str | None = None
    case_number: str | None = None
    citation: str | None = None
    date: str | None = None
    judges: list[str] = Field(default_factory=list)
    parties: list[str] = Field(default_factory=list)
    petitioner: str | None = None
    respondent: str | None = None
    acts_referenced: list[str] = Field(default_factory=list)
    sections_referenced: list[str] = Field(default_factory=list)
    articles_referenced: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    evidence: list[MetadataEvidence] = Field(default_factory=list)


class DocumentAnalysis(BaseModel):
    document_id: str
    metadata: LegalDocumentMetadata
    summary: str
    facts: list[str] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)
    arguments: list[str] = Field(default_factory=list)
    reasoning: list[str] = Field(default_factory=list)
    decision: str | None = None
    important_sections: list[str] = Field(default_factory=list)
    referenced_laws: list[str] = Field(default_factory=list)
    key_entities: list[str] = Field(default_factory=list)
    citations: list[dict] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class DocumentSummary(BaseModel):
    document_id: str
    title: str | None = None
    document_type: str | None = None
    court: str | None = None
    date: str | None = None
    chunk_count: int = 0
    created_at: str | None = None


class DocumentQuestion(BaseModel):
    query: str = Field(min_length=1, max_length=4000)


class CompareRequest(BaseModel):
    document_ids: list[str] = Field(min_length=2, max_length=2)


class DocumentComparison(BaseModel):
    document_ids: list[str]
    overview: str
    similarities: list[str] = Field(default_factory=list)
    differences: list[str] = Field(default_factory=list)
    legal_provisions: list[str] = Field(default_factory=list)
    reasoning_comparison: list[str] = Field(default_factory=list)
    outcome_comparison: str | None = None
    citations: list[dict] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
