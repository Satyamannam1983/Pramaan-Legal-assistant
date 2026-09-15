from __future__ import annotations

from pydantic import BaseModel, Field


class RetrievalExample(BaseModel):
    id: str
    query: str = Field(min_length=1)
    relevant_chunk_ids: list[str] = Field(default_factory=list)
    relevant_document_ids: list[str] = Field(default_factory=list)
    category: str


class EvaluationDataset(BaseModel):
    name: str
    version: str = "1"
    examples: list[RetrievalExample]


class RetrievalMetrics(BaseModel):
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    recall_at_10: float
    mrr: float
    ndcg_at_5: float
    count: int


class EvaluationStatus(BaseModel):
    status: str
    reason: str | None = None
    dataset_size: int = 0
    metrics: dict[str, RetrievalMetrics] = Field(default_factory=dict)


class FailureCase(BaseModel):
    failure: str
    expected_behavior: str
    actual_behavior: str
    passed: bool
