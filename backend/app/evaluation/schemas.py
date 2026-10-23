from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class ExpectedAnswerCharacteristics(BaseModel):
    """Optional testable answer expectations; never an implicit quality score."""

    expected_abstention: bool | None = None
    required_terms: list[str] = Field(default_factory=list)
    reference_answer: str | None = None
    expected_citation_document_ids: list[str] = Field(default_factory=list)


class RetrievalExample(BaseModel):
    id: str = Field(min_length=1)
    query: str = Field(min_length=1, max_length=4000)
    relevant_chunk_ids: list[str] = Field(default_factory=list)
    relevant_document_ids: list[str] = Field(default_factory=list)
    category: str = "uncategorized"
    expected_answer: ExpectedAnswerCharacteristics | None = None
    expected_abstention: bool | None = None

    @model_validator(mode="after")
    def combine_legacy_abstention_label(self) -> RetrievalExample:
        if self.expected_answer is None and self.expected_abstention is not None:
            self.expected_answer = ExpectedAnswerCharacteristics(
                expected_abstention=self.expected_abstention
            )
        elif (
            self.expected_answer is not None
            and self.expected_abstention is not None
            and self.expected_answer.expected_abstention is not None
            and self.expected_answer.expected_abstention != self.expected_abstention
        ):
            raise ValueError("Conflicting expected abstention labels.")
        return self


class EvaluationDataset(BaseModel):
    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    provenance: str = Field(min_length=1)
    examples: list[RetrievalExample] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_example_ids(self) -> EvaluationDataset:
        ids = [example.id for example in self.examples]
        if len(ids) != len(set(ids)):
            raise ValueError("Evaluation dataset contains duplicate IDs.")
        return self


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
