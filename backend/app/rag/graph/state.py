"""Serializable state for the bounded legal research graph."""
from __future__ import annotations

from typing import Any, TypedDict


class LegalResearchState(TypedDict, total=False):
    query: str
    original_query: str
    query_type: str
    retrieval_strategy: str
    retrieval_attempt: int
    strategies: list[str]
    retrieved_results: list[dict[str, Any]]
    evidence_score: float
    evidence_sufficient: bool
    evidence_reason: str
    refined_query: str
    answer: dict[str, Any]
    citations: list[dict[str, Any]]
    validation_result: dict[str, Any]
    limitations: list[str]
    nodes_executed: list[str]
    error: str | None
