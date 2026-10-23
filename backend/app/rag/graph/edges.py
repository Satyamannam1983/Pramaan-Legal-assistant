"""Conditional graph routing rules."""
from __future__ import annotations

from app.config import settings
from app.rag.graph.state import LegalResearchState

_MAX_GRAPH_RETRIEVAL_ATTEMPTS = 2


def route_after_evaluation(state: LegalResearchState) -> str:
    if state.get("evidence_sufficient"):
        return "generate_answer"
    configured_attempts = max(1, settings.max_retrieval_attempts)
    max_attempts = min(configured_attempts, _MAX_GRAPH_RETRIEVAL_ATTEMPTS)
    if state.get("retrieval_attempt", 0) < max_attempts:
        return "refine_query"
    return "abstain"


def route_after_validation(state: LegalResearchState) -> str:
    validation = state.get("validation_result", {})
    return "finalize" if validation.get("is_grounded", False) else "abstain"
