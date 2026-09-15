"""Nodes for the bounded legal research workflow."""
from __future__ import annotations

import logging
import re
from typing import Any

from app.config import settings
from app.rag.graph.state import LegalResearchState
from app.rag.generation.schemas import LegalAnswer
from app.rag.services.rag_service import LegalRAGService
from app.rag.services.retrieval_service import LegalRetrievalService, SearchResultItem

logger = logging.getLogger(__name__)


def _mark(state: LegalResearchState, name: str) -> list[str]:
    return [*state.get("nodes_executed", []), name]


def analyze_query(state: LegalResearchState) -> dict[str, Any]:
    query = state["query"].strip()
    query_type = "UNKNOWN"
    if re.search(r"\b(section|article)\s+[\w-]+", query, re.I):
        query_type = "SECTION_LOOKUP"
    elif re.search(r"\b(case|judgment|citation|AIR|SCC)\b", query, re.I):
        query_type = "CASE_LOOKUP"
    elif re.search(r"\b(compare|difference|versus|vs\.?)\b", query, re.I):
        query_type = "COMPARISON"
    elif re.search(r"\b(define|what is|meaning of)\b", query, re.I):
        query_type = "DEFINITION"
    elif re.search(r"\b(and|both|multiple|judgments|documents)\b", query, re.I):
        query_type = "MULTI_DOCUMENT"
    else:
        query_type = "FACTUAL"
    return {
        "original_query": state.get("original_query", query),
        "query": query,
        "query_type": query_type,
        "nodes_executed": _mark(state, "analyze_query"),
    }


def select_retrieval_strategy(state: LegalResearchState) -> dict[str, Any]:
    strategy = "EXACT_TERM" if state.get("query_type") in {"SECTION_LOOKUP", "CASE_LOOKUP"} else "STANDARD"
    return {
        "retrieval_strategy": strategy,
        "strategies": [*state.get("strategies", []), strategy],
        "nodes_executed": _mark(state, "select_retrieval_strategy"),
    }


async def retrieve_evidence(state: LegalResearchState) -> dict[str, Any]:
    query = state["query"]
    if state.get("retrieval_strategy") == "BROAD":
        query = f"{query} legal provisions principles remedies"
    service = LegalRetrievalService()
    results = await service.search(query, top_k=settings.final_top_k)
    serialized = [result.__dict__ for result in results]
    attempt = state.get("retrieval_attempt", 0) + 1
    logger.info(
        "legal_graph_retrieval attempt=%d strategy=%s results=%d",
        attempt, state.get("retrieval_strategy"), len(serialized),
    )
    return {
        "retrieved_results": serialized,
        "retrieval_attempt": attempt,
        "nodes_executed": _mark(state, "retrieve_evidence"),
    }


def evaluate_evidence(state: LegalResearchState) -> dict[str, Any]:
    results = state.get("retrieved_results", [])
    scores = [float(r.get("rerank_score", r.get("rrf_score", 0.0))) for r in results]
    score = max(scores, default=0.0)
    usable = [r for r in results if r.get("content") and r.get("document_id")]
    sufficient = bool(usable) and score >= settings.evidence_min_score
    reason = "strong_evidence" if sufficient else ("weak_evidence" if results else "no_evidence")
    return {
        "evidence_score": score,
        "evidence_sufficient": sufficient,
        "evidence_reason": reason,
        "nodes_executed": _mark(state, "evaluate_evidence"),
    }


def refine_query(state: LegalResearchState) -> dict[str, Any]:
    query = state.get("query", state["original_query"])
    refined = f"{query} legal provisions principles requirements remedies"
    return {
        "query": refined,
        "refined_query": refined,
        "retrieval_strategy": "BROAD",
        "strategies": [*state.get("strategies", []), "BROAD"],
        "nodes_executed": _mark(state, "refine_query"),
    }


async def generate_answer(state: LegalResearchState) -> dict[str, Any]:
    # The existing RAG service remains the single owner of context, LLM, and citations.
    results = [SearchResultItem(**item) for item in state.get("retrieved_results", [])]
    result = await LegalRAGService().answer(state["query"], search_results=results)
    answer = result.model_dump()
    return {
        "answer": answer,
        "citations": [c.model_dump() for c in result.citations],
        "limitations": result.limitations,
        "nodes_executed": _mark(state, "generate_answer"),
    }


def validate_answer(state: LegalResearchState) -> dict[str, Any]:
    answer = state.get("answer", {})
    citations = answer.get("citations", [])
    valid = bool(citations) or answer.get("confidence") == "low"
    validation = {"is_grounded": valid, "summary": "validated" if valid else "missing citations"}
    return {
        "validation_result": validation,
        "nodes_executed": _mark(state, "validate_answer"),
    }


def abstain(state: LegalResearchState) -> dict[str, Any]:
    answer = LegalAnswer(
        answer="I couldn't find sufficient evidence in the indexed legal sources to answer this question reliably.",
        confidence="low",
        limitations=[state.get("evidence_reason", "Evidence was insufficient.")],
    ).model_dump()
    return {
        "answer": answer,
        "citations": [],
        "limitations": answer["limitations"],
        "nodes_executed": _mark(state, "abstain"),
    }


def finalize(state: LegalResearchState) -> dict[str, Any]:
    return {"nodes_executed": _mark(state, "finalize")}
