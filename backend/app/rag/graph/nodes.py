"""Nodes for the bounded legal research workflow."""
from __future__ import annotations

import logging
import re
import math
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
    raw_query = state.get("query")
    if not isinstance(raw_query, str) or not raw_query.strip():
        raise ValueError("A non-empty query is required.")
    query = raw_query.strip()
    original_query = state.get("original_query")
    if not isinstance(original_query, str) or not original_query.strip():
        original_query = query
    if re.search(r"\b(compare|comparison|difference|versus|vs\.?)\b", query, re.I):
        query_type = "COMPARISON"
    elif re.search(r"\b(section|article)\s+[\w-]+", query, re.I):
        query_type = "SECTION_LOOKUP"
    elif re.search(r"\b(case|judgment|citation|AIR|SCC)\b", query, re.I):
        query_type = "CASE_LOOKUP"
    elif re.search(r"\b(define|what is|meaning of)\b", query, re.I):
        query_type = "DEFINITION"
    elif re.search(r"\b(and|both|multiple|judgments|documents)\b", query, re.I):
        query_type = "MULTI_DOCUMENT"
    else:
        query_type = "FACTUAL"
    attempt = state.get("retrieval_attempt", 0)
    if not isinstance(attempt, int) or isinstance(attempt, bool) or attempt < 0:
        attempt = 0
    return {
        "original_query": original_query.strip(),
        "query": query,
        "query_type": query_type,
        "retrieval_attempt": attempt,
        "strategies": list(state.get("strategies", [])),
        "nodes_executed": _mark(state, "analyze_query"),
    }


def select_retrieval_strategy(state: LegalResearchState) -> dict[str, Any]:
    strategy = "EXACT_TERM" if state.get("query_type") in {"SECTION_LOOKUP", "CASE_LOOKUP"} else "STANDARD"
    return {
        "retrieval_strategy": strategy,
        "strategies": [*state.get("strategies", []), strategy],
        "nodes_executed": _mark(state, "select_retrieval_strategy"),
    }


async def retrieve_evidence(
    state: LegalResearchState,
    retrieval_service: LegalRetrievalService | None = None,
) -> dict[str, Any]:
    query = state.get("query", "").strip()
    if not query:
        raise ValueError("A non-empty query is required for retrieval.")
    if state.get("retrieval_strategy") == "BROAD":
        query = f"{query} legal provisions principles requirements remedies"
    service = retrieval_service or LegalRetrievalService()
    attempt = state.get("retrieval_attempt", 0) + 1
    retrieval_error = None
    try:
        results = await service.search(query, top_k=settings.final_top_k)
        serialized = [result.__dict__ for result in results]
    except Exception as exc:
        logger.warning(
            "legal_graph_retrieval failed attempt=%d error_type=%s",
            attempt,
            type(exc).__name__,
        )
        serialized = []
        retrieval_error = "retrieval_failed"
    logger.info(
        "legal_graph_retrieval attempt=%d strategy=%s results=%d",
        attempt, state.get("retrieval_strategy"), len(serialized),
    )
    return {
        "retrieved_results": serialized,
        "search_results": serialized,
        "retrieval_attempt": attempt,
        "retrieval_error": retrieval_error,
        "nodes_executed": _mark(state, "retrieve_evidence"),
    }


def evaluate_evidence(state: LegalResearchState) -> dict[str, Any]:
    results = state.get("retrieved_results", [])
    scores = []
    for result in results:
        try:
            value = float(result.get("rerank_score") or result.get("rrf_score", 0.0))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            scores.append(value)
    score = max(scores, default=0.0)
    usable = [
        result for result in results
        if isinstance(result.get("content"), str)
        and result["content"].strip()
        and result.get("document_id")
        and result.get("chunk_id")
    ]
    sufficient = bool(usable) and score >= settings.evidence_min_score
    if sufficient:
        reason = "sufficient_evidence"
    elif state.get("retrieval_error"):
        reason = "retrieval_failed"
    else:
        reason = "weak_evidence" if results else "no_evidence"
    return {
        "evidence_score": score,
        "evidence_sufficient": sufficient,
        "evidence_reason": reason,
        "nodes_executed": _mark(state, "evaluate_evidence"),
    }


def refine_query(state: LegalResearchState) -> dict[str, Any]:
    query = state.get("original_query") or state.get("query", "")
    return {
        "query": query,
        "refined_query": query,
        "retrieval_strategy": "BROAD",
        "strategies": [*state.get("strategies", []), "BROAD"],
        "nodes_executed": _mark(state, "refine_query"),
    }


async def generate_answer(
    state: LegalResearchState,
    rag_service: LegalRAGService | None = None,
) -> dict[str, Any]:
    # The existing RAG service remains the single owner of context, LLM, and citations.
    try:
        results = [SearchResultItem(**item) for item in state.get("retrieved_results", [])]
        service = rag_service if rag_service is not None else LegalRAGService()
        result = await service.answer(state["query"], search_results=results)
    except Exception as exc:
        logger.warning("legal_graph_generation failed error_type=%s", type(exc).__name__)
        result = LegalAnswer(
            answer="I couldn't generate a verifiable answer from the retrieved legal sources.",
            confidence="low",
            limitations=["Answer generation failed; no generated answer was accepted."],
            grounding_valid=False,
            generation_mode="failed",
        )
        return {
            "answer": result.model_dump(),
            "citations": [],
            "limitations": result.limitations,
            "generation_error": "generation_failed",
            "generation_mode": result.generation_mode,
            "nodes_executed": _mark(state, "generate_answer"),
        }
    answer = result.model_dump()
    return {
        "answer": answer,
        "citations": [c.model_dump() for c in result.citations],
        "limitations": result.limitations,
        "generation_error": None,
        "generation_mode": result.generation_mode,
        "nodes_executed": _mark(state, "generate_answer"),
    }


def validate_answer(state: LegalResearchState) -> dict[str, Any]:
    answer = state.get("answer", {})
    citations = state.get("citations", [])
    retrieved = state.get("retrieved_results", [])
    available_sources = {
        (item.get("document_id"), item.get("chunk_id"))
        for item in retrieved
        if isinstance(item.get("document_id"), str)
        and item["document_id"].strip()
        and isinstance(item.get("chunk_id"), str)
        and item["chunk_id"].strip()
        and isinstance(item.get("content"), str)
        and item["content"].strip()
    }
    citation_ids: set[str] = set()
    citations_match_evidence = bool(citations)
    for citation in citations:
        citation_id = citation.get("citation_id", "")
        if (
            not isinstance(citation_id, str)
            or not re.fullmatch(r"SOURCE [1-9]\d*", citation_id)
            or citation_id in citation_ids
            or not citation.get("document_id")
            or not citation.get("chunk_id")
            or (citation.get("document_id"), citation.get("chunk_id"))
            not in available_sources
        ):
            citations_match_evidence = False
            break
        citation_ids.add(citation_id)
    valid = answer.get("grounding_valid", False) and citations_match_evidence
    if valid:
        reason = "validated"
    elif state.get("generation_error") or answer.get("generation_mode") == "failed":
        reason = "generation_failed"
    else:
        reason = "citation_validation_failed"
    validation = {
        "is_grounded": valid,
        "summary": reason,
    }
    return {
        "validation_result": validation,
        "evidence_reason": reason if not valid else state.get("evidence_reason", ""),
        "nodes_executed": _mark(state, "validate_answer"),
    }


def abstain(state: LegalResearchState) -> dict[str, Any]:
    reason = state.get("evidence_reason", "insufficient_evidence")
    if reason == "citation_validation_failed":
        message = "I couldn't verify the generated answer's citations, so I can't provide it reliably."
        limitation = "Generated answer rejected because its citations were missing or invalid."
    elif reason == "generation_failed":
        message = "I couldn't generate a verifiable answer from the retrieved legal sources."
        limitation = "Answer generation failed; no generated answer was accepted."
    elif reason == "retrieval_failed":
        message = "I couldn't retrieve legal sources at this time, so I can't answer reliably."
        limitation = "Legal source retrieval failed."
    elif reason == "no_evidence":
        message = "I couldn't find sufficient evidence in the indexed legal sources to answer this question reliably."
        limitation = "No relevant evidence was found in the indexed legal sources."
    elif reason == "weak_evidence":
        message = "I couldn't find sufficient evidence in the indexed legal sources to answer this question reliably."
        limitation = "The retrieved legal evidence was not strong enough to support an answer."
    else:
        message = "I couldn't find sufficient evidence in the indexed legal sources to answer this question reliably."
        limitation = "The available evidence was insufficient to support a reliable answer."
    answer = LegalAnswer(
        answer=message,
        confidence="low",
        limitations=[limitation],
        grounding_valid=False,
        abstained=True,
        generation_mode=state.get("generation_mode", "not_run"),
    ).model_dump()
    return {
        "answer": answer,
        "citations": [],
        "limitations": answer["limitations"],
        "nodes_executed": _mark(state, "abstain"),
    }


def finalize(state: LegalResearchState) -> dict[str, Any]:
    return {"nodes_executed": _mark(state, "finalize")}
