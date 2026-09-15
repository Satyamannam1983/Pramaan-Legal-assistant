from __future__ import annotations

import pytest

from app.rag.graph.graph import build_legal_research_graph
from app.rag.graph.nodes import analyze_query, evaluate_evidence


def test_query_analyzer_detects_section_lookup():
    state = analyze_query({"query": "What does Section 302 provide?"})
    assert state["query_type"] == "SECTION_LOOKUP"


def test_evidence_evaluator_is_deterministic():
    state = evaluate_evidence(
        {
            "retrieved_results": [
                {
                    "content": "A supported legal passage.",
                    "document_id": "doc-1",
                    "rerank_score": 0.8,
                }
            ]
        }
    )
    assert state["evidence_sufficient"] is True
    assert state["evidence_reason"] == "strong_evidence"


@pytest.mark.asyncio
async def test_graph_abstains_without_database():
    result = await build_legal_research_graph().ainvoke(
        {"query": "What is the current RBI interest rate?"}
    )
    assert result["answer"]["confidence"] == "low"
    assert "abstain" in result["nodes_executed"]
    assert result["retrieval_attempt"] <= 2
