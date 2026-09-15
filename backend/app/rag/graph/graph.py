"""Compiled bounded legal research graph."""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from app.rag.graph.edges import route_after_evaluation, route_after_validation
from app.rag.graph.nodes import (
    abstain,
    analyze_query,
    evaluate_evidence,
    finalize,
    generate_answer,
    refine_query,
    retrieve_evidence,
    select_retrieval_strategy,
    validate_answer,
)
from app.rag.graph.state import LegalResearchState


def build_legal_research_graph():
    workflow = StateGraph(LegalResearchState)
    workflow.add_node("analyze_query", analyze_query)
    workflow.add_node("select_retrieval_strategy", select_retrieval_strategy)
    workflow.add_node("retrieve_evidence", retrieve_evidence)
    workflow.add_node("evaluate_evidence", evaluate_evidence)
    workflow.add_node("refine_query", refine_query)
    workflow.add_node("generate_answer", generate_answer)
    workflow.add_node("validate_answer", validate_answer)
    workflow.add_node("abstain", abstain)
    workflow.add_node("finalize", finalize)
    workflow.add_edge(START, "analyze_query")
    workflow.add_edge("analyze_query", "select_retrieval_strategy")
    workflow.add_edge("select_retrieval_strategy", "retrieve_evidence")
    workflow.add_edge("retrieve_evidence", "evaluate_evidence")
    workflow.add_conditional_edges("evaluate_evidence", route_after_evaluation)
    workflow.add_edge("refine_query", "retrieve_evidence")
    workflow.add_edge("generate_answer", "validate_answer")
    workflow.add_conditional_edges("validate_answer", route_after_validation)
    workflow.add_edge("abstain", END)
    workflow.add_edge("finalize", END)
    return workflow.compile()
