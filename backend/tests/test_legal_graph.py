from __future__ import annotations

import pytest

from app.rag.generation.schemas import CitationRef, LegalAnswer
from app.rag.graph.graph import build_legal_research_graph
from app.rag.graph.nodes import analyze_query, evaluate_evidence, validate_answer
from app.rag.services.retrieval_service import SearchResultItem


def test_query_analyzer_detects_section_lookup():
    state = analyze_query({"query": "What does Section 302 provide?"})
    assert state["query_type"] == "SECTION_LOOKUP"

def test_comparison_classification_precedes_section_lookup():
    state = analyze_query({"query": "Compare Section 302 and Section 304"})
    assert state["query_type"] == "COMPARISON"


def test_query_analyzer_rejects_missing_or_empty_query():
    with pytest.raises(ValueError, match="non-empty query"):
        analyze_query({})
    with pytest.raises(ValueError, match="non-empty query"):
        analyze_query({"query": "  "})


def test_evidence_evaluator_is_deterministic():
    state = evaluate_evidence(
        {
            "retrieved_results": [
                {
                    "content": "A supported legal passage.",
                    "document_id": "doc-1",
                    "chunk_id": "doc-1:0",
                    "rerank_score": 0.8,
                }
            ]
        }
    )
    assert state["evidence_sufficient"] is True
    assert state["evidence_reason"] == "sufficient_evidence"


def test_low_confidence_without_citations_does_not_pass_validation():
    state = validate_answer({"answer": {"confidence": "low", "citations": []}})
    assert state["validation_result"]["is_grounded"] is False


@pytest.mark.asyncio
async def test_graph_abstains_without_database():
    result = await build_legal_research_graph().ainvoke(
        {"query": "What is the current RBI interest rate?"}
    )
    assert result["answer"]["confidence"] == "low"
    assert "abstain" in result["nodes_executed"]
    assert result["retrieval_attempt"] <= 2


@pytest.mark.asyncio
async def test_graph_reuses_injected_retrieval_service_for_retry():
    class EmptyRetrievalService:
        calls = 0

        async def search(self, query, top_k=None, document_id=None):
            self.calls += 1
            return []

    service = EmptyRetrievalService()
    await build_legal_research_graph(retrieval_service=service).ainvoke(
        {"query": "A question with no indexed evidence"}
    )
    assert service.calls == 2


@pytest.mark.asyncio
async def test_retrieve_node_exposes_same_results_for_answer_and_evidence():
    class RetrievalService:
        async def search(self, query, top_k=None, document_id=None):
            return [
                SearchResultItem(
                    chunk_id="chunk-1",
                    content="Evidence text",
                    document_id="doc-1",
                    rrf_score=0.05,
                )
            ]

    from app.rag.graph.nodes import retrieve_evidence

    result = await retrieve_evidence(
        {"query": "Question"},
        retrieval_service=RetrievalService(),
    )
    assert result["retrieved_results"] == result["search_results"]
    assert result["search_results"][0]["chunk_id"] == "chunk-1"


def _retrieval_result(score: float = 0.8) -> SearchResultItem:
    return SearchResultItem(
        chunk_id="doc-1:0",
        document_id="doc-1",
        content="Section 302 applies to the described offence.",
        section="Section 302",
        source_url="https://example.test/act",
        rerank_score=score,
        rrf_score=score,
    )


class _ValidRagService:
    calls = 0

    async def answer(self, query, search_results=None):
        self.calls += 1
        result = search_results[0]
        return LegalAnswer(
            answer="Section 302 applies as described [SOURCE 1].",
            citations=[CitationRef(
                citation_id="SOURCE 1",
                chunk_id=result.chunk_id,
                document_id=result.document_id,
                section=result.section,
                source_url=result.source_url,
            )],
            confidence="medium",
            grounding_valid=True,
            generation_mode="stub",
        )


@pytest.mark.asyncio
async def test_sufficient_evidence_generates_validated_answer_and_expected_flow():
    class Retrieval:
        async def search(self, query, top_k=None, document_id=None):
            return [_retrieval_result()]

    answer_service = _ValidRagService()
    result = await build_legal_research_graph(
        retrieval_service=Retrieval(),
        rag_service=answer_service,
    ).ainvoke({"query": "What does Section 302 say?"})

    assert result["answer"]["grounding_valid"] is True
    assert result["generation_mode"] == "stub"
    assert result["retrieval_attempt"] == 1
    assert result["nodes_executed"] == [
        "analyze_query",
        "select_retrieval_strategy",
        "retrieve_evidence",
        "evaluate_evidence",
        "generate_answer",
        "validate_answer",
        "finalize",
    ]


@pytest.mark.asyncio
async def test_insufficient_evidence_abstains_after_bounded_retry(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "max_retrieval_attempts", 2)

    class Retrieval:
        calls = 0

        async def search(self, query, top_k=None, document_id=None):
            self.calls += 1
            return []

    retrieval = Retrieval()
    result = await build_legal_research_graph(retrieval_service=retrieval).ainvoke(
        {"query": "Unsupported question"}
    )

    assert retrieval.calls == result["retrieval_attempt"] == 2
    assert result["answer"]["grounding_valid"] is False
    assert result["answer"]["limitations"] == [
        "No relevant evidence was found in the indexed legal sources."
    ]
    assert result["nodes_executed"].count("refine_query") == 1
    assert result["nodes_executed"][-1] == "abstain"


def test_retry_router_has_a_hard_two_attempt_ceiling(monkeypatch):
    from app.config import settings
    from app.rag.graph.edges import route_after_evaluation

    monkeypatch.setattr(settings, "max_retrieval_attempts", 100)
    assert route_after_evaluation({
        "evidence_sufficient": False,
        "retrieval_attempt": 1,
    }) == "refine_query"
    assert route_after_evaluation({
        "evidence_sufficient": False,
        "retrieval_attempt": 2,
    }) == "abstain"


@pytest.mark.asyncio
async def test_broader_retry_can_succeed(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "evidence_min_score", 0.5)

    class Retrieval:
        queries = []

        async def search(self, query, top_k=None, document_id=None):
            self.queries.append(query)
            return [_retrieval_result(0.2 if len(self.queries) == 1 else 0.8)]

    retrieval = Retrieval()
    answer_service = _ValidRagService()
    result = await build_legal_research_graph(
        retrieval_service=retrieval,
        rag_service=answer_service,
    ).ainvoke({"query": "Question"})

    assert len(retrieval.queries) == 2
    assert retrieval.queries[0] == "Question"
    assert "requirements remedies" in retrieval.queries[1]
    assert result["retrieval_attempt"] == 2
    assert answer_service.calls == 1
    assert result["answer"]["grounding_valid"] is True
    assert "abstain" not in result["nodes_executed"]


@pytest.mark.asyncio
async def test_retry_failure_abstains_with_retrieval_failure_status(monkeypatch, caplog):
    from app.config import settings

    monkeypatch.setattr(settings, "max_retrieval_attempts", 2)

    class Retrieval:
        calls = 0

        async def search(self, query, top_k=None, document_id=None):
            self.calls += 1
            raise RuntimeError("private database endpoint")

    retrieval = Retrieval()
    result = await build_legal_research_graph(retrieval_service=retrieval).ainvoke(
        {"query": "Question"}
    )

    assert retrieval.calls == 2
    assert result["evidence_reason"] == "retrieval_failed"
    assert "retrieve" in result["answer"]["answer"].lower()
    assert result["nodes_executed"][-1] == "abstain"
    assert "private database endpoint" not in caplog.text


@pytest.mark.asyncio
async def test_generation_failure_abstains_without_raising():
    class Retrieval:
        async def search(self, query, top_k=None, document_id=None):
            return [_retrieval_result()]

    class BrokenRag:
        async def answer(self, query, search_results=None):
            raise RuntimeError("secret provider response")

    result = await build_legal_research_graph(
        retrieval_service=Retrieval(),
        rag_service=BrokenRag(),
    ).ainvoke({"query": "Question"})

    assert result["generation_mode"] == "failed"
    assert result["validation_result"]["is_grounded"] is False
    assert result["nodes_executed"][-1] == "abstain"
    assert "couldn't generate" in result["answer"]["answer"].lower()


@pytest.mark.asyncio
async def test_citation_validation_failure_abstains():
    class Retrieval:
        async def search(self, query, top_k=None, document_id=None):
            return [_retrieval_result()]

    class InvalidCitationRag:
        async def answer(self, query, search_results=None):
            return LegalAnswer(
                answer="Unsupported [SOURCE 99].",
                confidence="low",
                grounding_valid=False,
                generation_mode="openai-compatible",
            )

    result = await build_legal_research_graph(
        retrieval_service=Retrieval(),
        rag_service=InvalidCitationRag(),
    ).ainvoke({"query": "Question"})

    assert result["evidence_reason"] == "citation_validation_failed"
    assert result["citations"] == []
    assert result["nodes_executed"][-1] == "abstain"
    assert "couldn't verify" in result["answer"]["answer"].lower()


def test_graph_rejects_citation_with_mismatched_retrieved_chunk():
    from app.rag.graph.nodes import validate_answer

    state = {
        "answer": {"grounding_valid": True},
        "citations": [{
            "citation_id": "SOURCE 1",
            "document_id": "doc-1",
            "chunk_id": "wrong-chunk",
        }],
        "retrieved_results": [{
            "document_id": "doc-1",
            "chunk_id": "doc-1:0",
            "content": "Retrieved evidence.",
        }],
    }

    result = validate_answer(state)

    assert result["validation_result"]["is_grounded"] is False
    assert result["evidence_reason"] == "citation_validation_failed"
