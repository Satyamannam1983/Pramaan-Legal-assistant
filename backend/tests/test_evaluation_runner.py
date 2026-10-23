import asyncio
import json
from types import SimpleNamespace

from app.evaluation import runner
from app.rag.generation.schemas import CitationRef, LegalAnswer
from app.rag.services.retrieval_service import SearchResultItem


def _write_datasets(tmp_path):
    retrieval = tmp_path / "retrieval.json"
    answers = tmp_path / "answers.json"
    retrieval.write_text(
        json.dumps({
            "name": "retrieval-fixture",
            "version": "1",
            "provenance": "unit test",
            "examples": [{
                "id": "r1",
                "query": "known question",
                "relevant_chunk_ids": ["doc-a:0"],
                "relevant_document_ids": ["doc-a"],
            }],
        }),
        encoding="utf-8",
    )
    answers.write_text(
        json.dumps({
            "name": "answer-fixture",
            "version": "1",
            "provenance": "abstention expectation only",
            "examples": [{
                "id": "a1",
                "query": "known question",
                "expected_answer": {
                    "expected_abstention": False,
                    "expected_citation_document_ids": ["doc-a"],
                },
            }],
        }),
        encoding="utf-8",
    )
    return retrieval, answers


def test_citation_integrity_checks_id_format_and_selected_source_provenance():
    retrieved = [{"document_id": "doc-a", "chunk_id": "doc-a:0", "content": "Evidence"}]
    state = {
        "answer": {"answer": "Claim [SOURCE 1]"},
        "citations": [{"citation_id": "SOURCE 1", "document_id": "doc-a", "chunk_id": "doc-a:0"}],
    }
    result = runner._citation_integrity(state, retrieved)
    assert result["citation_identifier_validity"] == 1
    assert result["provenance_validity"] == 1
    assert result["semantic_claim_support_evaluated"] is False


def test_citation_integrity_flags_wrong_source_and_unresolved_reference():
    retrieved = [{"document_id": "doc-a", "chunk_id": "doc-a:0", "content": "Evidence"}]
    state = {
        "answer": {"answer": "Claim [SOURCE 2]"},
        "citations": [{"citation_id": "SOURCE 2", "document_id": "doc-b", "chunk_id": "doc-b:0"}],
    }
    result = runner._citation_integrity(state, retrieved)
    assert result["citation_identifier_validity"] == 1
    assert result["provenance_validity"] == 0
    assert result["citation_record_integrity"] == 0
    assert result["unresolved_inline_reference_count"] == 0


def test_citation_integrity_detects_missing_and_malformed_references():
    missing = runner._citation_integrity(
        {"answer": {"answer": "Claim [SOURCE 1]"}, "citations": []},
        [],
    )
    assert missing["citation_identifier_validity"] is None
    assert missing["unresolved_inline_reference_count"] == 1

    malformed = runner._citation_integrity(
        {"answer": {"answer": "Claim [SOURCE nope]"}, "citations": []},
        [],
    )
    assert malformed["malformed_inline_reference_count"] == 1

    invalid_identifier = runner._citation_integrity(
        {
            "answer": {"answer": "Claim [SOURCE 0]"},
            "citations": [{
                "citation_id": "SOURCE 0",
                "document_id": "doc-a",
                "chunk_id": "doc-a:0",
            }],
        },
        [{"document_id": "doc-a", "chunk_id": "doc-a:0", "content": "Evidence"}],
    )
    assert invalid_identifier["citation_identifier_validity"] == 0
    assert invalid_identifier["provenance_validity"] == 1
    assert invalid_identifier["citation_record_integrity"] == 0


def test_runner_marks_unavailable_database_not_run_without_scores(tmp_path):
    retrieval, answers = _write_datasets(tmp_path)
    report = asyncio.run(runner.run_evaluation(
        retrieval,
        answers,
        tmp_path / "reports",
        database_available=False,
    ))
    assert report["status"] == "NOT RUN"
    assert report["dataset_size"] == 2
    assert report["evaluations"]["retrieval"]["status"] == "NOT RUN"
    assert report["evaluations"]["answer_generation"]["status"] == "NOT RUN"
    assert not report["evaluations"]["retrieval"]["metrics"]
    persisted = json.loads((tmp_path / "reports" / "evaluation_report.json").read_text(encoding="utf-8"))
    assert persisted["evaluations"]["retrieval"]["metrics"] == {}
    markdown = (tmp_path / "reports" / "evaluation_report.md").read_text(encoding="utf-8")
    assert "NOT RUN" in markdown
    assert "recall_at_1" not in markdown


class _Cursor:
    def __init__(self, rows):
        self.rows = rows

    async def to_list(self, length):
        return self.rows[:length]


class _Collection:
    def __init__(self, documents):
        self.documents = documents

    def find(self, query, projection):
        ids = query.get("chunk_id", query.get("document_id", {})).get("$in", [])
        field = "chunk_id" if "chunk_id" in query else "document_id"
        return _Cursor([row for row in self.documents if row.get(field) in ids])


class _Database(dict):
    pass


class _FakeRetrieval:
    async def search(self, query, top_k):
        return [SearchResultItem(
            chunk_id="doc-a:0",
            content="Relevant evidence passage.",
            document_id="doc-a",
            section="Section 1",
            rrf_score=0.8,
            rerank_score=0.8,
            final_rank=1,
        )]


class _FakeRag:
    async def answer(self, query, search_results):
        return LegalAnswer(
            answer="A supported response. [SOURCE 1]",
            citations=[CitationRef(
                citation_id="SOURCE 1",
                document_id=search_results[0].document_id,
                chunk_id=search_results[0].chunk_id,
                section=search_results[0].section,
            )],
            grounding_valid=True,
            generation_mode="stub",
        )


class _FailingRetrieval:
    async def search(self, query, top_k):
        raise RuntimeError("sensitive retrieval backend detail")


def test_runner_measures_rankings_and_graph_observations(monkeypatch, tmp_path):
    retrieval, answers = _write_datasets(tmp_path)
    database = _Database({
        runner.CHUNKS_COLLECTION: _Collection([{
            "chunk_id": "doc-a:0",
            "document_id": "doc-a",
        }]),
        runner.DOCUMENTS_COLLECTION: _Collection([{"document_id": "doc-a"}]),
    })

    async def connected():
        return True

    async def initialize():
        return None

    monkeypatch.setattr(runner, "check_connection", connected)
    monkeypatch.setattr(runner, "init_db", initialize)
    monkeypatch.setattr(runner, "get_db", lambda: database)
    monkeypatch.setattr(
        runner,
        "get_llm_provider",
        lambda: SimpleNamespace(provider_name="stub", model_name="unit-test"),
    )
    report = asyncio.run(runner.run_evaluation(
        retrieval,
        answers,
        tmp_path / "reports",
        retrieval_service=_FakeRetrieval(),
        rag_service=_FakeRag(),
    ))

    retrieval_stage = report["evaluations"]["retrieval"]
    assert retrieval_stage["status"] == "COMPLETED"
    assert retrieval_stage["metrics"]["recall_at_1"] == 1
    assert retrieval_stage["metrics"]["count"] == 1
    answer_stage = report["evaluations"]["answer_generation"]
    assert answer_stage["status"] == "PARTIAL"
    assert answer_stage["quality_evaluation"]["status"] == "BLOCKED"
    assert answer_stage["operational_metrics"]["citation_identifier_validity"] == 1
    assert answer_stage["operational_metrics"]["citation_provenance_validity"] == 1
    assert answer_stage["operational_metrics"]["citation_record_integrity"] == 1
    assert answer_stage["operational_metrics"]["abstention_accuracy"] == 1
    assert answer_stage["operational_metrics"]["expected_citation_document_coverage"] == 1
    assert answer_stage["diagnostics"]["latency_ms"]["count"] == 1
    assert report["configuration"]["answer_execution_path"] == "LangGraph question-answering workflow"


def test_runner_records_retrieval_failures_and_graph_abstention(monkeypatch, tmp_path):
    retrieval, answers = _write_datasets(tmp_path)
    database = _Database({
        runner.CHUNKS_COLLECTION: _Collection([{
            "chunk_id": "doc-a:0",
            "document_id": "doc-a",
        }]),
        runner.DOCUMENTS_COLLECTION: _Collection([{"document_id": "doc-a"}]),
    })

    async def connected():
        return True

    async def initialize():
        return None

    monkeypatch.setattr(runner, "check_connection", connected)
    monkeypatch.setattr(runner, "init_db", initialize)
    monkeypatch.setattr(runner, "get_db", lambda: database)
    monkeypatch.setattr(
        runner,
        "get_llm_provider",
        lambda: SimpleNamespace(provider_name="stub", model_name="unit-test"),
    )
    report = asyncio.run(runner.run_evaluation(
        retrieval,
        answers,
        tmp_path / "reports",
        retrieval_service=_FailingRetrieval(),
        rag_service=_FakeRag(),
    ))

    assert report["evaluations"]["retrieval"]["status"] == "BLOCKED"
    assert report["evaluations"]["retrieval"]["diagnostics"]["failed_requests"] == 1
    answer_stage = report["evaluations"]["answer_generation"]
    assert answer_stage["diagnostics"]["retrieval_failures"] == 1
    assert answer_stage["diagnostics"]["empty_retrievals"] == 1
    assert answer_stage["operational_metrics"]["abstention_accuracy"] == 0
    assert answer_stage["cases"][0]["abstained"] is True
    assert "sensitive retrieval backend detail" not in (tmp_path / "reports" / "evaluation_report.md").read_text(encoding="utf-8")


def test_runner_skips_answer_labels_without_indexed_source_documents(monkeypatch, tmp_path):
    retrieval, answers = _write_datasets(tmp_path)
    database = _Database({
        runner.CHUNKS_COLLECTION: _Collection([{
            "chunk_id": "doc-a:0",
            "document_id": "doc-a",
        }]),
        runner.DOCUMENTS_COLLECTION: _Collection([]),
    })

    async def connected():
        return True

    async def initialize():
        return None

    monkeypatch.setattr(runner, "check_connection", connected)
    monkeypatch.setattr(runner, "init_db", initialize)
    monkeypatch.setattr(runner, "get_db", lambda: database)
    monkeypatch.setattr(
        runner,
        "get_llm_provider",
        lambda: SimpleNamespace(provider_name="stub", model_name="unit-test"),
    )
    report = asyncio.run(runner.run_evaluation(
        retrieval,
        answers,
        tmp_path / "reports",
        retrieval_service=_FakeRetrieval(),
        rag_service=_FakeRag(),
    ))

    metrics = report["evaluations"]["answer_generation"]["operational_metrics"]
    assert metrics["abstention_cases"] == 0
    assert metrics["skipped_abstention_cases"] == 1
    assert metrics["abstention_accuracy"] is None
    assert metrics["unresolved_expected_source_cases"] == 1
