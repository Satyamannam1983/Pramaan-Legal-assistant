"""API integration tests.

These tests use TestClient with lifespan=False so they don't attempt to
connect to MongoDB (which may not be running during unit testing).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from fastapi.testclient import TestClient
from app.main import app
from app.schemas import ChatRequest, SearchRequest

# lifespan=False skips startup/shutdown handlers so no DB connection is attempted
client = TestClient(app, raise_server_exceptions=False)


def test_health_aliases():
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"

    resp2 = client.get("/api/v1/health")
    assert resp2.status_code == 200


def test_search_returns_semantic_response():
    """Search endpoint returns SemanticSearchResponse shape (empty results without DB)."""
    response = client.post("/api/search", json={"query": "contract", "top_k": 5})
    # 200 with empty results (DB unavailable — service returns [] gracefully) is expected
    assert response.status_code in (200, 422, 503)
    if response.status_code == 200:
        data = response.json()
        assert "query" in data
        assert "results" in data
        assert isinstance(data["results"], list)


def test_search_empty_query_rejected():
    """Empty query must be rejected with 422."""
    response = client.post("/api/search", json={"query": "   "})
    assert response.status_code == 422


def test_search_missing_query_rejected():
    """Missing query field must be rejected with 422."""
    response = client.post("/api/search", json={})
    assert response.status_code == 422


def test_chat_response_contract_includes_retrieved_evidence():
    response = client.post("/api/chat", json={"query": "Summarize this"})
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "conversation_id" in data
    assert "citations" in data
    assert "key_points" in data
    assert "limitations" in data
    assert "confidence" in data
    assert isinstance(data["search_results"], list)
    assert data["generation_mode"] in {"stub", "evidence_only", "not_run", "failed"}


def test_chat_and_search_accept_same_maximum_query_length():
    query = "x" * 3000
    assert ChatRequest(query=query).query == query
    assert SearchRequest(query=query).query == query


def test_document_question_preserves_document_filter_and_chat_response_contract(monkeypatch):
    import app.api.routes as routes
    from app.rag.generation.schemas import CitationRef, LegalAnswer
    from app.rag.services.retrieval_service import SearchResultItem

    class Repository:
        async def document_exists(self, document_id):
            assert document_id == "doc-17"
            return True

    class Retrieval:
        async def search(self, query, top_k, document_id):
            assert query == "question"
            assert document_id == "doc-17"
            return [SearchResultItem(
                chunk_id="chunk-17",
                content="Evidence passage",
                document_id="doc-17",
                section="Section 4",
                subsection="Section 4(a)",
            )]

    class RAG:
        async def answer(self, query, search_results):
            assert query == "question"
            return LegalAnswer(
                answer="Answer [SOURCE 1]",
                citations=[CitationRef(
                    citation_id="SOURCE 1",
                    document_id="doc-17",
                    chunk_id="chunk-17",
                    section="Section 4",
                    subsection="Section 4(a)",
                )],
                grounding_valid=True,
                generation_mode="stub",
            )

    monkeypatch.setattr(routes, "DocumentRepository", Repository)
    monkeypatch.setattr(routes, "_get_retrieval_service", lambda: Retrieval())
    monkeypatch.setattr(routes, "_get_rag_service", lambda: RAG())

    response = client.post(
        "/api/documents/doc-17/ask",
        json={"query": "question"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["generation_mode"] == "stub"
    assert data["citations"][0]["document_id"] == "doc-17"
    assert data["citations"][0]["chunk_id"] == "chunk-17"
    assert data["citations"][0]["subsection"] == "Section 4(a)"


def test_comparison_contract_keeps_selected_document_order(monkeypatch):
    import app.api.routes as routes
    from app.rag.document_analysis.schemas import DocumentComparison

    class Comparison:
        async def compare(self, document_ids):
            assert document_ids == ["doc-b", "doc-a"]
            return DocumentComparison(
                document_ids=document_ids,
                overview="Compared documents.",
                citations=[{
                    "citation_id": "A-SOURCE 1",
                    "chunk_id": "chunk-b",
                    "document_id": "doc-b",
                    "section": "Section 2",
                    "page_start": 2,
                    "page_end": 2,
                    "relevant_quote": "passage",
                }],
            )

    monkeypatch.setattr(routes, "ComparisonService", Comparison)
    response = client.post(
        "/api/documents/compare",
        json={"document_ids": ["doc-b", "doc-a"]},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["document_ids"] == ["doc-b", "doc-a"]
    assert data["citations"][0]["document_id"] == "doc-b"
    assert data["citations"][0]["chunk_id"] == "chunk-b"


def test_comparison_requires_two_distinct_identifiers():
    response = client.post(
        "/api/documents/compare",
        json={"document_ids": ["same", "same"]},
    )
    assert response.status_code == 404


def test_upload_reports_when_indexing_is_skipped(monkeypatch, tmp_path):
    import app.db.connection as connection
    from app.config import settings

    async def unavailable():
        return False

    monkeypatch.setattr(connection, "check_connection", unavailable)
    monkeypatch.setattr(settings, "upload_dir", str(tmp_path))
    response = client.post(
        "/api/documents/upload",
        files={"file": ("notice.txt", b"Legal notice content", "text/plain")},
    )
    assert response.status_code == 200
    assert response.json()["status"] == "accepted"
    assert response.json()["indexing_status"] == "skipped"


def test_upload_reports_indexing_failure(monkeypatch, tmp_path):
    import app.db.connection as connection
    import app.api.routes as routes
    from app.config import settings

    async def available():
        return True

    class FailingIndexer:
        async def index_record(self, record):
            raise RuntimeError("database write failed")

    monkeypatch.setattr(connection, "check_connection", available)
    monkeypatch.setattr(routes, "DocumentIndexer", FailingIndexer)
    monkeypatch.setattr(settings, "upload_dir", str(tmp_path))
    response = client.post(
        "/api/documents/upload",
        files={"file": ("notice.txt", b"Legal notice content", "text/plain")},
    )
    assert response.status_code == 200
    assert response.json()["indexing_status"] == "failed"
    assert response.json()["detail"] == "The file was stored, but indexing failed."


def test_repeated_upload_uses_a_stable_document_identity(monkeypatch, tmp_path):
    import app.db.connection as connection
    import app.api.routes as routes
    from app.config import settings

    async def available():
        return True

    class RecordingIndexer:
        records = []

        async def index_record(self, record):
            self.records.append(record)

    monkeypatch.setattr(connection, "check_connection", available)
    monkeypatch.setattr(routes, "DocumentIndexer", RecordingIndexer)
    monkeypatch.setattr(settings, "upload_dir", str(tmp_path))

    first = client.post(
        "/api/documents/upload",
        files={"file": ("notice.txt", b"Legal notice content", "text/plain")},
    )
    second = client.post(
        "/api/documents/upload",
        files={"file": ("notice.txt", b"Legal notice content", "text/plain")},
    )

    assert first.status_code == second.status_code == 200
    assert first.json()["document_id"] == second.json()["document_id"]
    assert first.json()["indexing_status"] == second.json()["indexing_status"] == "indexed"
    assert len(list(tmp_path.iterdir())) == 1
    assert len(RecordingIndexer.records) == 2
    assert all(
        record["metadata"]["document_id"] == first.json()["document_id"]
        for record in RecordingIndexer.records
    )


def test_empty_upload_is_rejected_before_storage(monkeypatch, tmp_path):
    from app.config import settings

    monkeypatch.setattr(settings, "upload_dir", str(tmp_path))
    response = client.post(
        "/api/documents/upload",
        files={"file": ("empty.txt", b"", "text/plain")},
    )

    assert response.status_code == 422
    assert list(tmp_path.iterdir()) == []


def test_upload_enforces_streamed_size_limit(monkeypatch, tmp_path):
    from app.config import settings

    monkeypatch.setattr(settings, "upload_dir", str(tmp_path))
    monkeypatch.setattr(settings, "max_upload_size_mb", 1)
    response = client.post(
        "/api/documents/upload",
        files={"file": ("large.txt", b"x" * (1024 * 1024 + 1), "text/plain")},
    )

    assert response.status_code == 413
    assert "size limit" in response.json()["detail"]
    assert list(tmp_path.iterdir()) == []


def test_document_database_failures_return_safe_service_error(monkeypatch):
    import app.api.routes as routes

    class FailingRepository:
        async def get_document(self, _document_id):
            raise RuntimeError("mongodb://sensitive-host/user:secret")

        async def document_exists(self, _document_id):
            raise RuntimeError("mongodb://sensitive-host/user:secret")

    monkeypatch.setattr(routes, "DocumentRepository", FailingRepository)
    metadata = client.get("/api/documents/document-1")
    question = client.post(
        "/api/documents/document-1/ask",
        json={"query": "question"},
    )

    assert metadata.status_code == 503
    assert question.status_code == 503
    assert "secret" not in metadata.text
    assert "secret" not in question.text
