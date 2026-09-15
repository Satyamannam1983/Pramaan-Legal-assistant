"""Tests for the retrieval pipeline with a real MongoDB instance.

These tests require a running MongoDB server.
Set TEST_DATABASE_URL to enable them:

  $env:TEST_DATABASE_URL = "mongodb://localhost:27017/pramaan_test"
  pytest backend/tests/test_retrieval.py -v

Without TEST_DATABASE_URL, all tests in this file are skipped.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from uuid import uuid4

import pytest
import pytest_asyncio

sys.path.insert(0, str(Path(__file__).parents[1]))

TEST_DB_URL = os.getenv("TEST_DATABASE_URL", "")
REQUIRES_DB = pytest.mark.skipif(
    not TEST_DB_URL,
    reason="TEST_DATABASE_URL not set — skipping MongoDB retrieval tests",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture(scope="module")
async def test_db():
    """Return a Motor database pointing at the test DB."""
    if not TEST_DB_URL:
        yield None
        return

    from motor.motor_asyncio import AsyncIOMotorClient  # noqa: PLC0415

    client = AsyncIOMotorClient(TEST_DB_URL, serverSelectionTimeoutMS=5000)
    db_name = TEST_DB_URL.rsplit("/", 1)[-1].split("?")[0] or "pramaan_test"
    db = client[db_name]

    # Create indexes
    await db.documents.create_index("document_id", unique=True)
    await db.document_chunks.create_index("chunk_id", unique=True)
    await db.document_chunks.create_index("document_id")

    yield db

    # Cleanup test collections
    await db.documents.drop()
    await db.document_chunks.drop()
    client.close()


@pytest.fixture
def stub_embedding_service():
    from app.rag.embeddings.stub import StubEmbeddingService  # noqa: PLC0415
    return StubEmbeddingService(dimension=64)


@pytest.fixture
def unique_doc_id():
    return f"test_doc_{uuid4().hex[:8]}"


# ---------------------------------------------------------------------------
# Repository tests
# ---------------------------------------------------------------------------

@REQUIRES_DB
@pytest.mark.asyncio
async def test_document_insert(test_db, unique_doc_id):
    """A document can be inserted and found."""
    from app.db.repository import DocumentRepository, DocumentData  # noqa: PLC0415

    repo = DocumentRepository(test_db)
    await repo.upsert_document(
        DocumentData(document_id=unique_doc_id, title="Test Statute", document_type="STATUTE")
    )
    assert await repo.document_exists(unique_doc_id)


@REQUIRES_DB
@pytest.mark.asyncio
async def test_chunk_insert_with_embedding(test_db, unique_doc_id, stub_embedding_service):
    """Chunks with embeddings can be inserted and counted."""
    from app.db.repository import DocumentRepository, DocumentData, ChunkData  # noqa: PLC0415

    repo = DocumentRepository(test_db)
    await repo.upsert_document(DocumentData(document_id=unique_doc_id, title="Test"))

    embedding = stub_embedding_service.embed_text("valid contract requirements")
    await repo.upsert_chunks([
        ChunkData(
            chunk_id=f"{unique_doc_id}:0",
            document_id=unique_doc_id,
            content="A valid contract requires offer, acceptance, and consideration.",
            embedding=embedding,
            chunk_index=0,
            section="Section 1",
            page_start=1,
            page_end=1,
            document_type="STATUTE",
            source="test",
        )
    ])

    count = await repo.count_chunks()
    assert count >= 1


@REQUIRES_DB
@pytest.mark.asyncio
async def test_duplicate_document_idempotent(test_db, unique_doc_id):
    """Upserting the same document twice does not create duplicates."""
    from app.db.repository import DocumentRepository, DocumentData  # noqa: PLC0415

    repo = DocumentRepository(test_db)
    await repo.upsert_document(DocumentData(document_id=unique_doc_id, title="Original"))
    await repo.upsert_document(DocumentData(document_id=unique_doc_id, title="Updated"))

    count = await test_db.documents.count_documents({"document_id": unique_doc_id})
    assert count == 1


@REQUIRES_DB
@pytest.mark.asyncio
async def test_similarity_search_returns_correct_chunk(test_db, stub_embedding_service):
    """Chunk with identical embedding to query should score ~1.0."""
    from app.db.repository import DocumentRepository, DocumentData, ChunkData  # noqa: PLC0415

    doc_id = f"search_test_{uuid4().hex[:8]}"
    query_text = "What are the requirements for a valid contract?"

    repo = DocumentRepository(test_db)
    await repo.upsert_document(DocumentData(document_id=doc_id, title="Contract Statute"))

    query_vec = stub_embedding_service.embed_text(query_text)
    await repo.upsert_chunks([
        ChunkData(
            chunk_id=f"{doc_id}:0",
            document_id=doc_id,
            content="A valid contract requires offer, acceptance, consideration, and free consent.",
            embedding=query_vec,   # identical vector → cosine similarity ≈ 1.0
            chunk_index=0,
            section="Section 1. Essential Elements",
            page_start=1,
            page_end=1,
            document_type="STATUTE",
            source="sample_statute",
        )
    ])

    results = await repo.similarity_search(query_vec, top_k=5)
    assert results, "Expected at least one result"

    our = [r for r in results if r.document_id == doc_id]
    assert our, "Inserted chunk not found"
    assert our[0].similarity_score > 0.99
    assert our[0].section == "Section 1. Essential Elements"
    assert our[0].page_start == 1


@REQUIRES_DB
@pytest.mark.asyncio
async def test_retrieval_result_full_provenance(test_db, stub_embedding_service):
    """Every retrieved result must carry complete provenance metadata."""
    from app.db.repository import DocumentRepository, DocumentData, ChunkData  # noqa: PLC0415

    doc_id = f"prov_test_{uuid4().hex[:8]}"
    repo = DocumentRepository(test_db)

    await repo.upsert_document(DocumentData(
        document_id=doc_id,
        title="Provenance Test Statute",
        document_type="STATUTE",
        source="test_source",
        source_url="https://example.com/statute",
        court="Supreme Court",
        date="2024-01-01",
    ))
    vec = stub_embedding_service.embed_text("provenance test query")
    await repo.upsert_chunks([
        ChunkData(
            chunk_id=f"{doc_id}:0",
            document_id=doc_id,
            content="Legal text with full provenance.",
            embedding=vec,
            chunk_index=0,
            section="Part I",
            page_start=3,
            page_end=4,
            document_type="STATUTE",
            source="test_source",
            source_url="https://example.com/statute",
            court="Supreme Court",
            date="2024-01-01",
        )
    ])

    results = await repo.similarity_search(vec, top_k=5)
    our = [r for r in results if r.document_id == doc_id]
    assert our

    r = our[0]
    assert r.title == "Provenance Test Statute"
    assert r.section == "Part I"
    assert r.page_start == 3
    assert r.page_end == 4
    assert r.source == "test_source"
    assert r.source_url == "https://example.com/statute"
    assert r.similarity_score > 0.0


@REQUIRES_DB
@pytest.mark.asyncio
async def test_vector_retriever_end_to_end(test_db, stub_embedding_service):
    """End-to-end: insert chunk → query via VectorRetriever → correct result."""
    import app.db.connection as conn_mod  # noqa: PLC0415
    from app.db.repository import DocumentRepository, DocumentData, ChunkData  # noqa: PLC0415
    from app.rag.retrieval.vector_retriever import VectorRetriever  # noqa: PLC0415

    doc_id = f"e2e_test_{uuid4().hex[:8]}"
    query = "anticipatory breach of contract"

    repo = DocumentRepository(test_db)
    await repo.upsert_document(DocumentData(document_id=doc_id, title="Contract Act"))
    embedding = stub_embedding_service.embed_text(query)
    await repo.upsert_chunks([
        ChunkData(
            chunk_id=f"{doc_id}:0",
            document_id=doc_id,
            content="Anticipatory breach occurs when a party repudiates the contract.",
            embedding=embedding,
            chunk_index=0,
            section="Section 7",
            page_start=2,
            page_end=2,
            document_type="STATUTE",
            source="sample",
        )
    ])

    # Point the retriever at the test DB
    original_get_db = conn_mod.get_db
    conn_mod._db = test_db

    try:
        retriever = VectorRetriever(embedding_service=stub_embedding_service, top_k=5)
        results = await retriever.retrieve(query, top_k=5)
    finally:
        conn_mod._db = None

    assert results
    our = [r for r in results if r.document_id == doc_id]
    assert our, "End-to-end retrieval did not return the indexed chunk"
    assert our[0].similarity_score > 0.99


@REQUIRES_DB
@pytest.mark.asyncio
async def test_retriever_empty_query_raises(stub_embedding_service):
    """Empty query must raise ValueError."""
    from app.rag.retrieval.vector_retriever import VectorRetriever  # noqa: PLC0415

    retriever = VectorRetriever(embedding_service=stub_embedding_service, top_k=5)
    with pytest.raises((ValueError, RuntimeError)):
        await retriever.retrieve("")
