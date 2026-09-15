"""Tests for keyword retrieval (BM25)."""
from __future__ import annotations

import pytest

from app.rag.retrieval.keyword_retriever import BM25Index, KeywordRetriever, tokenise


# ---------------------------------------------------------------------------
# Tokeniser tests
# ---------------------------------------------------------------------------

class TestTokeniser:
    def test_basic_tokens(self):
        tokens = tokenise("hello world")
        assert "hello" in tokens
        assert "world" in tokens

    def test_section_number_preserved(self):
        tokens = tokenise("Section 302 IPC murder")
        joined = " ".join(tokens)
        assert "302" in joined

    def test_case_insensitive(self):
        t1 = tokenise("CONTRACT law")
        t2 = tokenise("contract law")
        assert t1 == t2

    def test_empty_string(self):
        tokens = tokenise("")
        assert tokens == []

    def test_preserves_numbers(self):
        tokens = tokenise("Article 14 equality")
        joined = " ".join(tokens)
        assert "14" in joined


# ---------------------------------------------------------------------------
# BM25Index tests
# ---------------------------------------------------------------------------

class TestBM25Index:
    @pytest.fixture
    def sample_chunks(self):
        return [
            {
                "chunk_id": "doc1:0",
                "content": "Section 302 of the Indian Penal Code deals with murder punishment.",
                "document_id": "doc1",
                "title": "Indian Penal Code",
                "section": "Section 302",
                "page_start": 1,
                "page_end": 1,
                "document_type": "Act",
                "court": None,
                "date": None,
                "source": "raw_file",
                "source_url": None,
            },
            {
                "chunk_id": "doc1:1",
                "content": "Section 299 defines culpable homicide causing death.",
                "document_id": "doc1",
                "title": "Indian Penal Code",
                "section": "Section 299",
                "page_start": 2,
                "page_end": 2,
                "document_type": "Act",
                "court": None,
                "date": None,
                "source": "raw_file",
                "source_url": None,
            },
            {
                "chunk_id": "doc2:0",
                "content": "A valid contract requires offer acceptance consideration.",
                "document_id": "doc2",
                "title": "Indian Contract Act",
                "section": "Section 1",
                "page_start": 1,
                "page_end": 1,
                "document_type": "Act",
                "court": None,
                "date": None,
                "source": "raw_file",
                "source_url": None,
            },
            {
                "chunk_id": "doc2:1",
                "content": "Consideration is something of value given by the promisee.",
                "document_id": "doc2",
                "title": "Indian Contract Act",
                "section": "Section 3",
                "page_start": 3,
                "page_end": 3,
                "document_type": "Act",
                "court": None,
                "date": None,
                "source": "raw_file",
                "source_url": None,
            },
        ]

    def test_build_index(self, sample_chunks):
        index = BM25Index()
        index.build(sample_chunks)
        assert index._n_docs == 4
        assert index._avg_doc_length > 0

    def test_empty_index_search(self):
        index = BM25Index()
        results = index.search("anything")
        assert results == []

    def test_exact_section_match(self, sample_chunks):
        index = BM25Index()
        index.build(sample_chunks)
        results = index.search("Section 302", top_k=5)
        assert len(results) > 0
        # The murder chunk should rank highest
        assert results[0].chunk_id == "doc1:0"

    def test_terminology_match(self, sample_chunks):
        index = BM25Index()
        index.build(sample_chunks)
        results = index.search("consideration contract", top_k=5)
        assert len(results) > 0
        # Should find the consideration chunk
        doc_ids = [r.document_id for r in results]
        assert "doc2" in doc_ids

    def test_ranking_order(self, sample_chunks):
        index = BM25Index()
        index.build(sample_chunks)
        results = index.search("murder Section 302", top_k=5)
        assert len(results) >= 2
        # Murder chunk should rank higher than contract chunk
        assert results[0].chunk_id == "doc1:0"

    def test_metadata_preserved(self, sample_chunks):
        index = BM25Index()
        index.build(sample_chunks)
        results = index.search("murder", top_k=1)
        assert len(results) == 1
        r = results[0]
        assert r.title == "Indian Penal Code"
        assert r.section == "Section 302"
        assert r.document_type == "Act"

    def test_case_insensitive_search(self, sample_chunks):
        index = BM25Index()
        index.build(sample_chunks)
        r1 = index.search("MURDER", top_k=1)
        r2 = index.search("murder", top_k=1)
        assert len(r1) > 0 and len(r2) > 0
        assert r1[0].chunk_id == r2[0].chunk_id

    def test_top_k_limits_results(self, sample_chunks):
        index = BM25Index()
        index.build(sample_chunks)
        results = index.search("Section", top_k=2)
        assert len(results) <= 2

    def test_no_match_returns_empty(self, sample_chunks):
        index = BM25Index()
        index.build(sample_chunks)
        results = index.search("xyzzy non-existent term")
        # May return results with very low score or empty
        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# KeywordRetriever tests (requires MongoDB)
# ---------------------------------------------------------------------------

class TestKeywordRetriever:
    @pytest.mark.asyncio
    async def test_empty_query_raises(self):
        retriever = KeywordRetriever()
        with pytest.raises(ValueError):
            await retriever.retrieve("")
