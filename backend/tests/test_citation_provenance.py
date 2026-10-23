from __future__ import annotations

import pytest

from app.db.models import CHUNKS_COLLECTION, DOCUMENTS_COLLECTION
from app.db.repository import DocumentRepository


class _Cursor:
    def __init__(self, rows):
        self._rows = rows

    async def to_list(self, length=None):
        return self._rows


class _Collection:
    def __init__(self, rows):
        self._rows = rows

    def find(self, *_args, **_kwargs):
        return _Cursor(self._rows)


class _Database:
    def __init__(self, collections):
        self._collections = collections

    def __getitem__(self, collection_name):
        return self._collections[collection_name]


@pytest.mark.asyncio
async def test_vector_retrieval_excludes_orphan_chunks_and_preserves_subsection():
    database = _Database({
        CHUNKS_COLLECTION: _Collection([
            {
                "chunk_id": "stored-chunk",
                "document_id": "stored-document",
                "content": "Stored legal passage.",
                "embedding": [1.0, 0.0],
                "section": "Part I",
                "subsection": "Section 3(a)",
                "page_start": 4,
                "page_end": 5,
            },
            {
                "chunk_id": "orphan-chunk",
                "document_id": "deleted-document",
                "content": "Orphaned passage.",
                "embedding": [0.9, 0.1],
            },
        ]),
        DOCUMENTS_COLLECTION: _Collection([
            {"document_id": "stored-document", "title": "Stored document"},
        ]),
    })
    repository = DocumentRepository(database)

    results = await repository._python_cosine_search([1.0, 0.0], top_k=5)

    assert len(results) == 1
    assert results[0].document_id == "stored-document"
    assert results[0].chunk_id == "stored-chunk"
    assert results[0].subsection == "Section 3(a)"
    assert results[0].page_start == 4
