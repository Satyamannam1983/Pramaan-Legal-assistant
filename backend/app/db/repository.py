"""MongoDB data access layer — all DB queries live here, never in routes.

Similarity search strategy
--------------------------
For local / single-node MongoDB:
  Embeddings are loaded into Python and cosine similarity is computed with NumPy.
  This is efficient for corpora up to ~100k chunks.

For MongoDB Atlas (production):
  Set ATLAS_VECTOR_SEARCH=true in .env.
  The repository will use the $vectorSearch aggregation pipeline instead.
  You must create an Atlas Vector Search index named "embedding_index" on the
  document_chunks collection with path="embedding" and dimensions matching
  EMBEDDING_DIMENSION.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.db.connection import get_db
from app.db.models import CHUNKS_COLLECTION, DOCUMENTS_COLLECTION

logger = logging.getLogger(__name__)

# Set ATLAS_VECTOR_SEARCH=true to use $vectorSearch (MongoDB Atlas only)
_USE_ATLAS = os.getenv("ATLAS_VECTOR_SEARCH", "false").lower() == "true"


# ---------------------------------------------------------------------------
# Transfer objects (plain dataclasses — no MongoDB dependency upstream)
# ---------------------------------------------------------------------------

@dataclass
class DocumentData:
    document_id: str
    title: str | None = None
    document_type: str | None = None
    source: str | None = None
    source_url: str | None = None
    jurisdiction: str | None = None
    court: str | None = None
    case_name: str | None = None
    case_number: str | None = None
    citation: str | None = None
    date: str | None = None
    act_name: str | None = None
    language: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class ChunkData:
    chunk_id: str
    document_id: str
    content: str
    embedding: list[float] | None
    chunk_index: int
    section: str | None = None
    subsection: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    document_type: str | None = None
    court: str | None = None
    date: str | None = None
    source: str | None = None
    source_url: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class RetrievedChunk:
    chunk_id: str
    content: str
    document_id: str
    title: str | None
    section: str | None
    page_start: int | None
    page_end: int | None
    document_type: str | None
    court: str | None
    date: str | None
    source: str | None
    source_url: str | None
    similarity_score: float


# ---------------------------------------------------------------------------
# Repository
# ---------------------------------------------------------------------------

class DocumentRepository:
    """All persistence operations for documents and chunks (MongoDB)."""

    def __init__(self, db: AsyncIOMotorDatabase | None = None) -> None:
        self._db = db or get_db()

    # ------------------------------------------------------------------
    # Document operations
    # ------------------------------------------------------------------

    async def upsert_document(self, data: DocumentData) -> None:
        """Insert or update a document record by document_id."""
        doc = {
            "document_id": data.document_id,
            "title": data.title,
            "document_type": data.document_type,
            "source": data.source,
            "source_url": data.source_url,
            "jurisdiction": data.jurisdiction,
            "court": data.court,
            "case_name": data.case_name,
            "case_number": data.case_number,
            "citation": data.citation,
            "date": data.date,
            "act_name": data.act_name,
            "language": data.language,
            "metadata": data.metadata or {},
        }
        await self._db[DOCUMENTS_COLLECTION].update_one(
            {"document_id": data.document_id},
            {
                "$set": doc,
                "$setOnInsert": {"created_at": datetime.now(timezone.utc)},
            },
            upsert=True,
        )

    async def document_exists(self, document_id: str) -> bool:
        count = await self._db[DOCUMENTS_COLLECTION].count_documents(
            {"document_id": document_id}, limit=1
        )
        return count > 0

    async def list_documents(self) -> list[dict[str, Any]]:
        cursor = self._db[DOCUMENTS_COLLECTION].find({}, {"_id": 0}).sort("created_at", -1)
        return await cursor.to_list(length=1000)

    async def get_document(self, document_id: str) -> dict[str, Any] | None:
        return await self._db[DOCUMENTS_COLLECTION].find_one({"document_id": document_id}, {"_id": 0})

    async def get_chunks(self, document_id: str, limit: int = 100) -> list[ChunkData]:
        cursor = self._db[CHUNKS_COLLECTION].find(
            {"document_id": document_id}, {"_id": 0, "embedding": 0}
        ).sort("chunk_index", 1).limit(limit)
        rows = await cursor.to_list(length=limit)
        return [
            ChunkData(
                chunk_id=r["chunk_id"], document_id=r["document_id"],
                content=r.get("content", ""), embedding=None,
                chunk_index=r.get("chunk_index", 0), section=r.get("section"),
                subsection=r.get("subsection"), page_start=r.get("page_start"),
                page_end=r.get("page_end"), document_type=r.get("document_type"),
                court=r.get("court"), date=r.get("date"), source=r.get("source"),
                source_url=r.get("source_url"), metadata=r.get("metadata"),
            )
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Chunk operations
    # ------------------------------------------------------------------

    async def upsert_chunks(self, chunks: list[ChunkData]) -> None:
        """Bulk upsert chunks. Idempotent by chunk_id."""
        if not chunks:
            return

        from pymongo import UpdateOne  # noqa: PLC0415

        ops = []
        for c in chunks:
            doc = {
                "chunk_id": c.chunk_id,
                "document_id": c.document_id,
                "content": c.content,
                "embedding": c.embedding,
                "chunk_index": c.chunk_index,
                "section": c.section,
                "subsection": c.subsection,
                "page_start": c.page_start,
                "page_end": c.page_end,
                "document_type": c.document_type,
                "court": c.court,
                "date": c.date,
                "source": c.source,
                "source_url": c.source_url,
                "metadata": c.metadata or {},
            }
            ops.append(
                UpdateOne(
                    {"chunk_id": c.chunk_id},
                    {
                        "$set": doc,
                        "$setOnInsert": {"created_at": datetime.now(timezone.utc)},
                    },
                    upsert=True,
                )
            )

        if ops:
            await self._db[CHUNKS_COLLECTION].bulk_write(ops, ordered=False)

    async def delete_document_chunks(self, document_id: str) -> None:
        await self._db[CHUNKS_COLLECTION].delete_many({"document_id": document_id})

    # ------------------------------------------------------------------
    # Similarity search
    # ------------------------------------------------------------------

    async def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        document_id: str | None = None,
    ) -> list[RetrievedChunk]:
        """Cosine similarity search.

        Local MongoDB: loads all chunk embeddings and computes similarity in NumPy.
        MongoDB Atlas: uses $vectorSearch aggregation pipeline.
        """
        if _USE_ATLAS:
            return await self._atlas_vector_search(query_embedding, top_k, document_id)
        return await self._python_cosine_search(query_embedding, top_k, document_id)

    async def _python_cosine_search(
        self, query_embedding: list[float], top_k: int, document_id: str | None = None
    ) -> list[RetrievedChunk]:
        """NumPy cosine similarity — works on any MongoDB instance."""
        # Fetch all chunks that have embeddings
        query: dict[str, Any] = {"embedding": {"$exists": True, "$ne": None, "$not": {"$size": 0}}}
        if document_id:
            query["document_id"] = document_id
        cursor = self._db[CHUNKS_COLLECTION].find(
            query,
            {
                "_id": 0,
                "chunk_id": 1,
                "content": 1,
                "document_id": 1,
                "section": 1,
                "page_start": 1,
                "page_end": 1,
                "document_type": 1,
                "court": 1,
                "date": 1,
                "source": 1,
                "source_url": 1,
                "embedding": 1,
            },
        )
        chunks = await cursor.to_list(length=100_000)

        if not chunks:
            return []

        # Fetch document titles
        doc_ids = list({c["document_id"] for c in chunks})
        doc_cursor = self._db[DOCUMENTS_COLLECTION].find(
            {"document_id": {"$in": doc_ids}},
            {"_id": 0, "document_id": 1, "title": 1},
        )
        docs = await doc_cursor.to_list(length=None)
        title_map: dict[str, str | None] = {d["document_id"]: d.get("title") for d in docs}

        # Vectorised cosine similarity
        q = np.array(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return []

        # Build matrix [n_chunks × dim]
        embeddings = np.array([c["embedding"] for c in chunks], dtype=np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)  # avoid divide-by-zero
        similarities = (embeddings / norms) @ (q / q_norm)

        # Top-K indices (descending similarity)
        if top_k >= len(similarities):
            top_indices = np.argsort(similarities)[::-1]
        else:
            top_indices = np.argpartition(similarities, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]

        return [
            RetrievedChunk(
                chunk_id=chunks[i]["chunk_id"],
                content=chunks[i]["content"],
                document_id=chunks[i]["document_id"],
                title=title_map.get(chunks[i]["document_id"]),
                section=chunks[i].get("section"),
                page_start=chunks[i].get("page_start"),
                page_end=chunks[i].get("page_end"),
                document_type=chunks[i].get("document_type"),
                court=chunks[i].get("court"),
                date=chunks[i].get("date"),
                source=chunks[i].get("source"),
                source_url=chunks[i].get("source_url"),
                similarity_score=float(similarities[i]),
            )
            for i in top_indices
        ]

    async def _atlas_vector_search(
        self, query_embedding: list[float], top_k: int, document_id: str | None = None
    ) -> list[RetrievedChunk]:
        """MongoDB Atlas $vectorSearch — production path.

        Requires an Atlas Vector Search index named 'embedding_index'
        with path='embedding' and correct numDimensions.
        """
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "embedding_index",
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": top_k * 10,
                    "limit": top_k,
                    **({"filter": {"document_id": {"$eq": document_id}}} if document_id else {}),
                }
            },
            {
                "$lookup": {
                    "from": DOCUMENTS_COLLECTION,
                    "localField": "document_id",
                    "foreignField": "document_id",
                    "as": "doc",
                }
            },
            {"$unwind": {"path": "$doc", "preserveNullAndEmptyArrays": True}},
            {
                "$project": {
                    "_id": 0,
                    "chunk_id": 1,
                    "content": 1,
                    "document_id": 1,
                    "title": "$doc.title",
                    "section": 1,
                    "page_start": 1,
                    "page_end": 1,
                    "document_type": 1,
                    "court": 1,
                    "date": 1,
                    "source": 1,
                    "source_url": 1,
                    "score": {"$meta": "vectorSearchScore"},
                }
            },
        ]
        cursor = self._db[CHUNKS_COLLECTION].aggregate(pipeline)
        rows = await cursor.to_list(length=top_k)
        return [
            RetrievedChunk(
                chunk_id=r["chunk_id"],
                content=r["content"],
                document_id=r["document_id"],
                title=r.get("title"),
                section=r.get("section"),
                page_start=r.get("page_start"),
                page_end=r.get("page_end"),
                document_type=r.get("document_type"),
                court=r.get("court"),
                date=r.get("date"),
                source=r.get("source"),
                source_url=r.get("source_url"),
                similarity_score=float(r.get("score", 0.0)),
            )
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Counters
    # ------------------------------------------------------------------

    async def count_documents(self) -> int:
        return await self._db[DOCUMENTS_COLLECTION].count_documents({})

    async def count_chunks(self, document_id: str | None = None) -> int:
        return await self._db[CHUNKS_COLLECTION].count_documents(
            {"document_id": document_id} if document_id else {}
        )
