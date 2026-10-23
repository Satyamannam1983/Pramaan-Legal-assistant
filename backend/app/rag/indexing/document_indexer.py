"""Document indexer — processes JSONL files into MongoDB.

Pipeline
--------
data/processed/*.jsonl
    ↓  load chunks from JSONL
    ↓  generate embeddings (batched)
    ↓  upsert document record (MongoDB)
    ↓  upsert chunks + embeddings (MongoDB)
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from app.config import settings
from app.db.repository import ChunkData, DocumentData, DocumentRepository
from app.rag.embeddings.base import BaseEmbeddingService
from app.rag.embeddings.factory import get_embedding_service

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class IndexingStats:
    documents_found: int = 0
    documents_indexed: int = 0
    chunks_found: int = 0
    chunks_indexed: int = 0
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logger.warning("Skipping malformed JSONL line %d in %s: %s", lineno, path, exc)
    return records


def _extract_document_data(record: dict) -> DocumentData | None:
    metadata = record.get("metadata")
    if not isinstance(metadata, dict) or not metadata:
        logger.warning("JSONL record missing 'metadata' — skipping")
        return None
    doc_id = metadata.get("document_id")
    if not isinstance(doc_id, str) or not doc_id.strip() or ":" in doc_id:
        logger.warning("JSONL record missing document_id — skipping")
        return None
    return DocumentData(
        document_id=doc_id,
        title=metadata.get("title"),
        document_type=metadata.get("document_type"),
        source=metadata.get("source"),
        source_url=metadata.get("source_url"),
        jurisdiction=metadata.get("jurisdiction"),
        court=metadata.get("court"),
        case_name=metadata.get("case_name"),
        case_number=metadata.get("case_number"),
        citation=metadata.get("citation"),
        date=metadata.get("date"),
        act_name=metadata.get("act_name"),
        language=metadata.get("language"),
        metadata=metadata.get("metadata"),
    )


def _extract_chunk_data(
    record: dict,
    doc_id: str,
    doc_meta: dict,
    embeddings: list[list[float]],
) -> list[ChunkData]:
    all_chunks = record.get("all_chunks") or record.get("sample_chunks") or []
    if len(embeddings) != len(all_chunks):
        raise ValueError("Embedding result count does not match the document chunk count.")
    chunks: list[ChunkData] = []
    chunk_ids: set[str] = set()
    for i, (chunk, embedding) in enumerate(zip(all_chunks, embeddings)):
        if not isinstance(chunk, dict):
            raise ValueError(f"Chunk {i} must be an object.")
        if chunk.get("document_id") and chunk["document_id"] != doc_id:
            raise ValueError("A chunk references a different source document.")
        content = chunk.get("content") or chunk.get("text") or ""
        if not content.strip():
            raise ValueError(f"Chunk {i} has no readable text.")
        chunk_id = chunk.get("chunk_id") or f"{doc_id}:{i}"
        if not chunk_id.startswith(f"{doc_id}:") or chunk_id in chunk_ids:
            raise ValueError(f"Chunk {i} has an invalid or duplicate chunk ID.")
        chunk_ids.add(chunk_id)
        chunks.append(
            ChunkData(
                chunk_id=chunk_id,
                document_id=doc_id,
                content=content,
                embedding=embedding,
                chunk_index=chunk.get("chunk_index", i),
                section=chunk.get("section"),
                subsection=chunk.get("subsection"),
                page_start=chunk.get("page_start"),
                page_end=chunk.get("page_end"),
                document_type=chunk.get("document_type") or doc_meta.get("document_type"),
                court=chunk.get("court") or doc_meta.get("court"),
                date=chunk.get("date") or doc_meta.get("date"),
                source=chunk.get("source") or doc_meta.get("source"),
                source_url=chunk.get("source_url") or doc_meta.get("source_url"),
                metadata=None,
            )
        )
    return chunks


# ---------------------------------------------------------------------------
# DocumentIndexer
# ---------------------------------------------------------------------------

class DocumentIndexer:
    """Loads JSONL files, embeds chunks, and stores them in MongoDB."""

    def __init__(
        self,
        embedding_service: BaseEmbeddingService | None = None,
        processed_dir: str | Path | None = None,
    ) -> None:
        self._embedding_service = embedding_service or get_embedding_service()
        self._processed_dir = Path(processed_dir or settings.processed_dir)

    async def index_record(self, record: dict) -> int:
        """Embed and persist one processed-document record using the batch contract."""
        doc_data = _extract_document_data(record)
        if doc_data is None:
            raise ValueError("Processed document is missing required metadata.")

        doc_meta = record.get("metadata", {})
        chunks = record.get("all_chunks") or record.get("sample_chunks") or []
        if not chunks:
            raise ValueError("Processed document has no chunks to index.")
        texts = [chunk.get("content") or chunk.get("text") or "" for chunk in chunks]
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Every processed chunk must contain readable text.")
        embeddings = self._embedding_service.embed_documents(texts) if texts else []
        if len(embeddings) != len(texts):
            raise ValueError("Embedding result count does not match the document chunk count.")
        chunk_data = _extract_chunk_data(record, doc_data.document_id, doc_meta, embeddings)
        await self._persist_document(DocumentRepository(), doc_data, chunk_data)
        return len(chunk_data)

    @staticmethod
    async def _persist_document(
        repository: DocumentRepository,
        document: DocumentData,
        chunks: list[ChunkData],
    ) -> None:
        await repository.upsert_document(document)
        if chunks:
            await repository.upsert_chunks(chunks)
        await repository.delete_stale_chunks(
            document.document_id,
            [chunk.chunk_id for chunk in chunks],
        )

    async def index_all(
        self,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> IndexingStats:
        """Index all JSONL files in processed_dir."""
        jsonl_files = sorted(self._processed_dir.glob("*.jsonl"))
        stats = IndexingStats()

        if not jsonl_files:
            logger.warning("No JSONL files found in %s", self._processed_dir)
            return stats

        # Load all records
        all_records: list[dict] = []
        for path in jsonl_files:
            records = _load_jsonl(path)
            all_records.extend(r for r in records if "error" not in r)

        stats.documents_found = len(all_records)

        # Collect all chunk texts for batched embedding
        record_chunk_texts: list[list[str]] = []
        for record in all_records:
            chunks = record.get("all_chunks") or record.get("sample_chunks") or []
            texts = [c.get("content") or c.get("text") or "" for c in chunks]
            record_chunk_texts.append(texts)
            stats.chunks_found += len(texts)

        flat_texts = [t for texts in record_chunk_texts for t in texts]

        logger.info(
            "Generating embeddings for %d chunks across %d documents…",
            len(flat_texts),
            stats.documents_found,
        )

        flat_embeddings = (
            self._embedding_service.embed_documents(flat_texts) if flat_texts else []
        )
        if len(flat_embeddings) != len(flat_texts):
            message = "Embedding result count does not match the total chunk count."
            logger.error(message)
            stats.errors.append(message)
            return stats

        # Persist to MongoDB
        repo = DocumentRepository()
        idx = 0

        for record_idx, (record, texts) in enumerate(
            zip(all_records, record_chunk_texts)
        ):
            n = len(texts)
            embeddings = flat_embeddings[idx : idx + n]
            idx += n

            doc_data = _extract_document_data(record)
            if doc_data is None:
                stats.errors.append(f"Record {record_idx}: missing metadata")
                continue

            try:
                if not texts:
                    raise ValueError("Processed document has no chunks to index.")
                doc_meta = record.get("metadata", {})
                chunk_data = _extract_chunk_data(
                    record, doc_data.document_id, doc_meta, embeddings
                )
                await self._persist_document(repo, doc_data, chunk_data)
                stats.documents_indexed += 1
                stats.chunks_indexed += len(chunk_data)
            except Exception as exc:
                logger.error(
                    "Failed to index document %s: %s", doc_data.document_id, exc
                )
                stats.errors.append(f"{doc_data.document_id}: {exc}")

            if progress_callback:
                progress_callback(record_idx + 1, stats.documents_found)

        return stats
