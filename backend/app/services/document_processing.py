from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any
from uuid import uuid4
from app.config import settings
from app.schemas import DocumentMetadata, DocumentPage, DocumentChunk, DocumentProcessResponse
from app.rag.loaders import TXTLoader, HTMLLoader, JSONLoader, PDFLoader, LoaderError
from app.rag.processors import process_pages
from app.rag.legal_chunker import LegalChunker


class DocumentProcessingService:
    """Parse, clean, and chunk source documents for either indexing path."""

    @staticmethod
    def _metadata_text(metadata: dict, key: str) -> str | None:
        value = metadata.get(key)
        return str(value) if value is not None else None

    async def process(
        self,
        filename: str,
        content: bytes,
        *,
        include_all_chunks: bool = False,
        document_id: str | None = None,
        source: str | None = None,
        metadata_overrides: dict[str, Any] | None = None,
    ) -> DocumentProcessResponse:
        if not content or not content.strip():
            raise LoaderError("The document is empty.")
        suffix = Path(filename).suffix.lower()
        if not suffix:
            raise LoaderError("The document filename must include a supported extension.")
        ext = suffix[1:]
        loader = {"txt": TXTLoader, "text": TXTLoader, "html": HTMLLoader, "htm": HTMLLoader, "json": JSONLoader, "pdf": PDFLoader}.get(ext)
        if loader is None:
            raise LoaderError(f"Unsupported document format: .{ext}")
        loaded = loader().load(content, filename)
        pages = process_pages(loaded.pages)
        if not any(page.text for page in pages):
            raise LoaderError("The document contains no readable text.")
        doc_id = document_id or str(uuid4())
        if not doc_id.strip() or ":" in doc_id:
            raise LoaderError("The source document identifier is invalid.")
        chunks = LegalChunker(settings.chunk_size, settings.chunk_overlap).chunk(pages)
        if not chunks:
            raise LoaderError("The document contains no readable text.")
        source_metadata = {**loaded.metadata, **(metadata_overrides or {})}
        metadata_text = self._metadata_text
        document_source = metadata_text(source_metadata, "source") or source or "user_upload"
        title = metadata_text(source_metadata, "title") or Path(filename).stem.replace("_", " ").title()
        metadata = DocumentMetadata(
            document_id=doc_id, filename=filename, content_type=loaded.content_type,
            document_type=metadata_text(source_metadata, "document_type") or ext.upper(),
            title=title,
            source=document_source,
            source_url=metadata_text(source_metadata, "source_url"),
            jurisdiction=metadata_text(source_metadata, "jurisdiction"),
            court=metadata_text(source_metadata, "court"),
            case_name=metadata_text(source_metadata, "case_name"),
            case_number=metadata_text(source_metadata, "case_number"),
            citation=metadata_text(source_metadata, "citation"),
            date=metadata_text(source_metadata, "date"),
            act_name=metadata_text(source_metadata, "act_name"),
            section=metadata_text(source_metadata, "section"),
            language=metadata_text(source_metadata, "language"),
            size=len(content),
            metadata={
                key: (
                    json.dumps(value, ensure_ascii=False, sort_keys=True)
                    if isinstance(value, (dict, list))
                    else str(value)
                )
                for key, value in source_metadata.items()
                if key not in {"title", "source", "source_url"}
            },
            created_at=datetime.now(timezone.utc),
        )
        serialized_chunks = [
            DocumentChunk(
                chunk_id=f"{doc_id}:{c.index}", document_id=doc_id, content=c.text,
                text=c.text, page_start=c.page_start, page_end=c.page_end,
                section=c.section, subsection=None, chunk_index=c.index,
                document_type=metadata.document_type,
                court=metadata.court,
                date=metadata.date,
                source=document_source,
                source_url=metadata.source_url,
            ) for c in chunks
        ]
        return DocumentProcessResponse(
            metadata=metadata,
            pages=[DocumentPage(page_number=p.page_number, text=p.text) for p in pages],
            chunk_count=len(chunks),
            sample_chunks=serialized_chunks[:3],
            all_chunks=serialized_chunks if include_all_chunks else None,
        )
