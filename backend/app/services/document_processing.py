from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import uuid4
from app.config import settings
from app.schemas import DocumentMetadata, DocumentPage, DocumentChunk, DocumentProcessResponse
from app.rag.loaders import TXTLoader, HTMLLoader, JSONLoader, PDFLoader, LoaderError
from app.rag.processors import process_pages
from app.rag.legal_chunker import LegalChunker


@dataclass
class ProcessingResult:
    filename: str
    status: str = "queued"
    chunks: int = 0


class DocumentProcessingService:
    """Pipeline seam for parsing, cleaning, chunking and indexing documents."""

    async def process(
        self,
        filename: str,
        content: bytes,
        *,
        include_all_chunks: bool = False,
    ) -> DocumentProcessResponse:
        ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else "txt"
        loader = {"txt": TXTLoader, "text": TXTLoader, "html": HTMLLoader, "htm": HTMLLoader, "json": JSONLoader, "pdf": PDFLoader}.get(ext)
        if loader is None:
            raise LoaderError(f"Unsupported document format: .{ext}")
        loaded = loader().load(content, filename)
        pages = process_pages(loaded.pages)
        doc_id = str(uuid4())
        chunks = LegalChunker(settings.chunk_size, settings.chunk_overlap).chunk(pages)
        source_metadata = loaded.metadata
        metadata = DocumentMetadata(
            document_id=doc_id, filename=filename, content_type=loaded.content_type,
            document_type=str(source_metadata.get("document_type") or ext.upper()),
            title=source_metadata.get("title"),
            source=str(source_metadata.get("source") or "user_upload"),
            source_url=source_metadata.get("source_url"),
            jurisdiction=source_metadata.get("jurisdiction"),
            court=source_metadata.get("court"),
            case_name=source_metadata.get("case_name"),
            case_number=source_metadata.get("case_number"),
            citation=source_metadata.get("citation"),
            date=str(source_metadata["date"]) if source_metadata.get("date") else None,
            act_name=source_metadata.get("act_name"),
            section=source_metadata.get("section"),
            language=source_metadata.get("language"),
            size=len(content),
            metadata={key: str(value) for key, value in source_metadata.items() if key not in {"title", "source", "source_url"}},
            created_at=datetime.now(timezone.utc),
        )
        serialized_chunks = [
            DocumentChunk(
                chunk_id=f"{doc_id}:{c.index}", document_id=doc_id, content=c.text,
                text=c.text, page_start=c.page_start, page_end=c.page_end,
                section=c.section, subsection=None, chunk_index=c.index,
                document_type=ext.upper(), source="user_upload",
            ) for c in chunks
        ]
        return DocumentProcessResponse(
            metadata=metadata,
            pages=[DocumentPage(page_number=p.page_number, text=p.text) for p in pages],
            chunk_count=len(chunks),
            sample_chunks=serialized_chunks[:3],
            all_chunks=serialized_chunks if include_all_chunks else None,
        )
