"""Process raw documents and emit JSONL with ALL chunks for indexing.

Each JSONL line contains:
  - metadata: document-level metadata
  - pages: list of cleaned page objects
  - chunk_count: total number of chunks
  - sample_chunks: first 3 chunks (for API preview)
  - all_chunks: ALL chunks (used by index_documents.py)

Usage
-----
  python scripts/process_documents.py
  python scripts/process_documents.py data/raw/sample --output data/processed/sample.jsonl
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "backend"))

from app.config import settings
from app.rag.legal_chunker import LegalChunker
from app.rag.loaders import (
    TXTLoader, HTMLLoader, JSONLoader, PDFLoader, LoaderError,
)
from app.rag.processors import process_pages
from app.schemas import DocumentChunk, DocumentMetadata, DocumentPage
from datetime import datetime, timezone
from uuid import uuid4


LOADERS = {
    "txt": TXTLoader, "text": TXTLoader,
    "html": HTMLLoader, "htm": HTMLLoader,
    "json": JSONLoader,
    "pdf": PDFLoader,
}


async def process_file(path: Path) -> dict | None:
    ext = path.suffix.lstrip(".").lower()
    loader_cls = LOADERS.get(ext)
    if loader_cls is None:
        return {"filename": path.name, "error": f"Unsupported format: .{ext}"}

    try:
        loaded = loader_cls().load(path.read_bytes(), path.name)
    except LoaderError as exc:
        return {"filename": path.name, "error": str(exc)}

    pages = process_pages(loaded.pages)
    doc_id = path.stem  # Use filename stem as stable document_id
    meta = loaded.metadata

    chunks = LegalChunker(settings.chunk_size, settings.chunk_overlap).chunk(pages)

    all_chunks = [
        DocumentChunk(
            chunk_id=f"{doc_id}:{c.index}",
            document_id=doc_id,
            content=c.text,
            text=c.text,
            page_start=c.page_start,
            page_end=c.page_end,
            section=c.section,
            subsection=None,
            chunk_index=c.index,
            document_type=ext.upper(),
            source=str(meta.get("source") or "raw_file"),
            source_url=meta.get("source_url"),
        )
        for c in chunks
    ]

    metadata = DocumentMetadata(
        document_id=doc_id,
        filename=path.name,
        content_type=loaded.content_type,
        document_type=str(meta.get("document_type") or ext.upper()),
        title=meta.get("title") or path.stem.replace("_", " ").title(),
        source=str(meta.get("source") or "raw_file"),
        source_url=meta.get("source_url"),
        jurisdiction=meta.get("jurisdiction"),
        court=meta.get("court"),
        case_name=meta.get("case_name"),
        case_number=meta.get("case_number"),
        citation=meta.get("citation"),
        date=str(meta["date"]) if meta.get("date") else None,
        act_name=meta.get("act_name"),
        section=meta.get("section"),
        language=meta.get("language"),
        size=path.stat().st_size,
        metadata={k: str(v) for k, v in meta.items() if k not in {"title", "source", "source_url"}},
        created_at=datetime.now(timezone.utc),
    )

    return {
        "metadata": metadata.model_dump(mode="json"),
        "pages": [DocumentPage(page_number=p.page_number, text=p.text).model_dump() for p in pages],
        "chunk_count": len(chunks),
        "sample_chunks": [c.model_dump() for c in all_chunks[:3]],
        "all_chunks": [c.model_dump() for c in all_chunks],  # ← ALL chunks for indexing
    }


async def main():
    parser = argparse.ArgumentParser(description="Process raw documents into JSONL for indexing")
    parser.add_argument("input", nargs="?", default="data/raw/sample")
    parser.add_argument("--output", default="data/processed/sample.jsonl")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    files = [p for p in input_path.rglob("*") if p.is_file()]
    print(f"Processing {len(files)} file(s) from {input_path} -> {output_path}\n")

    results = []
    errors = []

    with open(output_path, "w", encoding="utf-8") as out:
        for path in sorted(files):
            result = await process_file(path)
            if result is None:
                continue
            out.write(json.dumps(result, ensure_ascii=False, default=str) + "\n")
            if "error" in result:
                errors.append(result)
                print(f"  [ERR] {path.name}: {result['error']}")
            else:
                results.append(result)
                chunks = result["chunk_count"]
                print(f"  [OK]  {path.name}: {chunks} chunk(s)")

    print(f"\nDocument ingestion complete")
    print(f"  Documents processed : {len(results)}")
    print(f"  Errors              : {len(errors)}")
    print(f"  Total chunks        : {sum(r['chunk_count'] for r in results)}")
    print(f"  Output              : {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
