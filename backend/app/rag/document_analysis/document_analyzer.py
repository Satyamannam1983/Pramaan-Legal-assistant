from __future__ import annotations

from app.config import settings
from app.db.repository import DocumentRepository
from app.rag.document_analysis.metadata_extractor import extract_metadata
from app.rag.document_analysis.schemas import DocumentAnalysis


class DocumentAnalyzer:
    def __init__(self, repository: DocumentRepository | None = None) -> None:
        self.repository = repository or DocumentRepository()

    async def analyze(self, document_id: str) -> DocumentAnalysis:
        document = await self.repository.get_document(document_id)
        chunks = await self.repository.get_chunks(document_id, limit=settings.document_analysis_top_k)
        if not document:
            raise ValueError("Document not found.")
        if not chunks:
            raise ValueError("Document has no indexed chunks.")
        text = "\n\n".join(c.content for c in chunks)
        metadata = extract_metadata(document_id, text, document)
        citations = [
            {
                "citation_id": f"SOURCE {i}",
                "chunk_id": c.chunk_id,
                "document_id": c.document_id,
                "section": c.section,
                "page_start": c.page_start,
                "page_end": c.page_end,
                "relevant_quote": c.content[:300],
            }
            for i, c in enumerate(chunks, 1)
        ]
        return DocumentAnalysis(
            document_id=document_id,
            metadata=metadata,
            summary=text[:1200],
            facts=[c.content[:400] for c in chunks if c.section and "fact" in c.section.lower()][:3],
            issues=[c.section for c in chunks if c.section and "issue" in c.section.lower()],
            reasoning=[c.content[:400] for c in chunks if c.section and "reason" in c.section.lower()][:3],
            decision=next((c.content[:600] for c in chunks if c.section and any(x in c.section.lower() for x in ("decision", "conclusion", "outcome"))), None),
            important_sections=sorted({c.section for c in chunks if c.section}),
            referenced_laws=metadata.acts_referenced + metadata.sections_referenced + metadata.articles_referenced,
            citations=citations,
            limitations=["Analysis is extractive and limited to indexed document chunks."],
        )
