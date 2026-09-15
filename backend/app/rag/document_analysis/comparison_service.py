from __future__ import annotations

from app.config import settings
from app.db.repository import DocumentRepository
from app.rag.document_analysis.schemas import DocumentComparison


class ComparisonService:
    def __init__(self, repository: DocumentRepository | None = None) -> None:
        self.repository = repository or DocumentRepository()

    async def compare(self, document_ids: list[str]) -> DocumentComparison:
        if len(document_ids) != 2 or document_ids[0] == document_ids[1]:
            raise ValueError("Exactly two distinct document IDs are required.")
        docs = [await self.repository.get_document(doc_id) for doc_id in document_ids]
        if any(doc is None for doc in docs):
            raise ValueError("One or more documents were not found.")
        chunks = [await self.repository.get_chunks(doc_id, limit=settings.document_comparison_top_k) for doc_id in document_ids]
        citations = []
        for label, rows in zip(("A", "B"), chunks):
            citations.extend({
                "citation_id": f"{label}-SOURCE {i}",
                "chunk_id": c.chunk_id,
                "document_id": c.document_id,
                "section": c.section,
                "page_start": c.page_start,
                "page_end": c.page_end,
                "relevant_quote": c.content[:300],
            } for i, c in enumerate(rows, 1))
        type_a = docs[0].get("document_type")
        type_b = docs[1].get("document_type")
        return DocumentComparison(
            document_ids=document_ids,
            overview=f"Compared {docs[0].get('title') or document_ids[0]} and {docs[1].get('title') or document_ids[1]}.",
            similarities=[f"Document types: {type_a or 'unknown'} and {type_b or 'unknown'}."],
            differences=[],
            legal_provisions=[],
            reasoning_comparison=[],
            outcome_comparison=None,
            citations=citations,
            limitations=["Comparison claims require human review; unsupported differences are omitted."],
        )
