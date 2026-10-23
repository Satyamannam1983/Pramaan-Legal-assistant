"""Citation service — maps LLM citation IDs back to real chunk metadata.

The LLM selects citation IDs ([SOURCE 1], [SOURCE 2], etc.) but the server
owns the actual source metadata. This service prevents citation fabrication.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from app.rag.generation.evidence_selector import EvidenceBundle, EvidenceItem

logger = logging.getLogger(__name__)


@dataclass
class Citation:
    """A validated citation with full metadata."""

    citation_id: str  # e.g. "SOURCE 1"
    chunk_id: str
    document_id: str
    title: str | None = None
    section: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    subsection: str | None = None
    document_type: str | None = None
    court: str | None = None
    date: str | None = None
    source: str | None = None
    source_url: str | None = None
    relevant_quote: str | None = None


# Pattern to match [SOURCE N] in LLM output
_VALID_SOURCE_PATTERN = re.compile(r"\[SOURCE\s+([1-9]\d*)\]")
_SOURCE_CANDIDATE_PATTERN = re.compile(
    r"\[\s*SOURCE[^\]\r\n]*(?:\]|\Z)",
    re.IGNORECASE,
)


class CitationService:
    """Maps LLM citation references to actual retrieved evidence.

    Security model:
    - Only citation IDs that were supplied to the LLM are valid.
    - The LLM cannot invent new citation IDs.
    - All citation metadata comes from the original retrieval results.
    """

    def __init__(self) -> None:
        self._valid_ids: dict[int, EvidenceItem] = {}
        self._invalid_ids: set[int] = set()

    def load_evidence(self, bundle: EvidenceBundle) -> None:
        """Load only uniquely identified, usable retrieved chunks as citation sources."""
        self._valid_ids = {}
        self._invalid_ids = set()
        for item in bundle.items:
            match = re.fullmatch(r"SOURCE ([1-9]\d*)", item.citation_id)
            usable = (
                isinstance(item.chunk_id, str)
                and bool(item.chunk_id.strip())
                and isinstance(item.document_id, str)
                and bool(item.document_id.strip())
                and isinstance(item.content, str)
                and bool(item.content.strip())
            )
            if not match or not usable:
                if match:
                    self._invalid_ids.add(int(match.group(1)))
                continue
            source_id = int(match.group(1))
            if source_id in self._valid_ids:
                self._valid_ids.pop(source_id)
                self._invalid_ids.add(source_id)
            elif source_id not in self._invalid_ids:
                self._valid_ids[source_id] = item

    @staticmethod
    def parse_references(text: str) -> tuple[list[int], list[str]]:
        """Return canonical source IDs and malformed source-reference tokens."""
        valid_ids: list[int] = []
        malformed: list[str] = []
        for match in _SOURCE_CANDIDATE_PATTERN.finditer(text):
            token = match.group(0)
            valid_match = _VALID_SOURCE_PATTERN.fullmatch(token)
            if valid_match:
                valid_ids.append(int(valid_match.group(1)))
            else:
                malformed.append(token.strip())
        return valid_ids, malformed

    def extract_citations(self, llm_output: str) -> list[Citation]:
        """Extract and validate citation IDs from LLM output.

        Returns only citations that map to actual retrieved evidence.
        Logs warnings for any invalid citation references.
        """
        citation_ids, _ = self.parse_references(llm_output)
        seen: set[int] = set()
        citations: list[Citation] = []

        for cid_str in citation_ids:
            cid = int(cid_str)
            if cid in seen:
                continue
            seen.add(cid)

            if cid not in self._valid_ids:
                logger.warning(
                    "LLM cited [SOURCE %d] but only %d sources were provided. "
                    "Citation ignored.",
                    cid, len(self._valid_ids),
                )
                continue

            item = self._valid_ids[cid]
            citations.append(
                Citation(
                    citation_id=f"SOURCE {cid}",
                    chunk_id=item.chunk_id,
                    document_id=item.document_id,
                    title=item.title,
                    section=item.section,
                    page_start=item.page_start,
                    page_end=item.page_end,
                    subsection=item.subsection,
                    document_type=item.document_type,
                    court=item.court,
                    date=item.date,
                    source=item.source,
                    source_url=item.source_url,
                )
            )

        return citations

    def validate_citations(self, citations: list[Citation]) -> bool:
        """Verify returned citation IDs and source identities against loaded evidence."""
        seen: set[str] = set()
        seen_sources: set[tuple[str, str]] = set()
        for citation in citations:
            match = re.fullmatch(r"SOURCE ([1-9]\d*)", citation.citation_id)
            if not match or citation.citation_id in seen:
                return False
            seen.add(citation.citation_id)
            item = self._valid_ids.get(int(match.group(1)))
            if item is None:
                return False
            if citation.document_id != item.document_id or citation.chunk_id != item.chunk_id:
                return False
            source_identity = (citation.document_id, citation.chunk_id)
            if source_identity in seen_sources:
                return False
            seen_sources.add(source_identity)
        return True

    def get_citation_by_id(self, citation_id: str) -> EvidenceItem | None:
        """Look up a citation ID and return the evidence item, or None."""
        match = re.fullmatch(r"SOURCE ([1-9]\d*)", citation_id)
        if not match:
            return None
        cid = int(match.group(1))
        return None if cid in self._invalid_ids else self._valid_ids.get(cid)

    @property
    def valid_citation_count(self) -> int:
        return len(self._valid_ids)
