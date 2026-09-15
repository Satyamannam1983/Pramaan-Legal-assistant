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
    source_url: str | None = None
    relevant_quote: str | None = None


# Pattern to match [SOURCE N] in LLM output
_SOURCE_PATTERN = re.compile(r"\[SOURCE\s+(\d+)\]")


class CitationService:
    """Maps LLM citation references to actual retrieved evidence.

    Security model:
    - Only citation IDs that were supplied to the LLM are valid.
    - The LLM cannot invent new citation IDs.
    - All citation metadata comes from the original retrieval results.
    """

    def __init__(self) -> None:
        self._valid_ids: dict[int, EvidenceItem] = {}

    def load_evidence(self, bundle: EvidenceBundle) -> None:
        """Load evidence bundle to establish valid citation IDs."""
        self._valid_ids = {}
        for item in bundle.items:
            # Extract numeric ID from "SOURCE N"
            match = re.match(r"SOURCE\s+(\d+)", item.citation_id)
            if match:
                self._valid_ids[int(match.group(1))] = item

    def extract_citations(self, llm_output: str) -> list[Citation]:
        """Extract and validate citation IDs from LLM output.

        Returns only citations that map to actual retrieved evidence.
        Logs warnings for any invalid citation references.
        """
        citation_ids = _SOURCE_PATTERN.findall(llm_output)
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
                    source_url=item.source_url,
                )
            )

        return citations

    def get_citation_by_id(self, citation_id: str) -> EvidenceItem | None:
        """Look up a citation ID and return the evidence item, or None."""
        match = re.match(r"SOURCE\s+(\d+)", citation_id)
        if not match:
            return None
        cid = int(match.group(1))
        return self._valid_ids.get(cid)

    @property
    def valid_citation_count(self) -> int:
        return len(self._valid_ids)
