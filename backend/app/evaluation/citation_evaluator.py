from __future__ import annotations

import re
from typing import Iterable

_CITATION_PATTERN = re.compile(r"^SOURCE [1-9]\d*$")
_INLINE_CITATION_PATTERN = re.compile(r"\[SOURCE\s+([^\]\r\n]*)\]", re.IGNORECASE)


def citation_validity(citations: Iterable[dict], source_ids: set[str]) -> float | None:
    """Canonical identifier and allowed-ID validity; does not prove source semantics."""
    citations = list(citations)
    if not citations:
        return None
    valid = 0
    seen: set[str] = set()
    for citation in citations:
        citation_id = citation.get("citation_id")
        source_identity = (citation.get("document_id"), citation.get("chunk_id"))
        if (
            isinstance(citation_id, str)
            and _CITATION_PATTERN.fullmatch(citation_id)
            and citation_id in source_ids
            and citation_id not in seen
            and isinstance(source_identity[0], str)
            and bool(source_identity[0].strip())
            and isinstance(source_identity[1], str)
            and bool(source_identity[1].strip())
        ):
            valid += 1
        if isinstance(citation_id, str):
            seen.add(citation_id)
    return valid / len(citations)


def citation_coverage(answer: str, citations: Iterable[dict]) -> float | None:
    """Measure inline ID-to-response-citation resolution, not claim support."""
    inline_ids = [f"SOURCE {match.group(1)}" for match in _INLINE_CITATION_PATTERN.finditer(answer)]
    if not inline_ids:
        return None
    returned_ids = {
        citation.get("citation_id")
        for citation in citations
        if isinstance(citation.get("citation_id"), str)
    }
    return sum(citation_id in returned_ids for citation_id in inline_ids) / len(inline_ids)


def metadata_matches(citation: dict, source: dict) -> bool:
    return all(
        citation.get(field) == source.get(field)
        for field in ("document_id", "chunk_id", "page_start", "page_end", "section", "source")
        if citation.get(field) is not None
    )
