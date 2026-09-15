from __future__ import annotations

import re
from typing import Iterable


def citation_validity(citations: Iterable[dict], source_ids: set[str]) -> float:
    citations = list(citations)
    if not citations:
        return 1.0
    return sum(1 for citation in citations if citation.get("citation_id") in source_ids) / len(citations)


def citation_coverage(answer: str, citations: Iterable[dict]) -> float:
    claims = [
        part.strip()
        for part in re.split(r"[.!?]\s+(?=[A-Z])", answer)
        if part.strip()
    ]
    if not claims:
        return 1.0
    cited_claims = sum(1 for claim in claims if re.search(r"\[SOURCE\s+\d+\]", claim))
    return cited_claims / len(claims)


def metadata_matches(citation: dict, source: dict) -> bool:
    return all(
        citation.get(field) == source.get(field)
        for field in ("document_id", "chunk_id", "page_start", "page_end", "section", "source")
        if citation.get(field) is not None
    )
