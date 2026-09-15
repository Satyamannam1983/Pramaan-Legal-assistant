"""Conservative, text-grounded metadata extraction."""
from __future__ import annotations

import re

from app.rag.document_analysis.schemas import LegalDocumentMetadata, MetadataEvidence


def extract_metadata(document_id: str, text: str, base: dict | None = None) -> LegalDocumentMetadata:
    base = base or {}
    def first(pattern: str) -> tuple[str | None, int]:
        match = re.search(pattern, text, re.I | re.M)
        return (match.group(1).strip(), match.start()) if match else (None, -1)

    court, court_pos = first(r"(IN THE [^\n]*COURT[^\n]*|SUPREME COURT OF INDIA|HIGH COURT[^\n]*)")
    case_number, case_pos = first(r"((?:Civil|Criminal|Writ|Crl\.)\s+(?:Appeal|Petition)[^\n]{0,80}\bNo\.?\s*[\w/-]+)")
    date, date_pos = first(r"(?:Decided|Date|Dated)\s*:\s*([0-9]{1,2}\s+[A-Za-z]+\s+[0-9]{4}|[0-9]{1,2}[/-][0-9]{1,2}[/-][0-9]{2,4})")
    parties, _ = first(r"([A-Z][A-Za-z .&'-]{1,60}\s+(?:v\.|vs\.|versus)\s+[A-Z][A-Za-z .&'-]{1,60})")
    sections = sorted(set(re.findall(r"\bSection\s+\d+[A-Za-z]?(?:\(\d+\))?", text, re.I)))
    articles = sorted(set(re.findall(r"\bArticle\s+\d+[A-Za-z]?(?:\(\d+\))?", text, re.I)))
    acts = sorted(set(re.findall(r"\b[A-Z][A-Za-z ]+(?:Act|Code|Rules?),?\s*(?:\d{4})?", text)))
    dtype = base.get("document_type")
    if not dtype:
        if re.search(r"\bjudgment|court|petition|appeal\b", text, re.I):
            dtype = "JUDGMENT"
        elif re.search(r"\bact\b|\bsection\s+\d+", text, re.I):
            dtype = "ACT"
        else:
            dtype = "UNKNOWN"
    evidence: list[MetadataEvidence] = []
    for field, value, pos in (("court", court, court_pos), ("case_number", case_number, case_pos), ("date", date, date_pos)):
        evidence.append(MetadataEvidence(field=field, value=value, confidence=0.9 if value else 0.0))
    return LegalDocumentMetadata(
        document_id=document_id,
        title=base.get("title"),
        document_type=dtype,
        court=court or base.get("court"),
        case_number=case_number or base.get("case_number"),
        citation=base.get("citation"),
        date=date or base.get("date"),
        parties=[parties] if parties else [],
        petitioner=parties.split(" v. ")[0].strip() if parties and " v. " in parties else None,
        respondent=parties.split(" v. ")[-1].strip() if parties and " v. " in parties else None,
        acts_referenced=acts[:30],
        sections_referenced=sections,
        articles_referenced=articles,
        keywords=sorted(set(re.findall(r"\b(?:contract|murder|bail|writ|evidence|jurisdiction|appeal)\b", text, re.I))),
        evidence=evidence,
    )
