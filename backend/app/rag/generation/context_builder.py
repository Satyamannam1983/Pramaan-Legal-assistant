"""Context builder — formats evidence into a deterministic context string for the LLM."""
from __future__ import annotations

import logging
import re

from app.rag.generation.evidence_selector import EvidenceBundle, EvidenceItem

logger = logging.getLogger(__name__)


def _sanitize_for_prompt(text: str) -> str:
    """Remove potential prompt injection patterns from retrieved text.

    This is a basic defense — retrieved legal documents should be treated
    as untrusted input. We strip known instruction-like patterns while
    preserving the actual content.
    """
    # Remove common injection patterns (case-insensitive)
    injection_patterns = [
        r"(?i)ignore\s+(all\s+)?previous\s+instructions",
        r"(?i)disregard\s+(all\s+)?prior\s+instructions",
        r"(?i)you\s+are\s+now\s+",
        r"(?i)forget\s+(all\s+)?previous\s+",
        r"(?i)new\s+instructions?\s*:",
        r"(?i)system\s*prompt\s*:",
        r"(?i)reveal\s+(your|the)\s+system\s+prompt",
    ]
    sanitized = text
    for pattern in injection_patterns:
        sanitized = re.sub(pattern, "[CONTENT REDACTED]", sanitized)
    return sanitized


class ContextBuilder:
    """Deterministic context formatter for legal RAG.

    Produces a stable, well-structured context string with citation IDs
    that the LLM must reference when making claims.
    """

    SYSTEM_PROMPT = """You are a legal research assistant for Indian law. Your role is to provide accurate, evidence-based answers grounded exclusively in the retrieved legal sources provided below.

CORE INSTRUCTIONS:
1. Answer ONLY using the evidence provided in the [SOURCES] section. Do not use any external knowledge.
2. Never invent, fabricate, or hallucinate statutes, sections, judgments, citations, dates, case names, or legal holdings.
3. When citing a source, use the exact citation ID format: [SOURCE 1], [SOURCE 2], etc.
4. Always distinguish between:
   - What the evidence directly states
   - What can be reasonably inferred from the evidence
   - What is uncertain or not addressed by the evidence
5. If the provided evidence is insufficient to answer the question, say so clearly. Do NOT guess or fill gaps with invented information.
6. Do not pretend to be a lawyer or give legal advice. You are a research tool.
7. Never fabricate a source citation. Only cite sources that appear in the [SOURCES] section.
8. Preserve important legal qualifications, exceptions, and limitations from the source text.
9. Prefer direct quotes from the source when stating specific legal provisions.
10. If a question asks about a topic not covered by the evidence, respond with: "The retrieved sources do not contain sufficient information to answer this question."

RESPONSE FORMAT: You MUST respond with a valid JSON object in the following exact format:
{
  "answer": "Your detailed answer here, citing [SOURCE N] where relevant",
  "key_points": ["Point 1 supported by evidence", "Point 2 supported by evidence"],
  "citations": [
    {
      "citation_id": "SOURCE 1",
      "relevant_quote": "Direct quote from SOURCE 1 that supports the answer"
    }
  ],
  "limitations": ["Any limitations of the evidence or your answer"],
  "confidence": "high" | "medium" | "low"
}

Confidence levels:
- "high": Evidence directly and clearly answers the question from authoritative sources
- "medium": Evidence partially answers the question or requires some inference
- "low": Evidence is tangential, weak, or insufficient for a definitive answer"""

    def build_context(self, bundle: EvidenceBundle) -> str:
        """Build the full context string from an evidence bundle.

        Returns a string containing:
        1. The [SOURCES] section with numbered evidence chunks
        2. Ready for the LLM to process
        """
        if not bundle.items:
            return ""

        parts: list[str] = []

        for item in bundle.items:
            sanitized_content = _sanitize_for_prompt(item.content)
            section_parts = [f"[{item.citation_id}]"]

            metadata_lines = []
            if item.title:
                metadata_lines.append(f"Document: {item.title}")
            if item.section:
                metadata_lines.append(f"Section: {item.section}")
            if item.page_start is not None:
                page_str = (
                    str(item.page_start)
                    if item.page_end is None or item.page_start == item.page_end
                    else f"{item.page_start}–{item.page_end}"
                )
                metadata_lines.append(f"Page: {page_str}")
            if item.court:
                metadata_lines.append(f"Court: {item.court}")
            if item.date:
                metadata_lines.append(f"Date: {item.date}")
            if item.source:
                metadata_lines.append(f"Source: {item.source}")
            if item.source_url:
                metadata_lines.append(f"URL: {item.source_url}")

            if metadata_lines:
                section_parts.append("\n".join(metadata_lines))

            section_parts.append(f"\n{sanitized_content}")
            parts.append("\n".join(section_parts))

        return "\n\n".join(parts)

    def build_user_prompt(self, query: str, context: str) -> str:
        """Build the user prompt combining the query and evidence context."""
        if not context:
            return (
                f"Question: {query}\n\n"
                "No evidence sources are available. "
                "Respond with a JSON object indicating insufficient evidence."
            )

        return (
            f"Question: {query}\n\n"
            f"[SOURCES]\n{context}\n[/SOURCES]\n\n"
            "Based ONLY on the sources above, provide your answer as a JSON object."
        )
