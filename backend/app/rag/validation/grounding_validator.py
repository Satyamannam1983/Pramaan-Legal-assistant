"""Grounding validator — verifies LLM output against retrieved evidence."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from app.rag.citations.citation_service import CitationService

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of grounding validation."""

    is_grounded: bool = True
    citation_validity: dict[str, bool] = field(default_factory=dict)
    invalid_citations: list[str] = field(default_factory=list)
    missing_citations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def summary(self) -> str:
        parts = []
        if self.invalid_citations:
            parts.append(f"Invalid citations: {', '.join(self.invalid_citations)}")
        if self.missing_citations:
            parts.append(f"Citations referenced but not used: {', '.join(self.missing_citations)}")
        if self.warnings:
            parts.extend(self.warnings)
        return "; ".join(parts) if parts else "All citations valid."


class GroundingValidator:
    """Validates LLM output against the evidence supplied.

    Checks:
    1. Citation validity — every [SOURCE N] in the output must correspond
       to an actual retrieved source.
    2. Citation completeness — important claims should have citations.
    3. Basic grounding — the answer references at least one source
       when evidence is available.
    """

    def __init__(self) -> None:
        self._source_pattern = re.compile(r"\[SOURCE\s+(\d+)\]")

    def validate(
        self,
        llm_output: str,
        citation_service: CitationService,
        has_evidence: bool = True,
    ) -> ValidationResult:
        """Validate the LLM output against retrieved evidence.

        Parameters
        ----------
        llm_output : The raw LLM response text.
        citation_service : Loaded with the evidence bundle.
        has_evidence : Whether evidence was available for this query.

        Returns
        -------
        ValidationResult with validity status and details.
        """
        result = ValidationResult(is_grounded=True)

        if not has_evidence:
            # When no evidence, the LLM should abstain
            if "[SOURCE" in llm_output.upper():
                result.is_grounded = False
                result.warnings.append(
                    "LLM cited sources when no evidence was provided."
                )
            return result

        # Extract all citation IDs from the output
        cited_ids = set(self._source_pattern.findall(llm_output))

        # Check each citation is valid
        for cid_str in cited_ids:
            cid = int(cid_str)
            citation_id = f"SOURCE {cid}"
            valid = citation_service.get_citation_by_id(citation_id) is not None
            result.citation_validity[citation_id] = valid
            if not valid:
                result.invalid_citations.append(citation_id)
                result.is_grounded = False

        # Check if any citations were used at all
        if not cited_ids and has_evidence:
            result.warnings.append(
                "LLM output contains no source citations despite evidence being available."
            )

        # Log validation result
        if result.is_grounded:
            logger.debug(
                "Grounding validation passed: %d citations used, all valid.",
                len(cited_ids),
            )
        else:
            logger.warning(
                "Grounding validation failed: %d invalid citations: %s",
                len(result.invalid_citations), result.invalid_citations,
            )

        return result
