"""Grounding validator — verifies LLM output against retrieved evidence."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from app.rag.citations.citation_service import CitationService

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of grounding validation."""

    is_grounded: bool = True
    citation_validity: dict[str, bool] = field(default_factory=dict)
    invalid_citations: list[str] = field(default_factory=list)
    malformed_citations: list[str] = field(default_factory=list)
    missing_citations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def summary(self) -> str:
        parts = []
        if self.invalid_citations:
            parts.append(f"Invalid citations: {', '.join(self.invalid_citations)}")
        if self.malformed_citations:
            parts.append(f"Malformed citations: {', '.join(self.malformed_citations)}")
        if self.missing_citations:
            parts.append(f"Missing citations: {', '.join(self.missing_citations)}")
        if self.warnings:
            parts.extend(self.warnings)
        return "; ".join(parts) if parts else "All citations valid."


class GroundingValidator:
    """Validates citation syntax and identifier membership, not semantic entailment.

    Checks:
    - Every source token is canonical and maps to usable retrieved evidence.
    - The answer has at least one valid citation when evidence is available.
    - Each supplied key point has at least one valid citation.

    This does not determine whether the cited passage supports the claim.
    """

    def validate(
        self,
        llm_output: str,
        citation_service: CitationService,
        has_evidence: bool = True,
        key_points: list[str] | None = None,
    ) -> ValidationResult:
        """Validate the LLM output against retrieved evidence.

        Parameters
        ----------
        llm_output : The raw LLM response text.
        citation_service : Loaded with the evidence bundle.
        has_evidence : Whether usable evidence was available for this query.
        key_points : Claim-like summary points that must each contain a valid source ID.

        Returns
        -------
        ValidationResult with validity status and details.
        """
        result = ValidationResult(is_grounded=True)
        cited_ids, malformed = citation_service.parse_references(llm_output)
        result.malformed_citations = malformed
        if malformed:
            result.is_grounded = False

        if not has_evidence:
            result.is_grounded = False
            result.warnings.append("No evidence was available to validate the output.")
            if cited_ids or malformed:
                result.warnings.append("Output cited sources when no evidence was provided.")
            result.missing_citations = []
            return result

        valid_ids: set[int] = set()
        for cid in set(cited_ids):
            citation_id = f"SOURCE {cid}"
            valid = citation_service.get_citation_by_id(citation_id) is not None
            result.citation_validity[citation_id] = valid
            if valid:
                valid_ids.add(cid)
            else:
                result.invalid_citations.append(citation_id)
                result.is_grounded = False

        if not valid_ids:
            result.is_grounded = False
            result.missing_citations.append("answer")

        for index, key_point in enumerate(key_points or [], start=1):
            point_ids, malformed_point_ids = citation_service.parse_references(key_point)
            result.malformed_citations.extend(malformed_point_ids)
            if malformed_point_ids:
                result.is_grounded = False
            valid_point_ids = {
                cid for cid in point_ids
                if citation_service.get_citation_by_id(f"SOURCE {cid}") is not None
            }
            invalid_point_ids = {
                cid for cid in point_ids
                if citation_service.get_citation_by_id(f"SOURCE {cid}") is None
            }
            for cid in sorted(invalid_point_ids):
                citation_id = f"SOURCE {cid}"
                result.citation_validity[citation_id] = False
                if citation_id not in result.invalid_citations:
                    result.invalid_citations.append(citation_id)
            for cid in valid_point_ids:
                result.citation_validity[f"SOURCE {cid}"] = True
            if key_point.strip() and not valid_point_ids:
                result.missing_citations.append(f"key_points[{index}]")
                result.is_grounded = False

        result.malformed_citations = list(dict.fromkeys(result.malformed_citations))

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
