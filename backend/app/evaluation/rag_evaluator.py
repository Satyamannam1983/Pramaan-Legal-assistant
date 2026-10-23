from __future__ import annotations

from app.evaluation.citation_evaluator import citation_coverage, citation_validity


def evaluate_answer(answer: dict, source_ids: set[str]) -> dict[str, float | None]:
    citations = answer.get("citations", [])
    return {
        "citation_validity": citation_validity(citations, source_ids),
        "citation_coverage": citation_coverage(answer.get("answer", ""), citations),
    }


def abstention_is_correct(answer: dict, requires_abstention: bool) -> bool | None:
    """Compare an explicit abstention flag; never infer abstention from prose."""
    abstained = answer.get("abstained")
    if not isinstance(abstained, bool):
        return None
    return abstained == requires_abstention
