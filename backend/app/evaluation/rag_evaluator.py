from __future__ import annotations

from app.evaluation.citation_evaluator import citation_coverage, citation_validity


def evaluate_answer(answer: dict, source_ids: set[str]) -> dict[str, float]:
    citations = answer.get("citations", [])
    return {
        "citation_validity": citation_validity(citations, source_ids),
        "citation_coverage": citation_coverage(answer.get("answer", ""), citations),
        "answerability": float(bool(answer.get("answer"))),
    }


def abstention_is_correct(answer: dict, requires_abstention: bool) -> bool:
    text = answer.get("answer", "").lower()
    abstained = answer.get("confidence") == "low" or any(
        marker in text for marker in ("insufficient", "couldn't find", "not contain")
    )
    return abstained if requires_abstention else not abstained
