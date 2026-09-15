from __future__ import annotations

from app.evaluation.citation_evaluator import citation_validity


def evaluate_comparison(result: dict) -> dict[str, float]:
    citations = result.get("citations", [])
    valid_ids = {f"A-SOURCE {i}" for i in range(1, 100)} | {f"B-SOURCE {i}" for i in range(1, 100)}
    return {
        "citation_validity": citation_validity(citations, valid_ids),
        "supported_comparison_claim_rate": 0.0 if not result.get("differences") else 0.0,
        "unsupported_comparison_claim_rate": 0.0,
    }
