from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable

from app.evaluation.schemas import RetrievalMetrics


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive.")
    return len(relevant.intersection(retrieved[:k])) / len(relevant) if relevant else 0.0


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    for rank, item in enumerate(retrieved, 1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, item in enumerate(retrieved[:k], 1)
        if item in relevant
    )
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(k, len(relevant)) + 1))
    return dcg / ideal if ideal else 0.0


def evaluate_rankings(rows: Iterable[tuple[list[str], set[str]]]) -> RetrievalMetrics:
    rows = list(rows)
    if not rows:
        raise ValueError("Cannot compute retrieval metrics without labeled examples.")
    if any(not relevant for _, relevant in rows):
        raise ValueError("Every ranking row must contain at least one relevance label.")
    return RetrievalMetrics(
        recall_at_1=sum(recall_at_k(r, g, 1) for r, g in rows) / len(rows),
        recall_at_3=sum(recall_at_k(r, g, 3) for r, g in rows) / len(rows),
        recall_at_5=sum(recall_at_k(r, g, 5) for r, g in rows) / len(rows),
        recall_at_10=sum(recall_at_k(r, g, 10) for r, g in rows) / len(rows),
        mrr=sum(reciprocal_rank(r, g) for r, g in rows) / len(rows),
        ndcg_at_5=sum(ndcg_at_k(r, g, 5) for r, g in rows) / len(rows),
        count=len(rows),
    )


def category_metrics(
    examples: Iterable[tuple[str, list[str], set[str]]],
) -> dict[str, RetrievalMetrics]:
    grouped: dict[str, list[tuple[list[str], set[str]]]] = defaultdict(list)
    for category, retrieved, relevant in examples:
        grouped[category].append((retrieved, relevant))
    return {category: evaluate_rankings(rows) for category, rows in grouped.items()}
