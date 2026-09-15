"""evaluate_retrieval.py — Compare Vector, BM25, Hybrid, and Hybrid+Reranker retrieval.

Computes Recall@1, Recall@3, Recall@5, MRR, and NDCG@5 for each approach.
Prints a comparison table.

Usage
-----
  python scripts/evaluate_retrieval.py
  python scripts/evaluate_retrieval.py --queries data/evaluation/retrieval_queries.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "backend"))

from app.config import settings
from app.db.connection import init_db, check_connection
from app.rag.retrieval.vector_retriever import VectorRetriever
from app.rag.retrieval.hybrid_retriever import HybridRetriever
from app.rag.retrieval.keyword_retriever import KeywordRetriever


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """1.0 if any relevant doc is in top-k, else 0.0 (binary recall per query)."""
    if not relevant_ids:
        return 0.0
    return 1.0 if any(rid in retrieved_ids[:k] for rid in relevant_ids) else 0.0


def mrr(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    """Mean Reciprocal Rank — 1/rank of first relevant result, or 0."""
    for i, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant_ids:
            return 1.0 / i
    return 0.0


def dcg_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """DCG@k — discounted cumulative gain with binary relevance."""
    score = 0.0
    for i, rid in enumerate(retrieved_ids[:k], start=1):
        if rid in relevant_ids:
            score += 1.0 / math.log2(i + 1)
    return score


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """NDCG@k — normalised DCG."""
    actual_dcg = dcg_at_k(retrieved_ids, relevant_ids, k)
    # Ideal: all relevant docs at top positions
    ideal_relevant = min(len(relevant_ids), k)
    ideal_dcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_relevant + 1))
    if ideal_dcg == 0:
        return 0.0
    return actual_dcg / ideal_dcg


# ---------------------------------------------------------------------------
# Retrieval methods
# ---------------------------------------------------------------------------

async def vector_only(query: str, k: int, retriever: VectorRetriever) -> list[str]:
    """A. Vector only → top k."""
    results = await retriever.retrieve(query, top_k=k)
    return [r.chunk_id or r.document_id for r in results]


async def keyword_only(query: str, k: int, retriever: KeywordRetriever) -> list[str]:
    """B. BM25 keyword-only retrieval."""
    results = await retriever.retrieve(query, top_k=k)
    return [r.chunk_id or r.document_id for r in results]


async def hybrid_only(query: str, k: int, retriever: HybridRetriever) -> list[str]:
    """B. Vector + BM25 → RRF → top k."""
    results = await retriever.retrieve(query, top_k=k)
    return [r.chunk_id or r.document_id for r in results]


async def hybrid_reranked(
    query: str, k: int, retriever: HybridRetriever
) -> list[str]:
    """C. Vector + BM25 → RRF → Reranker → top k."""
    results = await retriever.retrieve(query, top_k=k)
    # Apply reranking if available
    if settings.reranker_provider == "cross_encoder" and settings.reranker_model:
        try:
            from app.rag.reranking.cross_encoder import CrossEncoderReranker  # noqa: PLC0415
            from app.rag.reranking.base import RerankResult  # noqa: PLC0415

            reranker = CrossEncoderReranker()
            candidates = [
                RerankResult(
                    chunk_id=r.chunk_id,
                    rerank_score=0.0,
                    content=r.content,
                    document_id=r.document_id,
                    title=r.title,
                    vector_score=r.vector_score,
                    keyword_score=r.keyword_score,
                    rrf_score=r.rrf_score,
                )
                for r in results
            ]
            reranked = await reranker.rerank(query, candidates, top_k=k)
            return [r.chunk_id or r.document_id for r in reranked]
        except Exception as exc:
            print(f"  [WARN] Reranker failed, using RRF scores: {exc}")
    # Fallback: already sorted by RRF
    return [r.chunk_id or r.document_id for r in results]


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

async def main(queries_path: Path) -> None:
    print("=" * 70)
    print("Pramaan — Retrieval Evaluation: Vector vs BM25 vs Hybrid vs Hybrid+Reranker")
    print("=" * 70)

    if not queries_path.exists():
        print(f"[ERR] Queries file not found: {queries_path}")
        sys.exit(1)

    with open(queries_path, encoding="utf-8") as fh:
        queries = json.load(fh)

    print(f"\nQueries file : {queries_path}")
    print(f"Queries      : {len(queries)}")
    print(f"Embedding    : {settings.embedding_model}")
    print(f"Reranker     : {settings.reranker_model or 'none'}")

    print("\nConnecting to MongoDB...")
    if not await check_connection():
        print("[NOT RUN] MongoDB unavailable; retrieval metrics are unavailable.")
        return
    await init_db()
    print("[OK] MongoDB connected\n")

    # Initialise retrievers
    vector_retriever = VectorRetriever(top_k=20)
    keyword_retriever = KeywordRetriever(top_k=20)
    hybrid_retriever = HybridRetriever(vector_top_k=20, keyword_top_k=20)

    k_values = [1, 3, 5, 10]
    methods = ["Vector", "BM25", "Hybrid", "Hybrid + Reranker"]

    # Accumulators for metrics
    hits: dict[str, dict[int, int]] = {m: {k: 0 for k in k_values} for m in methods}
    mrr_scores: dict[str, list[float]] = {m: [] for m in methods}
    ndcg_scores: dict[str, dict[int, list[float]]] = {
        m: {k: [] for k in k_values} for m in methods
    }
    total = len(queries)
    errors = 0

    print(f"{'Query':<50} {'Expected':<18} {'Vector':<18} {'BM25':<18} {'Hybrid':<18} {'Hyb+RR':<18}")
    print("-" * 122)

    for item in queries:
        query: str = item["query"]
        # Prefer chunk-level gold labels; document IDs are an explicit fallback
        # for datasets that intentionally evaluate document-level retrieval.
        relevant_ids: list[str] = item.get("relevant_chunk_ids") or item.get(
            "relevant_document_ids", []
        )

        # Run all three methods
        try:
            v_ids = await vector_only(query, max(k_values), vector_retriever)
        except Exception as exc:
            print(f"  [ERR] Vector failed: {exc}")
            v_ids = []
            errors += 1

        try:
            b_ids = await keyword_only(query, max(k_values), keyword_retriever)
        except Exception as exc:
            print(f"  [ERR] BM25 failed: {exc}")
            b_ids = []
            errors += 1

        try:
            h_ids = await hybrid_only(query, max(k_values), hybrid_retriever)
        except Exception as exc:
            print(f"  [ERR] Hybrid failed: {exc}")
            h_ids = []
            errors += 1

        try:
            hr_ids = await hybrid_reranked(query, max(k_values), hybrid_retriever)
        except Exception as exc:
            print(f"  [ERR] Hybrid+Reranker failed: {exc}")
            hr_ids = []
            errors += 1

        method_ids = {
            "Vector": v_ids,
            "BM25": b_ids,
            "Hybrid": h_ids,
            "Hybrid + Reranker": hr_ids,
        }

        # Compute metrics
        for method_name, doc_ids in method_ids.items():
            for k in k_values:
                hits[method_name][k] += recall_at_k(doc_ids, relevant_ids, k)
            mrr_scores[method_name].append(mrr(doc_ids, relevant_ids))
            for k in k_values:
                ndcg_scores[method_name][k].append(
                    ndcg_at_k(doc_ids, relevant_ids, k)
                )

        # Print per-query line
        q_short = query[:47] + "..." if len(query) > 47 else query
        exp = relevant_ids[0] if relevant_ids else "-"
        v_top = v_ids[0] if v_ids else "-"
        b_top = b_ids[0] if b_ids else "-"
        h_top = h_ids[0] if h_ids else "-"
        hr_top = hr_ids[0] if hr_ids else "-"
        print(
            f"  {q_short:<48} {exp:<18} {v_top:<18} {b_top:<18} {h_top:<18} {hr_top:<18}"
        )

    # Compute final metrics
    print("\n" + "=" * 70)
    print("Retrieval Evaluation Results")
    print("=" * 70)
    print(f"\nQueries: {total}  (errors: {errors})\n")

    # Print comparison table
    print(f"{'Method':<25} {'Recall@1':<10} {'Recall@3':<10} {'Recall@5':<10} {'Recall@10':<10} {'MRR':<10} {'NDCG@5':<10}")
    print("-" * 75)

    for method_name in methods:
        recall_vals = []
        for k in k_values:
            r = hits[method_name][k] / total if total else 0.0
            recall_vals.append(r)
        avg_mrr = sum(mrr_scores[method_name]) / total if total else 0.0
        ndcg5 = (
            sum(ndcg_scores[method_name][5]) / total if total else 0.0
        )
        print(
            f"  {method_name:<23} {recall_vals[0]:<10.3f} {recall_vals[1]:<10.3f} "
            f"{recall_vals[2]:<10.3f} {recall_vals[3]:<10.3f} {avg_mrr:<10.3f} {ndcg5:<10.3f}"
        )

    print("\n" + "=" * 70)
    print("Evaluation complete.")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", default="data/evaluation/retrieval_gold.json")
    args = parser.parse_args()
    asyncio.run(main(Path(args.queries)))
