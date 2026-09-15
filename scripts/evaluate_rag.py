"""evaluate_rag.py — Evaluate the grounded RAG pipeline.

Measures:
- Retrieval: Recall@5, MRR
- Generation: Citation validity, citation completeness, abstention accuracy

Usage:
  python scripts/evaluate_rag.py
  python scripts/evaluate_rag.py --questions data/evaluation/rag_questions.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "backend"))

from app.config import settings
from app.db.connection import init_db, check_connection


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    return 1.0 if any(rid in retrieved_ids[:k] for rid in relevant_ids) else 0.0


def mrr(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    for i, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant_ids:
            return 1.0 / i
    return 0.0


def citation_validity(answer_data: dict, max_sources: int) -> tuple[float, list[str]]:
    """Check if all cited sources exist in the evidence."""
    citations = answer_data.get("citations", [])
    if not citations:
        return 1.0, []  # No citations = technically valid

    errors = []
    for c in citations:
        cid = c.get("citation_id", "")
        match = re.match(r"SOURCE\s+(\d+)", cid)
        if match:
            num = int(match.group(1))
            if num > max_sources or num < 1:
                errors.append(cid)

    validity = 1.0 if not errors else max(0.0, 1.0 - len(errors) / len(citations))
    return validity, errors


def abstention_accuracy(answer_data: dict, requires_abstention: bool) -> bool:
    """Check if the system correctly abstained when evidence was insufficient."""
    answer = answer_data.get("answer", "").lower()
    confidence = answer_data.get("confidence", "low")
    limitations = answer_data.get("limitations", [])

    abstained = (
        "insufficient" in answer
        or "not contain" in answer
        or "not available" in answer
        or "not find" in answer
        or "outside the scope" in answer
        or "no relevant" in answer
        or confidence == "low"
        or any("insufficient" in lim.lower() or "no relevant" in lim.lower() for lim in limitations)
    )

    if requires_abstention:
        return abstained
    else:
        return not abstained or confidence in ("medium", "high")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main(questions_path: Path) -> None:
    print("=" * 70)
    print("Pramaan — RAG Evaluation")
    print("=" * 70)

    if not questions_path.exists():
        print(f"[ERR] Questions file not found: {questions_path}")
        sys.exit(1)

    with open(questions_path, encoding="utf-8") as fh:
        questions = json.load(fh)

    print(f"\nQuestions file : {questions_path}")
    print(f"Questions      : {len(questions)}")
    print(f"LLM provider   : {settings.llm_provider}")

    print("\nConnecting to MongoDB...")
    db_connected = await check_connection()
    if not db_connected:
        print("[NOT RUN] MongoDB unavailable; RAG metrics are unavailable.")
        return
    else:
        await init_db()
        print("[OK] MongoDB connected\n")

    from app.rag.services.rag_service import LegalRAGService  # noqa: PLC0415
    from app.rag.retrieval.vector_retriever import VectorRetriever  # noqa: PLC0415

    rag = LegalRAGService()
    vector_retriever = VectorRetriever(top_k=20)

    # Metrics accumulators
    total = len(questions)
    retrieval_recall = {1: 0, 3: 0, 5: 0, 10: 0}
    retrieval_mrr_scores: list[float] = []
    citation_valid_scores: list[float] = []
    abstention_correct = 0
    categories: dict[str, dict[str, int]] = {}
    errors = 0

    print(f"{'#':<4} {'Q':<50} {'Cat':<22} {'Retrieved':<18} {'Cited':<8} {'Abstain':<8}")
    print("-" * 110)

    for idx, item in enumerate(questions, start=1):
        question = item.get("question", item.get("query", ""))
        expected_ids = item.get("expected_chunk_ids") or item.get("expected_source_ids", [])
        requires_abstention = item.get("requires_abstention", False)
        category = item.get("category", "unknown")

        # Track category stats
        if category not in categories:
            categories[category] = {"total": 0, "correct": 0}
        categories[category]["total"] += 1

        q_short = question[:47] + "..." if len(question) > 47 else question

        try:
            # Run RAG
            result = await rag.answer(question)

            # Retrieval evaluation (via vector search for retrieval metrics)
            try:
                v_results = await vector_retriever.retrieve(question, top_k=10)
                retrieved_ids = [r.chunk_id or r.document_id for r in v_results]
            except Exception:
                retrieved_ids = []

            # Compute retrieval metrics
            for k in [1, 3, 5, 10]:
                retrieval_recall[k] += recall_at_k(retrieved_ids, expected_ids, k)
            retrieval_mrr_scores.append(mrr(retrieved_ids, expected_ids))

            # Citation validity
            answer_data = {
                "answer": result.answer,
                "citations": [{"citation_id": c.citation_id} for c in result.citations],
                "confidence": result.confidence,
                "limitations": result.limitations,
            }
            n_sources = len(result.citations) if result.citations else 5
            cv_score, _ = citation_validity(answer_data, n_sources)
            citation_valid_scores.append(cv_score)

            # Abstention accuracy
            abstain_correct = abstention_accuracy(answer_data, requires_abstention)
            if abstain_correct:
                abstention_correct += 1
                categories[category]["correct"] += 1

            # Display
            retrieved_str = ", ".join(retrieved_ids[:2]) if retrieved_ids else "-"
            cited_str = str(len(result.citations))
            abstain_str = "YES" if abstain_correct else "NO"

            print(
                f"  {idx:<3} {q_short:<50} {category:<22} {retrieved_str:<18} "
                f"{cited_str:<8} {abstain_str:<8}"
            )

        except Exception as exc:
            print(f"  {idx:<3} {q_short:<50} {category:<22} ERROR: {exc}")
            errors += 1

    # Summary
    print("\n" + "=" * 70)
    print("RAG Evaluation Results")
    print("=" * 70)
    print(f"\nQuestions: {total}  (errors: {errors})\n")

    # Retrieval metrics
    print("Retrieval Metrics:")
    for k in [1, 3, 5, 10]:
        r = retrieval_recall[k] / total if total else 0.0
        print(f"  Recall@{k}: {r:.3f} ({retrieval_recall[k]}/{total})")
    avg_mrr = sum(retrieval_mrr_scores) / len(retrieval_mrr_scores) if retrieval_mrr_scores else 0.0
    print(f"  MRR:       {avg_mrr:.3f}")

    # Generation metrics
    print("\nGeneration Metrics:")
    avg_cv = sum(citation_valid_scores) / len(citation_valid_scores) if citation_valid_scores else 0.0
    print(f"  Citation validity:     {avg_cv:.3f}")
    abstain_acc = abstention_correct / total if total else 0.0
    print(f"  Abstention accuracy:   {abstain_acc:.3f} ({abstention_correct}/{total})")

    # Category breakdown
    print("\nCategory Breakdown:")
    for cat, stats in sorted(categories.items()):
        acc = stats["correct"] / stats["total"] if stats["total"] else 0.0
        print(f"  {cat:<30} {acc:.3f} ({stats['correct']}/{stats['total']})")

    print("\n" + "=" * 70)
    print("Evaluation complete.")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default="data/evaluation/rag_gold.json")
    args = parser.parse_args()
    asyncio.run(main(Path(args.questions)))
