"""Compare linear and bounded graph execution.

Meaningful retrieval metrics require a reachable MongoDB instance. When it is
unavailable this script reports evaluation as unavailable rather than treating
empty retrieval as a quality score.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "backend"))

from app.db.connection import check_connection
from app.rag.graph.graph import build_legal_research_graph
from app.rag.services.rag_service import LegalRAGService


async def main(question: str) -> None:
    if not await check_connection():
        print("Graph evaluation unavailable: MongoDB is not reachable.")
        return

    linear = await LegalRAGService().answer(question)
    graph = await build_legal_research_graph().ainvoke({"query": question})
    print("Linear RAG:")
    print(f"  citation_validity: {1.0 if linear.citations else 0.0:.3f}")
    print("LangGraph RAG:")
    print(f"  retrieval_success: {bool(graph.get('retrieved_results'))}")
    print(f"  citation_validity: {bool(graph.get('validation_result', {}).get('is_grounded'))}")
    print(f"  abstention: {graph.get('answer', {}).get('confidence') == 'low'}")
    print(f"  retrieval_attempts: {graph.get('retrieval_attempt', 0)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="?", default="What is consideration in contract law?")
    args = parser.parse_args()
    asyncio.run(main(args.question))
