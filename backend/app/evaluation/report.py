from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def build_report(*, status: str, dataset_size: int, metrics: dict[str, Any], limitations: list[str]) -> dict:
    from app.config import settings
    return {
        "status": status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset_size": dataset_size,
        "configuration": {
            "embedding_model": settings.embedding_model,
            "reranker_model": settings.reranker_model or None,
            "llm_model": settings.llm_model,
            "rrf_k": settings.rrf_k,
            "vector_top_k": settings.vector_top_k,
            "keyword_top_k": settings.keyword_top_k,
            "rerank_top_k": settings.rerank_top_k,
        },
        "metrics": metrics,
        "limitations": limitations,
    }


def write_report(report: dict, output_dir: str | Path) -> None:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "evaluation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    lines = ["# Legal RAG Evaluation", "", f"Status: **{report['status']}**", f"Dataset size: **{report['dataset_size']}**", ""]
    for name, metrics in report.get("metrics", {}).items():
        lines.append(f"## {name}")
        lines.append("```json")
        lines.append(json.dumps(metrics, indent=2))
        lines.append("```")
    lines.extend(["", "## Limitations", *[f"- {item}" for item in report.get("limitations", [])]])
    (destination / "evaluation_report.md").write_text("\n".join(lines), encoding="utf-8")
