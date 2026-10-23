from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from app.evaluation.schemas import EvaluationDataset


def load_retrieval_dataset(path: str | Path) -> EvaluationDataset:
    """Load the versioned JSON dataset (and legacy JSON arrays) deterministically."""
    source = Path(path)
    with source.open(encoding="utf-8") as handle:
        raw: Any = json.load(handle)
    if isinstance(raw, list):
        raw = {
            "name": source.stem,
            "version": "legacy-unversioned",
            "provenance": "Legacy array format; add dataset provenance before evaluation.",
            "examples": [
                {
                    **item,
                    "query": item.get("query", item.get("question", "")),
                    "expected_abstention": item.get("requires_abstention"),
                    "expected_answer": item.get("expected_answer") or {
                        "expected_abstention": item.get("requires_abstention"),
                        "expected_citation_document_ids": item.get(
                            "expected_source_ids", []
                        ),
                    },
                }
                for item in raw
            ],
        }
    if not isinstance(raw, dict):
        raise ValueError("Evaluation dataset must be a JSON object or legacy JSON array.")
    if "examples" not in raw:
        raise ValueError("Evaluation dataset must include an examples array.")
    dataset = EvaluationDataset.model_validate(raw)
    return dataset


def dataset_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
