from __future__ import annotations

import json
from pathlib import Path

from app.evaluation.schemas import EvaluationDataset, RetrievalExample


def load_retrieval_dataset(path: str | Path) -> EvaluationDataset:
    source = Path(path)
    with source.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, list):
        raise ValueError("Retrieval dataset must be a JSON array.")
    examples = [RetrievalExample.model_validate(item) for item in raw]
    ids = [item.id for item in examples]
    if len(ids) != len(set(ids)):
        raise ValueError("Retrieval dataset contains duplicate IDs.")
    return EvaluationDataset(name=source.stem, examples=examples)
