import pytest

from app.evaluation.citation_evaluator import citation_coverage, citation_validity
from app.evaluation.dataset import load_retrieval_dataset
from app.evaluation.retrieval_evaluator import evaluate_rankings


def test_retrieval_metrics():
    metrics = evaluate_rankings([(["a", "b", "c"], {"b"})])
    assert metrics.recall_at_1 == 0
    assert metrics.recall_at_3 == 1
    assert metrics.mrr == pytest.approx(0.5)
    assert metrics.ndcg_at_5 > 0


def test_citation_metrics():
    citations = [{"citation_id": "SOURCE 1"}]
    assert citation_validity(citations, {"SOURCE 1"}) == 1
    assert citation_coverage("Supported claim. [SOURCE 1]", citations) == 1


def test_dataset_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "gold.json"
    path.write_text(
        '[{"id":"q1","query":"a","category":"x"},{"id":"q1","query":"b","category":"x"}]',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_retrieval_dataset(path)
