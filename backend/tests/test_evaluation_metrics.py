import pytest

from app.evaluation.citation_evaluator import citation_coverage, citation_validity
from app.evaluation.dataset import load_retrieval_dataset
from app.evaluation.rag_evaluator import abstention_is_correct, evaluate_answer
from app.evaluation.retrieval_evaluator import evaluate_rankings


def test_recall_is_fraction_of_all_relevant_items():
    metrics = evaluate_rankings([(["a", "b", "c"], {"b", "c"})])
    assert metrics.recall_at_1 == 0
    assert metrics.recall_at_3 == 1
    assert metrics.mrr == pytest.approx(0.5)
    assert metrics.ndcg_at_5 > 0


def test_recall_does_not_degrade_to_hit_rate():
    metrics = evaluate_rankings([(["relevant-a"], {"relevant-a", "relevant-b"})])
    assert metrics.recall_at_1 == pytest.approx(0.5)


def test_empty_or_unlabeled_rankings_are_not_scored():
    with pytest.raises(ValueError, match="without labeled examples"):
        evaluate_rankings([])
    with pytest.raises(ValueError, match="relevance label"):
        evaluate_rankings([(["some-result"], set())])


def test_citation_metrics_distinguish_missing_ids():
    citations = [{"citation_id": "SOURCE 1", "document_id": "doc", "chunk_id": "chunk"}]
    assert citation_validity(citations, {"SOURCE 1"}) == 1
    assert citation_coverage("Supported claim. [SOURCE 1]", citations) == 1
    assert citation_coverage("Claim without reference.", citations) is None
    assert citation_validity([], {"SOURCE 1"}) is None


def test_answer_evaluator_does_not_score_answerability_or_infer_abstention():
    assert evaluate_answer({"answer": "Some text", "citations": []}, set()) == {
        "citation_validity": None,
        "citation_coverage": None,
    }
    assert abstention_is_correct({"confidence": "low", "answer": "uncertain"}, True) is None
    assert abstention_is_correct({"abstained": True}, True) is True
    assert abstention_is_correct({"abstained": False}, True) is False


def test_dataset_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "gold.json"
    path.write_text(
        '[{"id":"q1","query":"a","category":"x"},{"id":"q1","query":"b","category":"x"}]',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_retrieval_dataset(path)


def test_dataset_loads_versioned_expected_answer_characteristics(tmp_path):
    path = tmp_path / "gold.json"
    path.write_text(
        '{"name":"fixture","version":"2","provenance":"local",'
        '"examples":[{"id":"q1","query":"question",'
        '"expected_answer":{"expected_abstention":true,"required_terms":["evidence"]}}]}',
        encoding="utf-8",
    )
    dataset = load_retrieval_dataset(path)
    assert dataset.version == "2"
    assert dataset.examples[0].expected_answer.expected_abstention is True
    assert dataset.examples[0].expected_answer.required_terms == ["evidence"]


def test_legacy_dataset_migrates_expected_source_ids(tmp_path):
    path = tmp_path / "legacy.json"
    path.write_text(
        '[{"id":"q1","question":"question","expected_source_ids":["doc-a"],'
        '"requires_abstention":false}]',
        encoding="utf-8",
    )
    example = load_retrieval_dataset(path).examples[0]
    assert example.query == "question"
    assert example.expected_answer.expected_abstention is False
    assert example.expected_answer.expected_citation_document_ids == ["doc-a"]
