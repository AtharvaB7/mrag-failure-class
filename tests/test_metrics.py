import pytest

from evaluation.metrics import accuracy, retrieval_recall_at_k


def test_accuracy_all_correct():
    assert accuracy(["A", "B", "C"], ["A", "B", "C"]) == 1.0


def test_accuracy_partial():
    assert accuracy(["A", "B", "C", "D"], ["A", "B", "X", "Y"]) == 0.5


def test_accuracy_length_mismatch_raises():
    with pytest.raises(ValueError):
        accuracy(["A", "B"], ["A"])


def test_accuracy_empty_raises():
    with pytest.raises(ValueError):
        accuracy([], [])


def test_retrieval_recall_basic():
    retrieved = {
        "q1": ["d1", "d2", "d3"],
        "q2": ["d5", "d6"],
    }
    gt = {
        "q1": ["d3"],  # in top 3 -> hit
        "q2": ["d9"],  # not present -> miss
    }
    result = retrieval_recall_at_k(retrieved, gt, k=3)
    assert result.per_instance == {"q1": True, "q2": False}
    assert result.recall_at_k == 0.5


def test_retrieval_recall_respects_k_truncation():
    retrieved = {"q1": ["d1", "d2", "d3", "d4"]}
    gt = {"q1": ["d4"]}
    # d4 is rank 4, so k=3 should NOT count it as retrieved
    result_k3 = retrieval_recall_at_k(retrieved, gt, k=3)
    assert result_k3.per_instance["q1"] is False

    result_k4 = retrieval_recall_at_k(retrieved, gt, k=4)
    assert result_k4.per_instance["q1"] is True


def test_retrieval_recall_skips_instances_with_no_gt():
    retrieved = {"q1": ["d1"], "q2": ["d2"]}
    gt = {"q1": [], "q2": ["d2"]}
    result = retrieval_recall_at_k(retrieved, gt, k=1)
    assert "q1" not in result.per_instance
    assert result.recall_at_k == 1.0  # only q2 counted, and it's a hit


def test_retrieval_recall_mismatched_instance_ids_raises():
    with pytest.raises(ValueError):
        retrieval_recall_at_k({"q1": ["d1"]}, {"q2": ["d1"]}, k=1)


def test_retrieval_recall_all_empty_gt_raises():
    with pytest.raises(ValueError):
        retrieval_recall_at_k({"q1": ["d1"]}, {"q1": []}, k=1)
