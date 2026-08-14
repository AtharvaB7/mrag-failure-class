"""Accuracy and retrieval recall metrics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Sequence


def accuracy(predictions: Sequence[str], ground_truth: Sequence[str]) -> float:
    """Exact-match accuracy over answer_choice letters (A/B/C/D)."""
    if len(predictions) != len(ground_truth):
        raise ValueError(
            f"predictions ({len(predictions)}) and ground_truth ({len(ground_truth)}) "
            "must be the same length"
        )
    if not predictions:
        raise ValueError("Cannot compute accuracy over an empty set")
    correct = sum(p == g for p, g in zip(predictions, ground_truth))
    return correct / len(predictions)


@dataclass
class RetrievalRecallResult:
    per_instance: dict[str, bool]  # instance_id -> whether >=1 gt doc was retrieved in top_k
    recall_at_k: float


def retrieval_recall_at_k(
    retrieved_ids_by_instance: dict[str, Sequence[Hashable]],
    gt_ids_by_instance: dict[str, Sequence[Hashable]],
    k: int,
) -> RetrievalRecallResult:
    """Fraction of instances for which at least one ground-truth doc_id
    appears in the top-k retrieved doc_ids. This is the metric that directly
    tells you "retrieval missed the evidence" vs. not for failure-mode
    labeling.

    Instances with an empty gt_ids list are skipped (no evidence to recall)
    and do not count toward the denominator.
    """
    if set(retrieved_ids_by_instance.keys()) != set(gt_ids_by_instance.keys()):
        missing = set(gt_ids_by_instance.keys()) - set(retrieved_ids_by_instance.keys())
        extra = set(retrieved_ids_by_instance.keys()) - set(gt_ids_by_instance.keys())
        raise ValueError(
            f"instance id sets don't match. missing from retrieved: {missing}, "
            f"extra in retrieved: {extra}"
        )

    per_instance: dict[str, bool] = {}
    for inst_id, gt_ids in gt_ids_by_instance.items():
        if not gt_ids:
            continue
        top_k_retrieved = set(retrieved_ids_by_instance[inst_id][:k])
        per_instance[inst_id] = bool(top_k_retrieved & set(gt_ids))

    if not per_instance:
        raise ValueError("No instances had non-empty ground-truth ids; recall undefined")

    recall = sum(per_instance.values()) / len(per_instance)
    return RetrievalRecallResult(per_instance=per_instance, recall_at_k=recall)
