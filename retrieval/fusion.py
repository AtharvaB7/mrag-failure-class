"""Reciprocal Rank Fusion (RRF) for combining sparse and dense retrieval rankings.

RRF score for a document d given a set of ranked lists R:
    score(d) = sum_{r in R} 1 / (k + rank_r(d))
where rank_r(d) is the 1-indexed rank of d in ranked list r (documents not
present in a given list simply don't contribute a term for that list).

Reference: Cormack, Clarke, Buettcher (2009), "Reciprocal Rank Fusion
Outperforms Condorcet and Individual Rank Learning Methods."
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Sequence


DEFAULT_K = 60  # standard RRF constant from the original paper


@dataclass(frozen=True)
class RankedResult:
    doc_id: Hashable
    score: float


def _ranks_from_list(ranked_ids: Sequence[Hashable]) -> dict[Hashable, int]:
    """Map doc_id -> 1-indexed rank from an ordered (best-first) id sequence.

    If a doc_id appears more than once, the first (best) occurrence wins.
    """
    ranks: dict[Hashable, int] = {}
    for i, doc_id in enumerate(ranked_ids):
        if doc_id not in ranks:
            ranks[doc_id] = i + 1
    return ranks


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[Hashable]],
    k: int = DEFAULT_K,
    weights: Sequence[float] | None = None,
) -> list[RankedResult]:
    """Fuse multiple ranked lists of doc_ids into a single ranked list via RRF.

    Args:
        ranked_lists: each element is a best-first sequence of doc_ids from one
            retriever (e.g. [sparse_ranked_ids, dense_ranked_ids]).
        k: RRF constant (default 60, the standard value from the original paper).
        weights: optional per-list weight multiplier, same length as ranked_lists.
            Defaults to all-ones (unweighted RRF).

    Returns:
        List of RankedResult sorted by descending fused score. Ties are broken
        by the doc_id's best (lowest) rank across all lists, then by doc_id
        string representation, for deterministic output.

    Raises:
        ValueError: if k <= 0, or weights length mismatches ranked_lists length.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if weights is not None and len(weights) != len(ranked_lists):
        raise ValueError(
            f"weights length ({len(weights)}) must match ranked_lists length "
            f"({len(ranked_lists)})"
        )
    if weights is None:
        weights = [1.0] * len(ranked_lists)

    fused_scores: dict[Hashable, float] = {}
    best_rank: dict[Hashable, int] = {}

    for ranked_ids, weight in zip(ranked_lists, weights):
        ranks = _ranks_from_list(ranked_ids)
        for doc_id, rank in ranks.items():
            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + weight * (
                1.0 / (k + rank)
            )
            if doc_id not in best_rank or rank < best_rank[doc_id]:
                best_rank[doc_id] = rank

    results = [RankedResult(doc_id, score) for doc_id, score in fused_scores.items()]
    results.sort(key=lambda r: (-r.score, best_rank[r.doc_id], str(r.doc_id)))
    return results


def top_k_ids(fused: Sequence[RankedResult], k: int) -> list[Hashable]:
    """Convenience: extract just the top-k doc_ids from a fused result list."""
    return [r.doc_id for r in fused[:k]]
