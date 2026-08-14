"""Contamination check: per-scenario comparison of no-retrieval vs. hybrid
accuracy. A suspiciously small gap suggests the model may already have that
scenario's answer memorized from pretraining, which would confound any
"retrieval doesn't help" conclusion for that scenario specifically.

Per the project's methodological stance: flag, never silently exclude. The
caller decides what to do with flagged scenarios.
"""
from __future__ import annotations

from dataclasses import dataclass


DEFAULT_SUSPICIOUS_GAP_THRESHOLD = 0.05  # 5 percentage points


@dataclass
class ScenarioContaminationResult:
    scenario: str
    no_retrieval_accuracy: float
    hybrid_accuracy: float
    gap: float  # hybrid - no_retrieval
    n_instances: int
    flagged: bool


def check_contamination(
    no_retrieval_accuracy_by_scenario: dict[str, float],
    hybrid_accuracy_by_scenario: dict[str, float],
    n_instances_by_scenario: dict[str, int],
    suspicious_gap_threshold: float = DEFAULT_SUSPICIOUS_GAP_THRESHOLD,
) -> list[ScenarioContaminationResult]:
    """Compare no-retrieval vs. hybrid accuracy per scenario and flag any
    scenario where hybrid retrieval doesn't improve accuracy by at least
    `suspicious_gap_threshold`.

    A negative gap (hybrid accuracy is LOWER than no-retrieval) is also
    flagged -- that's not evidence of contamination per se, but it's exactly
    as noteworthy and should not be silently treated as "gap is fine because
    it's not small."

    Raises:
        ValueError: if the three dicts don't share the same scenario key set,
            or if suspicious_gap_threshold < 0.
    """
    if suspicious_gap_threshold < 0:
        raise ValueError(f"suspicious_gap_threshold must be >= 0, got {suspicious_gap_threshold}")

    keys_a = set(no_retrieval_accuracy_by_scenario)
    keys_b = set(hybrid_accuracy_by_scenario)
    keys_c = set(n_instances_by_scenario)
    if not (keys_a == keys_b == keys_c):
        raise ValueError(
            "scenario key sets must match across all three inputs; "
            f"no_retrieval={keys_a}, hybrid={keys_b}, n_instances={keys_c}"
        )

    results = []
    for scenario in sorted(keys_a):
        no_ret = no_retrieval_accuracy_by_scenario[scenario]
        hyb = hybrid_accuracy_by_scenario[scenario]
        gap = hyb - no_ret
        flagged = gap < suspicious_gap_threshold
        results.append(
            ScenarioContaminationResult(
                scenario=scenario,
                no_retrieval_accuracy=no_ret,
                hybrid_accuracy=hyb,
                gap=gap,
                n_instances=n_instances_by_scenario[scenario],
                flagged=flagged,
            )
        )
    return results
