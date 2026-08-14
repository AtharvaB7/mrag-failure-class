"""Failure-mode transition matrix -- the project's main result.

Tracks, instance by instance, how the failure-mode label changes as the
retrieval setting moves through the ordered sequence
no_retrieval -> sparse -> dense -> hybrid.

Built SEPARATELY per VLM (never pooled), since it's an open research question
(RQ4) whether failure-mode shifts are model-dependent.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Sequence

from evaluation.failure_modes import FailureMode


SETTINGS_ORDER = ("no_retrieval", "sparse", "dense", "hybrid")


@dataclass
class InstanceLabels:
    """Failure-mode label for one instance across all 4 settings, for one VLM."""

    instance_id: str
    labels: dict[str, FailureMode]  # setting name -> label

    def __post_init__(self) -> None:
        missing = set(SETTINGS_ORDER) - set(self.labels)
        if missing:
            raise ValueError(
                f"InstanceLabels for {self.instance_id!r} missing settings: {sorted(missing)}"
            )
        extra = set(self.labels) - set(SETTINGS_ORDER)
        if extra:
            raise ValueError(
                f"InstanceLabels for {self.instance_id!r} has unknown settings: {sorted(extra)}"
            )


@dataclass
class TransitionMatrix:
    """Transition counts for ONE (from_setting, to_setting) pair adjacent in
    SETTINGS_ORDER, i.e. one step of no_retrieval->sparse, sparse->dense, or
    dense->hybrid.
    """

    from_setting: str
    to_setting: str
    counts: dict[tuple[FailureMode, FailureMode], int] = field(default_factory=Counter)

    def total(self) -> int:
        return sum(self.counts.values())

    def as_dense_matrix(self) -> tuple[list[FailureMode], list[list[int]]]:
        """Return (ordered_labels, matrix) where matrix[i][j] = count of
        instances that were ordered_labels[i] at from_setting and
        ordered_labels[j] at to_setting. Labels are ordered by FailureMode
        enum definition order for reproducible output.
        """
        labels = list(FailureMode)
        idx = {label: i for i, label in enumerate(labels)}
        matrix = [[0] * len(labels) for _ in labels]
        for (from_label, to_label), count in self.counts.items():
            matrix[idx[from_label]][idx[to_label]] = count
        return labels, matrix


@dataclass
class ModelTransitionMatrices:
    """All 3 adjacent-step transition matrices for one VLM."""

    model_name: str
    transitions: dict[tuple[str, str], TransitionMatrix]


def build_transition_matrices(
    model_name: str, instances: Sequence[InstanceLabels]
) -> ModelTransitionMatrices:
    """Build the 3 adjacent-step transition matrices
    (no_retrieval->sparse, sparse->dense, dense->hybrid) for one VLM from a
    sequence of per-instance labels across all 4 settings.

    Raises:
        ValueError: if instances is empty, or contains duplicate instance_ids.
    """
    if not instances:
        raise ValueError("Cannot build transition matrices from an empty instance list")

    seen_ids = set()
    for inst in instances:
        if inst.instance_id in seen_ids:
            raise ValueError(f"Duplicate instance_id in input: {inst.instance_id!r}")
        seen_ids.add(inst.instance_id)

    transitions: dict[tuple[str, str], TransitionMatrix] = {}
    for from_setting, to_setting in zip(SETTINGS_ORDER, SETTINGS_ORDER[1:]):
        tm = TransitionMatrix(from_setting=from_setting, to_setting=to_setting)
        for inst in instances:
            from_label = inst.labels[from_setting]
            to_label = inst.labels[to_setting]
            tm.counts[(from_label, to_label)] += 1
        transitions[(from_setting, to_setting)] = tm

    return ModelTransitionMatrices(model_name=model_name, transitions=transitions)


def summarize_conversions(
    tm: TransitionMatrix,
) -> dict[str, int]:
    """Convenience summary of a single transition step:
    - stayed_correct, stayed_same_failure, became_correct (fixed by retrieval),
      became_incorrect (retrieval broke something that was working),
      changed_failure_type (was one failure mode, now a different one).
    """
    stayed_correct = 0
    became_correct = 0
    became_incorrect = 0
    stayed_same_failure = 0
    changed_failure_type = 0

    for (from_label, to_label), count in tm.counts.items():
        from_correct = from_label == FailureMode.CORRECT
        to_correct = to_label == FailureMode.CORRECT
        if from_correct and to_correct:
            stayed_correct += count
        elif from_correct and not to_correct:
            became_incorrect += count
        elif not from_correct and to_correct:
            became_correct += count
        else:
            if from_label == to_label:
                stayed_same_failure += count
            else:
                changed_failure_type += count

    return {
        "stayed_correct": stayed_correct,
        "became_correct": became_correct,
        "became_incorrect": became_incorrect,
        "stayed_same_failure": stayed_same_failure,
        "changed_failure_type": changed_failure_type,
    }
