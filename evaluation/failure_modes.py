"""Fixed 7-category failure-mode taxonomy, plus rule-based prefiltering.

Only INCORRECT predictions get a failure-mode label. Correct predictions are
labeled CORRECT (not one of the 7 failure categories) so downstream code
(transition matrix) has a uniform per-instance-per-setting label space.

Rule-based prefilter handles what's mechanically determinable without an LLM:
- CORRECT: prediction matches ground truth.
- RETRIEVAL_MISSED_EVIDENCE: incorrect AND no gt evidence doc was in the
  retrieved set (retrieval recall directly tells you this).
- UNDETERMINED: everything else (incorrect, but evidence WAS retrieved, or
  this is a no-retrieval setting where "missed evidence" isn't a meaningful
  category) -- these fall through to the stratified LLM-assisted annotation
  pass described in the proposal.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class FailureMode(str, Enum):
    CORRECT = "correct"
    MISSING_KNOWLEDGE = "missing_factual_cultural_knowledge"
    RETRIEVAL_MISSED_EVIDENCE = "retrieval_missed_evidence"
    EVIDENCE_NOT_USED = "evidence_retrieved_but_not_used"
    VISUAL_GROUNDING_ERROR = "visual_grounding_error"
    OCR_ERROR = "ocr_error"
    HALLUCINATION_DESPITE_EVIDENCE = "hallucination_despite_correct_evidence"
    MULTI_HOP_FAILURE = "multi_hop_reasoning_failure"
    UNDETERMINED = "undetermined"  # rule-prefilter couldn't decide; needs LLM/manual pass


# Failure modes an LLM-assisted annotation pass is responsible for
# distinguishing among (i.e. everything the rule-based prefilter cannot
# mechanically determine).
LLM_ANNOTATED_MODES = frozenset(
    {
        FailureMode.MISSING_KNOWLEDGE,
        FailureMode.EVIDENCE_NOT_USED,
        FailureMode.VISUAL_GROUNDING_ERROR,
        FailureMode.OCR_ERROR,
        FailureMode.HALLUCINATION_DESPITE_EVIDENCE,
        FailureMode.MULTI_HOP_FAILURE,
    }
)


@dataclass
class PrefilterInput:
    instance_id: str
    predicted_choice: str
    ground_truth_choice: str
    is_retrieval_setting: bool  # False for the no-retrieval condition
    evidence_retrieved: bool | None  # None when is_retrieval_setting is False


def rule_based_prefilter(inp: PrefilterInput) -> FailureMode:
    """Mechanically determine CORRECT / RETRIEVAL_MISSED_EVIDENCE / UNDETERMINED.

    Raises:
        ValueError: if is_retrieval_setting is True but evidence_retrieved is
            None (that combination is a caller bug -- retrieval recall must
            be computed for any retrieval setting).
    """
    if inp.predicted_choice == inp.ground_truth_choice:
        return FailureMode.CORRECT

    if inp.is_retrieval_setting:
        if inp.evidence_retrieved is None:
            raise ValueError(
                "evidence_retrieved must be set (True/False) when is_retrieval_setting=True"
            )
        if not inp.evidence_retrieved:
            return FailureMode.RETRIEVAL_MISSED_EVIDENCE

    return FailureMode.UNDETERMINED
