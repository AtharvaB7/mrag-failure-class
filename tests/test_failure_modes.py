import pytest

from evaluation.failure_modes import (
    FailureMode,
    LLM_ANNOTATED_MODES,
    PrefilterInput,
    rule_based_prefilter,
)


def test_correct_prediction_labeled_correct_regardless_of_setting():
    inp = PrefilterInput(
        instance_id="q1",
        predicted_choice="B",
        ground_truth_choice="B",
        is_retrieval_setting=True,
        evidence_retrieved=False,  # shouldn't matter, prediction was right
    )
    assert rule_based_prefilter(inp) == FailureMode.CORRECT


def test_incorrect_no_retrieval_setting_is_undetermined():
    inp = PrefilterInput(
        instance_id="q1",
        predicted_choice="A",
        ground_truth_choice="B",
        is_retrieval_setting=False,
        evidence_retrieved=None,
    )
    assert rule_based_prefilter(inp) == FailureMode.UNDETERMINED


def test_incorrect_retrieval_setting_evidence_missed():
    inp = PrefilterInput(
        instance_id="q1",
        predicted_choice="A",
        ground_truth_choice="B",
        is_retrieval_setting=True,
        evidence_retrieved=False,
    )
    assert rule_based_prefilter(inp) == FailureMode.RETRIEVAL_MISSED_EVIDENCE


def test_incorrect_retrieval_setting_evidence_present_is_undetermined():
    # Evidence WAS retrieved but the model still got it wrong -- rule-based
    # prefilter can't distinguish evidence-not-used vs. hallucination vs.
    # visual-grounding vs. multi-hop; that needs the LLM/manual pass.
    inp = PrefilterInput(
        instance_id="q1",
        predicted_choice="A",
        ground_truth_choice="B",
        is_retrieval_setting=True,
        evidence_retrieved=True,
    )
    assert rule_based_prefilter(inp) == FailureMode.UNDETERMINED


def test_retrieval_setting_without_evidence_flag_raises():
    inp = PrefilterInput(
        instance_id="q1",
        predicted_choice="A",
        ground_truth_choice="B",
        is_retrieval_setting=True,
        evidence_retrieved=None,
    )
    with pytest.raises(ValueError):
        rule_based_prefilter(inp)


def test_llm_annotated_modes_excludes_mechanically_determined_ones():
    assert FailureMode.CORRECT not in LLM_ANNOTATED_MODES
    assert FailureMode.RETRIEVAL_MISSED_EVIDENCE not in LLM_ANNOTATED_MODES
    assert FailureMode.UNDETERMINED not in LLM_ANNOTATED_MODES
    assert FailureMode.VISUAL_GROUNDING_ERROR in LLM_ANNOTATED_MODES
    assert FailureMode.HALLUCINATION_DESPITE_EVIDENCE in LLM_ANNOTATED_MODES


def test_all_seven_failure_categories_from_proposal_present():
    expected = {
        FailureMode.MISSING_KNOWLEDGE,
        FailureMode.RETRIEVAL_MISSED_EVIDENCE,
        FailureMode.EVIDENCE_NOT_USED,
        FailureMode.VISUAL_GROUNDING_ERROR,
        FailureMode.OCR_ERROR,
        FailureMode.HALLUCINATION_DESPITE_EVIDENCE,
        FailureMode.MULTI_HOP_FAILURE,
    }
    assert expected.issubset(set(FailureMode))
