import pytest

from evaluation.failure_modes import FailureMode
from evaluation.transition_matrix import (
    InstanceLabels,
    SETTINGS_ORDER,
    build_transition_matrices,
    summarize_conversions,
)


FM = FailureMode


def _labels(instance_id, no_ret, sparse, dense, hybrid):
    return InstanceLabels(
        instance_id=instance_id,
        labels={
            "no_retrieval": no_ret,
            "sparse": sparse,
            "dense": dense,
            "hybrid": hybrid,
        },
    )


def test_instance_labels_requires_all_four_settings():
    with pytest.raises(ValueError):
        InstanceLabels(instance_id="q1", labels={"no_retrieval": FM.CORRECT})


def test_instance_labels_rejects_unknown_setting():
    with pytest.raises(ValueError):
        InstanceLabels(
            instance_id="q1",
            labels={
                "no_retrieval": FM.CORRECT,
                "sparse": FM.CORRECT,
                "dense": FM.CORRECT,
                "hybrid": FM.CORRECT,
                "extra_setting": FM.CORRECT,
            },
        )


def test_settings_order_matches_proposal():
    assert SETTINGS_ORDER == ("no_retrieval", "sparse", "dense", "hybrid")


def test_build_transition_matrices_produces_three_adjacent_steps():
    instances = [
        _labels("q1", FM.MISSING_KNOWLEDGE, FM.CORRECT, FM.CORRECT, FM.CORRECT),
    ]
    result = build_transition_matrices("qwen2vl", instances)
    assert set(result.transitions.keys()) == {
        ("no_retrieval", "sparse"),
        ("sparse", "dense"),
        ("dense", "hybrid"),
    }
    assert result.model_name == "qwen2vl"


def test_missing_knowledge_converts_to_correct_tracked_correctly():
    # This is the exact pattern hypothesized in "Expected Outcomes":
    # retrieval converts missing-knowledge errors into successes.
    instances = [
        _labels("q1", FM.MISSING_KNOWLEDGE, FM.CORRECT, FM.CORRECT, FM.CORRECT),
        _labels("q2", FM.MISSING_KNOWLEDGE, FM.MISSING_KNOWLEDGE, FM.CORRECT, FM.CORRECT),
    ]
    result = build_transition_matrices("qwen2vl", instances)
    step1 = result.transitions[("no_retrieval", "sparse")]
    assert step1.counts[(FM.MISSING_KNOWLEDGE, FM.CORRECT)] == 1
    assert step1.counts[(FM.MISSING_KNOWLEDGE, FM.MISSING_KNOWLEDGE)] == 1
    assert step1.total() == 2


def test_visual_grounding_error_untouched_by_retrieval():
    instances = [
        _labels(
            "q1",
            FM.VISUAL_GROUNDING_ERROR,
            FM.VISUAL_GROUNDING_ERROR,
            FM.VISUAL_GROUNDING_ERROR,
            FM.VISUAL_GROUNDING_ERROR,
        )
    ]
    result = build_transition_matrices("qwen2vl", instances)
    for step in result.transitions.values():
        assert step.total() == 1
        # every step: stayed exactly the same failure mode
        assert step.counts[(FM.VISUAL_GROUNDING_ERROR, FM.VISUAL_GROUNDING_ERROR)] == 1


def test_retrieval_can_introduce_new_hallucination_failure():
    # RQ3: retrieval makes a failure WORSE / different -- correct with no
    # retrieval, but hallucinates once (misleading) evidence is retrieved.
    instances = [
        _labels("q1", FM.CORRECT, FM.HALLUCINATION_DESPITE_EVIDENCE, FM.CORRECT, FM.CORRECT)
    ]
    result = build_transition_matrices("qwen2vl", instances)
    step1 = result.transitions[("no_retrieval", "sparse")]
    assert step1.counts[(FM.CORRECT, FM.HALLUCINATION_DESPITE_EVIDENCE)] == 1


def test_as_dense_matrix_shape_and_indexing():
    instances = [_labels("q1", FM.CORRECT, FM.CORRECT, FM.CORRECT, FM.CORRECT)]
    result = build_transition_matrices("qwen2vl", instances)
    step1 = result.transitions[("no_retrieval", "sparse")]
    labels, matrix = step1.as_dense_matrix()
    n = len(labels)
    assert len(matrix) == n
    assert all(len(row) == n for row in matrix)
    correct_idx = labels.index(FM.CORRECT)
    assert matrix[correct_idx][correct_idx] == 1


def test_empty_instance_list_raises():
    with pytest.raises(ValueError):
        build_transition_matrices("qwen2vl", [])


def test_duplicate_instance_id_raises():
    instances = [
        _labels("q1", FM.CORRECT, FM.CORRECT, FM.CORRECT, FM.CORRECT),
        _labels("q1", FM.CORRECT, FM.CORRECT, FM.CORRECT, FM.CORRECT),
    ]
    with pytest.raises(ValueError):
        build_transition_matrices("qwen2vl", instances)


def test_matrices_built_separately_per_model_not_pooled():
    # Calling build_transition_matrices twice with different model_name and
    # DIFFERENT instance patterns must not let one model's counts leak into
    # the other's -- this directly encodes the "never pooled" requirement.
    qwen_instances = [
        _labels("q1", FM.MISSING_KNOWLEDGE, FM.CORRECT, FM.CORRECT, FM.CORRECT)
    ]
    llava_instances = [
        _labels(
            "q1",
            FM.VISUAL_GROUNDING_ERROR,
            FM.VISUAL_GROUNDING_ERROR,
            FM.VISUAL_GROUNDING_ERROR,
            FM.VISUAL_GROUNDING_ERROR,
        )
    ]
    qwen_result = build_transition_matrices("qwen2vl", qwen_instances)
    llava_result = build_transition_matrices("llava_next", llava_instances)

    qwen_step1 = qwen_result.transitions[("no_retrieval", "sparse")]
    llava_step1 = llava_result.transitions[("no_retrieval", "sparse")]

    assert qwen_step1.counts[(FM.MISSING_KNOWLEDGE, FM.CORRECT)] == 1
    assert (FM.MISSING_KNOWLEDGE, FM.CORRECT) not in llava_step1.counts
    assert llava_step1.counts[(FM.VISUAL_GROUNDING_ERROR, FM.VISUAL_GROUNDING_ERROR)] == 1


def test_summarize_conversions_categorizes_correctly():
    instances = [
        _labels("q1", FM.CORRECT, FM.CORRECT, FM.CORRECT, FM.CORRECT),  # stayed_correct
        _labels(
            "q2", FM.MISSING_KNOWLEDGE, FM.CORRECT, FM.CORRECT, FM.CORRECT
        ),  # became_correct
        _labels("q3", FM.CORRECT, FM.HALLUCINATION_DESPITE_EVIDENCE, FM.CORRECT, FM.CORRECT),  # became_incorrect
        _labels(
            "q4",
            FM.VISUAL_GROUNDING_ERROR,
            FM.VISUAL_GROUNDING_ERROR,
            FM.CORRECT,
            FM.CORRECT,
        ),  # stayed_same_failure
        _labels(
            "q5",
            FM.MISSING_KNOWLEDGE,
            FM.HALLUCINATION_DESPITE_EVIDENCE,
            FM.CORRECT,
            FM.CORRECT,
        ),  # changed_failure_type
    ]
    result = build_transition_matrices("qwen2vl", instances)
    summary = summarize_conversions(result.transitions[("no_retrieval", "sparse")])
    assert summary == {
        "stayed_correct": 1,
        "became_correct": 1,
        "became_incorrect": 1,
        "stayed_same_failure": 1,
        "changed_failure_type": 1,
    }
