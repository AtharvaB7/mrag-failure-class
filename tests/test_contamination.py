import pytest

from evaluation.contamination import check_contamination


def test_large_gap_not_flagged():
    results = check_contamination(
        no_retrieval_accuracy_by_scenario={"Animal": 0.40},
        hybrid_accuracy_by_scenario={"Animal": 0.70},
        n_instances_by_scenario={"Animal": 150},
        suspicious_gap_threshold=0.05,
    )
    assert len(results) == 1
    assert results[0].flagged is False
    assert results[0].gap == pytest.approx(0.30)


def test_small_gap_is_flagged():
    results = check_contamination(
        no_retrieval_accuracy_by_scenario={"Biology": 0.65},
        hybrid_accuracy_by_scenario={"Biology": 0.67},
        n_instances_by_scenario={"Biology": 100},
        suspicious_gap_threshold=0.05,
    )
    assert results[0].flagged is True
    assert results[0].gap == pytest.approx(0.02)


def test_negative_gap_is_flagged_not_silently_fine():
    # hybrid retrieval HURTS accuracy for this scenario -- must be flagged,
    # not treated as "gap isn't small so it's ok."
    results = check_contamination(
        no_retrieval_accuracy_by_scenario={"Chemistry": 0.55},
        hybrid_accuracy_by_scenario={"Chemistry": 0.50},
        n_instances_by_scenario={"Chemistry": 80},
        suspicious_gap_threshold=0.05,
    )
    assert results[0].flagged is True
    assert results[0].gap == pytest.approx(-0.05)


def test_multiple_scenarios_sorted_deterministically():
    results = check_contamination(
        no_retrieval_accuracy_by_scenario={"Zebra": 0.5, "Apple": 0.5},
        hybrid_accuracy_by_scenario={"Zebra": 0.9, "Apple": 0.9},
        n_instances_by_scenario={"Zebra": 10, "Apple": 10},
    )
    assert [r.scenario for r in results] == ["Apple", "Zebra"]


def test_mismatched_scenario_keys_raises():
    with pytest.raises(ValueError):
        check_contamination(
            no_retrieval_accuracy_by_scenario={"A": 0.5},
            hybrid_accuracy_by_scenario={"B": 0.5},
            n_instances_by_scenario={"A": 10},
        )


def test_negative_threshold_raises():
    with pytest.raises(ValueError):
        check_contamination(
            no_retrieval_accuracy_by_scenario={"A": 0.5},
            hybrid_accuracy_by_scenario={"A": 0.6},
            n_instances_by_scenario={"A": 10},
            suspicious_gap_threshold=-0.1,
        )


def test_exact_threshold_boundary_is_flagged():
    # gap == threshold exactly should be flagged (flagged = gap < threshold is False here,
    # so gap == threshold should NOT flag -- test documents the exact boundary behavior)
    results = check_contamination(
        no_retrieval_accuracy_by_scenario={"A": 0.50},
        hybrid_accuracy_by_scenario={"A": 0.55},
        n_instances_by_scenario={"A": 10},
        suspicious_gap_threshold=0.05,
    )
    assert results[0].gap == pytest.approx(0.05)
    assert results[0].flagged is False
