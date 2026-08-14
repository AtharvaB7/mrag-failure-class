import pytest

from retrieval.fusion import RankedResult, reciprocal_rank_fusion, top_k_ids


def test_single_list_preserves_order():
    fused = reciprocal_rank_fusion([["a", "b", "c"]])
    assert [r.doc_id for r in fused] == ["a", "b", "c"]


def test_agreement_boosts_score_above_disagreement():
    # doc "x" is top-ranked in BOTH lists; doc "y" is top in only one.
    # x's fused RRF score must exceed y's.
    list1 = ["x", "y", "z"]
    list2 = ["x", "z", "y"]
    fused = reciprocal_rank_fusion([list1, list2])
    scores = {r.doc_id: r.score for r in fused}
    assert scores["x"] > scores["y"]
    assert scores["x"] > scores["z"]


def test_doc_in_only_one_list_still_included():
    list1 = ["a", "b"]
    list2 = ["c", "d"]
    fused = reciprocal_rank_fusion([list1, list2])
    ids = {r.doc_id for r in fused}
    assert ids == {"a", "b", "c", "d"}


def test_disjoint_equal_rank_docs_tie_and_are_deterministic():
    # a is rank 1 in list1 only, c is rank 1 in list2 only -> equal RRF score.
    # Tie-break falls to best_rank (both rank 1) then string doc_id.
    fused1 = reciprocal_rank_fusion([["a", "b"], ["c", "d"]])
    fused2 = reciprocal_rank_fusion([["a", "b"], ["c", "d"]])
    assert [r.doc_id for r in fused1] == [r.doc_id for r in fused2]
    assert fused1[0].doc_id == "a"  # "a" < "c" lexicographically


def test_weights_scale_contribution():
    # Weighting list2 heavily should let its top doc win over list1's top doc
    # even though list1 ranks its doc first.
    list1 = ["a", "b"]
    list2 = ["b", "a"]
    fused_unweighted = reciprocal_rank_fusion([list1, list2])
    assert fused_unweighted[0].doc_id == "a"  # tie broken by best_rank/str, a==rank1 in list1

    fused_weighted = reciprocal_rank_fusion([list1, list2], weights=[0.01, 10.0])
    assert fused_weighted[0].doc_id == "b"


def test_k_must_be_positive():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"]], k=0)
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"]], k=-5)


def test_weights_length_mismatch_raises():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"], ["b"]], weights=[1.0])


def test_duplicate_doc_id_in_single_list_uses_best_rank():
    # If "a" appears twice in one list (shouldn't normally happen but code
    # should be robust), the first (best) occurrence's rank should be used.
    fused = reciprocal_rank_fusion([["a", "b", "a"]])
    scores = {r.doc_id: r.score for r in fused}
    from retrieval.fusion import DEFAULT_K

    assert scores["a"] == pytest.approx(1.0 / (DEFAULT_K + 1))


def test_top_k_ids_truncates():
    fused = [RankedResult("a", 0.9), RankedResult("b", 0.5), RankedResult("c", 0.1)]
    assert top_k_ids(fused, 2) == ["a", "b"]


def test_known_rrf_value_matches_hand_computation():
    # doc "a": rank 1 in list1, rank 2 in list2. k=60.
    # expected score = 1/(60+1) + 1/(60+2)
    fused = reciprocal_rank_fusion([["a", "z"], ["y", "a"]], k=60)
    scores = {r.doc_id: r.score for r in fused}
    expected = 1 / 61 + 1 / 62
    assert scores["a"] == pytest.approx(expected)
