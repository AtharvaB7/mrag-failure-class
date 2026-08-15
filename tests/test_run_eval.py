import json
from pathlib import Path

import pytest
from PIL import Image
from omegaconf import OmegaConf

from scripts.run_eval import (
    _load_completed_ids,
    _rewrite_dropping_corrupt_lines,
    build_retrievers,
    load_cached_instances,
)


def _make_cache_dir(tmp_path, extra_records=None):
    cache_dir = tmp_path / "cache"
    (cache_dir / "corpus").mkdir(parents=True)
    (cache_dir / "query_images").mkdir(parents=True)

    Image.new("RGB", (8, 8), (1, 1, 1)).save(cache_dir / "corpus" / "hashA.png")
    Image.new("RGB", (8, 8), (2, 2, 2)).save(cache_dir / "corpus" / "hashB.png")
    Image.new("RGB", (8, 8), (3, 3, 3)).save(cache_dir / "query_images" / "q1.png")
    Image.new("RGB", (8, 8), (4, 4, 4)).save(cache_dir / "query_images" / "q2.png")

    records = [
        {
            "id": "q1", "aspect": "Scope", "scenario": "Animal",
            "question": "What animal?", "choices": {"A": "a", "B": "b", "C": "c", "D": "d"},
            "answer_choice": "A", "query_image_path": "query_images/q1.png",
            "gt_image_hashes": ["hashA"],
        },
        {
            "id": "q2", "aspect": "Scope", "scenario": "Animal",
            "question": "What breed?", "choices": {"A": "a", "B": "b", "C": "c", "D": "d"},
            "answer_choice": "B", "query_image_path": "query_images/q2.png",
            "gt_image_hashes": ["hashB"],
        },
    ]
    with open(cache_dir / "instances.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return cache_dir, records


def test_load_cached_instances_reads_all_records(tmp_path):
    cache_dir, records = _make_cache_dir(tmp_path)
    loaded = load_cached_instances(cache_dir)
    assert len(loaded) == 2
    assert loaded[0]["id"] == "q1"
    assert loaded[1]["id"] == "q2"


def test_load_cached_instances_respects_limit(tmp_path):
    cache_dir, records = _make_cache_dir(tmp_path)
    loaded = load_cached_instances(cache_dir, limit=1)
    assert len(loaded) == 1
    assert loaded[0]["id"] == "q1"


def test_build_retrievers_sparse_only_works_even_if_corpus_images_missing(tmp_path):
    # Regression test for a previously-reported bug: a sparse-only run must
    # not require corpus image FILES to exist on disk -- only the text
    # surrogate (question text) built from instances.jsonl, which never
    # touches the corpus/ directory at all.
    cache_dir, records = _make_cache_dir(tmp_path)
    import shutil

    shutil.rmtree(cache_dir / "corpus")  # simulate missing/incomplete corpus images
    cfg = OmegaConf.create({"sparse_enabled": True, "dense_enabled": False})
    sparse, dense = build_retrievers(cfg, cache_dir, records)
    assert sparse is not None
    assert dense is None
    results = sparse.query("What animal is this?")
    assert set(results) == {"hashA", "hashB"}


def test_build_retrievers_dense_with_missing_corpus_image_raises_with_details(tmp_path):
    cache_dir, records = _make_cache_dir(tmp_path)
    (cache_dir / "corpus" / "hashB.png").unlink()  # simulate one missing file
    cfg = OmegaConf.create(
        {
            "sparse_enabled": False,
            "dense_enabled": True,
            "image_encoder": {"hf_id": "fake", "batch_size": 2},
        }
    )
    with pytest.raises(FileNotFoundError, match="hashB"):
        build_retrievers(cfg, cache_dir, records)


def test_build_retrievers_dense_does_not_pass_batch_size_to_encoder_constructor(tmp_path, monkeypatch):
    # Regression test for a real TypeError hit in production: build_retrievers
    # was passing batch_size= into SiglipImageEncoder's constructor, which
    # doesn't accept it (batch_size belongs to encode_images(), invoked via
    # DenseRetriever). Confirm the constructor is called with only hf_id.
    cache_dir, records = _make_cache_dir(tmp_path)

    captured_kwargs = {}

    class FakeEncoder:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)

        def encode_images(self, images, batch_size=32):
            import numpy as np

            return np.zeros((len(images), 4))

    import retrieval.encoders.siglip as siglip_module

    monkeypatch.setattr(siglip_module, "SiglipImageEncoder", FakeEncoder)

    cfg = OmegaConf.create(
        {
            "sparse_enabled": False,
            "dense_enabled": True,
            "image_encoder": {"hf_id": "fake-id", "batch_size": 2},
        }
    )
    build_retrievers(cfg, cache_dir, records)
    assert captured_kwargs == {"hf_id": "fake-id"}


def test_build_retrievers_no_retrieval_returns_none_none(tmp_path):
    cache_dir, records = _make_cache_dir(tmp_path)
    cfg = OmegaConf.create({"sparse_enabled": False, "dense_enabled": False})
    sparse, dense = build_retrievers(cfg, cache_dir, records)
    assert sparse is None
    assert dense is None


def test_build_retrievers_sparse_only_builds_bm25_over_corpus(tmp_path):
    cache_dir, records = _make_cache_dir(tmp_path)
    cfg = OmegaConf.create({"sparse_enabled": True, "dense_enabled": False})
    sparse, dense = build_retrievers(cfg, cache_dir, records)
    assert sparse is not None
    assert dense is None
    results = sparse.query("What animal is this? What breed?")
    assert set(results) == {"hashA", "hashB"}


def test_build_retrievers_corpus_text_uses_question_not_scenario_label(tmp_path):
    # Regression test for a real bug found via production predictions: using
    # f"{scenario} {aspect}" (e.g. "Animal Scope") as the corpus surrogate
    # text shares almost no vocabulary with real questions, so every BM25
    # score was 0.0 and retrieval silently returned the SAME doc_ids for
    # every query regardless of content (confirmed: retrieved_doc_ids was
    # byte-identical across all 1353 real instances in one run). This test
    # asserts retrieval actually differentiates between two distinguishable
    # queries once real question text is used as the surrogate.
    #
    # NOTE: uses a 6-doc corpus, not the 2-doc default from _make_cache_dir --
    # BM25's IDF term log((N - n + 0.5) / (n + 0.5)) is exactly 0 when a term
    # appears in precisely half the corpus (see test_sparse.py), which a
    # tiny 2-doc corpus triggers by construction and would make this test
    # spuriously fail regardless of the surrogate-text fix being correct.
    cache_dir, records = _make_cache_dir(tmp_path)
    extra_hashes = ["hashC", "hashD", "hashE", "hashF"]
    for h in extra_hashes:
        Image.new("RGB", (8, 8), (9, 9, 9)).save(cache_dir / "corpus" / f"{h}.png")
    for i, h in enumerate(extra_hashes):
        records.append(
            {
                "id": f"filler{i}", "aspect": "Scope", "scenario": "Animal",
                "question": "totally unrelated filler content about nothing",
                "choices": {"A": "a", "B": "b", "C": "c", "D": "d"},
                "answer_choice": "A", "query_image_path": "query_images/q1.png",
                "gt_image_hashes": [h],
            }
        )

    cfg = OmegaConf.create({"sparse_enabled": True, "dense_enabled": False})
    sparse, _ = build_retrievers(cfg, cache_dir, records)

    results_for_q1_text = sparse.query("What animal?")
    results_for_q2_text = sparse.query("What breed?")
    assert results_for_q1_text[0] == "hashA"
    assert results_for_q2_text[0] == "hashB"


def test_build_retrievers_corpus_text_aggregates_multiple_questions_per_hash(tmp_path):
    # When multiple instances share a gt_image, its surrogate text should
    # include all of their distinct questions (deduplicated), not just one.
    cache_dir, records = _make_cache_dir(tmp_path)
    records.append(
        {
            "id": "q3", "aspect": "Scope", "scenario": "Animal",
            "question": "What color is it?", "choices": {"A": "a", "B": "b", "C": "c", "D": "d"},
            "answer_choice": "C", "query_image_path": "query_images/q1.png",
            "gt_image_hashes": ["hashA"],  # shares hashA with q1
        }
    )
    cfg = OmegaConf.create({"sparse_enabled": True, "dense_enabled": False})
    sparse, _ = build_retrievers(cfg, cache_dir, records)
    hasha_doc = next(d for d in sparse.docs if d.doc_id == "hashA")
    assert "animal" in hasha_doc.text.lower()
    assert "color" in hasha_doc.text.lower()


def test_build_retrievers_corpus_deduplicates_shared_hashes(tmp_path):
    cache_dir, records = _make_cache_dir(tmp_path)
    # add a third instance that shares hashA
    records.append(
        {
            "id": "q3", "aspect": "Scope", "scenario": "Animal",
            "question": "Another question", "choices": {"A": "a", "B": "b", "C": "c", "D": "d"},
            "answer_choice": "C", "query_image_path": "query_images/q1.png",
            "gt_image_hashes": ["hashA"],
        }
    )
    cfg = OmegaConf.create({"sparse_enabled": True, "dense_enabled": False})
    sparse, _ = build_retrievers(cfg, cache_dir, records)
    # hashA should appear exactly once in the corpus despite 2 instances referencing it
    assert len(sparse.docs) == 2


# --- resume / crash-tolerance tests -----------------------------------

def test_load_completed_ids_missing_file_returns_empty_set(tmp_path):
    assert _load_completed_ids(tmp_path / "does_not_exist.jsonl") == set()


def test_load_completed_ids_reads_valid_ids(tmp_path):
    path = tmp_path / "predictions.jsonl"
    with open(path, "w") as f:
        f.write(json.dumps({"id": "q1", "correct": True}) + "\n")
        f.write(json.dumps({"id": "q2", "correct": False}) + "\n")
    assert _load_completed_ids(path) == {"q1", "q2"}


def test_load_completed_ids_skips_corrupt_lines(tmp_path, capsys):
    path = tmp_path / "predictions.jsonl"
    with open(path, "w") as f:
        f.write(json.dumps({"id": "q1", "correct": True}) + "\n")
        f.write("\x00" * 50 + "\n")  # simulated corruption
        f.write(json.dumps({"id": "q2", "correct": False}) + "\n")
    completed = _load_completed_ids(path)
    assert completed == {"q1", "q2"}
    assert "1 corrupted" in capsys.readouterr().out


def test_load_completed_ids_skips_blank_lines(tmp_path):
    path = tmp_path / "predictions.jsonl"
    with open(path, "w") as f:
        f.write(json.dumps({"id": "q1", "correct": True}) + "\n")
        f.write("\n")
        f.write(json.dumps({"id": "q2", "correct": False}) + "\n")
    assert _load_completed_ids(path) == {"q1", "q2"}


def test_rewrite_dropping_corrupt_lines_removes_only_bad_lines(tmp_path):
    path = tmp_path / "predictions.jsonl"
    good1 = json.dumps({"id": "q1", "correct": True})
    good2 = json.dumps({"id": "q2", "correct": False})
    with open(path, "w") as f:
        f.write(good1 + "\n")
        f.write("\x00" * 50 + "\n")
        f.write(good2 + "\n")

    _rewrite_dropping_corrupt_lines(path)

    lines = path.read_text().strip().split("\n")
    assert lines == [good1, good2]


def test_rewrite_dropping_corrupt_lines_missing_file_is_a_noop(tmp_path):
    path = tmp_path / "does_not_exist.jsonl"
    _rewrite_dropping_corrupt_lines(path)  # must not raise
    assert not path.exists()


def test_rewrite_dropping_corrupt_lines_all_valid_unchanged(tmp_path):
    path = tmp_path / "predictions.jsonl"
    good1 = json.dumps({"id": "q1"})
    good2 = json.dumps({"id": "q2"})
    path.write_text(good1 + "\n" + good2 + "\n")
    _rewrite_dropping_corrupt_lines(path)
    assert path.read_text().strip().split("\n") == [good1, good2]


# --- run_predictions_loop integration tests (fake model, no GPU) -------

class _FakeModel:
    """Deterministic fake VLM: always predicts the ground-truth answer,
    unless instance.id is in fail_on_ids, in which case it raises to
    simulate a transient generation failure.
    """

    def __init__(self, fail_on_ids: set[str] = frozenset()):
        self.fail_on_ids = fail_on_ids
        self.calls = []

    def answer(self, instance, retrieved_images):
        self.calls.append(instance.id)
        if instance.id in self.fail_on_ids:
            raise RuntimeError(f"simulated failure on {instance.id}")
        return instance.answer_choice


def test_run_predictions_loop_writes_all_records_and_correct_accuracy(tmp_path):
    from scripts.run_eval import run_predictions_loop

    cache_dir, records = _make_cache_dir(tmp_path)
    predictions_path = tmp_path / "predictions.jsonl"
    errors_path = tmp_path / "errors.jsonl"
    model = _FakeModel()
    cfg = OmegaConf.create({})

    n_errors, all_recs = run_predictions_loop(
        records=records,
        remaining_records=records,
        cache_dir=cache_dir,
        sparse=None,
        dense=None,
        model=model,
        retrieval_cfg=cfg,
        top_k=5,
        predictions_path=predictions_path,
        errors_path=errors_path,
    )
    assert n_errors == 0
    assert len(all_recs) == 2
    assert all(r["correct"] for r in all_recs)


def test_run_predictions_loop_one_failure_does_not_stop_the_rest(tmp_path):
    from scripts.run_eval import run_predictions_loop

    cache_dir, records = _make_cache_dir(tmp_path)
    predictions_path = tmp_path / "predictions.jsonl"
    errors_path = tmp_path / "errors.jsonl"
    model = _FakeModel(fail_on_ids={"q1"})
    cfg = OmegaConf.create({})

    n_errors, all_recs = run_predictions_loop(
        records=records,
        remaining_records=records,
        cache_dir=cache_dir,
        sparse=None,
        dense=None,
        model=model,
        retrieval_cfg=cfg,
        top_k=5,
        predictions_path=predictions_path,
        errors_path=errors_path,
    )
    # q1 failed and is logged as an error, NOT written to predictions.jsonl;
    # q2 still got processed and written despite q1's failure.
    assert n_errors == 1
    assert len(all_recs) == 1
    assert all_recs[0]["id"] == "q2"
    assert model.calls == ["q1", "q2"]  # loop continued past the failure

    error_lines = errors_path.read_text().strip().split("\n")
    assert len(error_lines) == 1
    error_rec = json.loads(error_lines[0])
    assert error_rec["id"] == "q1"
    assert "simulated failure" in error_rec["error"]


def test_run_predictions_loop_resumes_and_appends_not_overwrites(tmp_path):
    from scripts.run_eval import run_predictions_loop

    cache_dir, records = _make_cache_dir(tmp_path)
    predictions_path = tmp_path / "predictions.jsonl"
    errors_path = tmp_path / "errors.jsonl"

    # Simulate a prior crashed session that already completed q1.
    predictions_path.write_text(
        json.dumps(
            {
                "id": "q1", "scenario": "Animal", "predicted_choice": "A",
                "answer_choice": "A", "correct": True,
                "retrieved_doc_ids": [], "evidence_retrieved": None,
            }
        )
        + "\n"
    )

    completed_ids = _load_completed_ids(predictions_path)
    assert completed_ids == {"q1"}
    remaining = [r for r in records if r["id"] not in completed_ids]
    assert [r["id"] for r in remaining] == ["q2"]

    model = _FakeModel()
    n_errors, all_recs = run_predictions_loop(
        records=records,
        remaining_records=remaining,
        cache_dir=cache_dir,
        sparse=None,
        dense=None,
        model=model,
        retrieval_cfg=OmegaConf.create({}),
        top_k=5,
        predictions_path=predictions_path,
        errors_path=errors_path,
    )
    # q1's original prediction is preserved (not re-run/overwritten), q2 was
    # newly appended -- model.answer() was only ever called for q2.
    assert model.calls == ["q2"]
    assert n_errors == 0
    assert {r["id"] for r in all_recs} == {"q1", "q2"}
    assert len(all_recs) == 2
