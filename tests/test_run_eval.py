import json
from pathlib import Path

import pytest
from PIL import Image
from omegaconf import OmegaConf

from scripts.run_eval import build_retrievers, load_cached_instances


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
