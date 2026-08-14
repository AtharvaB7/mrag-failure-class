import json
from pathlib import Path

from PIL import Image
from omegaconf import OmegaConf

from scripts.run_eval import build_retrievers, load_cached_instances


def _make_cache_dir(tmp_path):
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
    results = sparse.query("Animal Scope")
    assert set(results) == {"hashA", "hashB"}


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
