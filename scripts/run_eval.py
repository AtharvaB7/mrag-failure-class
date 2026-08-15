"""Main eval entry point. Runs ONE model under ONE retrieval setting over the
downloaded MRAG-Bench instances (data/cache, built by data/download.py) and
writes per-instance predictions + accuracy to disk.

Usage:
    python -m scripts.run_eval model=qwen2vl retrieval=no_retrieval
    python -m scripts.run_eval model=qwen2vl retrieval=hybrid
    python -m scripts.run_eval model=llava_next retrieval=sparse

Run ALL 8 combinations (2 models x 4 retrieval settings) to get the full
grid the transition matrix needs. Start with just no_retrieval for ONE model
on a small subset first (see --limit) before running the full grid.

Output: outputs/<date>/<time>/predictions.jsonl, one line per instance:
    {"id": ..., "scenario": ..., "predicted_choice": ..., "answer_choice": ...,
     "correct": bool, "retrieved_doc_ids": [...], "evidence_retrieved": bool}
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import hydra
from omegaconf import DictConfig
from PIL import Image

from data.schema import MRAGBenchInstance
from evaluation.metrics import accuracy
from retrieval.dense import DenseCorpusDoc, DenseRetriever
from retrieval.hybrid import HybridRetriever
from retrieval.sparse import BM25Retriever, SparseCorpusDoc


def _find_corpus_image(cache_dir: Path, content_hash: str) -> Path:
    """Locate a corpus image by content hash regardless of the file extension
    it was saved with (data/download.py defaults to JPEG for speed, but may
    be run with image_format="PNG"). Raises if not found or ambiguous.
    """
    matches = list((cache_dir / "corpus").glob(f"{content_hash}.*"))
    if not matches:
        raise FileNotFoundError(f"No corpus image found for hash {content_hash!r} in {cache_dir / 'corpus'}")
    if len(matches) > 1:
        raise ValueError(f"Multiple corpus files found for hash {content_hash!r}: {matches}")
    return matches[0]


def load_cached_instances(cache_dir: Path, limit: int | None = None) -> list[dict]:
    """Load instances.jsonl records (NOT full MRAGBenchInstance objects yet
    -- images are loaded lazily per-instance below since the full corpus of
    images doesn't need to be resident at once).
    """
    records = []
    with open(cache_dir / "instances.jsonl") as f:
        for line in f:
            records.append(json.loads(line))
            if limit is not None and len(records) >= limit:
                break
    return records


def _load_completed_ids(predictions_path: Path) -> set[str]:
    """Read an existing predictions.jsonl (if any) and return the set of
    instance ids that already have a VALID recorded prediction, so a resumed
    run can skip them. Corrupted lines (e.g. null-byte blocks from a process
    killed mid-write) are silently treated as not-completed -- that instance
    will simply be re-run, which is the correct behavior; we don't try to
    repair the corrupt line in place.
    """
    if not predictions_path.exists():
        return set()
    completed = set()
    n_corrupt = 0
    with open(predictions_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                completed.add(rec["id"])
            except (json.JSONDecodeError, KeyError):
                n_corrupt += 1
    if n_corrupt:
        print(
            f"  WARNING: {n_corrupt} corrupted line(s) found in existing "
            f"{predictions_path} and will be skipped (not counted as completed, "
            f"not repaired in place -- their instances will be re-run and "
            f"appended fresh)."
        )
    return completed


def _rewrite_dropping_corrupt_lines(predictions_path: Path) -> None:
    """Rewrite predictions.jsonl keeping only valid JSON lines, so corrupt
    lines from a previous crash don't linger forever mixed in with fresh
    appended data. Called once at the start of a resumed run.
    """
    if not predictions_path.exists():
        return
    valid_lines = []
    with open(predictions_path) as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                json.loads(stripped)
                valid_lines.append(stripped)
            except json.JSONDecodeError:
                pass
    with open(predictions_path, "w") as f:
        for line in valid_lines:
            f.write(line + "\n")


def build_model(model_cfg: DictConfig):
    if model_cfg.name == "qwen2vl":
        from models.qwen2vl import Qwen2VLModel

        return Qwen2VLModel(hf_id=model_cfg.hf_id, max_new_tokens=model_cfg.max_new_tokens)
    elif model_cfg.name == "llava_next":
        from models.llava_next import LlavaNextModel

        return LlavaNextModel(hf_id=model_cfg.hf_id, max_new_tokens=model_cfg.max_new_tokens)
    raise ValueError(f"Unknown model name: {model_cfg.name!r}")


def build_retrievers(retrieval_cfg: DictConfig, cache_dir: Path, records: list[dict]):
    """Build sparse and/or dense retrievers over the corpus, per retrieval_cfg
    flags. Returns (sparse_retriever_or_none, dense_retriever_or_none).

    Corpus text for BM25: each corpus image's surrogate caption is the set
    of question texts from instances it served as gt_images evidence for
    (see inline comment below for why scenario/aspect labels don't work).
    """
    sparse = None
    dense = None

    if retrieval_cfg.sparse_enabled or retrieval_cfg.dense_enabled:
        # build corpus doc list: one entry per unique gt_image hash.
        # Text surrogate (hash_to_text) is needed for sparse; image path
        # resolution (hash_to_image_path) is needed ONLY for dense -- these
        # are independent and must not block each other. Resolving image
        # paths when only sparse_enabled is set makes a sparse-only run
        # depend on every corpus image file existing on disk, which it has
        # no reason to.
        #
        # IMPORTANT: the text surrogate for a corpus image is the set of
        # QUESTION TEXTS from every instance it was gt_images evidence for
        # -- NOT scenario/aspect labels. An earlier version used
        # f"{scenario} {aspect}" (e.g. "Scope angle"), which shares almost
        # no vocabulary with real questions like "Can you identify this
        # animal?" -- BM25 scored every doc 0.0 for every query, so
        # retrieval silently returned the same alphabetically-first 5
        # doc_ids for EVERY question regardless of content (confirmed via
        # real prediction files: retrieved_doc_ids was byte-identical across
        # all 1353 instances). Using the originating question(s) as the
        # surrogate caption gives real lexical overlap, since MRAG-Bench
        # questions are deliberately generic/repetitive within a scenario.
        hash_to_questions: dict[str, set[str]] = {}
        for rec in records:
            for h in rec["gt_image_hashes"]:
                hash_to_questions.setdefault(h, set()).add(rec["question"])
        hash_to_text = {h: " ".join(sorted(qs)) for h, qs in hash_to_questions.items()}

        if retrieval_cfg.sparse_enabled:
            sparse_docs = [SparseCorpusDoc(h, text) for h, text in hash_to_text.items()]
            sparse = BM25Retriever(docs=sparse_docs).build()

        if retrieval_cfg.dense_enabled:
            missing = []
            hash_to_image_path = {}
            for h in hash_to_text:
                try:
                    hash_to_image_path[h] = _find_corpus_image(cache_dir, h)
                except FileNotFoundError:
                    missing.append(h)
            if missing:
                raise FileNotFoundError(
                    f"{len(missing)} of {len(hash_to_text)} corpus images referenced in "
                    f"instances.jsonl are missing from {cache_dir / 'corpus'}. "
                    f"This means data/cache is incomplete or was copied/interrupted "
                    f"partway -- re-run `python -m data.download` rather than patching "
                    f"around it. First few missing hashes: {missing[:5]}"
                )

            from retrieval.encoders.siglip import SiglipImageEncoder

            encoder = SiglipImageEncoder(
                hf_id=retrieval_cfg.image_encoder.hf_id,
            )
            dense_docs = [
                DenseCorpusDoc(h, Image.open(path)) for h, path in hash_to_image_path.items()
            ]
            dense = DenseRetriever(
                encoder=encoder, docs=dense_docs, batch_size=retrieval_cfg.image_encoder.batch_size
            ).build()

    return sparse, dense


def run_predictions_loop(
    records: list[dict],
    remaining_records: list[dict],
    cache_dir: Path,
    sparse,
    dense,
    model,
    retrieval_cfg: DictConfig,
    top_k: int,
    predictions_path: Path,
    errors_path: Path,
) -> tuple[int, list[dict]]:
    """Run model.answer() over remaining_records, appending results to
    predictions_path and errors_path. Returns (n_errors, all_records_in_file)
    after re-reading the full file (so callers get a correct total even
    across resumed sessions).

    A single instance's failure (bad image, transient generation error, etc)
    is caught, logged to errors_path, and does NOT stop the loop -- this is
    the core crash-tolerance property this function exists to provide.
    """
    n_errors = 0
    with open(predictions_path, "a") as out_f, open(errors_path, "a") as err_f:
        for rec in remaining_records:
            try:
                query_image = Image.open(cache_dir / rec["query_image_path"])
                instance = MRAGBenchInstance(
                    id=rec["id"],
                    aspect=rec["aspect"],
                    scenario=rec["scenario"],
                    image=query_image,
                    gt_images=[],
                    question=rec["question"],
                    choices=rec["choices"],
                    answer_choice=rec["answer_choice"],
                )

                retrieved_ids = []
                if sparse is not None and dense is not None:
                    hybrid = HybridRetriever(
                        sparse=sparse,
                        dense=dense,
                        rrf_k=retrieval_cfg.get("rrf_k", 60),
                        sparse_weight=retrieval_cfg.get("sparse_weight", 1.0),
                        dense_weight=retrieval_cfg.get("dense_weight", 1.0),
                    )
                    retrieved_ids = hybrid.query(rec["question"], query_image, top_k=top_k)
                elif sparse is not None:
                    retrieved_ids = sparse.query(rec["question"], top_k=top_k)
                elif dense is not None:
                    retrieved_ids = dense.query(query_image, top_k=top_k)

                retrieved_images = [
                    Image.open(_find_corpus_image(cache_dir, h)) for h in retrieved_ids
                ]

                predicted = model.answer(instance, retrieved_images)
                correct = predicted == instance.answer_choice

                evidence_retrieved = None
                if retrieved_ids:
                    gt_hashes = set(rec["gt_image_hashes"])
                    evidence_retrieved = bool(set(retrieved_ids) & gt_hashes)

                out_f.write(
                    json.dumps(
                        {
                            "id": rec["id"],
                            "scenario": rec["scenario"],
                            "predicted_choice": predicted,
                            "answer_choice": instance.answer_choice,
                            "correct": correct,
                            "retrieved_doc_ids": retrieved_ids,
                            "evidence_retrieved": evidence_retrieved,
                        }
                    )
                    + "\n"
                )
                out_f.flush()
                os.fsync(out_f.fileno())
            except Exception as e:
                n_errors += 1
                err_f.write(
                    json.dumps({"id": rec["id"], "error": f"{type(e).__name__}: {e}"}) + "\n"
                )
                err_f.flush()
                print(f"  ERROR on instance {rec['id']}: {type(e).__name__}: {e}")

    all_recs = []
    with open(predictions_path) as f:
        for line in f:
            line = line.strip()
            if line:
                all_recs.append(json.loads(line))
    return n_errors, all_recs


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    cache_dir = Path(cfg.data.cache_dir)
    if not (cache_dir / "instances.jsonl").exists():
        raise FileNotFoundError(
            f"{cache_dir}/instances.jsonl not found. Run `python -m data.download "
            f"--out_dir {cache_dir}` first."
        )

    limit = cfg.get("limit", None)
    records = load_cached_instances(cache_dir, limit=limit)
    print(f"Loaded {len(records)} instances from {cache_dir}")

    sparse, dense = build_retrievers(cfg.retrieval, cache_dir, records)
    top_k = cfg.retrieval.get("top_k", 5)

    model = build_model(cfg.model)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"
    errors_path = output_dir / "errors.jsonl"

    # RESUME SUPPORT: if predictions.jsonl already has (valid) entries for
    # some instance ids -- e.g. from a run that got killed by an OOM, a
    # Colab disconnect, or a preemption -- skip those ids and append the
    # rest, instead of starting over or silently producing a second,
    # differently-named output that has to be manually stitched together
    # later. Corrupt lines from a previous crash are dropped up front so
    # they don't linger mixed in with fresh data.
    _rewrite_dropping_corrupt_lines(predictions_path)
    completed_ids = _load_completed_ids(predictions_path)
    remaining_records = [r for r in records if r["id"] not in completed_ids]
    if completed_ids:
        print(
            f"Resuming: {len(completed_ids)} instances already completed in "
            f"{predictions_path}, {len(remaining_records)} remaining."
        )

    n_errors, all_recs = run_predictions_loop(
        records=records,
        remaining_records=remaining_records,
        cache_dir=cache_dir,
        sparse=sparse,
        dense=dense,
        model=model,
        retrieval_cfg=cfg.retrieval,
        top_k=top_k,
        predictions_path=predictions_path,
        errors_path=errors_path,
    )

    acc = sum(r["correct"] for r in all_recs) / len(all_recs) if all_recs else 0.0
    print(
        f"model={cfg.model.name} retrieval={cfg.retrieval.name} "
        f"n_total={len(all_recs)}/{len(records)} n_errors_this_session={n_errors} "
        f"accuracy={acc:.4f}"
    )
    if len(all_recs) < len(records):
        print(
            f"  INCOMPLETE: {len(records) - len(all_recs)} instances still missing. "
            f"Re-run the exact same command to resume."
        )
    print(f"Predictions written to {predictions_path}")
    if n_errors:
        print(f"Errors logged to {errors_path}")


if __name__ == "__main__":
    main()
