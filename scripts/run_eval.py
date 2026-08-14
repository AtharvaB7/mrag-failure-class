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

    Corpus text for BM25 uses each instance's own question as a placeholder
    caption source is WRONG -- corpus docs need their own captions, not the
    query's. Since MRAG-Bench gt_images have no captions, we use each
    corpus image's originating instance's scenario+aspect as a lightweight
    text surrogate for sparse retrieval. This is a known-weak baseline by
    design (per the proposal, BM25-over-captions is the weaker baseline).
    """
    sparse = None
    dense = None

    if retrieval_cfg.sparse_enabled or retrieval_cfg.dense_enabled:
        # build corpus doc list: one entry per unique gt_image hash
        hash_to_text = {}
        hash_to_image_path = {}
        for rec in records:
            for h in rec["gt_image_hashes"]:
                if h not in hash_to_text:
                    hash_to_text[h] = f"{rec['scenario']} {rec['aspect']}"
                    hash_to_image_path[h] = _find_corpus_image(cache_dir, h)

        if retrieval_cfg.sparse_enabled:
            sparse_docs = [SparseCorpusDoc(h, text) for h, text in hash_to_text.items()]
            sparse = BM25Retriever(docs=sparse_docs).build()

        if retrieval_cfg.dense_enabled:
            from retrieval.encoders.siglip import SiglipImageEncoder

            encoder = SiglipImageEncoder(
                hf_id=retrieval_cfg.image_encoder.hf_id,
                batch_size=retrieval_cfg.image_encoder.batch_size,
            )
            dense_docs = [
                DenseCorpusDoc(h, Image.open(path)) for h, path in hash_to_image_path.items()
            ]
            dense = DenseRetriever(
                encoder=encoder, docs=dense_docs, batch_size=retrieval_cfg.image_encoder.batch_size
            ).build()

    return sparse, dense


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

    all_correct = []
    with open(predictions_path, "w") as out_f:
        for rec in records:
            query_image = Image.open(cache_dir / rec["query_image_path"])
            instance = MRAGBenchInstance(
                id=rec["id"],
                aspect=rec["aspect"],
                scenario=rec["scenario"],
                image=query_image,
                gt_images=[],  # not needed at eval time, only gt_image_hashes for recall
                question=rec["question"],
                choices=rec["choices"],
                answer_choice=rec["answer_choice"],
            )

            retrieved_ids = []
            if sparse is not None and dense is not None:
                hybrid = HybridRetriever(
                    sparse=sparse,
                    dense=dense,
                    rrf_k=cfg.retrieval.get("rrf_k", 60),
                    sparse_weight=cfg.retrieval.get("sparse_weight", 1.0),
                    dense_weight=cfg.retrieval.get("dense_weight", 1.0),
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
            all_correct.append(correct)

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

    acc = sum(all_correct) / len(all_correct) if all_correct else 0.0
    print(f"model={cfg.model.name} retrieval={cfg.retrieval.name} n={len(records)} accuracy={acc:.4f}")
    print(f"Predictions written to {predictions_path}")


if __name__ == "__main__":
    main()
