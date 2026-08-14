# RAG Failure Classification: Multimodal RAG Failure-Mode Analysis

Diagnostic study of how failure modes shift as retrieval quality improves
(no retrieval → sparse → dense → hybrid) on MRAG-Bench, using a per-instance
failure-mode transition matrix as the main result.

## Status: repo skeleton + CPU/no-GPU-testable logic complete, tested, passing

**80/80 tests passing** as of this commit. Run with `pytest` from repo root.

### What IS tested (real logic, real assertions, no mocking of the thing under test)
- `retrieval/fusion.py` — RRF math, including a hand-computed exact-value check
- `retrieval/sparse.py` — real `rank_bm25` over real tokenized text
- `retrieval/dense.py` — image-to-image retrieval interface, batching, and
  cosine-similarity ranking, tested against a deterministic fake encoder
  (embeddings are known vectors at known angles) since no real SigLIP weights
  are available in this environment
- `retrieval/hybrid.py` — RRF fusion of sparse + dense end-to-end
- `data/schema.py` — MRAG-Bench row parsing + gt_images corpus dedup via
  content hashing, tested against real PIL images
- `evaluation/metrics.py` — accuracy, retrieval recall@k
- `evaluation/contamination.py` — per-scenario no-retrieval vs. hybrid gap
  flagging
- `evaluation/failure_modes.py` — rule-based prefilter (CORRECT /
  RETRIEVAL_MISSED_EVIDENCE / UNDETERMINED)
- `evaluation/transition_matrix.py` — the core novel result: instance-level
  failure-mode tracking across all 4 settings, built separately per model
- `configs/` — Hydra composition and CLI-style overrides
  (`model=llava_next retrieval=dense`, dotted-path scalar overrides)

### What is NOT yet tested (needs your A100 Colab runtime)
- Actual model loading/inference (Qwen2-VL, LLaVA-NeXT)
- Actual SigLIP image encoding (the `ImageEncoder` Protocol in `dense.py` is
  ready to accept a real implementation — build `retrieval/encoders/siglip.py`
  next, handling both raw-tensor and `BaseModelOutputWithPooling` return
  shapes from `get_image_features()`, and using `dtype=` not `torch_dtype=`)
- The actual MRAG-Bench download (schema in `data/schema.py` matches your
  confirmed `load_dataset(...).column_names` output, but hasn't been
  round-tripped against the real dataset yet)
- LLM-assisted annotation pass (DeepSeek backend, pluggable)

## One real bug caught while writing tests (documented, not silently fixed)

`tests/test_sparse.py` originally used a 4-document corpus where a query
term ("tiger") appeared in exactly 2 of the 4 docs. BM25's standard IDF term,
`log((N - n + 0.5) / (n + 0.5))`, is exactly **zero** when a term appears in
precisely half the corpus — so every document scored 0.0 and the "ranking"
was arbitrary tie-break order, not relevance. This wasn't a bug in
`retrieval/sparse.py`; it was an artifact of an unrealistically small test
corpus. Fixed by using an 8-document corpus where the term distribution
doesn't hit this degenerate case, and added
`test_bm25_zero_score_when_term_appears_in_exactly_half_corpus` so this
property is documented and never mistaken for a real bug again.

## Environment notes

Built and tested in a sandboxed environment with pypi.org access but no
network access to huggingface.co and no GPU. All model-weight-dependent code
is written against a `Protocol`/interface so its non-GPU logic (batching,
ranking, config wiring) could be verified now; actual inference needs to run
on your Colab A100.

## Critical design decision baked into `retrieval/dense.py`

Dense retrieval is **image-to-image**: it embeds the MRAG-Bench `image`
column (the query image) and searches a corpus of `gt_images`-derived
images by visual similarity. It does **not** embed question text against an
image corpus — MRAG-Bench questions are deliberately generic ("Can you
identify this animal?") and text-vs-image retrieval on this benchmark
collapses to near-random (~1-2% recall). If a text-vs-image ablation is
ever wanted, it must be built as a separately-named module, never as the
default `DenseRetriever`.

## Repo layout

```
data/          MRAG-Bench schema + loading (download script TBD)
models/        VLM wrappers behind a common interface (TBD)
retrieval/     sparse (BM25), dense (image-to-image), hybrid (RRF)
evaluation/    metrics, contamination check, failure-mode taxonomy, transition matrix
scripts/       entry points (TBD, Hydra-configured)
configs/       model=qwen2vl|llava_next, retrieval=no_retrieval|sparse|dense|hybrid
tests/         80 passing tests, run before any GPU work
```

## Next steps (in order — don't skip ahead)

1. Run `load_dataset("uclanlp/MRAG-Bench", split="test")[0]` on your Colab
   runtime one more time to confirm PIL image objects behave as expected
   with `data/schema.py`'s `hash_image_bytes` (untested against the real
   dataset yet — only against synthetic PIL images here).
2. Build `data/download.py` — atomic-write download script.
3. Build `retrieval/encoders/siglip.py` — real SigLIP wrapper implementing
   the `ImageEncoder` Protocol, fp16, batch_size=32 default, handles the
   `pooler_output` vs. raw-tensor API drift.
4. Build `models/` VLM wrappers (Qwen2-VL primary, LLaVA-NeXT secondary) —
   common interface, tested with a fake model class for prompt/multi-image
   wiring before any real GPU run.
5. Verify one model + one retrieval setting on a small subset before running
   the full 8-run grid (2 models × 4 retrieval settings).
