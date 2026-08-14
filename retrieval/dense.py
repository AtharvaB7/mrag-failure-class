"""Dense retrieval: IMAGE-to-IMAGE embedding similarity.

CRITICAL DESIGN CONSTRAINT (do not violate this):
MRAG-Bench questions are deliberately generic and never describe the query
image's specific visual content. Embedding the question TEXT and searching
against image embeddings collapses to near-random retrieval (~1-2% recall
observed empirically). The correct design embeds the QUERY IMAGE itself and
retrieves corpus images by visual similarity.

This module is encoder-agnostic: `ImageEncoder` is a Protocol so that in this
sandbox (no GPU, no network to huggingface.co) we can test all the batching,
indexing, and search logic against a fake encoder, and swap in a real SigLIP
encoder (retrieval/encoders/siglip.py, built separately, on your A100 Colab
runtime) without touching this file.

If a text-vs-image baseline is ever wanted for comparison, it must be built
as a SEPARATELY labeled ablation (e.g. `TextToImageDenseRetriever` in a
clearly-named separate module) — never as the default `DenseRetriever` here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Hashable, Protocol, Sequence

import numpy as np


class ImageEncoder(Protocol):
    """Anything that can turn a batch of images into L2-normalizable embeddings.

    Real implementation (built later, GPU-side): wraps a SigLIP image tower,
    handles both raw-tensor and BaseModelOutputWithPooling return shapes from
    `get_image_features()` depending on transformers version (see project
    notes: check `hasattr(output, "pooler_output")`), and loads in fp16.
    """

    def encode_images(self, images: Sequence[object], batch_size: int = 32) -> np.ndarray:
        """Return an (N, D) float array of embeddings for N input images."""
        ...


@dataclass
class DenseCorpusDoc:
    doc_id: Hashable
    image: object  # PIL.Image in real use; any opaque object for testing


def _l2_normalize(mat: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.clip(norms, eps, None)


@dataclass
class DenseRetriever:
    """Image-to-image retrieval: encode a corpus of images once, then rank by
    cosine similarity to a query image's embedding.

    Embedding computation is batched (default batch_size=32) rather than done
    in one forward pass over the whole corpus, since encoding a multi-thousand
    image corpus in a single pass will OOM on an A100 that also has a 7-8B VLM
    resident.
    """

    encoder: ImageEncoder
    docs: list[DenseCorpusDoc] = field(default_factory=list)
    batch_size: int = 32
    _doc_embeddings: np.ndarray | None = field(default=None, init=False, repr=False)

    def build(self) -> "DenseRetriever":
        if not self.docs:
            raise ValueError("Cannot build dense index over an empty corpus")
        images = [d.image for d in self.docs]
        embs = self.encoder.encode_images(images, batch_size=self.batch_size)
        if embs.shape[0] != len(self.docs):
            raise ValueError(
                f"Encoder returned {embs.shape[0]} embeddings for {len(self.docs)} images"
            )
        self._doc_embeddings = _l2_normalize(embs.astype(np.float32))
        return self

    def query(self, query_image: object, top_k: int | None = None) -> list[Hashable]:
        """Return doc_ids ranked best-first by cosine similarity to query_image.

        NOTE: query_image is an actual image object (the MRAG-Bench `image`
        column), never question text.
        """
        if self._doc_embeddings is None:
            raise RuntimeError("Call .build() before .query()")
        query_emb = self.encoder.encode_images([query_image], batch_size=1)
        query_emb = _l2_normalize(query_emb.astype(np.float32))[0]
        sims = self._doc_embeddings @ query_emb
        order = sorted(
            range(len(self.docs)), key=lambda i: (-sims[i], str(self.docs[i].doc_id))
        )
        if top_k is not None:
            order = order[:top_k]
        return [self.docs[i].doc_id for i in order]
