"""Hybrid retrieval: RRF fusion of sparse (text-vs-text) and dense (image-vs-image) rankings.

Both sub-retrievers must already be built (`.build()` called) against corpora
that share the SAME doc_id space, since fusion operates over doc_ids.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable

from retrieval.dense import DenseRetriever
from retrieval.fusion import DEFAULT_K, reciprocal_rank_fusion, top_k_ids
from retrieval.sparse import BM25Retriever


@dataclass
class HybridRetriever:
    sparse: BM25Retriever
    dense: DenseRetriever
    rrf_k: int = DEFAULT_K
    sparse_weight: float = 1.0
    dense_weight: float = 1.0

    def query(
        self,
        query_text: str,
        query_image: object,
        top_k: int | None = None,
    ) -> list[Hashable]:
        """Retrieve using BOTH question text (sparse) and query image (dense),
        fused via RRF. This is the only place question text and query image
        are combined for retrieval purposes.
        """
        sparse_ids = self.sparse.query(query_text)
        dense_ids = self.dense.query(query_image)
        fused = reciprocal_rank_fusion(
            [sparse_ids, dense_ids],
            k=self.rrf_k,
            weights=[self.sparse_weight, self.dense_weight],
        )
        ids = top_k_ids(fused, top_k) if top_k is not None else [r.doc_id for r in fused]
        return ids
