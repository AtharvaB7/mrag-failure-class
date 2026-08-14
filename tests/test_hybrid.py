import numpy as np

from retrieval.dense import DenseCorpusDoc, DenseRetriever
from retrieval.hybrid import HybridRetriever
from retrieval.sparse import BM25Retriever, SparseCorpusDoc


class FakeImageEncoder:
    def __init__(self, embedding_by_tag):
        self.embedding_by_tag = embedding_by_tag

    def encode_images(self, images, batch_size=32):
        return np.stack([self.embedding_by_tag[img] for img in images])


def _unit(v):
    v = np.array(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def _build_hybrid():
    sparse_docs = [
        SparseCorpusDoc("shared_doc", "a tiger in the jungle"),
        SparseCorpusDoc("sparse_only", "some unrelated caption about a car"),
    ]
    sparse = BM25Retriever(docs=sparse_docs).build()

    embeddings = {
        "shared_doc": _unit([1.0, 0.0]),
        "dense_only": _unit([0.0, 1.0]),
        "query_img": _unit([0.95, 0.05]),
    }
    encoder = FakeImageEncoder(embeddings)
    dense_docs = [
        DenseCorpusDoc("shared_doc", "shared_doc"),
        DenseCorpusDoc("dense_only", "dense_only"),
    ]
    dense = DenseRetriever(encoder=encoder, docs=dense_docs).build()
    return HybridRetriever(sparse=sparse, dense=dense)


def test_doc_ranked_in_both_beats_single_source_docs():
    hybrid = _build_hybrid()
    results = hybrid.query(query_text="tiger animal", query_image="query_img")
    assert results[0] == "shared_doc"
    assert set(results) == {"shared_doc", "sparse_only", "dense_only"}


def test_top_k_truncates_fused_results():
    hybrid = _build_hybrid()
    results = hybrid.query(query_text="tiger animal", query_image="query_img", top_k=1)
    assert results == ["shared_doc"]


def test_weights_affect_fusion_outcome():
    hybrid = _build_hybrid()
    hybrid.dense_weight = 100.0
    hybrid.sparse_weight = 0.001
    results = hybrid.query(query_text="unrelated car caption", query_image="query_img")
    # with dense massively upweighted, shared_doc (close to query_img) should
    # still win even though sparse text query favors "sparse_only"
    assert results[0] == "shared_doc"
