import numpy as np
import pytest

from retrieval.dense import DenseCorpusDoc, DenseRetriever


class FakeImageEncoder:
    """Deterministic fake encoder: maps each opaque 'image' object (here just
    an int tag or small vector) to a fixed embedding via a lookup, so we can
    test batching, indexing, and cosine-similarity ranking logic exactly,
    without any real model weights.
    """

    def __init__(self, embedding_by_tag: dict[object, np.ndarray], record_batch_sizes: list | None = None):
        self.embedding_by_tag = embedding_by_tag
        self.record_batch_sizes = record_batch_sizes

    def encode_images(self, images, batch_size: int = 32) -> np.ndarray:
        if self.record_batch_sizes is not None:
            # record how many "batches" this call would be split into
            n_batches = (len(images) + batch_size - 1) // batch_size
            self.record_batch_sizes.append((len(images), batch_size, n_batches))
        return np.stack([self.embedding_by_tag[img] for img in images])


def _unit(v):
    v = np.array(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_query_returns_most_similar_image_first():
    # 4 corpus images with embeddings at known angles from a "cat-like" axis
    embeddings = {
        "cat_photo_1": _unit([1.0, 0.0]),
        "cat_photo_2": _unit([0.9, 0.1]),
        "dog_photo": _unit([0.0, 1.0]),
        "car_photo": _unit([-1.0, 0.0]),
    }
    encoder = FakeImageEncoder(embeddings)
    docs = [DenseCorpusDoc(doc_id=tag, image=tag) for tag in embeddings]
    retriever = DenseRetriever(encoder=encoder, docs=docs).build()

    # Query image embeds identically to cat_photo_1's direction
    query_embeddings = dict(embeddings)
    query_embeddings["query_cat"] = _unit([1.0, 0.05])
    encoder.embedding_by_tag = query_embeddings

    results = retriever.query("query_cat", top_k=2)
    assert results[0] == "cat_photo_1"
    assert set(results) == {"cat_photo_1", "cat_photo_2"}


def test_query_uses_image_not_text():
    # The whole point: query() takes an image-like object, and ranking is
    # purely a function of the encoder's embedding of that object -- there is
    # no text-based scoring path in this module at all. We verify this by
    # confirming a query "image" that happens to be a string of question text
    # is treated as just another opaque encodable object (encoder controls
    # meaning), not specially parsed/tokenized.
    embeddings = {
        "img_a": _unit([1.0, 0.0, 0.0]),
        "img_b": _unit([0.0, 1.0, 0.0]),
        "not really an image, just text": _unit([1.0, 0.01, 0.0]),
    }
    encoder = FakeImageEncoder(embeddings)
    docs = [
        DenseCorpusDoc(doc_id="a", image="img_a"),
        DenseCorpusDoc(doc_id="b", image="img_b"),
    ]
    retriever = DenseRetriever(encoder=encoder, docs=docs).build()
    results = retriever.query("not really an image, just text")
    assert results[0] == "a"


def test_embedding_computation_is_batched():
    embeddings = {i: _unit(np.random.RandomState(i).randn(8)) for i in range(100)}
    record = []
    encoder = FakeImageEncoder(embeddings, record_batch_sizes=record)
    docs = [DenseCorpusDoc(doc_id=i, image=i) for i in range(100)]
    DenseRetriever(encoder=encoder, docs=docs, batch_size=32).build()

    # exactly one call was made to encode the whole corpus (our fake encoder
    # simulates batching internally), but we assert the retriever *passed*
    # batch_size=32 through rather than silently using some default/whole-corpus value
    assert record[0] == (100, 32, 4)  # 100 images, batch_size 32 -> 4 batches


def test_query_before_build_raises():
    encoder = FakeImageEncoder({})
    retriever = DenseRetriever(encoder=encoder, docs=[DenseCorpusDoc("a", "a")])
    with pytest.raises(RuntimeError):
        retriever.query("a")


def test_build_on_empty_corpus_raises():
    encoder = FakeImageEncoder({})
    with pytest.raises(ValueError):
        DenseRetriever(encoder=encoder, docs=[]).build()


def test_encoder_returning_wrong_number_of_embeddings_raises():
    class BadEncoder:
        def encode_images(self, images, batch_size=32):
            return np.zeros((len(images) - 1, 4))  # deliberately wrong count

    docs = [DenseCorpusDoc("a", "a"), DenseCorpusDoc("b", "b")]
    with pytest.raises(ValueError):
        DenseRetriever(encoder=BadEncoder(), docs=docs).build()
