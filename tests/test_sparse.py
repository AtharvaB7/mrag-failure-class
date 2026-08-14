import pytest

from retrieval.sparse import BM25Retriever, SparseCorpusDoc, tokenize


def test_tokenize_lowercases_and_strips_punctuation():
    assert tokenize("Can you identify THIS animal?!") == [
        "can",
        "you",
        "identify",
        "this",
        "animal",
    ]


def test_tokenize_handles_numbers():
    assert tokenize("Species #42 lives here") == ["species", "42", "lives", "here"]


def test_tokenize_empty_string():
    assert tokenize("") == []


def _build_retriever():
    # NOTE: BM25's standard IDF term is log((N - n + 0.5) / (n + 0.5)), which
    # is exactly 0 when a query term appears in precisely half the corpus
    # documents (n = N/2) -- e.g. with a 4-doc corpus and a term in 2 of them.
    # A too-small/adversarial corpus can silently zero out the very term
    # you're testing relevance on, making every doc score 0.0 and the
    # "ranking" degenerate to arbitrary tie-break order. Use a corpus large
    # enough (8 docs, "tiger" in only 2) that IDF is meaningfully positive.
    docs = [
        SparseCorpusDoc("d1", "a golden retriever dog running on grass"),
        SparseCorpusDoc("d2", "a bengal tiger in the jungle"),
        SparseCorpusDoc("d3", "a house cat sleeping on a couch"),
        SparseCorpusDoc("d4", "a siberian tiger walking in snow"),
        SparseCorpusDoc("d5", "a parrot perched on a branch"),
        SparseCorpusDoc("d6", "a horse grazing in a field"),
        SparseCorpusDoc("d7", "a dolphin swimming in the ocean"),
        SparseCorpusDoc("d8", "a rabbit hopping through a meadow"),
    ]
    return BM25Retriever(docs=docs).build()


def test_query_ranks_relevant_doc_highest():
    retriever = _build_retriever()
    results = retriever.query("what kind of tiger is this")
    assert results[0] in ("d2", "d4")
    assert set(results[:2]) == {"d2", "d4"}


def test_query_top_k_truncates():
    retriever = _build_retriever()
    results = retriever.query("dog", top_k=2)
    assert len(results) == 2


def test_bm25_zero_score_when_term_appears_in_exactly_half_corpus():
    # Documents this BM25 IDF property explicitly so it's never mistaken for
    # a bug again: a term present in exactly N/2 of the corpus docs gets
    # IDF == 0 under the standard formula, and contributes nothing to the score.
    docs = [
        SparseCorpusDoc("d1", "shared term here"),
        SparseCorpusDoc("d2", "shared term here"),
        SparseCorpusDoc("d3", "nothing relevant at all"),
        SparseCorpusDoc("d4", "nothing relevant at all"),
    ]
    retriever = BM25Retriever(docs=docs).build()
    scores = retriever._bm25.get_scores(["shared", "term"])
    assert all(s == pytest.approx(0.0) for s in scores)


def test_query_before_build_raises():
    retriever = BM25Retriever(docs=[SparseCorpusDoc("d1", "text")])
    with pytest.raises(RuntimeError):
        retriever.query("text")


def test_build_on_empty_corpus_raises():
    with pytest.raises(ValueError):
        BM25Retriever(docs=[]).build()


def test_query_with_no_tokenizable_content_returns_empty():
    retriever = _build_retriever()
    assert retriever.query("???!!!") == []


def test_returns_all_doc_ids_present():
    retriever = _build_retriever()
    results = retriever.query("dog cat tiger")
    assert set(results) == {"d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8"}
