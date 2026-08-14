"""Sparse (BM25) retrieval: question TEXT against corpus TEXT (captions/metadata).

This is intentionally text-vs-text. It is a legitimate but weaker baseline
compared to image-to-image dense retrieval (see retrieval/dense.py) because
MRAG-Bench questions are deliberately generic ("Can you identify this
animal?") and rarely describe the specific visual content — BM25 over
captions is the best a pure-text retriever can do here.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Hashable

from rank_bm25 import BM25Okapi


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase, alphanumeric-only tokenization. Deterministic and dependency-free."""
    return _TOKEN_RE.findall(text.lower())


@dataclass
class SparseCorpusDoc:
    doc_id: Hashable
    text: str  # caption / metadata text associated with a corpus image


@dataclass
class BM25Retriever:
    """Wraps rank_bm25.BM25Okapi over a fixed corpus of (doc_id, text) pairs."""

    docs: list[SparseCorpusDoc] = field(default_factory=list)
    _bm25: BM25Okapi | None = field(default=None, init=False, repr=False)
    _tokenized_corpus: list[list[str]] = field(default_factory=list, init=False, repr=False)

    def build(self) -> "BM25Retriever":
        if not self.docs:
            raise ValueError("Cannot build BM25 index over an empty corpus")
        self._tokenized_corpus = [tokenize(d.text) for d in self.docs]
        self._bm25 = BM25Okapi(self._tokenized_corpus)
        return self

    def query(self, query_text: str, top_k: int | None = None) -> list[Hashable]:
        """Return doc_ids ranked best-first by BM25 score for query_text.

        Empty tokenized query (e.g. all stopword-stripped-to-nothing edge case
        or empty string input) returns an empty list rather than raising, since
        a degenerate query has no meaningful ranking.
        """
        if self._bm25 is None:
            raise RuntimeError("Call .build() before .query()")
        tokens = tokenize(query_text)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        order = sorted(
            range(len(self.docs)), key=lambda i: (-scores[i], str(self.docs[i].doc_id))
        )
        if top_k is not None:
            order = order[:top_k]
        return [self.docs[i].doc_id for i in order]
