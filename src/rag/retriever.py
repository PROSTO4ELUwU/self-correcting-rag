"""Dense retriever: sentence-transformer embeddings + FAISS index.

This replaces the toy "pass the context in by hand" setup with a real
retrieval step: embed a corpus of passages once, then for each question embed
it and pull the top-k most similar passages (cosine similarity via inner
product on normalised vectors).

Default encoder: all-MiniLM-L6-v2 (small, fast, good baseline).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

_DEFAULT_MODEL = "all-MiniLM-L6-v2"


@lru_cache(maxsize=2)
def _encoder(name: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(name)


@dataclass
class Retrieved:
    text: str
    score: float
    index: int


class DenseRetriever:
    def __init__(self, model_name: str = _DEFAULT_MODEL) -> None:
        self.model_name = model_name
        self._passages: list[str] = []
        self._index = None

    def build_index(self, passages: list[str]) -> "DenseRetriever":
        import faiss
        self._passages = list(passages)
        emb = _encoder(self.model_name).encode(
            self._passages, normalize_embeddings=True, show_progress_bar=False
        ).astype("float32")
        self._index = faiss.IndexFlatIP(emb.shape[1])
        self._index.add(emb)
        return self

    def search(self, query: str, k: int = 3) -> list[Retrieved]:
        if self._index is None:
            raise RuntimeError("Call build_index(...) first.")
        q = _encoder(self.model_name).encode(
            [query], normalize_embeddings=True, show_progress_bar=False
        ).astype("float32")
        scores, idxs = self._index.search(q, k)
        return [Retrieved(self._passages[i], float(s), int(i))
                for s, i in zip(scores[0], idxs[0]) if i >= 0]

    def retrieve_context(self, query: str, k: int = 3) -> str:
        return "\n\n".join(r.text for r in self.search(query, k))
