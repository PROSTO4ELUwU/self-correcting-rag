"""Feature extraction for the hallucination critic.

The critic decides: "Is this answer grounded in the context, or made up?"
Instead of throwing a black-box LLM at it, we engineer interpretable signals
that correlate with hallucination. This keeps the project offline, fast, and
*explainable* -- you can show WHY an answer was flagged.

Signals:
  - lexical_overlap   : fraction of answer tokens present in the context
  - novel_token_ratio : fraction of answer content-tokens NOT in the context
  - tfidf_cosine      : TF-IDF cosine similarity(answer, context)
  - number_support    : fraction of numbers in the answer that appear in context
  - has_negation      : whether the answer contains a negation (flip risk)
  - entity_support    : fraction of capitalised entities supported by context
  - len_ratio         : answer length / context length
"""
from __future__ import annotations

import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_NUMBER_RE = re.compile(r"\d[\d,\.]*")
_ENTITY_RE = re.compile(r"\b[A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)*\b")
_NEGATIONS = {"not", "no", "never", "neither", "nor", "cannot", "n't"}

_STOP = {
    "the", "a", "an", "is", "was", "are", "were", "be", "been", "of", "and",
    "to", "in", "on", "for", "it", "its", "by", "with", "as", "at", "that",
    "this", "from", "about", "approximately", "roughly", "around", "over",
}


def _tokens(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOP]


def _numbers(text: str) -> set[str]:
    return {re.sub(r"[^\d]", "", n) for n in _NUMBER_RE.findall(text) if re.sub(r"[^\d]", "", n)}


def _entities(text: str) -> list[str]:
    return [e for e in _ENTITY_RE.findall(text)
            if e.split()[0] not in {"The", "It", "A", "An", "This"}]


class FeatureExtractor:
    """Computes a fixed-length numeric feature vector for (answer, context)."""

    FEATURE_NAMES = [
        "lexical_overlap", "novel_token_ratio", "tfidf_cosine",
        "number_support", "has_negation", "entity_support", "len_ratio",
    ]

    def __init__(self) -> None:
        self._tfidf = TfidfVectorizer(token_pattern=r"[a-z0-9]+", lowercase=True)
        self._fitted = False

    def fit(self, contexts: list[str], answers: list[str]) -> "FeatureExtractor":
        self._tfidf.fit(list(contexts) + list(answers))
        self._fitted = True
        return self

    def _tfidf_cosine(self, answer: str, context: str) -> float:
        m = self._tfidf.transform([answer, context])
        a, c = m[0], m[1]
        denom = (np.sqrt(a.multiply(a).sum()) * np.sqrt(c.multiply(c).sum()))
        if denom == 0:
            return 0.0
        return float(a.multiply(c).sum() / denom)

    def transform_one(self, answer: str, context: str) -> np.ndarray:
        ans_tokens = _tokens(answer)
        ctx_tokens = set(_tokens(context))

        if ans_tokens:
            overlap = sum(1 for t in ans_tokens if t in ctx_tokens) / len(ans_tokens)
        else:
            overlap = 1.0
        novel = 1.0 - overlap

        ans_nums = _numbers(answer)
        ctx_nums = _numbers(context)
        if ans_nums:
            number_support = len(ans_nums & ctx_nums) / len(ans_nums)
        else:
            number_support = 1.0  # no numeric claim -> nothing to contradict

        has_neg = float(any(n in set(_TOKEN_RE.findall(answer.lower())) or n in answer.lower()
                            for n in _NEGATIONS))

        ans_ents = _entities(answer)
        ctx_lower = context.lower()
        if ans_ents:
            entity_support = sum(1 for e in ans_ents if e.lower() in ctx_lower) / len(ans_ents)
        else:
            entity_support = 1.0

        len_ratio = len(answer) / max(1, len(context))
        tfidf_cos = self._tfidf_cosine(answer, context) if self._fitted else 0.0

        return np.array([
            overlap, novel, tfidf_cos, number_support,
            has_neg, entity_support, len_ratio,
        ], dtype=np.float64)

    def transform(self, answers: list[str], contexts: list[str]) -> np.ndarray:
        return np.vstack([self.transform_one(a, c) for a, c in zip(answers, contexts)])
