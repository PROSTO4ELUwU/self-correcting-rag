"""Answer generators behind one small interface.

Keeping the generator swappable is the whole point: the RAG system never cares
whether the answer came from a TF-IDF lookup or GPT-4. Three are bundled:

    ExtractiveGenerator - offline, always grounded. Returns the context sentence
        closest to the question. A safe baseline.
    StubLLMGenerator - fakes an LLM that gets it right most of the time and
        hallucinates the rest, so the demo and the Phase-4 experiments run with
        no API key.
    OpenAIGenerator - the real thing, used only when OPENAI_API_KEY is set.

Contract: generate(question, context) -> str, and sample(question, context, n)
-> list[str] for the best-of-n / self-consistency strategies.
"""
from __future__ import annotations

import os
import random
import re

from ..data.synth import corrupt_number, corrupt_entity


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 10]


class ExtractiveGenerator:
    """Pick the context sentence most relevant to the question. Always grounded."""

    def generate(self, question: str, context: str) -> str:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        sents = _sentences(context)
        if not sents:
            return context.strip()
        vec = TfidfVectorizer(token_pattern=r"[a-z0-9]+", lowercase=True)
        try:
            m = vec.fit_transform(sents + [question])
        except ValueError:
            return sents[0]
        sims = cosine_similarity(m[-1], m[:-1]).ravel()
        return sents[int(sims.argmax())]

    def sample(self, question: str, context: str, n: int = 5) -> list[str]:
        # deterministic, so every draw is the same grounded sentence
        return [self.generate(question, context)] * n


class StubLLMGenerator:
    """Simulated LLM: grounded most of the time, hallucinates with prob `p`."""

    def __init__(self, p_hallucinate: float = 0.5, seed: int = 0) -> None:
        self.p = p_hallucinate
        self.rng = random.Random(seed)
        self._base = ExtractiveGenerator()

    def generate(self, question: str, context: str) -> str:
        base = self._base.generate(question, context)
        if self.rng.random() < self.p:
            bad = corrupt_number(base, context, self.rng) or \
                corrupt_entity(base, context, self.rng)
            if bad:
                return bad
        return base

    def sample(self, question: str, context: str, n: int = 5) -> list[str]:
        # each draw rolls the hallucination dice again, so the n drafts vary --
        # which is exactly what best-of-n / self-consistency need to chew on
        return [self.generate(question, context) for _ in range(n)]


class OpenAIGenerator:
    """Real LLM generator (optional). Requires OPENAI_API_KEY in the environment."""

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY not set.")
        self.model = model

    def generate(self, question: str, context: str) -> str:
        from openai import OpenAI
        client = OpenAI()
        msg = [
            {"role": "system", "content": "Answer ONLY from the context in one sentence."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ]
        resp = client.chat.completions.create(model=self.model, messages=msg, temperature=0)
        return resp.choices[0].message.content.strip()

    def sample(self, question: str, context: str, n: int = 5) -> list[str]:
        from openai import OpenAI
        client = OpenAI()
        msg = [
            {"role": "system", "content": "Answer ONLY from the context in one sentence."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ]
        # temperature > 0 so the n completions actually differ
        resp = client.chat.completions.create(
            model=self.model, messages=msg, temperature=0.8, n=n)
        return [c.message.content.strip() for c in resp.choices]
