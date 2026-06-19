"""Answer generators behind a common interface, so the RAG system is model-agnostic.

- ExtractiveGenerator: offline default. Returns the retrieved-context sentence
  most relevant to the question. Always grounded -> a safe baseline.
- StubLLMGenerator: simulates a real LLM that *sometimes hallucinates* (corrupts
  a number/entity). Used to demo the self-correcting loop end-to-end without an
  API key.
- OpenAIGenerator: optional real LLM, used only if OPENAI_API_KEY is set.

All expose: generate(question, context) -> str
"""
from __future__ import annotations

import os
import random
import re

from ..data.synth import corrupt_number, corrupt_entity


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 10]


class ExtractiveGenerator:
    """Grounded baseline: pick the context sentence most relevant to the question."""

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
