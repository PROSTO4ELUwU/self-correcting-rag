"""Toy RAG pipeline with a self-correcting loop driven by the critic.

Flow:
    question --> retrieve(context) --> generate(answer)
                 --> critic.is_hallucinated(answer, context)?
                       no -> return answer
                       yes -> regenerate (extractive fallback) -> re-check
                              (repeat up to max_iters)

The "generator" here is intentionally simple and offline: it returns a stored
candidate answer that may be faithful OR hallucinated, simulating a real LLM
that sometimes makes things up. The point of the project is the *correction
loop and the learned critic*, not the base generator -- swap in any LLM later.

The extractive fallback grounds the answer by returning the most relevant
sentence(s) from the retrieved context, which is guaranteed to be supported.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ..critic.model import HallucinationCritic


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


@dataclass
class CorrectionTrace:
    question: str
    context: str
    initial_answer: str
    final_answer: str
    iterations: int
    was_corrected: bool
    proba_history: list = field(default_factory=list)


class SelfCorrectingRAG:
    def __init__(self, critic: HallucinationCritic, max_iters: int = 2) -> None:
        self.critic = critic
        self.max_iters = max_iters

    def _extractive_fallback(self, question: str, context: str) -> str:
        """Return the context sentence most relevant to the question -> grounded."""
        sentences = _split_sentences(context)
        if not sentences:
            return context
        vec = TfidfVectorizer(token_pattern=r"[a-z0-9]+", lowercase=True)
        try:
            mat = vec.fit_transform(sentences + [question])
        except ValueError:
            return sentences[0]
        sims = cosine_similarity(mat[-1], mat[:-1]).ravel()
        return sentences[int(sims.argmax())]

    def answer(self, question: str, context: str, candidate: str) -> CorrectionTrace:
        proba_hist = [round(self.critic.predict_proba(candidate, context), 3)]
        current = candidate
        corrected = False
        it = 0
        while it < self.max_iters and self.critic.is_hallucinated(current, context):
            current = self._extractive_fallback(question, context)
            proba_hist.append(round(self.critic.predict_proba(current, context), 3))
            corrected = True
            it += 1
            if not self.critic.is_hallucinated(current, context):
                break
        return CorrectionTrace(
            question=question, context=context,
            initial_answer=candidate, final_answer=current,
            iterations=it, was_corrected=corrected, proba_history=proba_hist,
        )
