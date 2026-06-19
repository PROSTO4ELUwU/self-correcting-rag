"""End-to-end RAG: retrieve -> generate -> self-correct.

Ties together the Phase-3 retriever, a generator, and the critic-driven
self-correction loop into a single `answer(question)` call.

    retriever.retrieve_context(question) -> context
    generator.generate(question, context) -> candidate answer
    SelfCorrectingRAG(critic).answer(...) -> flag & repair hallucinations
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .retriever import DenseRetriever
from .pipeline import SelfCorrectingRAG, CorrectionTrace


@dataclass
class RAGAnswer:
    question: str
    context: str
    trace: CorrectionTrace
    retrieved_scores: list = field(default_factory=list)


class EndToEndRAG:
    def __init__(self, retriever: DenseRetriever, generator, critic,
                 top_k: int = 3, max_iters: int = 2) -> None:
        self.retriever = retriever
        self.generator = generator
        self.loop = SelfCorrectingRAG(critic, max_iters=max_iters)
        self.top_k = top_k

    def answer(self, question: str) -> RAGAnswer:
        hits = self.retriever.search(question, k=self.top_k)
        context = "\n\n".join(h.text for h in hits)
        candidate = self.generator.generate(question, context)
        trace = self.loop.answer(question, context, candidate)
        return RAGAnswer(question=question, context=context, trace=trace,
                         retrieved_scores=[round(h.score, 3) for h in hits])
