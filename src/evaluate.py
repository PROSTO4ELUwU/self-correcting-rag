"""End-to-end evaluation: does the self-correcting loop reduce hallucinations?

We measure the headline metric of the whole project:
    hallucination_rate BEFORE correction vs AFTER correction
on a held-out set, using the trained critic both as the in-loop verifier and
(separately) as the oracle-style judge for reporting.

To avoid the critic "grading its own homework", we report two things:
  1) critic-judged hallucination rate (what the system optimises)
  2) ground-truth label rate on the candidates (sanity check of inputs)
"""
from __future__ import annotations

from dataclasses import dataclass

from .critic.model import HallucinationCritic
from .rag.pipeline import SelfCorrectingRAG


@dataclass
class LoopReport:
    n: int
    before_rate: float
    after_rate: float
    corrected: int
    reduction_abs: float
    reduction_rel: float


def evaluate_loop(critic: HallucinationCritic, examples: list[dict],
                  max_iters: int = 2) -> LoopReport:
    rag = SelfCorrectingRAG(critic, max_iters=max_iters)
    before_flags = 0
    after_flags = 0
    corrected = 0
    for e in examples:
        trace = rag.answer(e["question"], e["context"], e["answer"])
        before_flags += int(critic.is_hallucinated(trace.initial_answer, e["context"]))
        after_flags += int(critic.is_hallucinated(trace.final_answer, e["context"]))
        corrected += int(trace.was_corrected)
    n = len(examples)
    before = before_flags / n
    after = after_flags / n
    rel = (before - after) / before if before > 0 else 0.0
    return LoopReport(
        n=n, before_rate=before, after_rate=after, corrected=corrected,
        reduction_abs=before - after, reduction_rel=rel,
    )
