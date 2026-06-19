"""Decoding strategies that use the critic as a reward signal.

The self-correction loop in `pipeline.py` is reactive: generate once, and only
fix the answer if the critic complains. Phase 4 turns the critic into something
the *generation* step optimises against, so we stop bad answers before they are
returned instead of patching them afterwards.

Two strategies live here. BestOfN samples N candidates and keeps the one the
critic trusts most - plain rejection sampling with reward = 1 - P(halluc).
SelfConsistency samples N candidates and returns whichever answer shows up most
often; it's a reward-free baseline that only works when the grounded answer is
the modal one, which makes it a useful sanity check against the reward-driven
version.

Both expect a generator exposing `sample(question, context, n)` (see
`generator.py`). Any critic with `predict_proba(answer, context)` plugs in.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field


@dataclass
class Selection:
    """What a strategy picked, plus enough trace to explain the choice."""
    answer: str
    reward: float # 1 - P(hallucinated) of the winner
    n_candidates: int
    candidates: list = field(default_factory=list) # (answer, reward) pairs
    strategy: str = ""


def _reward(critic, answer: str, context: str) -> float:
    # grounding reward: high when the critic thinks the answer is supported
    return 1.0 - critic.predict_proba(answer, context)


class BestOfN:
    """Reward-guided decoding: draw N answers, return the best-scoring one.

    This is the simplest form of "optimise the answer for the critic". With a
    real LLM you would raise the sampling temperature to get diverse drafts;
    here the stub generator already injects hallucinations at random, so the
    candidates differ run to run.
    """

    def __init__(self, generator, critic, n: int = 5) -> None:
        self.generator = generator
        self.critic = critic
        self.n = n

    def select(self, question: str, context: str) -> Selection:
        drafts = self.generator.sample(question, context, self.n)
        scored = [(d, _reward(self.critic, d, context)) for d in drafts]
        # ties broken by the first occurrence, which keeps it deterministic
        best, best_r = max(scored, key=lambda t: t[1])
        return Selection(
            answer=best, reward=round(best_r, 3), n_candidates=len(drafts),
            candidates=[(d, round(r, 3)) for d, r in scored], strategy="best_of_n",
        )


class SelfConsistency:
    """Majority vote over N samples -- a reward-free baseline.

    No critic involved in picking the winner; we just return the most common
    draft. It tends to surface the grounded answer when the generator repeats
    it more often than any single hallucination, and falls apart when the
    hallucinations agree with each other. We still report the winner's reward
    so it can be compared against BestOfN on equal footing.
    """

    def __init__(self, generator, critic=None, n: int = 5) -> None:
        self.generator = generator
        self.critic = critic
        self.n = n

    def select(self, question: str, context: str) -> Selection:
        drafts = self.generator.sample(question, context, self.n)
        winner, _ = Counter(drafts).most_common(1)[0]
        reward = _reward(self.critic, winner, context) if self.critic else 0.0
        return Selection(
            answer=winner, reward=round(reward, 3), n_candidates=len(drafts),
            candidates=[(d, drafts.count(d)) for d in dict.fromkeys(drafts)],
            strategy="self_consistency",
        )
