"""Self-training: grow the critic's training set from unlabelled answers.

Labelling grounding examples by hand is the expensive part. Self-training side-
steps it: train on a small labelled seed, let that critic score a pile of
unlabelled (answer, context) pairs, and promote only the predictions it is very
sure about to pseudo-labels. Fold those back into the training set and refit.
Repeat for a few rounds.

The confidence gate (hi / lo) is what keeps it honest -- we only trust a
pseudo-label when P(hallucinated) is near 1 or near 0, and leave the ambiguous
middle out. Run `experiments/run_phase4.py` to see a small seed climb toward the
fully-supervised ceiling this way.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RoundStats:
    round: int
    train_size: int
    pseudo_added: int
    pseudo_correct: int # vs the held-back true label, for reporting only
    pool_left: int

    @property
    def pseudo_accuracy(self) -> float:
        return self.pseudo_correct / self.pseudo_added if self.pseudo_added else 0.0


def self_train(critic_factory, seed_labeled: list[dict], unlabeled: list[dict],
               hi: float = 0.85, lo: float = 0.15, rounds: int = 3):
    """Iteratively expand `seed_labeled` with confident pseudo-labels.

    critic_factory : zero-arg callable returning a fresh, untrained critic
        (e.g. ``lambda: HallucinationCritic(0.5)``).
    seed_labeled : small set of dicts with answer / context / label.
    unlabeled : larger pool of dicts. Any ``label`` they carry is used only to
        measure pseudo-label accuracy, never to train.
    hi, lo : confidence thresholds for promoting a prediction.
    rounds : number of refit cycles.

    Returns (final_critic, list[RoundStats]).
    """
    train = [dict(e) for e in seed_labeled]
    pool = [dict(e) for e in unlabeled]
    history: list[RoundStats] = []

    for r in range(1, rounds + 1):
        critic = critic_factory().fit(train)
        added, correct, leftover = [], 0, []
        for ex in pool:
            p = critic.predict_proba(ex["answer"], ex["context"])
            if p >= hi:
                pseudo = 1
            elif p <= lo:
                pseudo = 0
            else:
                leftover.append(ex) # too unsure, look again next round
                continue
            if "label" in ex and ex["label"] == pseudo:
                correct += 1
            promoted = dict(ex)
            promoted["label"] = pseudo
            added.append(promoted)

        train.extend(added)
        pool = leftover
        history.append(RoundStats(r, len(train), len(added), correct, len(pool)))
        if not added: # nothing new to learn from
            break

    final = critic_factory().fit(train)
    return final, history
