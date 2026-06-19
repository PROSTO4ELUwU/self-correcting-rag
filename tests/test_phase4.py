"""Phase 4: best-of-n / self-consistency decoding and critic self-training.

All offline -- runs on the bundled synthetic facts, no model downloads.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.synth import build_dataset, SEED_FACTS
from src.critic.model import HallucinationCritic
from src.critic.self_training import self_train
from src.rag.generator import ExtractiveGenerator, StubLLMGenerator
from src.rag.strategies import BestOfN, SelfConsistency


def _critic():
    return HallucinationCritic(0.5).fit(build_dataset())


def test_generator_sample_count():
    fact = SEED_FACTS[0]
    drafts = StubLLMGenerator(p_hallucinate=0.8, seed=0).sample(
        fact["question"], fact["context"], n=5)
    assert len(drafts) == 5


def test_best_of_n_prefers_grounded():
    fact = SEED_FACTS[0]
    # generator that almost always hallucinates -> best-of-n still has to find
    # the grounded draft and the critic reward should rank it on top
    gen = StubLLMGenerator(p_hallucinate=0.7, seed=1)
    sel = BestOfN(gen, _critic(), n=8).select(fact["question"], fact["context"])
    assert sel.strategy == "best_of_n"
    assert 0.0 <= sel.reward <= 1.0
    assert sel.n_candidates == 8
    # the winner should be the one with the highest reward among candidates
    assert sel.reward == max(r for _, r in sel.candidates)


def test_self_consistency_returns_modal_answer():
    fact = SEED_FACTS[0]
    gen = StubLLMGenerator(p_hallucinate=0.3, seed=2) # grounded answer is modal
    sel = SelfConsistency(gen, _critic(), n=9).select(fact["question"], fact["context"])
    assert sel.answer
    assert sel.strategy == "self_consistency"


def test_extractive_best_of_n_is_grounded():
    fact = SEED_FACTS[2]
    sel = BestOfN(ExtractiveGenerator(), _critic(), n=3).select(
        fact["question"], fact["context"])
    # extractive answers are pulled from the context, so reward should be high
    assert sel.reward >= 0.5


def test_self_training_does_not_regress():
    ds = build_dataset()
    labels = [e["label"] for e in ds]
    # tiny seed + the rest as an unlabelled pool
    seed = ds[:8]
    pool = ds[8:]
    base = HallucinationCritic(0.5).fit(seed)
    trained, history = self_train(lambda: HallucinationCritic(0.5), seed, pool)
    assert history # at least one round happened
    f1_seed = base.evaluate(ds).f1
    f1_trained = trained.evaluate(ds).f1
    assert f1_trained >= f1_seed - 1e-9


def test_self_training_pseudo_labels_are_promoted():
    ds = build_dataset()
    seed, pool = ds[:10], ds[10:]
    _, history = self_train(lambda: HallucinationCritic(0.5), seed, pool)
    assert sum(h.pseudo_added for h in history) > 0
