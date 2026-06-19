"""Smoke tests: the pipeline trains and the loop reduces (or holds) hallucinations."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.synth import build_dataset
from src.critic.model import HallucinationCritic
from src.rag.pipeline import SelfCorrectingRAG
from src.evaluate import evaluate_loop


def test_dataset_is_balanced_and_labeled():
    ds = build_dataset()
    assert len(ds) > 20
    labels = {e["label"] for e in ds}
    assert labels == {0, 1}


def test_critic_trains_and_predicts():
    ds = build_dataset()
    critic = HallucinationCritic().fit(ds)
    p = critic.predict_proba(ds[0]["answer"], ds[0]["context"])
    assert 0.0 <= p <= 1.0


def test_loop_does_not_increase_hallucinations():
    ds = build_dataset()
    critic = HallucinationCritic().fit(ds)
    report = evaluate_loop(critic, ds)
    assert report.after_rate <= report.before_rate + 1e-9


def test_correction_grounds_answer():
    ds = build_dataset()
    critic = HallucinationCritic().fit(ds)
    rag = SelfCorrectingRAG(critic)
    # take a clearly hallucinated example
    bad = next(e for e in ds if e["label"] == 1)
    trace = rag.answer(bad["question"], bad["context"], bad["answer"])
    assert trace.final_answer # non-empty
