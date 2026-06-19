"""Reproducible end-to-end demo.

Run:  python -m experiments.run_demo
Steps:
  1. Build synthetic labeled dataset (grounded vs hallucinated).
  2. Train/test split.
  3. Train the hallucination critic.
  4. Report critic classification metrics + feature importances.
  5. Run the self-correcting loop and report hallucination-rate before/after.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.synth import build_dataset          # noqa: E402
from src.critic.model import HallucinationCritic   # noqa: E402
from src.evaluate import evaluate_loop             # noqa: E402


def main() -> dict:
    ds = build_dataset(n_per_fact=6, seed=13)
    labels = [e["label"] for e in ds]
    train, test = train_test_split(ds, test_size=0.3, random_state=7, stratify=labels)

    critic = HallucinationCritic(threshold=0.5).fit(train)
    metrics = critic.evaluate(test)
    importance = critic.feature_importance()
    loop = evaluate_loop(critic, test, max_iters=2)

    print("=" * 60)
    print("SELF-CORRECTING RAG  —  demo run")
    print("=" * 60)
    print(f"dataset: {len(ds)} examples "
          f"(train={len(train)}, test={len(test)}, hallucinated={sum(labels)})")
    print("\n[1] Critic classification metrics (held-out test):")
    print("   ", metrics)
    print("    confusion [[TN,FP],[FN,TP]]:", metrics.confusion)
    print("\n[2] Feature importances (what the critic relies on):")
    for k, v in sorted(importance.items(), key=lambda x: -x[1]):
        print(f"    {k:<18} {v:.3f}")
    print("\n[3] Self-correction loop (hallucination rate):")
    print(f"    before = {loop.before_rate:.1%}   after = {loop.after_rate:.1%}")
    print(f"    corrected {loop.corrected}/{loop.n} answers")
    print(f"    absolute reduction = {loop.reduction_abs:.1%} | "
          f"relative reduction = {loop.reduction_rel:.1%}")
    print("=" * 60)

    return {
        "dataset_size": len(ds),
        "critic": metrics.__dict__,
        "feature_importance": importance,
        "loop": loop.__dict__,
    }


if __name__ == "__main__":
    result = main()
    out = ROOT / "experiments" / "last_run_metrics.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"\nSaved metrics -> {out.relative_to(ROOT)}")
