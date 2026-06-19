"""Real-data experiment on SQuAD v2 + a hard stress test, with plots.

Run:  python -m experiments.run_real

Produces:
  - console report (main metrics + stress-test gap)
  - experiments/real_run_metrics.json
  - assets/results.png   (feature importance, before/after, per-source recall, calibration)
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.real import build_squad_dataset, build_hard_stress_set  # noqa: E402
from src.critic.model import HallucinationCritic                      # noqa: E402
from src.evaluate import evaluate_loop                                # noqa: E402


def per_source_recall(critic, examples):
    by = defaultdict(lambda: [0, 0])
    for e in examples:
        pred = int(critic.is_hallucinated(e["answer"], e["context"]))
        correct = int(pred == e["label"])
        by[e["source"]][0] += correct
        by[e["source"]][1] += 1
    return {s: c / t for s, (c, t) in by.items()}


def main() -> dict:
    # ---- main dataset (detectable hallucinations) ----
    ds = build_squad_dataset(n_contexts=220, seed=17, include_hard=False)
    y = [e["label"] for e in ds]
    train, test = train_test_split(ds, test_size=0.3, random_state=7, stratify=y)
    critic = HallucinationCritic(threshold=0.5).fit(train)
    metrics = critic.evaluate(test)
    loop = evaluate_loop(critic, test, max_iters=2)
    src_acc = per_source_recall(critic, test)

    # ---- hard stress test (recombined vs grounded) ----
    hard = build_hard_stress_set(n_contexts=220, seed=17)
    hard_metrics = critic.evaluate(hard)
    hard_recall = per_source_recall(critic, hard)

    print("=" * 64)
    print("SELF-CORRECTING RAG — real data (SQuAD v2)")
    print("=" * 64)
    print(f"dataset: {len(ds)} | sources: {dict(Counter(e['source'] for e in ds))}")
    print(f"\n[1] Critic (held-out test):  {metrics}")
    print(f"    confusion [[TN,FP],[FN,TP]]: {metrics.confusion}")
    print("\n[2] Per-source accuracy:")
    for s, a in sorted(src_acc.items()):
        print(f"    {s:<16} {a:.0%}")
    print(f"\n[3] Self-correction loop: before={loop.before_rate:.1%} "
          f"after={loop.after_rate:.1%} (rel. reduction {loop.reduction_rel:.1%})")
    print("\n[4] HARD stress test (recombined-in-context vs grounded):")
    print(f"    {hard_metrics}")
    print(f"    recombined_hard caught: {hard_recall.get('recombined_hard', 0):.0%} "
          f"<-- the gap a semantic/NLI critic (Phase 2) must close")
    print("=" * 64)

    _make_plots(critic, test, metrics, loop, src_acc, hard_recall)

    result = {
        "dataset_size": len(ds),
        "critic": metrics.__dict__,
        "per_source_accuracy": src_acc,
        "loop": loop.__dict__,
        "hard_stress": {"metrics": hard_metrics.__dict__, "recall": hard_recall},
    }
    (ROOT / "experiments" / "real_run_metrics.json").write_text(json.dumps(result, indent=2))
    return result


def _make_plots(critic, test, metrics, loop, src_acc, hard_recall):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Self-Correcting RAG — results on SQuAD v2", fontsize=15, weight="bold")

    # (a) feature importance
    fi = critic.feature_importance()
    items = sorted(fi.items(), key=lambda x: x[1])
    axes[0, 0].barh([k for k, _ in items], [v for _, v in items], color="#4C78A8")
    axes[0, 0].set_title("Critic feature importance")
    axes[0, 0].set_xlabel("importance")

    # (b) before/after hallucination rate
    axes[0, 1].bar(["before", "after"], [loop.before_rate, loop.after_rate],
                   color=["#E45756", "#54A24B"])
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].set_title(f"Hallucination rate (−{loop.reduction_rel:.0%} relative)")
    for i, v in enumerate([loop.before_rate, loop.after_rate]):
        axes[0, 1].text(i, v + 0.02, f"{v:.0%}", ha="center", weight="bold")

    # (c) per-source detection (main + hard)
    combined = {**src_acc}
    if "recombined_hard" in hard_recall:
        combined["recombined_hard"] = hard_recall["recombined_hard"]
    keys = list(combined.keys())
    colors = ["#54A24B" if combined[k] > 0.6 else "#E45756" for k in keys]
    axes[1, 0].bar(keys, [combined[k] for k in keys], color=colors)
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].set_title("Accuracy by hallucination type")
    axes[1, 0].tick_params(axis="x", rotation=30)

    # (d) calibration curve
    answers = [e["answer"] for e in test]
    contexts = [e["context"] for e in test]
    X = critic.extractor.transform(answers, contexts)
    proba = critic.clf.predict_proba(X)[:, 1]
    yt = np.array([e["label"] for e in test])
    frac_pos, mean_pred = calibration_curve(yt, proba, n_bins=8, strategy="quantile")
    axes[1, 1].plot([0, 1], [0, 1], "--", color="gray", label="perfect")
    axes[1, 1].plot(mean_pred, frac_pos, "o-", color="#4C78A8", label="critic")
    axes[1, 1].set_title("Critic calibration")
    axes[1, 1].set_xlabel("predicted P(hallucinated)")
    axes[1, 1].set_ylabel("observed fraction")
    axes[1, 1].legend()

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = ROOT / "assets" / "results.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"\nSaved figure -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
