"""Phase 2 experiment: feature critic vs NLI critic vs hybrid ensemble.

Run: python -m experiments.run_phase2
Produces a console ablation table, experiments/phase2_metrics.json, and
assets/phase2_comparison.png (F1 on the main set + recall on hard cases).

The headline finding: a small zero-shot NLI model detects the *contradiction*
hallucinations the lexical critic is blind to, lifting hard-case recall ~6% ->
~40%+, while the ensemble keeps the feature critic's strong main-set F1.
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.real import build_squad_dataset, build_hard_stress_set # noqa: E402
from src.critic.model import HallucinationCritic # noqa: E402
from src.critic.nli_critic import NLICritic, EnsembleCritic # noqa: E402


def _balanced_sample(examples, n_per_class, seed=0):
    rng = random.Random(seed)
    pos = [e for e in examples if e["label"] == 1]
    neg = [e for e in examples if e["label"] == 0]
    rng.shuffle(pos); rng.shuffle(neg)
    out = pos[:n_per_class] + neg[:n_per_class]
    rng.shuffle(out)
    return out


def _hard_recall(critic, examples):
    pos = [e for e in examples if e["label"] == 1]
    return sum(critic.is_hallucinated(e["answer"], e["context"]) for e in pos) / len(pos)


def main() -> dict:
    main_ds = build_squad_dataset(n_contexts=220, seed=17, include_hard=False)
    y = [e["label"] for e in main_ds]
    train, test = train_test_split(main_ds, test_size=0.3, random_state=7, stratify=y)
    # subsample the main test set to keep NLI inference fast
    test_s = _balanced_sample(test, 60)
    hard_s = _balanced_sample(build_hard_stress_set(220, 17), 70)

    feature = HallucinationCritic(0.5).fit(train)
    nli = NLICritic().fit()
    ensemble = EnsembleCritic(HallucinationCritic(0.5).fit(train), NLICritic())

    critics = {"feature": feature, "NLI": nli, "ensemble": ensemble}
    rows = {}
    t0 = time.time()
    for name, c in critics.items():
        main_m = c.evaluate(test_s)
        hard_m = c.evaluate(hard_s)
        rows[name] = {
            "main_f1": main_m.f1, "main_auc": main_m.roc_auc,
            "hard_auc": hard_m.roc_auc, "hard_recall": _hard_recall(c, hard_s),
            "hard_precision": hard_m.precision,
        }

    print("=" * 78)
    print("PHASE 2 ABLATION - feature vs NLI vs ensemble")
    print("=" * 78)
    print(f"{'critic':10} | {'main F1':>8} {'main AUC':>9} | "
          f"{'hard AUC':>9} {'hard recall':>12} {'hard prec':>10}")
    print("-" * 78)
    for name, r in rows.items():
        print(f"{name:10} | {r['main_f1']:8.3f} {r['main_auc']:9.3f} | "
              f"{r['hard_auc']:9.3f} {r['hard_recall']:11.0%} {r['hard_precision']:10.3f}")
    print("=" * 78)
    print(f"(evaluated in {time.time() - t0:.0f}s; main test n={len(test_s)}, hard n={len(hard_s)})")
    print("\nTakeaway: the lexical critic owns the main set; the NLI critic recovers "
          "contradiction\nhallucinations it misses; the ensemble keeps both strengths.")

    _plot(rows)
    out = ROOT / "experiments" / "phase2_metrics.json"
    out.write_text(json.dumps(rows, indent=2))
    return rows


def _plot(rows: dict):
    names = list(rows.keys())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Phase 2 - adding an NLI critic", fontsize=14, weight="bold")

    ax1.bar(names, [rows[n]["main_f1"] for n in names], color="#4C78A8")
    ax1.set_ylim(0, 1); ax1.set_title("Main-set F1 (easy + medium cases)")
    for i, n in enumerate(names):
        ax1.text(i, rows[n]["main_f1"] + 0.02, f"{rows[n]['main_f1']:.2f}",
                 ha="center", weight="bold")

    colors = ["#E45756", "#F58518", "#54A24B"]
    ax2.bar(names, [rows[n]["hard_recall"] for n in names], color=colors[:len(names)])
    ax2.set_ylim(0, 1); ax2.set_title("Hard-case recall (recombined-in-context)")
    for i, n in enumerate(names):
        ax2.text(i, rows[n]["hard_recall"] + 0.02, f"{rows[n]['hard_recall']:.0%}",
                 ha="center", weight="bold")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = ROOT / "assets" / "phase2_comparison.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"\nSaved figure -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
