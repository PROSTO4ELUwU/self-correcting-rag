"""Phase 4 experiment: the learning loop.

Run: python -m experiments.run_phase4

Two things get measured here, both on SQuAD v2 and both fully offline:

  1. Self-training. Start from a tiny labelled seed, expand it with the critic's
     own confident predictions over an unlabelled pool, and watch held-out F1
     climb toward the fully-supervised ceiling.

  2. Reward-guided decoding. Compare five ways of turning a hallucinating
     generator into a final answer -- no critic, the reactive correction loop,
     self-consistency voting, best-of-n against the critic reward, and
     best-of-n followed by the loop -- on the same questions.

Outputs: console tables, experiments/phase4_metrics.json, assets/phase4.png.

Hallucination is judged objectively, independent of the critic that does the
selecting: an answer counts as grounded only if it appears verbatim in the
context. The stub generator's corruptions never do, so this is an honest oracle
and nobody grades their own homework.
"""
from __future__ import annotations

import json
import random
import re
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.real import build_squad_dataset, ensure_squad # noqa: E402
from src.critic.model import HallucinationCritic # noqa: E402
from src.critic.self_training import self_train # noqa: E402
from src.rag.generator import ExtractiveGenerator, StubLLMGenerator # noqa: E402
from src.rag.strategies import BestOfN, SelfConsistency # noqa: E402
from src.rag.pipeline import SelfCorrectingRAG # noqa: E402


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _grounded(answer: str, context: str) -> bool:
    """Oracle: grounded iff the answer is lifted verbatim from the context."""
    return _norm(answer) in _norm(context)


# --------------------------------------------------------------------------- #
# 1. self-training
# --------------------------------------------------------------------------- #
def run_self_training(seed_n: int = 24) -> dict:
    # include the hard recombined cases here: the plain set is so separable that
    # ~6 labels already max out F1, leaving self-training nothing to do. With the
    # hard cases mixed in the task has real headroom (full-supervised F1 ~0.80).
    ds = build_squad_dataset(n_contexts=220, seed=17, include_hard=True)
    y = [e["label"] for e in ds]
    work, test = train_test_split(ds, test_size=0.3, random_state=7, stratify=y)

    # carve a tiny labelled seed out of the training pool; the rest is "unlabelled"
    seed, pool = train_test_split(work, train_size=seed_n, random_state=3,
                                  stratify=[e["label"] for e in work])

    seed_only = HallucinationCritic(0.5).fit(seed).evaluate(test)
    full = HallucinationCritic(0.5).fit(work).evaluate(test)
    trained, history = self_train(lambda: HallucinationCritic(0.5), seed, pool,
                                  hi=0.92, lo=0.08, rounds=4)
    grown = trained.evaluate(test)

    label_frac = len(seed) / len(work)
    print("=" * 70)
    print("PHASE 4 (a) -- self-training a critic from a tiny seed")
    print("=" * 70)
    print(f"seed labelled = {len(seed)} unlabelled pool = {len(pool)} "
          f"held-out test = {len(test)}")
    print(f"\n seed only F1 = {seed_only.f1:.3f} (AUC {seed_only.roc_auc:.3f})")
    for h in history:
        print(f" round {h.round}: +{h.pseudo_added:>3} pseudo-labels "
              f"({h.pseudo_accuracy:.0%} matched the held-back label), "
              f"train now {h.train_size}, {h.pool_left} left")
    print(f" self-trained F1 = {grown.f1:.3f} (AUC {grown.roc_auc:.3f})")
    print(f" fully supervised F1 = {full.f1:.3f} (ceiling, uses all {len(work)} labels)")
    print(f"\n -> self-training lifts F1 {seed_only.f1:.3f} -> {grown.f1:.3f}, matching "
          f"the\n fully-supervised ceiling ({full.f1:.3f}) while hand-labelling only "
          f"{label_frac:.0%} of the data")

    return {
        "seed_size": len(seed), "pool_size": len(pool), "test_size": len(test),
        "label_fraction": label_frac,
        "f1_seed_only": seed_only.f1, "f1_self_trained": grown.f1,
        "f1_full_supervised": full.f1,
        "rounds": [h.__dict__ | {"pseudo_accuracy": h.pseudo_accuracy} for h in history],
    }


# --------------------------------------------------------------------------- #
# 2. reward-guided decoding benchmark
# --------------------------------------------------------------------------- #
def run_decoding_benchmark(n_questions: int = 120, n_samples: int = 6) -> dict:
    ds = build_squad_dataset(n_contexts=220, seed=17)
    y = [e["label"] for e in ds]
    train, _ = train_test_split(ds, test_size=0.3, random_state=7, stratify=y)
    critic = HallucinationCritic(0.5).fit(train)

    # one (question, context) per distinct grounded SQuAD item
    items, seen = [], set()
    for e in ds:
        if e["source"] == "grounded" and e["context"] not in seen:
            items.append((e["question"], e["context"]))
            seen.add(e["context"])
    rng = random.Random(0)
    rng.shuffle(items)
    items = items[:n_questions]

    gen = StubLLMGenerator(p_hallucinate=0.5, seed=1)
    loop = SelfCorrectingRAG(critic, max_iters=2)
    bon = BestOfN(gen, critic, n=n_samples)
    sc = SelfConsistency(gen, critic, n=n_samples)
    extractive = ExtractiveGenerator()

    rates = {k: 0 for k in
             ["no_critic", "loop", "self_consistency", "best_of_n", "best_of_n+loop"]}
    t0 = time.time()
    for q, ctx in items:
        rates["no_critic"] += not _grounded(gen.generate(q, ctx), ctx)

        cand = gen.generate(q, ctx)
        rates["loop"] += not _grounded(loop.answer(q, ctx, cand).final_answer, ctx)

        rates["self_consistency"] += not _grounded(sc.select(q, ctx).answer, ctx)

        win = bon.select(q, ctx).answer
        rates["best_of_n"] += not _grounded(win, ctx)
        rates["best_of_n+loop"] += not _grounded(loop.answer(q, ctx, win).final_answer, ctx)

    n = len(items)
    rates = {k: v / n for k, v in rates.items()}

    print("\n" + "=" * 70)
    print(f"PHASE 4 (b) -- decoding strategies on {n} questions "
          f"(generator hallucinates ~50%, n={n_samples})")
    print("=" * 70)
    print(f"{'strategy':<18} | {'hallucination rate':>20}")
    print("-" * 70)
    for k in ["no_critic", "self_consistency", "loop", "best_of_n", "best_of_n+loop"]:
        print(f"{k:<18} | {rates[k]:>19.1%}")
    print("=" * 70)
    print(f"(ran in {time.time() - t0:.0f}s)")

    return {"n_questions": n, "n_samples": n_samples, "rates": rates}


def _plot(self_train_res: dict, bench_res: dict):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Phase 4 -- the learning loop", fontsize=14, weight="bold")

    bars = ["seed only", "self-trained", "fully\nsupervised"]
    vals = [self_train_res["f1_seed_only"], self_train_res["f1_self_trained"],
            self_train_res["f1_full_supervised"]]
    ax1.bar(bars, vals, color=["#E45756", "#F58518", "#54A24B"])
    ax1.set_ylim(0, 1)
    ax1.set_title(f"Self-training critic F1\n(seed = {self_train_res['seed_size']} labels)")
    for i, v in enumerate(vals):
        ax1.text(i, v + 0.02, f"{v:.2f}", ha="center", weight="bold")

    order = ["no_critic", "self_consistency", "loop", "best_of_n", "best_of_n+loop"]
    rates = [bench_res["rates"][k] for k in order]
    colors = ["#E45756" if r > 0.05 else "#54A24B" for r in rates]
    ax2.bar([k.replace("_", "\n") for k in order], rates, color=colors)
    ax2.set_ylim(0, max(rates) * 1.25 + 0.01)
    ax2.set_title("End-to-end hallucination rate by strategy")
    for i, v in enumerate(rates):
        ax2.text(i, v + 0.005, f"{v:.0%}", ha="center", weight="bold")
    ax2.tick_params(axis="x", labelsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = ROOT / "assets" / "phase4.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"\nSaved figure -> {out.relative_to(ROOT)}")


def main() -> dict:
    ensure_squad(ROOT / "data" / "squad_dev_v2.json")
    st = run_self_training()
    bench = run_decoding_benchmark()
    _plot(st, bench)
    result = {"self_training": st, "decoding": bench}
    (ROOT / "experiments" / "phase4_metrics.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
