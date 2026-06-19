"""Phase 3 experiment: a real end-to-end RAG (retriever + generator + critic).

Run: python -m experiments.run_phase3
  1. Build a FAISS index over SQuAD v2 paragraphs (dense retriever).
  2. Measure retrieval quality: hit@1/3/5 (gold paragraph in top-k).
  3. Run the full pipeline on real questions with a generator that sometimes
     hallucinates, and measure the hallucination rate before vs after the
     self-correcting loop.
Outputs: console report, experiments/phase3_metrics.json, assets/phase3.png
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.real import ensure_squad # noqa: E402
from src.rag.retriever import DenseRetriever # noqa: E402
from src.rag.generator import StubLLMGenerator # noqa: E402
from src.rag.rag_system import EndToEndRAG # noqa: E402
from src.critic.model import HallucinationCritic # noqa: E402
from src.data.real import build_squad_dataset # noqa: E402
from sklearn.model_selection import train_test_split # noqa: E402


def _load_corpus(n_paras: int = 600):
    data = json.load(open(ensure_squad(ROOT / "data" / "squad_dev_v2.json"), encoding="utf-8"))
    paras, qa = [], []
    for art in data["data"]:
        for p in art["paragraphs"]:
            idx = len(paras)
            paras.append(p["context"])
            for q in p["qas"]:
                if not q["is_impossible"] and q["answers"]:
                    qa.append((q["question"], idx))
                    break
    paras = paras[:n_paras]
    qa = [(q, i) for q, i in qa if i < n_paras]
    return paras, qa


def main() -> dict:
    paras, qa = _load_corpus(600)
    t0 = time.time()
    retriever = DenseRetriever().build_index(paras)
    print(f"Indexed {len(paras)} passages in {time.time() - t0:.0f}s")

    # retrieval quality
    eval_q = qa[:300]
    hit = {}
    for k in (1, 3, 5):
        ok = 0
        for q, gold in eval_q:
            hits = retriever.search(q, k=k)
            ok += any(h.index == gold for h in hits)
        hit[k] = ok / len(eval_q)

    # end-to-end with a generator that hallucinates ~half the time
    train = train_test_split(build_squad_dataset(220, 17), test_size=0.3,
                             random_state=7, stratify=[e["label"] for e in
                             build_squad_dataset(220, 17)])[0]
    critic = HallucinationCritic(0.5).fit(train)
    rag = EndToEndRAG(retriever, StubLLMGenerator(p_hallucinate=0.5, seed=1),
                      critic, top_k=3, max_iters=2)

    sample = [q for q, _ in qa[300:400]]
    before = after = corrected = 0
    examples = []
    for q in sample:
        a = rag.answer(q)
        b = critic.is_hallucinated(a.trace.initial_answer, a.context)
        f = critic.is_hallucinated(a.trace.final_answer, a.context)
        before += b; after += f; corrected += a.trace.was_corrected
        if b and len(examples) < 3:
            examples.append(a)
    n = len(sample)

    print("=" * 70)
    print("PHASE 3 - end-to-end RAG (retriever + generator + critic)")
    print("=" * 70)
    print(f"Retrieval: hit@1={hit[1]:.1%} hit@3={hit[3]:.1%} hit@5={hit[5]:.1%}")
    print(f"End-to-end on {n} real questions (generator hallucinates ~50%):")
    print(f" hallucination rate before={before / n:.1%} after={after / n:.1%}")
    print(f" loop corrected {corrected}/{n} answers")
    print("\nExample catches:")
    for e in examples:
        print(f" Q: {e.question}")
        print(f" candidate: {e.trace.initial_answer[:90]}")
        print(f" P(halluc): {e.trace.proba_history} -> final: {e.trace.final_answer[:80]}")
    print("=" * 70)

    _plot(hit, before / n, after / n)
    result = {"retrieval_hit": hit, "n_questions": n,
              "before_rate": before / n, "after_rate": after / n, "corrected": corrected}
    (ROOT / "experiments" / "phase3_metrics.json").write_text(json.dumps(result, indent=2))
    return result


def _plot(hit: dict, before: float, after: float):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Phase 3 - end-to-end RAG", fontsize=14, weight="bold")
    ks = list(hit.keys())
    ax1.bar([f"hit@{k}" for k in ks], [hit[k] for k in ks], color="#4C78A8")
    ax1.set_ylim(0, 1); ax1.set_title("Dense retrieval quality (SQuAD v2)")
    for i, k in enumerate(ks):
        ax1.text(i, hit[k] + 0.02, f"{hit[k]:.0%}", ha="center", weight="bold")
    ax2.bar(["before", "after"], [before, after], color=["#E45756", "#54A24B"])
    ax2.set_ylim(0, 1); ax2.set_title("Hallucination rate (end-to-end loop)")
    for i, v in enumerate([before, after]):
        ax2.text(i, v + 0.02, f"{v:.0%}", ha="center", weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = ROOT / "assets" / "phase3.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"\nSaved figure -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
