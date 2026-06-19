"""Interactive command-line demo of the self-correcting loop (fully offline).

Run:  python -m demo.cli
Trains the critic on the bundled synthetic facts, then lets you type a question
and a candidate answer; the loop flags + repairs hallucinations live.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.synth import build_dataset, SEED_FACTS
from src.critic.model import HallucinationCritic
from src.rag.pipeline import SelfCorrectingRAG


def build_rag() -> SelfCorrectingRAG:
    critic = HallucinationCritic(threshold=0.5).fit(build_dataset())
    return SelfCorrectingRAG(critic, max_iters=2)


def run_example(rag: SelfCorrectingRAG, question: str, context: str, candidate: str) -> None:
    trace = rag.answer(question, context, candidate)
    print(f"\nQ: {question}")
    print(f"candidate : {candidate}")
    print(f"P(halluc) : {trace.proba_history}")
    if trace.was_corrected:
        print(f"⚠️  flagged as hallucinated -> corrected after {trace.iterations} step(s)")
        print(f"final     : {trace.final_answer}")
    else:
        print("✅ grounded, returned as-is")


def main() -> None:
    print("Training critic on bundled facts...")
    rag = build_rag()
    fact = SEED_FACTS[0]
    print("\n--- canned examples ---")
    run_example(rag, fact["question"], fact["context"], fact["answer"])
    run_example(rag, fact["question"], fact["context"],
                "The Eiffel Tower is 450 metres tall and was completed in 1889.")

    print("\n--- interactive (Ctrl-C to quit) ---")
    print(f"context: {fact['context']}\n")
    try:
        while True:
            cand = input("candidate answer> ").strip()
            if not cand:
                continue
            run_example(rag, fact["question"], fact["context"], cand)
    except (KeyboardInterrupt, EOFError):
        print("\nbye!")


if __name__ == "__main__":
    main()
