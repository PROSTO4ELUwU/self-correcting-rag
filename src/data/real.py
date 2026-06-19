"""Real-data loader: build a hallucination-detection dataset from SQuAD v2.

Why SQuAD v2? It pairs real questions with real Wikipedia paragraphs and gold
answer spans. We turn it into a *grounding* dataset (is the answer supported by
THIS context?) using automatic labels -- no manual annotation:

  - GROUNDED (label 0):
        the context sentence that contains the gold answer span. It is, by
        construction, supported by the context.

  - HALLUCINATED (label 1), two natural sources:
        (a) "wrong-context": an answer-sentence taken from a *different*
            paragraph -> fluent, real text, but not supported here.
        (b) "corrupted": a grounded sentence with a number/entity swap
            (reuses src.data.synth corruptions) -> locally unsupported claim.

This is harder and more honest than the toy synthetic set: wrong-context
answers are real sentences with real entities, so the critic must rely on
genuine grounding signal, not surface weirdness.

Download (once):
    data/squad_dev_v2.json  from the official SQuAD-explorer repo.
"""
from __future__ import annotations

import json
import os
import random
import re
import urllib.request
from pathlib import Path

from .synth import corrupt_number, corrupt_entity

SQUAD_URL = ("https://raw.githubusercontent.com/rajpurkar/"
             "SQuAD-explorer/master/dataset/dev-v2.0.json")

_NUM_RE = re.compile(r"\d[\d,\.]*")
_ENT_RE = re.compile(r"\b[A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)*\b")


def corrupt_recombine(sentence: str, context: str, rng: random.Random) -> str | None:
    """HARD case: build a false claim using ONLY tokens that appear in the context.

    Swap a number (or entity) in the grounded sentence for a *different* one that
    also occurs elsewhere in the same context. Lexical-overlap and number-support
    features stay high (the token is "in context"), but the claim is now wrong --
    exactly the failure mode that needs semantic / NLI reasoning to catch.
    """
    # try numbers first
    sent_nums = _NUM_RE.findall(sentence)
    ctx_nums = [n for n in _NUM_RE.findall(context) if n not in sent_nums]
    if sent_nums and ctx_nums:
        return sentence.replace(rng.choice(sent_nums), rng.choice(ctx_nums), 1)
    # fall back to entity recombination
    def ents(t):
        return [e for e in _ENT_RE.findall(t)
                if e.split()[0] not in {"The", "It", "A", "An", "This", "In", "He", "She"}]
    sent_ents = ents(sentence)
    ctx_ents = [e for e in ents(context) if e not in sent_ents]
    if sent_ents and ctx_ents:
        return sentence.replace(rng.choice(sent_ents), rng.choice(ctx_ents), 1)
    return None


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 15]


def ensure_squad(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        urllib.request.urlretrieve(SQUAD_URL, path)
    return path


def _answer_sentence(context: str, answer_text: str) -> str | None:
    for s in _sentences(context):
        if answer_text and answer_text.lower() in s.lower():
            return s
    return None


def build_squad_dataset(n_contexts: int = 220, seed: int = 17,
                        path: str | Path = "data/squad_dev_v2.json",
                        include_hard: bool = False) -> list[dict]:
    """Return a balanced list of grounded/hallucinated examples from SQuAD v2.

    include_hard=False (default): the *main* dataset -- grounded, wrong-context
        and out-of-context-token corruptions. These are detectable from grounding
        signal, so this is the set the feature critic is trained/scored on.
    include_hard=True: also add `recombined_hard` examples (false claims built
        from in-context tokens). Use this to build the STRESS TEST that exposes
        the limits of a purely lexical critic -- see build_hard_stress_set.
    """
    rng = random.Random(seed)
    data = json.load(open(ensure_squad(path), encoding="utf-8"))

    # collect (question, context, answer_sentence) for answerable questions
    pool: list[dict] = []
    for art in data["data"]:
        for para in art["paragraphs"]:
            ctx = para["context"]
            for qa in para["qas"]:
                if qa["is_impossible"] or not qa["answers"]:
                    continue
                ans = qa["answers"][0]["text"]
                sent = _answer_sentence(ctx, ans)
                if sent:
                    pool.append({"question": qa["question"], "context": ctx,
                                 "answer": sent, "gold_span": ans})
                    break  # one per paragraph keeps contexts diverse

    rng.shuffle(pool)
    pool = pool[:n_contexts]
    if len(pool) < 10:
        raise RuntimeError("Not enough usable SQuAD paragraphs parsed.")

    examples: list[dict] = []
    for i, item in enumerate(pool):
        ctx = item["context"]
        # GROUNDED
        examples.append({"question": item["question"], "context": ctx,
                         "answer": item["answer"], "label": 0, "source": "grounded"})

        # HALLUCINATED (a): answer-sentence borrowed from a different paragraph
        j = rng.randrange(len(pool))
        while j == i:
            j = rng.randrange(len(pool))
        borrowed = pool[j]["answer"]
        if borrowed.lower() not in ctx.lower():
            examples.append({"question": item["question"], "context": ctx,
                             "answer": borrowed, "label": 1, "source": "wrong_context"})

        # HALLUCINATED (b): corrupt the grounded sentence (out-of-context token)
        corrupt = corrupt_number(item["answer"], ctx, rng) or \
            corrupt_entity(item["answer"], ctx, rng)
        if corrupt and corrupt != item["answer"]:
            examples.append({"question": item["question"], "context": ctx,
                             "answer": corrupt, "label": 1, "source": "corrupted"})

        # HALLUCINATED (c) HARD (opt-in): recombine in-context tokens -> false claim
        if include_hard:
            hard = corrupt_recombine(item["answer"], ctx, rng)
            if hard and hard != item["answer"]:
                examples.append({"question": item["question"], "context": ctx,
                                 "answer": hard, "label": 1, "source": "recombined_hard"})

    rng.shuffle(examples)
    return examples


def build_hard_stress_set(n_contexts: int = 220, seed: int = 17,
                          path: str | Path = "data/squad_dev_v2.json") -> list[dict]:
    """Held-out stress test: only `recombined_hard` (label 1) + `grounded` (label 0).

    These two classes are nearly identical at the lexical level, so a feature/
    overlap critic cannot separate them well. Use it to quantify the gap that a
    semantic NLI critic (Phase 2) must close.
    """
    full = build_squad_dataset(n_contexts, seed, path, include_hard=True)
    return [e for e in full if e["source"] in ("recombined_hard", "grounded")]


if __name__ == "__main__":
    ds = build_squad_dataset()
    pos = sum(e["label"] for e in ds)
    from collections import Counter
    print(f"SQuAD-derived dataset: {len(ds)} examples | "
          f"hallucinated={pos} | grounded={len(ds) - pos}")
    print("sources:", Counter(e["source"] for e in ds))
