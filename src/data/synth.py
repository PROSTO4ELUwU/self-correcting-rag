"""Synthetic dataset generation for training the hallucination critic.

Core idea (weak supervision):
We do NOT hand-label thousands of examples. Instead we take FAITHFUL
(answer, context) pairs and programmatically *corrupt* the answers to
produce HALLUCINATED examples with known labels. This gives us a balanced,
fully-labeled dataset for free.

Each example is a dict:
    {
        "question": str,
        "context":  str,   # the retrieved source passage(s)
        "answer":   str,   # candidate answer to judge
        "label":    int,   # 1 = hallucinated / unsupported, 0 = grounded
        "corruption": str, # which corruption was applied ("none" if grounded)
    }
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, asdict
from typing import Callable

# ---------------------------------------------------------------------------
# A small knowledge base of (passage, question, faithful answer) triples.
# Kept compact & offline so the project runs with zero external downloads.
# In the "real" version you'd swap this for SQuAD / HotpotQA / your own docs.
# ---------------------------------------------------------------------------
SEED_FACTS: list[dict] = [
    {
        "context": "The Eiffel Tower was completed in 1889 and stands 330 metres tall. "
                   "It was designed by the engineer Gustave Eiffel for the World's Fair in Paris.",
        "question": "How tall is the Eiffel Tower and when was it completed?",
        "answer": "The Eiffel Tower is 330 metres tall and was completed in 1889.",
    },
    {
        "context": "Insulin is a hormone produced by the pancreas that regulates blood sugar. "
                   "It was first isolated in 1921 by Frederick Banting and Charles Best.",
        "question": "What does insulin do and who isolated it?",
        "answer": "Insulin regulates blood sugar and was first isolated in 1921 by Banting and Best.",
    },
    {
        "context": "The Amazon River discharges about 209,000 cubic metres of water per second, "
                   "more than the next seven largest rivers combined. It flows mainly through Brazil.",
        "question": "How much water does the Amazon River discharge?",
        "answer": "The Amazon River discharges about 209,000 cubic metres of water per second.",
    },
    {
        "context": "Python 3.0 was released in December 2008. It was a major, backwards-incompatible "
                   "release that introduced print as a function and better Unicode support.",
        "question": "When was Python 3.0 released and what changed?",
        "answer": "Python 3.0 was released in December 2008 and made print a function with better Unicode.",
    },
    {
        "context": "The human heart has four chambers: two atria and two ventricles. "
                   "It beats around 100,000 times per day on average.",
        "question": "How many chambers does the human heart have?",
        "answer": "The human heart has four chambers and beats around 100,000 times per day.",
    },
    {
        "context": "Mount Everest is 8,849 metres tall and is located in the Himalayas on the "
                   "border between Nepal and the Tibet region of China.",
        "question": "How tall is Mount Everest and where is it?",
        "answer": "Mount Everest is 8,849 metres tall and lies on the Nepal-China border.",
    },
    {
        "context": "The speed of light in a vacuum is approximately 299,792 kilometres per second. "
                   "This constant is denoted by the letter c in physics.",
        "question": "What is the speed of light?",
        "answer": "The speed of light in a vacuum is about 299,792 kilometres per second.",
    },
    {
        "context": "Shakespeare wrote Romeo and Juliet around 1595. The play is a tragedy about two "
                   "young lovers from feuding families in Verona.",
        "question": "Who wrote Romeo and Juliet and when?",
        "answer": "Shakespeare wrote Romeo and Juliet around 1595, a tragedy set in Verona.",
    },
    {
        "context": "Photosynthesis converts carbon dioxide and water into glucose and oxygen using "
                   "light energy, mainly in the chloroplasts of plant cells.",
        "question": "What does photosynthesis produce?",
        "answer": "Photosynthesis produces glucose and oxygen from carbon dioxide and water.",
    },
    {
        "context": "The Great Wall of China is over 21,000 kilometres long and was built over many "
                   "dynasties, with major construction during the Ming dynasty.",
        "question": "How long is the Great Wall of China?",
        "answer": "The Great Wall of China is over 21,000 kilometres long.",
    },
]

_NUMBER_RE = re.compile(r"\d[\d,\.]*")


# ---------------------------------------------------------------------------
# Corruption functions: each takes a faithful answer + context and returns a
# corrupted (hallucinated) answer. They model the real failure modes of RAG.
# ---------------------------------------------------------------------------
def corrupt_number(answer: str, context: str, rng: random.Random) -> str | None:
    """Swap a number for a plausible-but-wrong one (most common RAG error)."""
    nums = _NUMBER_RE.findall(answer)
    if not nums:
        return None
    target = rng.choice(nums)
    digits = re.sub(r"[^\d]", "", target)
    if not digits:
        return None
    val = int(digits)
    delta = max(1, int(val * rng.uniform(0.15, 0.6)))
    new_val = val + (delta if rng.random() < 0.5 else -delta)
    new_val = abs(new_val) or val + 1
    formatted = format(new_val, ",") if "," in target else str(new_val)
    return answer.replace(target, formatted, 1)


def corrupt_entity(answer: str, context: str, rng: random.Random) -> str | None:
    """Replace a capitalised entity with an unrelated one."""
    fake_entities = ["Albert Einstein", "Tokyo", "the Pacific Ocean", "Isaac Newton",
                     "London", "Leonardo da Vinci", "the Sahara", "Marie Curie"]
    caps = re.findall(r"\b[A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)*\b", answer)
    caps = [c for c in caps if c.split()[0] not in {"The", "It", "A", "An"}]
    if not caps:
        return None
    target = rng.choice(caps)
    replacement = rng.choice([e for e in fake_entities if e not in target])
    return answer.replace(target, replacement, 1)


def corrupt_unsupported_fact(answer: str, context: str, rng: random.Random) -> str | None:
    """Append a fluent but completely unsupported claim."""
    additions = [
        " It is widely regarded as the largest of its kind in the world.",
        " This was later confirmed by NASA in 2017.",
        " Most experts agree it costs over two billion dollars.",
        " It holds the official Guinness World Record.",
        " The discovery won a Nobel Prize the following year.",
    ]
    return answer + rng.choice(additions)


def corrupt_negation(answer: str, context: str, rng: random.Random) -> str | None:
    """Flip the meaning by inserting a negation."""
    for verb in [" is ", " was ", " produces ", " regulates ", " discharges ", " has "]:
        if verb in answer:
            return answer.replace(verb, verb + "not ", 1)
    return None


CORRUPTIONS: dict[str, Callable] = {
    "number": corrupt_number,
    "entity": corrupt_entity,
    "unsupported": corrupt_unsupported_fact,
    "negation": corrupt_negation,
}


@dataclass
class Example:
    question: str
    context: str
    answer: str
    label: int
    corruption: str

    def to_dict(self) -> dict:
        return asdict(self)


def build_dataset(n_per_fact: int = 6, seed: int = 13) -> list[dict]:
    """Generate a balanced, labeled dataset of grounded vs hallucinated answers."""
    rng = random.Random(seed)
    examples: list[Example] = []

    for fact in SEED_FACTS:
        # 1) the faithful answer -> label 0
        examples.append(Example(fact["question"], fact["context"],
                                 fact["answer"], 0, "none"))

        # 2) several corrupted variants -> label 1
        names = list(CORRUPTIONS.keys())
        rng.shuffle(names)
        added = 0
        for name in names:
            if added >= n_per_fact:
                break
            corrupted = CORRUPTIONS[name](fact["answer"], fact["context"], rng)
            if corrupted and corrupted != fact["answer"]:
                examples.append(Example(fact["question"], fact["context"],
                                        corrupted, 1, name))
                added += 1

        # 3) light paraphrase of the faithful answer -> extra label-0 signal
        para = fact["answer"].replace("about ", "approximately ").replace("around ", "roughly ")
        if para != fact["answer"]:
            examples.append(Example(fact["question"], fact["context"], para, 0, "paraphrase"))

    rng.shuffle(examples)
    return [e.to_dict() for e in examples]


if __name__ == "__main__":
    ds = build_dataset()
    pos = sum(e["label"] for e in ds)
    print(f"Generated {len(ds)} examples | hallucinated={pos} | grounded={len(ds) - pos}")
    for e in ds[:3]:
        print(e)
