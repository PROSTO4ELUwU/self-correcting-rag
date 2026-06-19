"""Phase 2 critic: a Natural Language Inference (NLI) hallucination detector.

Instead of lexical/overlap features, this critic asks an NLI model whether each
claim in the answer is *entailed* by the context. A claim that is not entailed
(neutral or contradicted) is unsupported -> a hallucination. This reasons about
meaning, so it catches "recombined-in-context" hallucinations that look
lexically identical to grounded answers -- the exact failure mode of the
feature baseline (see experiments/run_real.py).

Model: cross-encoder/nli-deberta-v3-xsmall (labels: contradiction/entailment/neutral).
Zero-shot: no training needed; `fit` only (optionally) calibrates the threshold.

Same interface as HallucinationCritic so the self-correcting loop is unchanged:
    predict_proba(answer, context) -> float in [0, 1]
    is_hallucinated(answer, context) -> bool
    evaluate(examples) -> CriticMetrics
"""
from __future__ import annotations

import re
from functools import lru_cache

import numpy as np

from .model import CriticMetrics # reuse the metrics dataclass
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    accuracy_score, confusion_matrix,
)

_MODEL_NAME = "cross-encoder/nli-deberta-v3-xsmall"


def _split_claims(answer: str) -> list[str]:
    parts = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if len(s.strip()) > 3]
    return parts or [answer.strip()]


@lru_cache(maxsize=1)
def _load_model():
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tok = AutoTokenizer.from_pretrained(_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(_MODEL_NAME)
    model.eval()
    # map label name -> index, robust to ordering
    label2id = {v.lower(): k for k, v in model.config.id2label.items()}
    return tok, model, label2id


class NLICritic:
    def __init__(self, threshold: float = 0.05, max_length: int = 512) -> None:
        self.threshold = threshold
        self.max_length = max_length

    def fit(self, examples: list[dict] | None = None) -> "NLICritic":
        # zero-shot; nothing to train. Kept for interface compatibility.
        _load_model()
        return self

    def _nli_probs(self, premise: str, hypothesis: str) -> dict:
        import torch
        tok, model, label2id = _load_model()
        with torch.no_grad():
            x = tok(premise, hypothesis, return_tensors="pt",
                    truncation=True, max_length=self.max_length)
            probs = model(**x).logits.softmax(-1)[0]
        return {name: float(probs[idx]) for name, idx in label2id.items()}

    def predict_proba(self, answer: str, context: str) -> float:
        """P(hallucinated) = max P(contradiction) over the answer's claims.

        A factual error (wrong number/entity recombined from the context) is
        *contradicted* by the source, which is exactly what an NLI model detects
        and what lexical-overlap features cannot. Contradiction (not "lack of
        entailment") is the reliable signal: grounded long sentences often score
        as NLI-neutral, so using entailment directly causes false positives.
        """
        claims = _split_claims(answer)
        return float(max(self._nli_probs(context, c)["contradiction"] for c in claims))

    def is_hallucinated(self, answer: str, context: str) -> bool:
        return self.predict_proba(answer, context) >= self.threshold

    def evaluate(self, examples: list[dict]) -> CriticMetrics:
        proba = np.array([self.predict_proba(e["answer"], e["context"]) for e in examples])
        y = np.array([e["label"] for e in examples])
        pred = (proba >= self.threshold).astype(int)
        return CriticMetrics(
            accuracy=accuracy_score(y, pred),
            precision=precision_score(y, pred, zero_division=0),
            recall=recall_score(y, pred, zero_division=0),
            f1=f1_score(y, pred, zero_division=0),
            roc_auc=roc_auc_score(y, proba) if len(set(y)) > 1 else 0.5,
            confusion=confusion_matrix(y, pred).tolist(),
        )


class EnsembleCritic:
    """Hybrid critic: lexical feature critic OR NLI-contradiction critic.

    P(hallucinated) = max(feature_critic, nli_critic):
      - the feature critic catches *unsupported / wrong-context* answers
        (low lexical overlap) cheaply;
      - the NLI critic catches *contradictions* (recombined-in-context factual
        errors) the feature critic is blind to.
    Together they cover both failure modes. Same interface as the other critics.
    """

    def __init__(self, feature_critic, nli_critic: "NLICritic",
                 threshold: float = 0.5) -> None:
        self.feature = feature_critic
        self.nli = nli_critic
        self.threshold = threshold

    def fit(self, examples: list[dict]) -> "EnsembleCritic":
        self.feature.fit(examples)
        self.nli.fit(examples)
        return self

    def predict_proba(self, answer: str, context: str) -> float:
        return max(self.feature.predict_proba(answer, context),
                   self.nli.predict_proba(answer, context))

    def is_hallucinated(self, answer: str, context: str) -> bool:
        # OR of two independent detectors, each with its own calibrated threshold:
        # the feature critic flags low-overlap / wrong-context answers, the NLI
        # critic flags contradictions. Either firing => hallucination.
        return (self.feature.is_hallucinated(answer, context) or
                self.nli.is_hallucinated(answer, context))

    def evaluate(self, examples: list[dict]) -> CriticMetrics:
        proba = np.array([self.predict_proba(e["answer"], e["context"]) for e in examples])
        y = np.array([e["label"] for e in examples])
        pred = (proba >= self.threshold).astype(int)
        return CriticMetrics(
            accuracy=accuracy_score(y, pred),
            precision=precision_score(y, pred, zero_division=0),
            recall=recall_score(y, pred, zero_division=0),
            f1=f1_score(y, pred, zero_division=0),
            roc_auc=roc_auc_score(y, proba) if len(set(y)) > 1 else 0.5,
            confusion=confusion_matrix(y, pred).tolist(),
        )
