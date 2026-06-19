"""The trainable hallucination critic.

A scikit-learn classifier that, given engineered features describing how well
an answer is grounded in its context, predicts P(hallucinated). This is the
"reward model" / verifier at the heart of the self-correcting loop.

Baseline = GradientBoostingClassifier (strong on small tabular feature sets).
The interface is model-agnostic: swap in a fine-tuned transformer later by
re-implementing predict_proba over (answer, context) pairs.
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    accuracy_score, confusion_matrix,
)

from .features import FeatureExtractor


@dataclass
class CriticMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    confusion: list

    def __str__(self) -> str:
        return (f"acc={self.accuracy:.3f} precision={self.precision:.3f} "
                f"recall={self.recall:.3f} f1={self.f1:.3f} auc={self.roc_auc:.3f}")


class HallucinationCritic:
    def __init__(self, threshold: float = 0.5) -> None:
        self.extractor = FeatureExtractor()
        self.clf = GradientBoostingClassifier(random_state=0)
        self.threshold = threshold

    def fit(self, examples: list[dict]) -> "HallucinationCritic":
        answers = [e["answer"] for e in examples]
        contexts = [e["context"] for e in examples]
        y = np.array([e["label"] for e in examples])
        self.extractor.fit(contexts, answers)
        X = self.extractor.transform(answers, contexts)
        self.clf.fit(X, y)
        return self

    def predict_proba(self, answer: str, context: str) -> float:
        X = self.extractor.transform([answer], [context])
        return float(self.clf.predict_proba(X)[0, 1])

    def is_hallucinated(self, answer: str, context: str) -> bool:
        return self.predict_proba(answer, context) >= self.threshold

    def evaluate(self, examples: list[dict]) -> CriticMetrics:
        answers = [e["answer"] for e in examples]
        contexts = [e["context"] for e in examples]
        y = np.array([e["label"] for e in examples])
        X = self.extractor.transform(answers, contexts)
        proba = self.clf.predict_proba(X)[:, 1]
        pred = (proba >= self.threshold).astype(int)
        return CriticMetrics(
            accuracy=accuracy_score(y, pred),
            precision=precision_score(y, pred, zero_division=0),
            recall=recall_score(y, pred, zero_division=0),
            f1=f1_score(y, pred, zero_division=0),
            roc_auc=roc_auc_score(y, proba) if len(set(y)) > 1 else 0.5,
            confusion=confusion_matrix(y, pred).tolist(),
        )

    def feature_importance(self) -> dict:
        return dict(zip(self.extractor.FEATURE_NAMES,
                        [round(float(v), 4) for v in self.clf.feature_importances_]))

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str | Path) -> "HallucinationCritic":
        with open(path, "rb") as f:
            return pickle.load(f)
