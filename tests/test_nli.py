"""Tests for the NLI critic & ensemble.

These need torch + transformers and download a small model, so they are skipped
unless those deps are installed AND RUN_NLI_TESTS=1 is set (keeps default CI fast
and offline). Run locally with:  RUN_NLI_TESTS=1 pytest tests/test_nli.py
"""
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

pytest.importorskip("torch")
pytest.importorskip("transformers")
if os.environ.get("RUN_NLI_TESTS") != "1":
    pytest.skip("set RUN_NLI_TESTS=1 to run NLI model tests", allow_module_level=True)

from src.critic.nli_critic import NLICritic, EnsembleCritic
from src.critic.model import HallucinationCritic
from src.data.synth import build_dataset

CTX = "The Eiffel Tower was completed in 1889 and stands 330 metres tall."


def test_nli_flags_contradiction():
    nli = NLICritic().fit()
    grounded = nli.predict_proba("The Eiffel Tower is 330 metres tall.", CTX)
    wrong = nli.predict_proba("The Eiffel Tower is 900 metres tall.", CTX)
    assert wrong > grounded


def test_ensemble_interface():
    ds = build_dataset()
    ens = EnsembleCritic(HallucinationCritic().fit(ds), NLICritic())
    p = ens.predict_proba("The Eiffel Tower is 330 metres tall.", CTX)
    assert 0.0 <= p <= 1.0
    assert isinstance(ens.is_hallucinated("The Eiffel Tower is 900 metres tall.", CTX), bool)
