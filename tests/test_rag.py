"""Tests for the Phase-3 retriever and end-to-end RAG.

Need faiss + sentence-transformers (download a small model), so skipped unless
installed AND RUN_RAG_TESTS=1. Run: RUN_RAG_TESTS=1 pytest tests/test_rag.py
"""
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

pytest.importorskip("faiss")
pytest.importorskip("sentence_transformers")
if os.environ.get("RUN_RAG_TESTS") != "1":
    pytest.skip("set RUN_RAG_TESTS=1 to run retrieval tests", allow_module_level=True)

from src.rag.retriever import DenseRetriever
from src.rag.generator import ExtractiveGenerator, StubLLMGenerator
from src.rag.rag_system import EndToEndRAG
from src.critic.model import HallucinationCritic
from src.data.synth import build_dataset

PASSAGES = [
    "The Eiffel Tower was completed in 1889 and stands 330 metres tall.",
    "Photosynthesis converts carbon dioxide and water into glucose and oxygen.",
    "Mount Everest is 8,849 metres tall and lies on the Nepal-China border.",
]


def test_retriever_finds_relevant_passage():
    r = DenseRetriever().build_index(PASSAGES)
    hits = r.search("How tall is the Eiffel Tower?", k=1)
    assert "Eiffel" in hits[0].text


def test_end_to_end_returns_grounded_answer():
    r = DenseRetriever().build_index(PASSAGES)
    critic = HallucinationCritic().fit(build_dataset())
    rag = EndToEndRAG(r, ExtractiveGenerator(), critic, top_k=2)
    ans = rag.answer("How tall is Mount Everest?")
    assert ans.trace.final_answer
    assert ans.retrieved_scores
