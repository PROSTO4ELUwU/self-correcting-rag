# Self-Correcting RAG 🛡️

A retrieval-augmented generation (RAG) system that **catches and fixes its own
hallucinations** using a *trainable critic model*, instead of blindly trusting
whatever the language model writes.

> **The problem.** RAG systems answer questions over your documents — but they
> routinely state unsupported facts with total confidence (wrong numbers, swapped
> entities, invented claims). In medicine, law, finance and enterprise search this
> is the #1 blocker to actually deploying LLMs. There was even a real court case
> where lawyers were fined for submitting AI-invented case citations.
>
> **This project** adds a learned "fact-checker" in the loop: before an answer is
> returned, a trained critic verifies that every claim is grounded in the retrieved
> source. If not, the system rewrites the answer and checks again.

---

## Why this is interesting (not just another "chat with your PDF")

Most RAG portfolio projects are a thin wrapper around an LLM API. This one has
**real ML engineering** around it:

- 🧪 **A self-supervised dataset** — we don't hand-label data. We take faithful
  answers and *programmatically corrupt* them (number swaps, entity swaps,
  unsupported additions, negations) to create labeled hallucination examples for free.
- 🤖 **A trained verifier model** — a classifier that predicts `P(hallucinated)`
  from interpretable grounding features. This is effectively a lightweight
  **reward / verifier model**.
- 🔁 **A self-correction loop** — flag → regenerate (grounded fallback) → re-verify,
  iterating until the answer is supported.
- 📊 **A measurable headline result** — hallucination rate *before vs after*
  correction, not a vague "it works".

---

## Architecture

```
question
   │
   ▼
[retrieve] ──► context (source passages)
   │
   ▼
[generate] ──► candidate answer  (may be faithful OR hallucinated)
   │
   ▼
[CRITIC] ── P(hallucinated) ≥ threshold? ──┐
   │ no                                     │ yes
   ▼                                        ▼
 return answer                     [regenerate: grounded
                                    extractive fallback]
                                        │
                                        └──► re-check (loop, max_iters)
```

The **critic** scores grounding with engineered, *explainable* features:
`lexical_overlap`, `novel_token_ratio`, `tfidf_cosine`, `number_support`,
`has_negation`, `entity_support`, `len_ratio`. A `GradientBoostingClassifier`
learns to combine them.

## Results (offline synthetic baseline)

From `python -m experiments.run_demo` on the bundled synthetic dataset:

| Metric (held-out test)       | Value |
|------------------------------|-------|
| Critic F1                    | high (separable baseline) |
| Hallucination rate *before*  | ~73%  |
| Hallucination rate *after*   | ~0%   |

> ⚠️ **Honest caveat:** the bundled dataset is small and intentionally easy so the
> repo runs instantly with **zero downloads or API keys**. Near-perfect numbers
> here just prove the pipeline is wired correctly end-to-end. The interesting work
> (and the realistic, harder numbers) comes from the roadmap below — swapping in
> real datasets and a transformer critic.

## Quickstart

```bash
pip install -r requirements.txt
python -m experiments.run_demo   # trains critic + prints before/after metrics
pytest -q                        # run smoke tests
```

## Project structure

```
src/
  data/synth.py        # self-supervised dataset (corrupt faithful answers)
  critic/features.py   # interpretable grounding features
  critic/model.py      # trainable hallucination critic + metrics
  rag/pipeline.py      # RAG + self-correction loop
  evaluate.py          # before/after hallucination-rate evaluation
experiments/run_demo.py
tests/test_smoke.py
```

## Roadmap

See [`ROADMAP.md`](ROADMAP.md). Short version:
1. Plug in real QA data (SQuAD / HotpotQA / RAGTruth) instead of synthetic.
2. Replace the feature-based critic with a **fine-tuned transformer** (DeBERTa / NLI).
3. Real generator (local SLM or an LLM API) + real retriever (FAISS over embeddings).
4. Self-training / RL loop where grounding score is the reward.
5. Live demo (Gradio/Streamlit) + a proper benchmark table vs baselines.

## License

MIT — see [`LICENSE`](LICENSE).
