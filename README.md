# Self-Correcting RAG 🛡️

![CI](https://github.com/PROSTO4ELUwU/self-correcting-rag/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A retrieval-augmented generation (RAG) system that **catches and fixes its own
hallucinations** using a *trainable critic model*, instead of blindly trusting
whatever the language model writes.

> **The problem.** RAG systems answer questions over your documents — but they
> routinely state unsupported facts with total confidence (wrong numbers, swapped
> entities, invented claims). In medicine, law, finance and enterprise search this
> is the #1 blocker to deploying LLMs. There was even a real court case where
> lawyers were fined for submitting AI-invented case citations.
>
> **This project** adds a learned "fact-checker" in the loop: before an answer is
> returned, a trained critic verifies that every claim is grounded in the retrieved
> source. If not, the system rewrites the answer and checks again.

---

## Why this is interesting (not just another "chat with your PDF")

Most RAG portfolio projects are a thin wrapper around an LLM API. This one has
**real ML engineering** around it:

- 🧪 **Self-supervised labels** — no manual annotation. We derive a grounding
  dataset from **SQuAD v2**: a grounded answer-sentence (label 0) vs answers
  borrowed from a *different* passage and token-corrupted answers (label 1).
- 🤖 **A trained verifier model** — a classifier predicting `P(hallucinated)`
  from interpretable grounding features (a lightweight reward/verifier model).
- 🔁 **A self-correction loop** — flag → regenerate (grounded fallback) → re-verify.
- 📊 **Measurable results + an honest failure analysis** (see below).

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
`has_negation`, `entity_support`, `len_ratio`, combined by a
`GradientBoostingClassifier`.

## Results — real data (SQuAD v2)

From `python -m experiments.run_real` (640 examples, 30% held out):

| Metric (held-out test)            | Value |
|-----------------------------------|-------|
| Critic F1                         | **0.99** |
| Critic ROC-AUC                    | **1.00** |
| Hallucination rate **before** loop| **66.7%** |
| Hallucination rate **after** loop | **0.5%** |
| Relative reduction                | **−99%** |

![results](assets/results.png)

### Honest failure analysis (this is the interesting part) 🔬

I added a **stress test** of *hard* hallucinations: false claims built **only
from words already in the context** (e.g. swapping two real in-context numbers).
These are lexically almost identical to grounded answers.

| Hallucination type                 | Caught by critic |
|------------------------------------|------------------|
| Borrowed from wrong context        | 100% |
| Out-of-context token swap          | 100% |
| **Recombined in-context (hard)**   | **~6%** |

The feature-based critic is excellent at surface/lexical grounding but **cannot
catch semantically-recombined hallucinations** — they need entailment reasoning,
not word overlap. This directly motivates the **Phase-2 NLI/transformer critic**
(see [`ROADMAP.md`](ROADMAP.md)). Knowing *why* a baseline fails is the point.

## Quickstart

```bash
pip install -r requirements.txt

python -m experiments.run_demo   # offline synthetic demo (no downloads)
python -m experiments.run_real   # real SQuAD v2 + stress test + plots
pytest -q                        # tests
python -m demo.cli               # interactive CLI: catch & fix a hallucination
```

### Web demo

```bash
pip install -r requirements-demo.txt
python app.py                    # Gradio UI in the browser
```

## Project structure

```
src/
  data/synth.py        # offline self-supervised dataset (corrupt faithful answers)
  data/real.py         # SQuAD v2 grounding dataset + hard stress set
  critic/features.py   # interpretable grounding features
  critic/model.py      # trainable hallucination critic + metrics
  rag/pipeline.py      # RAG + self-correction loop
  evaluate.py          # before/after hallucination-rate evaluation
experiments/run_demo.py   experiments/run_real.py
demo/cli.py   app.py (Gradio)
tests/test_smoke.py
.github/workflows/ci.yml
```

## Roadmap

See [`ROADMAP.md`](ROADMAP.md). Next up: **Phase 2 — replace the feature critic
with a fine-tuned DeBERTa/NLI model** to close the hard-case gap above.

## License

MIT — see [`LICENSE`](LICENSE).
