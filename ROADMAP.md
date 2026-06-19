# Roadmap

This repo ships a **working, fully-offline baseline** so the whole loop runs in
seconds with no API keys. Below is how to grow it into a research-grade,
portfolio-defining project. Tackle phases in order — each is a self-contained,
demoable milestone.

## Phase 1 — Real data (replace synthetic) 🎯 *start here*
- [ ] Add a loader for **RAGTruth** (a real hallucination-annotated RAG dataset)
      and/or **SQuAD v2** (has unanswerable questions = natural hallucination traps).
- [ ] Keep the synthetic corruptions as *data augmentation* on top of real data.
- [ ] Re-run `run_demo` and record the (now harder, more realistic) metrics.

## Phase 2 — Transformer critic 🤖
- [ ] Replace `GradientBoostingClassifier` with a fine-tuned **DeBERTa-v3** or an
      **NLI model** (entailment between context and each answer claim).
- [ ] Keep the same `HallucinationCritic` interface (`predict_proba`,
      `is_hallucinated`) so the loop code doesn't change.
- [ ] Compare feature-baseline vs transformer in an ablation table.

## Phase 3 — Real RAG components 🔍
- [ ] Real retriever: embed a document corpus with sentence-transformers, index
      with **FAISS**, retrieve top-k.
- [ ] Real generator: a local small LM (Qwen / Phi) or an LLM API behind an
      interface so it's swappable.
- [ ] Claim-level checking: split answers into atomic claims and verify each.

## Phase 4 — Learning loop 🔁
- [ ] Self-training: use high-confidence critic labels to expand the training set.
- [ ] Reward-style optimisation: treat grounding score as a reward and prefer
      answers that maximise it (rejection sampling / best-of-n first, RL later).

## Phase 5 — Evaluation & demo 📊
- [ ] Benchmark table: hallucination rate, answer quality, latency vs baselines
      (no-critic RAG, self-consistency, etc.).
- [ ] Calibration plot for the critic (reliability diagram).
- [ ] **Live demo** (Gradio/Streamlit): type a question, watch the loop catch and
      fix a hallucination in real time. Add a GIF to the README.
- [ ] GitHub Actions CI running `pytest` on every push.

## Stretch ideas
- [ ] Per-claim citations: highlight exactly which source sentence supports each claim.
- [ ] Active learning: surface the examples the critic is least sure about.
- [ ] Write it up as a short technical blog post / paper.
