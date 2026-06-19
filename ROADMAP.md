# Roadmap

This repo ships a **working, fully-offline baseline** so the whole loop runs in
seconds with no API keys. Below is how to grow it into a research-grade,
portfolio-defining project. Tackle phases in order — each is a self-contained,
demoable milestone.

## Phase 1 — Real data (replace synthetic) ✅ *done*
- [x] Loader for **SQuAD v2** (`src/data/real.py`) building a grounding dataset.
- [x] Keep synthetic corruptions as *data augmentation* on top of real data.
- [x] `experiments/run_real.py` records realistic metrics (F1≈0.99) + plots.
- [x] Added a **hard stress test** (recombined in-context claims) exposing the
      lexical critic's limit (~6% recall) — motivates Phase 2.
- [ ] Stretch: also add **RAGTruth** as a second real source.

## Phase 2 — Transformer critic 🤖 ✅ *done*
- [x] NLI critic (`cross-encoder/nli-deberta-v3-xsmall`) scoring contradiction
      between context and each answer claim (`src/critic/nli_critic.py`).
- [x] Kept the critic interface (`predict_proba`, `is_hallucinated`) so the loop
      is unchanged.
- [x] **Hybrid ensemble** (feature OR NLI) — `experiments/run_phase2.py`.
- [x] Ablation table + plot: hard-case recall **4% → 44%** while main F1 stays 0.97.
- [ ] Stretch: fine-tune DeBERTa on the grounding labels; claim decomposition
      into atomic facts; try a larger NLI model to push recall further.

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

## Phase 5 — Evaluation & demo 📊 *partially done*
- [x] Calibration plot for the critic (reliability diagram) — in `run_real`.
- [x] Per-source detection breakdown + results figure (`assets/results.png`).
- [x] **Live demo**: interactive CLI (`demo/cli.py`) + Gradio app (`app.py`).
- [x] GitHub Actions CI running `pytest` + offline demo on every push.
- [ ] Full benchmark table vs baselines (no-critic RAG, self-consistency).
- [ ] Record a GIF of the demo and embed it in the README.

## Stretch ideas
- [ ] Per-claim citations: highlight exactly which source sentence supports each claim.
- [ ] Active learning: surface the examples the critic is least sure about.
- [ ] Write it up as a short technical blog post / paper.
