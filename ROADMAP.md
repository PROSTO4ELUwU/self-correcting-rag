# Roadmap

The repo ships a working, fully-offline baseline so the whole loop runs in
seconds with no API keys. The phases below grow it into something closer to a
research project. They're meant to be tackled in order; each one is a
self-contained, demoable milestone.

## Phase 1 - real data (replace synthetic) [done]
- [x] SQuAD v2 loader (`src/data/real.py`) building a grounding dataset.
- [x] Keep synthetic corruptions as data augmentation on top of real data.
- [x] `experiments/run_real.py` records realistic metrics (F1 ~0.99) + plots.
- [x] Hard stress test (recombined in-context claims) exposing the lexical
      critic's limit (~6% recall), which motivates Phase 2.
- [ ] Stretch: add RAGTruth as a second real source.

## Phase 2 - transformer critic [done]
- [x] NLI critic (`cross-encoder/nli-deberta-v3-xsmall`) scoring contradiction
      between context and each answer claim (`src/critic/nli_critic.py`).
- [x] Kept the critic interface (`predict_proba`, `is_hallucinated`) so the loop
      doesn't change.
- [x] Hybrid ensemble (feature OR NLI) - `experiments/run_phase2.py`.
- [x] Ablation table + plot: hard-case recall 4% -> 44% while main F1 stays 0.97.
- [ ] Stretch: fine-tune DeBERTa on the grounding labels; decompose claims into
      atomic facts; try a larger NLI model.

## Phase 3 - real RAG components [done]
- [x] Dense retriever: sentence-transformer embeddings + FAISS
      (`src/rag/retriever.py`), hit@3 ~83% on SQuAD v2.
- [x] Swappable generators (`src/rag/generator.py`): extractive baseline,
      hallucinating stub for the demo, optional `OpenAIGenerator`.
- [x] End-to-end retrieve -> generate -> self-correct (`src/rag/rag_system.py`),
      cutting end-to-end hallucination rate 42% -> 0% (`experiments/run_phase3.py`).
- [ ] Stretch: claim-level checking (split answers into atomic claims, verify each).

## Phase 4 - learning loop [done]
- [x] Reward-guided decoding: best-of-n against the critic reward
      (`src/rag/strategies.py`), plus a self-consistency baseline.
- [x] Self-training: promote the critic's confident predictions to pseudo-labels
      and refit (`src/critic/self_training.py`). Reaches the fully-supervised F1
      from a 24-label seed (~4% of the data).
- [x] Benchmark table across strategies in `experiments/run_phase4.py`
      (no critic / self-consistency / loop / best-of-n / best-of-n + loop).
- [ ] Stretch: proper RL (PPO-style) instead of rejection sampling once a real
      generator is wired in.

## Phase 5 - evaluation & demo [mostly done]
- [x] Calibration plot for the critic (reliability diagram) - in `run_real`.
- [x] Per-source detection breakdown + results figure (`assets/results.png`).
- [x] Live demo: interactive CLI (`demo/cli.py`) + Gradio app (`app.py`).
- [x] GitHub Actions CI running `pytest` + the offline demo on every push.
- [x] Benchmark table vs baselines (no-critic RAG, self-consistency) - Phase 4.
- [x] Demo GIF (`assets/demo.gif`, rendered by `experiments/make_demo_gif.py`),
      embedded in the README.

## Stretch ideas
- [ ] Per-claim citations: highlight which source sentence supports each claim.
- [ ] Active learning: surface the examples the critic is least sure about.
- [ ] Write it up as a short technical post.
