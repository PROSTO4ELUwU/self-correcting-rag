"""Gradio web demo for the self-correcting RAG loop.

Run: pip install -r requirements-demo.txt && python app.py
Paste a context passage, a question and a candidate answer; the trained critic
scores grounding and the loop repairs hallucinations in real time.
"""
from __future__ import annotations

import gradio as gr

from src.data.synth import build_dataset, SEED_FACTS
from src.critic.model import HallucinationCritic
from src.rag.pipeline import SelfCorrectingRAG

_critic = HallucinationCritic(threshold=0.5).fit(build_dataset())
_rag = SelfCorrectingRAG(_critic, max_iters=2)


def judge(context: str, question: str, candidate: str):
    if not context.strip() or not candidate.strip():
        return "Please provide both a context and a candidate answer.", "", ""
    trace = _rag.answer(question, context, candidate)
    verdict = ("Hallucination detected, corrected"
               if trace.was_corrected else "Grounded, answer accepted")
    proba = " -> ".join(f"{p:.2f}" for p in trace.proba_history)
    return verdict, proba, trace.final_answer


_example = SEED_FACTS[0]

with gr.Blocks(title="Self-Correcting RAG") as demo:
    gr.Markdown("# Self-Correcting RAG\n"
                "A trained critic checks whether an answer is grounded in the "
                "source, and repairs it if not.")
    context = gr.Textbox(label="Context (source passage)", lines=4,
                         value=_example["context"])
    question = gr.Textbox(label="Question", value=_example["question"])
    candidate = gr.Textbox(label="Candidate answer (try making one up!)",
                           value="The Eiffel Tower is 450 metres tall and was completed in 1889.")
    btn = gr.Button("Check & correct", variant="primary")
    verdict = gr.Textbox(label="Verdict")
    proba = gr.Textbox(label="P(hallucinated) over correction steps")
    final = gr.Textbox(label="Final (grounded) answer")
    btn.click(judge, [context, question, candidate], [verdict, proba, final])

if __name__ == "__main__":
    demo.launch()
