"""Render assets/demo.gif -- a terminal recording of the loop in action.

Run: python -m experiments.make_demo_gif

Trains the critic on the bundled facts, runs one hallucinated answer through the
self-correcting loop, and paints the real P(hallucinated) trace onto a fake
terminal frame by frame. No screen recorder needed, and the numbers are genuine
(straight out of the trained critic), so the GIF can't drift from the code.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.synth import build_dataset, SEED_FACTS # noqa: E402
from src.critic.model import HallucinationCritic # noqa: E402
from src.rag.pipeline import SelfCorrectingRAG # noqa: E402

# terminal palette (a muted dark theme)
BG = (13, 17, 23)
FG = (201, 209, 217)
GREEN = (63, 185, 80)
RED = (248, 81, 73)
YELLOW = (210, 153, 34)
BLUE = (88, 166, 255)
GREY = (110, 118, 129)

W, H = 860, 460
PAD = 26
LINE = 26
FONT_DIR = Path(__file__).resolve().parents[0] # placeholder, set below


def _font(size: int, bold: bool = False):
    import matplotlib
    base = Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf"
    name = "DejaVuSansMono-Bold.ttf" if bold else "DejaVuSansMono.ttf"
    return ImageFont.truetype(str(base / name), size)


F = None
FB = None


def _wrap(text: str, width: int = 78) -> list[str]:
    out, line = [], ""
    for word in text.split():
        if len(line) + len(word) + 1 > width:
            out.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        out.append(line)
    return out


def _frame(lines: list[tuple[str, tuple]]) -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    # window chrome
    d.rectangle([0, 0, W, 34], fill=(22, 27, 34))
    for i, c in enumerate([RED, YELLOW, GREEN]):
        d.ellipse([PAD + i * 22, 12, PAD + i * 22 + 11, 23], fill=c)
    d.text((W // 2 - 90, 9), "self-correcting-rag", font=F, fill=GREY)
    y = 50
    for text, color in lines:
        d.text((PAD, y), text, font=(FB if color in (GREEN, RED, YELLOW) else F), fill=color)
        y += LINE
    return img


def main() -> None:
    global F, FB
    F = _font(16)
    FB = _font(16, bold=True)

    critic = HallucinationCritic(0.5).fit(build_dataset())
    rag = SelfCorrectingRAG(critic, max_iters=2)

    fact = SEED_FACTS[0]
    bad = "The Eiffel Tower is 450 metres tall and was completed in 1889."
    trace = rag.answer(fact["question"], fact["context"], bad)

    ctx_lines = _wrap(fact["context"])
    p0 = trace.proba_history[0]
    p1 = trace.proba_history[-1]

    header = [
        ("$ python -m demo.cli", BLUE),
        ("", FG),
        ("context:", GREY),
    ] + [(" " + l, FG) for l in ctx_lines] + [
        ("", FG),
        (f"Q: {fact['question']}", FG),
        ("", FG),
    ]

    # build a small storyboard; each entry is the full set of lines for a frame
    frames_spec = [
        header + [("candidate> " + bad, FG)],
        header + [("candidate> " + bad, FG), ("", FG),
                  (f" critic: P(hallucinated) = {p0:.2f}", RED)],
        header + [("candidate> " + bad, FG), ("", FG),
                  (f" critic: P(hallucinated) = {p0:.2f}", RED),
                  (" flagged -- 450m is not supported, regenerating...", YELLOW)],
        header + [("candidate> " + bad, FG), ("", FG),
                  (f" critic: P(hallucinated) = {p0:.2f}", RED),
                  (" flagged -- 450m is not supported, regenerating...", YELLOW),
                  ("", FG),
                  (" fixed: " + trace.final_answer, GREEN)],
        header + [("candidate> " + bad, FG), ("", FG),
                  (f" critic: P(hallucinated) = {p0:.2f}", RED),
                  (" flagged -- 450m is not supported, regenerating...", YELLOW),
                  ("", FG),
                  (" fixed: " + trace.final_answer, GREEN),
                  (f" critic: P(hallucinated) = {p1:.2f} -> accepted", GREEN)],
    ]

    images = [_frame(spec) for spec in frames_spec]
    # hold the last frame longer, then loop
    durations = [900, 1100, 1300, 1300, 2600]

    out = ROOT / "assets" / "demo.gif"
    out.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(out, save_all=True, append_images=images[1:],
                   duration=durations, loop=0, optimize=True)
    print(f"Saved {out.relative_to(ROOT)} ({out.stat().st_size // 1024} KB, "
          f"{len(images)} frames)")


if __name__ == "__main__":
    main()
