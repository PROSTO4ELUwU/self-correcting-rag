"""Render the demo gifs - a terminal recording of the loop in action.

Run:  python -m experiments.make_demo_gif

Trains the critic on the bundled facts, runs one hallucinated answer through the
self-correcting loop, and paints the real P(hallucinated) trace onto a fake
terminal frame by frame. No screen recorder needed, and the numbers are genuine
(straight out of the trained critic), so the gif can't drift from the code.

Writes two language variants: assets/demo.gif (English) and assets/demo_ru.gif
(Russian). The critic always runs on the English example, so the probabilities
are identical; only the on-screen captions are translated.
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.synth import build_dataset, SEED_FACTS          # noqa: E402
from src.critic.model import HallucinationCritic              # noqa: E402
from src.rag.pipeline import SelfCorrectingRAG                # noqa: E402

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

F = None
FB = None


def _font(size: int, bold: bool = False):
    import matplotlib
    base = Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf"
    name = "DejaVuSansMono-Bold.ttf" if bold else "DejaVuSansMono.ttf"
    return ImageFont.truetype(str(base / name), size)


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
    d.rectangle([0, 0, W, 34], fill=(22, 27, 34))
    for i, c in enumerate([RED, YELLOW, GREEN]):
        d.ellipse([PAD + i * 22, 12, PAD + i * 22 + 11, 23], fill=c)
    d.text((W // 2 - 90, 9), "self-correcting-rag", font=F, fill=GREY)
    y = 50
    for text, color in lines:
        d.text((PAD, y), text, font=(FB if color in (GREEN, RED, YELLOW) else F), fill=color)
        y += LINE
    return img


# English uses the live trace text; Russian uses faithful translations of the
# same example (the critic still runs on English, so p0/p1 are real either way).
STRINGS = {
    "en": {
        "ctx_label": "context:",
        "context": SEED_FACTS[0]["context"],
        "question": "Q: " + SEED_FACTS[0]["question"],
        "prompt": "candidate> ",
        "candidate": "The Eiffel Tower is 450 metres tall and was completed in 1889.",
        "score": "  critic: P(hallucinated) = {p:.2f}",
        "flag": "  flagged -- 450m is not supported, regenerating...",
        "fixed": "  fixed:  {ans}",
        "accept": "  critic: P(hallucinated) = {p:.2f}  -> accepted",
        "out": "demo.gif",
    },
    "ru": {
        "ctx_label": "контекст:",
        "context": ("Эйфелева башня была построена в 1889 году и имеет высоту "
                    "330 метров. Её спроектировал инженер Гюстав Эйфель для "
                    "Всемирной выставки в Париже."),
        "question": "В: Какова высота Эйфелевой башни и когда её построили?",
        "prompt": "ответ> ",
        "candidate": "Эйфелева башня имеет высоту 450 метров, построена в 1889 году.",
        "score": "  критик: P(галлюцинация) = {p:.2f}",
        "flag": "  помечено: 450 м не подтверждается контекстом, перегенерация...",
        "fixed": "  исправлено:  Эйфелева башня имеет высоту 330 метров, построена в 1889 году.",
        "accept": "  критик: P(галлюцинация) = {p:.2f}  -> принято",
        "out": "demo_ru.gif",
    },
}


def _storyboard(s: dict, p0: float, p1: float) -> list[list[tuple]]:
    header = [
        ("$ python -m demo.cli", BLUE),
        ("", FG),
        (s["ctx_label"], GREY),
    ] + [("  " + l, FG) for l in _wrap(s["context"])] + [
        ("", FG),
        (s["question"], FG),
        ("", FG),
    ]
    cand = (s["prompt"] + s["candidate"], FG)
    score0 = (s["score"].format(p=p0), RED)
    flag = (s["flag"], YELLOW)
    fixed = (s["fixed"], GREEN)
    accept = (s["accept"].format(p=p1), GREEN)
    return [
        header + [cand],
        header + [cand, ("", FG), score0],
        header + [cand, ("", FG), score0, flag],
        header + [cand, ("", FG), score0, flag, ("", FG), fixed],
        header + [cand, ("", FG), score0, flag, ("", FG), fixed, accept],
    ]


def main() -> None:
    global F, FB
    F = _font(16)
    FB = _font(16, bold=True)

    critic = HallucinationCritic(0.5).fit(build_dataset())
    rag = SelfCorrectingRAG(critic, max_iters=2)
    fact = SEED_FACTS[0]
    trace = rag.answer(fact["question"], fact["context"],
                       STRINGS["en"]["candidate"])
    p0, p1 = trace.proba_history[0], trace.proba_history[-1]
    durations = [900, 1100, 1300, 1300, 2600]

    for lang, s in STRINGS.items():
        images = [_frame(spec) for spec in _storyboard(s, p0, p1)]
        out = ROOT / "assets" / s["out"]
        out.parent.mkdir(parents=True, exist_ok=True)
        images[0].save(out, save_all=True, append_images=images[1:],
                       duration=durations, loop=0, optimize=True)
        print(f"Saved {out.relative_to(ROOT)}  ({out.stat().st_size // 1024} KB, "
              f"{len(images)} frames)")


if __name__ == "__main__":
    main()
