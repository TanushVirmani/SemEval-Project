"""
Gradio web app for SemEval-2026 Task 13 (Subtask A):
binary classification of code as human-written vs machine-generated.

Uses microsoft/codebert-base with a 2-class sequence classification head.
Place fine-tuned weights in ./checkpoints/ or set MODEL_CHECKPOINT env var.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".cache/matplotlib")
)

import matplotlib

from model_utils import (
    DEFAULT_CHECKPOINT,
    LABELS,
    add_language_prefix,
    load_classifier,
    validate_checkpoint,
)

matplotlib.use("Agg")

import gradio as gr
import lizard
import matplotlib.pyplot as plt
import torch

MODEL_CHECKPOINT = str(DEFAULT_CHECKPOINT)

LANGUAGE_OPTIONS = ["Python", "C++", "Java"]

LIZARD_FILENAMES = {
    "Python": "snippet.py",
    "C++": "snippet.cpp",
    "Java": "snippet.java",
}

COMPLEXITY_THRESHOLDS = {
    "low": 5,
    "moderate": 10,
}

# Sample inputs for gr.Examples: [language, code]
EXAMPLE_HUMAN_PYTHON = [
    "Python",
    """def two_sum(nums, target):
    seen = {}
    for i, n in enumerate(nums):
        if target - n in seen:
            return [seen[target - n], i]
        seen[n] = i
    return []""",
]

EXAMPLE_MACHINE_PYTHON = [
    "Python",
    '''def calculate_fibonacci_sequence(n: int) -> list:
    """
    Generate a Fibonacci sequence up to n terms.
    Args:
        n: The number of terms to generate.
    Returns:
        A list containing the Fibonacci sequence.
    """
    if n <= 0:
        return []
    sequence = [0, 1]
    for i in range(2, n):
        sequence.append(sequence[i - 1] + sequence[i - 2])
    return sequence[:n]''',
]


def _language_prefix(language: str, code: str) -> str:
    """Optional language hint prepended for models trained with metadata."""
    return add_language_prefix(language, code)


@lru_cache(maxsize=1)
def load_model_and_tokenizer():
    return load_classifier(MODEL_CHECKPOINT)


def predict(language: str, code: str) -> tuple[str, str, str, str]:
    """
    Classify code snippet and return prediction plus confidence scores.

    Returns:
        prediction_label, human_confidence, machine_confidence, details_markdown
    """
    if not code or not code.strip():
        return (
            "—",
            "—",
            "—",
            "**Error:** Please paste a code snippet to analyze.",
        )

    try:
        model, tokenizer, device = load_model_and_tokenizer()
    except (ValueError, OSError) as exc:
        return "—", "—", "—", f"**Detector unavailable:** {exc}"
    text = _language_prefix(language, code)

    max_length = int(
        json.loads((Path(MODEL_CHECKPOINT) / "training_metadata.json").read_text())[
            "max_length"
        ]
    )
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=True,
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
        probabilities = torch.softmax(logits, dim=-1)[0].cpu().tolist()

    human_conf = probabilities[0]
    machine_conf = probabilities[1]
    predicted_id = 0 if human_conf >= machine_conf else 1
    prediction = LABELS[predicted_id]
    confidence = max(human_conf, machine_conf)

    details = (
        f"**Language:** {language}  \n"
        f"**Prediction:** {prediction}  \n"
        f"**Top confidence:** {confidence:.1%}  \n"
        f"**Input limit:** {max_length} tokens; longer snippets are truncated.  \n\n"
        "_SemEval-2026 Task 13 Subtask A — binary human vs machine code detection._"
    )

    return (
        prediction,
        f"{human_conf:.1%}",
        f"{machine_conf:.1%}",
        details,
    )


def _complexity_rating(cyclomatic: float) -> str:
    if cyclomatic <= COMPLEXITY_THRESHOLDS["low"]:
        return "Low"
    if cyclomatic <= COMPLEXITY_THRESHOLDS["moderate"]:
        return "Moderate"
    return "High"


def _complexity_color(cyclomatic: int) -> str:
    if cyclomatic <= COMPLEXITY_THRESHOLDS["low"]:
        return "#2ecc71"
    if cyclomatic <= COMPLEXITY_THRESHOLDS["moderate"]:
        return "#f39c12"
    return "#e74c3c"


def analyze_complexity(language: str, code: str) -> tuple[str, plt.Figure | None]:
    """Compute cyclomatic complexity metrics and return a bar chart."""
    if not code or not code.strip():
        return "Paste code to see complexity metrics.", None

    filename = LIZARD_FILENAMES.get(language, "snippet.py")
    try:
        analysis = lizard.analyze_file.analyze_source_code(filename, code)
    except Exception as exc:
        return f"**Could not parse code:** {exc}", None

    functions = analysis.function_list
    loc = analysis.nloc
    token_count = analysis.token_count
    avg_cc = analysis.average_cyclomatic_complexity
    max_cc = max((fn.cyclomatic_complexity for fn in functions), default=0)

    if not functions:
        summary = (
            f"**Lines of code:** {loc}  \n"
            f"**Tokens:** {token_count}  \n"
            f"**Functions detected:** 0  \n\n"
            "_No functions or methods were found. Try pasting a complete function or class._"
        )
        return summary, None

    per_function_rows = "\n".join(
        f"| `{fn.name}` | {fn.cyclomatic_complexity} | {_complexity_rating(fn.cyclomatic_complexity)} | {fn.nloc} |"
        for fn in functions
    )
    summary = (
        f"**Overall rating:** {_complexity_rating(max_cc)} (max cyclomatic complexity: **{max_cc}**)  \n"
        f"**Average cyclomatic complexity:** {avg_cc:.1f}  \n"
        f"**Lines of code:** {loc}  \n"
        f"**Functions / methods:** {len(functions)}  \n\n"
        "| Function | Cyclomatic | Rating | LOC |\n"
        "| --- | ---: | --- | ---: |\n"
        f"{per_function_rows}\n\n"
        "_Cyclomatic complexity counts independent paths through a function. "
        "Values above 10 often indicate code that is harder to test and maintain._"
    )

    labels = [fn.name if len(fn.name) <= 24 else f"{fn.name[:21]}…" for fn in functions]
    values = [fn.cyclomatic_complexity for fn in functions]
    colors = [_complexity_color(value) for value in values]

    fig_height = max(3.5, len(functions) * 0.55 + 1.5)
    fig, ax = plt.subplots(figsize=(8, fig_height))
    bars = ax.barh(labels, values, color=colors, edgecolor="#333333", linewidth=0.5)
    ax.axvline(
        COMPLEXITY_THRESHOLDS["moderate"],
        color="#e74c3c",
        linestyle="--",
        linewidth=1.2,
        label=f"High threshold ({COMPLEXITY_THRESHOLDS['moderate']})",
    )
    ax.axvline(
        COMPLEXITY_THRESHOLDS["low"],
        color="#f39c12",
        linestyle="--",
        linewidth=1.2,
        label=f"Moderate threshold ({COMPLEXITY_THRESHOLDS['low']})",
    )
    ax.set_xlabel("Cyclomatic complexity")
    ax.set_title(f"Cyclomatic complexity by function ({language})")
    ax.legend(loc="lower right", fontsize=8)
    ax.invert_yaxis()

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_width() + 0.15,
            bar.get_y() + bar.get_height() / 2,
            str(value),
            va="center",
            fontsize=9,
        )

    fig.tight_layout()
    return summary, fig


def build_interface() -> gr.Blocks:
    try:
        validate_checkpoint(MODEL_CHECKPOINT)
        checkpoint_status = f"Trained checkpoint available at `{MODEL_CHECKPOINT}`"
    except ValueError:
        checkpoint_status = "Detector unavailable: run train_model.py first. Complexity analysis remains available."

    with gr.Blocks(
        title="SemEval-2026 Task 13 — Code Authorship Detector", theme=gr.themes.Soft()
    ) as demo:
        gr.Markdown(
            """
            # SemEval-2026 Task 13: Machine-Generated Code Detector

            Detect whether a code snippet is **human-written** or **machine-generated**
            using [microsoft/codebert-base](https://huggingface.co/microsoft/codebert-base),
            and analyze its **cyclomatic complexity** with an interactive chart.

            Supports **Python**, **C++**, and **Java**. This is an experimental detector; confidence scores are uncalibrated estimates, not proof of authorship.
            """
        )
        gr.Markdown(f"**Model status:** {checkpoint_status}")

        with gr.Row():
            with gr.Column(scale=1):
                language = gr.Dropdown(
                    choices=LANGUAGE_OPTIONS,
                    value="Python",
                    label="Programming language",
                    info="Select the language of the submitted code.",
                )
                code_input = gr.Textbox(
                    label="Code snippet",
                    placeholder="Paste your code here…",
                    lines=16,
                    max_lines=30,
                )
                analyze_btn = gr.Button("Analyze code", variant="primary")

                gr.Examples(
                    examples=[EXAMPLE_HUMAN_PYTHON, EXAMPLE_MACHINE_PYTHON],
                    inputs=[language, code_input],
                    label="Illustrative snippets (not labeled test data)",
                )

            with gr.Column(scale=1):
                prediction = gr.Label(
                    label="Prediction",
                    value={},
                )
                with gr.Row():
                    human_confidence = gr.Textbox(
                        label="Human-written confidence",
                        interactive=False,
                    )
                    machine_confidence = gr.Textbox(
                        label="Machine-generated confidence",
                        interactive=False,
                    )
                details = gr.Markdown(label="Details")

        gr.Markdown("## Code complexity")
        with gr.Row():
            complexity_summary = gr.Markdown(
                "Run **Analyze code** to compute cyclomatic complexity."
            )
            complexity_plot = gr.Plot(label="Complexity graph")

        def run_inference(lang: str, code: str):
            pred, human_conf, machine_conf, detail_md = predict(lang, code)
            complexity_md, complexity_fig = analyze_complexity(lang, code)

            if pred == "—":
                if complexity_fig is not None:
                    plt.close(complexity_fig)
                return (
                    gr.update(value={}),
                    human_conf,
                    machine_conf,
                    detail_md,
                    complexity_md,
                    complexity_fig,
                )

            if complexity_fig is not None:
                plt.close(complexity_fig)
            scores = {
                "Human-written": float(human_conf.strip("%")) / 100,
                "Machine-generated": float(machine_conf.strip("%")) / 100,
            }
            return (
                scores,
                human_conf,
                machine_conf,
                detail_md,
                complexity_md,
                complexity_fig,
            )

        outputs = [
            prediction,
            human_confidence,
            machine_confidence,
            details,
            complexity_summary,
            complexity_plot,
        ]
        analyze_btn.click(
            fn=run_inference,
            inputs=[language, code_input],
            outputs=outputs,
        )
        code_input.submit(
            fn=run_inference,
            inputs=[language, code_input],
            outputs=outputs,
        )

        gr.Markdown(
            """
            ---
            **Note:** For competitive SemEval performance, fine-tune CodeBERT on the
            official [Task 13 dataset](https://huggingface.co/datasets/DaniilOr/SemEval-2026-Task13)
            and save weights to `./checkpoints/`, or set `MODEL_CHECKPOINT`.
            """
        )

    return demo


if __name__ == "__main__":
    app = build_interface()
    app.launch(
        server_name="127.0.0.1",
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
        ssr_mode=False,
    )
