"""
Gradio web app for SemEval-2026 Task 13 (Subtask A):
binary classification of code as human-written vs machine-generated.

Uses microsoft/codebert-base with a 2-class sequence classification head.
Place fine-tuned weights in ./checkpoints/ or set MODEL_CHECKPOINT env var.
"""

from __future__ import annotations

import os
from functools import lru_cache

import matplotlib

matplotlib.use("Agg")

import gradio as gr
import lizard
import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = os.environ.get("MODEL_NAME", "microsoft/codebert-base")
MODEL_CHECKPOINT = os.environ.get("MODEL_CHECKPOINT", "./checkpoints")
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "512"))

LABELS = {
    0: "Human-written",
    1: "Machine-generated",
}

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
    '''def two_sum(nums, target):
    seen = {}
    for i, n in enumerate(nums):
        if target - n in seen:
            return [seen[target - n], i]
        seen[n] = i
    return []''',
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
    prefixes = {
        "Python": "# language: python\n",
        "C++": "// language: cpp\n",
        "Java": "// language: java\n",
    }
    return prefixes.get(language, "") + code.strip()


@lru_cache(maxsize=1)
def load_model_and_tokenizer():
    checkpoint_path = MODEL_CHECKPOINT
    load_path = checkpoint_path if os.path.isdir(checkpoint_path) else MODEL_NAME

    tokenizer = AutoTokenizer.from_pretrained(load_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        load_path,
        num_labels=2,
        id2label=LABELS,
        label2id={v: k for k, v in LABELS.items()},
    )
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, tokenizer, device


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

    model, tokenizer, device = load_model_and_tokenizer()
    text = _language_prefix(language, code)

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
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
        f"**Top confidence:** {confidence:.1%}  \n\n"
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
    checkpoint_status = (
        f"Loaded from `{MODEL_CHECKPOINT}`"
        if os.path.isdir(MODEL_CHECKPOINT)
        else f"Using base weights from `{MODEL_NAME}` (fine-tune for best accuracy)"
    )

    with gr.Blocks(title="SemEval-2026 Task 13 — Code Authorship Detector") as demo:
        gr.Markdown(
            """
            # SemEval-2026 Task 13: Machine-Generated Code Detector

            Detect whether a code snippet is **human-written** or **machine-generated**
            using [microsoft/codebert-base](https://huggingface.co/microsoft/codebert-base),
            and analyze its **cyclomatic complexity** with an interactive chart.

            Supports training languages from Subtask A: **Python**, **C++**, and **Java**.
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
                    label="Sample examples",
                )

            with gr.Column(scale=1):
                prediction = gr.Label(
                    label="Prediction",
                    value={"Human-written": 0.5, "Machine-generated": 0.5},
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
                return (
                    gr.update(value={"Human-written": 0.0, "Machine-generated": 0.0}),
                    human_conf,
                    machine_conf,
                    detail_md,
                    complexity_md,
                    complexity_fig,
                )

            scores = {
                "Human-written": float(human_conf.strip("%")) / 100,
                "Machine-generated": float(machine_conf.strip("%")) / 100,
            }
            return scores, human_conf, machine_conf, detail_md, complexity_md, complexity_fig

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
        server_port=7860,
        theme=gr.themes.Soft(),
        ssr_mode=False,
    )
