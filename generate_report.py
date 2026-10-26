"""Generate a factual PDF from saved evaluation results, with optional second benchmark."""

import argparse
import json
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
)

BASE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-dir", type=Path, default=BASE / "results")
    parser.add_argument("--secondary-metrics-dir", type=Path)
    parser.add_argument("--output", type=Path, default=BASE / "Project_Report.pdf")
    args = parser.parse_args()
    path = args.metrics_dir / "evaluation_metrics.json"
    if not path.exists():
        raise SystemExit(
            "Run evaluate_model.py first; the report requires real results."
        )
    metrics = json.loads(path.read_text())
    training = metrics["training"]
    styles = getSampleStyleSheet()
    styles["BodyText"].leading = 15
    story = []

    def paragraph(text, style="BodyText"):
        story.extend([Paragraph(text, styles[style]), Spacer(1, 9)])

    paragraph("Code Authorship Detection", "Title")
    paragraph("Measured results and reproducible methodology", "Heading2")
    backend = training.get("backend", "codebert")
    paragraph("Model and training", "Heading2")
    if training.get("training_origin") == "third_party":
        paragraph(
            f"The active detector is an existing fine-tuned model imported from {training['source_model_card']}. "
            "Its authors identify the class mapping, but do not document the exact training dataset. "
            "Training/test overlap cannot be independently checked. Scores therefore describe this "
            "comparison and must not be claimed as independently verified unseen-data performance."
        )
    elif backend == "efficient":
        paragraph(
            "The current detector combines lexical TF-IDF features, structural code statistics and trained "
            "classifiers. Candidate models and fixed ensembles are selected using validation macro F1. "
            "It is a CPU-efficient alternative to the original fine-tuned CodeBERT model."
        )
        paragraph(
            f"Selected model: {training['selected']}. Training snippets/views: {training['train_rows']:,}; "
            f"validation snippets/views: {training['validation_rows']:,}. Random seed: {training['seed']}."
        )
        if training.get("track") == "broader":
            paragraph(
                "This academic/demo track uses official SemEval Subtask A training examples plus CodeMirage "
                "training data. CodeMirage adds more languages and production-style code. This configuration "
                "is not eligible for an official SemEval submission, which restricts external training data. "
                "CodeMirage attribution and license: HanxiGuo/CodeMirage, CC BY-NC-ND 4.0."
            )
        if training.get("fragment_views"):
            paragraph(
                "Training and validation include full-file and function/contiguous-fragment views. Fragments inherit "
                "their parent label. Related code is grouped using normalized fingerprints before evaluation "
                "splits; held-out parent files and the public benchmark are excluded from training."
            )
    else:
        paragraph(
            "The original model fine-tunes microsoft/codebert-base for labels 0=human-written and "
            "1=machine-generated. CodeBERT pretraining alone does not supply a trained authorship classifier."
        )
        paragraph(
            f"Training snippets: {training['train_rows']:,}; validation snippets: {training['validation_rows']:,}; "
            f"optimization steps: {training['steps_completed']}; maximum tokens: {training['max_length']}."
        )
    paragraph("Evaluation methodology", "Heading2")
    paragraph(
        "Training data fits model parameters; validation data selects the classifier. Evaluation rejects "
        "exact test snippets used for fitting or validation. Class-balanced holdouts have an approximately "
        "50% majority-class baseline. Snippet/view counts are not necessarily independent parent-file counts. "
        "The original public SemEval test sample has been inspected during development and its rechecks "
        "are exploratory; they do not constitute a fresh final benchmark."
    )

    def add_metrics(m, root, title):
        paragraph(title, "Heading2")
        paragraph(
            f"Evaluation role: {m.get('evaluation_role', 'original public test sample')}. "
            f"Samples: {m['test_samples']:,}. Accuracy: {m['accuracy']:.2%}; macro precision: "
            f"{m['macro_precision']:.4f}; macro recall: {m['macro_recall']:.4f}; macro F1: {m['macro_f1']:.4f}. "
            f"Majority-class accuracy baseline: {m['majority_class_accuracy']:.2%}."
        )
        if m.get("parent_files"):
            paragraph(
                f"Parent files: {m['parent_files']:,}. Averaging decision scores across each parent's views gives "
                f"{m['parent_accuracy']:.2%} parent-level accuracy and {m['parent_macro_f1']:.4f} macro F1. "
                "Views from the same parent are correlated; parent-level metrics provide a separate check."
            )
        rows = [["Class", "Precision", "Recall", "F1", "Support"]]
        for label in ["Human-written", "Machine-generated"]:
            entry = m["classification_report"][label]
            rows.append(
                [
                    label,
                    f"{entry['precision']:.4f}",
                    f"{entry['recall']:.4f}",
                    f"{entry['f1-score']:.4f}",
                    str(int(entry["support"])),
                ]
            )
        table = Table(rows, colWidths=[140, 75, 75, 75, 75])
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e4edf5")),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
                    ("TOPPADDING", (0, 0), (-1, -1), 8),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ]
            )
        )
        story.extend([table, Spacer(1, 12)])
        fig = root / "confusion_matrix.png"
        if fig.exists():
            story.append(
                Image(
                    str(fig), width=5.2 * inch, height=3.9 * inch, kind="proportional"
                )
            )
        if m["accuracy"] < m["majority_class_accuracy"]:
            paragraph(
                "This result is below the majority-class accuracy baseline. The detector should not be "
                "presented as reliable on this benchmark, even if another dataset gives higher accuracy."
            )

    add_metrics(metrics, args.metrics_dir, "Primary benchmark")
    if args.secondary_metrics_dir:
        secondary = json.loads(
            (args.secondary_metrics_dir / "evaluation_metrics.json").read_text()
        )
        add_metrics(
            secondary,
            args.secondary_metrics_dir,
            "Separate benchmark - different distribution",
        )
    paragraph("Application, limitations and reproducibility", "Heading2")
    paragraph(
        "The Gradio app displays predictions and uncalibrated model probabilities. These are estimates, "
        "not proof of authorship. Lizard computes cyclomatic complexity separately; it does not establish "
        "authorship or Big-O runtime. Exact and normalized deduplication cannot establish that all "
        "semantically related solutions or repository families are independent."
    )
    paragraph(
        "See README.md and IMPROVEMENTS.md for commands, checkpoint selection, data provenance and benchmark "
        "limitations. Saved JSON metrics, per-example predictions and training metadata record the evidence "
        "behind every reported result. The original CodeBERT checkpoint and results are retained separately."
    )
    paragraph("Sources", "Heading2")
    paragraph(
        "Official task: https://github.com/mbzuai-nlp/SemEval-2026-Task13<br/>"
        "Official data: https://huggingface.co/datasets/DaniilOr/SemEval-2026-Task13<br/>"
        "Broader demo data: https://huggingface.co/datasets/HanxiGuo/CodeMirage<br/>"
        "Original encoder: https://huggingface.co/microsoft/codebert-base"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    SimpleDocTemplate(
        str(args.output), leftMargin=42, rightMargin=42, topMargin=38, bottomMargin=38
    ).build(story)
    print(args.output)


if __name__ == "__main__":
    main()
