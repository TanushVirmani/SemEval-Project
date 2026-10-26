"""Generate a factual project report from saved evaluation results."""

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
OUTPUT = BASE / "Project_Report.pdf"


def main():
    metrics_path = BASE / "results/evaluation_metrics.json"
    if not metrics_path.exists():
        raise SystemExit(
            "Run evaluate_model.py first; a report requires real evaluation results."
        )
    metrics = json.loads(metrics_path.read_text())
    styles = getSampleStyleSheet()
    styles["BodyText"].leading = 15
    story = []

    def paragraph(text, style="BodyText"):
        story.append(Paragraph(text, styles[style]))
        story.append(Spacer(1, 9))

    paragraph("Code Authorship Detection", "Title")
    paragraph("SemEval-2026 Task 13 - Subtask A", "Heading2")
    paragraph("Model and experiment", "Heading2")
    training = metrics["training"]
    paragraph(
        "This project fine-tunes microsoft/codebert-base to classify source code as human-written "
        "(label 0) or machine-generated (label 1). CodeBERT pretraining alone does not provide a "
        "trained authorship classifier. The application loads only a trained local checkpoint."
    )
    paragraph(
        f"The recorded run used {training['train_rows']:,} training snippets and "
        f"{training['validation_rows']:,} validation snippets, with random seed {training['seed']}, "
        f"{training['steps_completed']} optimization steps and a maximum token length of "
        f"{training['max_length']}. Best checkpoint selection uses validation macro F1. "
        f"Smoke run: {training['smoke_run']}."
    )
    if training.get("best_validation_macro_f1") is not None:
        paragraph(
            f"Selected validation macro F1: {training['best_validation_macro_f1']:.4f}. "
            f"Selected checkpoint: {Path(training['best_checkpoint']).name}."
        )
    paragraph("Data and methodology", "Heading2")
    paragraph(
        "Source: DaniilOr/SemEval-2026-Task13, configuration A. Training and validation examples "
        "are sampled with class stratification from their respective official splits. Identical "
        "trimmed snippets are removed across splits. All model fitting uses training data; "
        "validation selects the checkpoint, and test data is used only for final evaluation. "
        "Training, inference and evaluation share the same language-prefix preprocessing."
    )
    paragraph(
        "The test set is the organizer-provided public sample of the test data, not the full "
        "competition test set. This initial experiment uses a small training subset and must "
        "not be presented as a full-data or leaderboard result."
    )
    paragraph("Held-out results", "Heading2")
    paragraph(
        f"Test samples: {metrics['test_samples']:,}. Accuracy: {metrics['accuracy']:.2%}. "
        f"Macro precision: {metrics['macro_precision']:.4f}; macro recall: "
        f"{metrics['macro_recall']:.4f}; macro F1: {metrics['macro_f1']:.4f}."
    )
    paragraph(
        f"For context, always predicting the majority class gives "
        f"{metrics['majority_class_accuracy']:.2%} accuracy on this sample. "
        "Macro F1 and per-class recall should be considered alongside accuracy because the class distribution is uneven."
    )
    rows = [["Class", "Precision", "Recall", "F1", "Support"]]
    for label in ["Human-written", "Machine-generated"]:
        entry = metrics["classification_report"][label]
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
    story.extend([table, Spacer(1, 14)])
    figure = BASE / "results/confusion_matrix.png"
    if figure.exists():
        story.append(
            Image(str(figure), width=5.2 * inch, height=3.9 * inch, kind="proportional")
        )
    paragraph("Interpretation", "Heading2")
    paragraph(
        "The measured test accuracy is below the majority-class accuracy baseline. "
        "Despite strong validation scores, this initial model generalizes poorly and should "
        "not be presented as a reliable authorship detector. The training subset contains "
        "1,825 Python examples out of 2,000, while the public test sample covers eight "
        "languages. Distribution shift may contribute to the gap, but its causal contribution "
        "has not been isolated. Future improvements should be selected using validation "
        "data and assessed on a fresh held-out set."
    )
    paragraph("Application and limitations", "Heading2")
    paragraph(
        "The Gradio app accepts Python, C++ and Java, displays model probabilities, and uses "
        "Lizard for per-function cyclomatic complexity. Complexity measures independent control-flow "
        "paths; it does not establish authorship or Big-O runtime. Probabilities are not calibrated "
        "certainty and should not be used as proof of misconduct. Long code is truncated. "
        "Exact deduplication cannot rule out near-duplicate solutions or shared programming problems."
    )
    paragraph("Reproducibility", "Heading2")
    paragraph(
        "Run prepare_data.py, train_model.py, evaluate_model.py, then generate_report.py. "
        "See README.md for exact commands and subset sizes. Data manifests and training metadata "
        "record random seeds, row counts and SHA-256 hashes. Results include a per-example CSV, "
        "machine-readable metrics and a confusion matrix."
    )
    paragraph("References", "Heading2")
    paragraph(
        "Official task: https://github.com/mbzuai-nlp/SemEval-2026-Task13<br/>"
        "Official dataset: https://huggingface.co/datasets/DaniilOr/SemEval-2026-Task13<br/>"
        "Base model: https://huggingface.co/microsoft/codebert-base"
    )
    SimpleDocTemplate(
        str(OUTPUT), leftMargin=42, rightMargin=42, topMargin=38, bottomMargin=38
    ).build(story)
    print(OUTPUT)


if __name__ == "__main__":
    main()
