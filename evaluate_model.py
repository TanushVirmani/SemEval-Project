"""Evaluate a trained classifier on an independent labeled CSV."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from model_utils import (
    BASE_DIR,
    DEFAULT_CHECKPOINT,
    LABELS,
    read_labeled_csv,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--test-csv",
        type=Path,
        default=Path(os.environ.get("TEST_CSV", BASE_DIR / "data/test.csv")),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.environ.get("EVALUATION_OUTPUT_DIR", BASE_DIR / "results")),
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--evaluation-role",
        default="unspecified",
        help="Describe whether this is a fresh holdout or a reused development benchmark.",
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("Batch size must be positive.")
    from detector import Detector, validate_detector

    checkpoint = validate_detector(args.checkpoint)
    df = read_labeled_csv(args.test_csv)
    metadata = json.loads((checkpoint / "training_metadata.json").read_text())
    forbidden = set(metadata["training_code_hashes"]) | set(
        metadata["validation_code_hashes"]
    )
    if any(
        hashlib.sha256(c.strip().encode()).hexdigest() in forbidden for c in df.code
    ):
        raise ValueError(
            "Evaluation data overlaps training or validation; supply an independent test set."
        )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import (
        ConfusionMatrixDisplay,
        accuracy_score,
        classification_report,
        confusion_matrix,
    )

    detector = Detector(checkpoint)
    probabilities = []
    for start in range(0, len(df), args.batch_size):
        batch = df.iloc[start : start + args.batch_size]
        probabilities.extend(
            detector.probabilities(batch.code, batch.language).tolist()
        )
        print(
            f"Evaluated {min(start + args.batch_size, len(df))}/{len(df)}", flush=True
        )
    predictions = [int(p[1] > p[0]) for p in probabilities]
    report = classification_report(
        df.label,
        predictions,
        labels=[0, 1],
        target_names=list(LABELS.values()),
        zero_division=0,
        output_dict=True,
    )
    matrix = confusion_matrix(df.label, predictions, labels=[0, 1])
    summary = {
        "evaluation_role": args.evaluation_role,
        "test_samples": len(df),
        "accuracy": accuracy_score(df.label, predictions),
        "majority_class_accuracy": float(df.label.value_counts().max() / len(df)),
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "macro_f1": report["macro avg"]["f1-score"],
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "checkpoint": str(checkpoint),
        "test_csv": str(args.test_csv.resolve()),
        "test_sha256": hashlib.sha256(args.test_csv.read_bytes()).hexdigest(),
        "training": {
            k: v for k, v in metadata.items() if not k.endswith("code_hashes")
        },
    }
    if "parent_hash" in df.columns:
        import pandas as pd

        grouped = pd.DataFrame(
            {
                "parent_hash": df.parent_hash,
                "label": df.label,
                "machine_score": [p[1] for p in probabilities],
            }
        )
        if grouped.groupby("parent_hash").label.nunique().max() > 1:
            raise ValueError("Parent file has conflicting labels.")
        parents = grouped.groupby("parent_hash").agg(
            label=("label", "first"), machine_score=("machine_score", "mean")
        )
        parent_predictions = (parents.machine_score > 0.5).astype(int)
        summary["parent_files"] = len(parents)
        summary["parent_accuracy"] = float(
            accuracy_score(parents.label, parent_predictions)
        )
        summary["parent_classification_report"] = classification_report(
            parents.label,
            parent_predictions,
            labels=[0, 1],
            target_names=list(LABELS.values()),
            zero_division=0,
            output_dict=True,
        )
        summary["parent_macro_f1"] = summary["parent_classification_report"][
            "macro avg"
        ]["f1-score"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "evaluation_metrics.json").write_text(
        json.dumps(summary, indent=2)
    )
    text = (
        f"Evaluation role: {args.evaluation_role}\nTest samples: {len(df)}\nAccuracy: {summary['accuracy']:.4f}\n"
        f"Macro Precision: {summary['macro_precision']:.4f}\nMacro Recall: {summary['macro_recall']:.4f}\n"
        f"Macro F1-score: {summary['macro_f1']:.4f}\nSmoke training run: {metadata.get('smoke_run', False)}\n\n"
    )
    text += f"Majority-class accuracy baseline: {summary['majority_class_accuracy']:.4f}\n\n"
    if "parent_files" in summary:
        text += f"Parent files: {summary['parent_files']}\nParent accuracy: {summary['parent_accuracy']:.4f}\nParent macro F1: {summary['parent_macro_f1']:.4f}\n\n"
    text += classification_report(
        df.label,
        predictions,
        labels=[0, 1],
        target_names=list(LABELS.values()),
        digits=4,
        zero_division=0,
    )
    (args.output_dir / "evaluation_metrics.txt").write_text(text)
    df["predicted_label"] = predictions
    df["predicted_class"] = [LABELS[p] for p in predictions]
    df["correct"] = df.label == df.predicted_label
    df["human_probability"] = [p[0] for p in probabilities]
    df["machine_probability"] = [p[1] for p in probabilities]
    df.to_csv(args.output_dir / "evaluation_predictions.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay(matrix, display_labels=list(LABELS.values())).plot(
        ax=ax, cmap="Blues", values_format="d"
    )
    plt.title(f"{detector.backend} - held-out evaluation")
    plt.tight_layout()
    plt.savefig(args.output_dir / "confusion_matrix.png", dpi=180, bbox_inches="tight")
    plt.close()
    print(text)
    print(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
