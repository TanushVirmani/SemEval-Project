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
    add_language_prefix,
    load_classifier,
    read_labeled_csv,
    validate_checkpoint,
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
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("Batch size must be positive.")
    checkpoint = validate_checkpoint(args.checkpoint)
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
    import torch
    from sklearn.metrics import (
        ConfusionMatrixDisplay,
        accuracy_score,
        classification_report,
        confusion_matrix,
    )

    model, tokenizer, device = load_classifier(checkpoint)
    probabilities = []
    texts = [add_language_prefix(r.language, r.code) for r in df.itertuples()]
    with torch.inference_mode():
        for start in range(0, len(texts), args.batch_size):
            inputs = tokenizer(
                texts[start : start + args.batch_size],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=metadata["max_length"],
            )
            logits = model(**{k: v.to(device) for k, v in inputs.items()}).logits
            probabilities.extend(torch.softmax(logits, dim=-1).cpu().tolist())
            print(
                f"Evaluated {min(start + args.batch_size, len(texts))}/{len(texts)}",
                flush=True,
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
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "evaluation_metrics.json").write_text(
        json.dumps(summary, indent=2)
    )
    text = (
        f"Test samples: {len(df)}\nAccuracy: {summary['accuracy']:.4f}\n"
        f"Macro Precision: {summary['macro_precision']:.4f}\nMacro Recall: {summary['macro_recall']:.4f}\n"
        f"Macro F1-score: {summary['macro_f1']:.4f}\nSmoke training run: {metadata['smoke_run']}\n\n"
    )
    text += f"Majority-class accuracy baseline: {summary['majority_class_accuracy']:.4f}\n\n"
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
    plt.title("CodeBERT — held-out evaluation")
    plt.tight_layout()
    plt.savefig(args.output_dir / "confusion_matrix.png", dpi=180, bbox_inches="tight")
    plt.close()
    print(text)
    print(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
