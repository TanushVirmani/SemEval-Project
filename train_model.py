"""Fine-tune CodeBERT on official Subtask A training data."""

import argparse
import hashlib
import json
from pathlib import Path

from model_utils import BASE_DIR, LABELS, add_language_prefix, read_labeled_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, default=BASE_DIR / "data/train.csv")
    parser.add_argument(
        "--validation-csv", type=Path, default=BASE_DIR / "data/validation.csv"
    )
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR / "checkpoints")
    parser.add_argument("--epochs", type=float, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Positive value for a pipeline smoke run only",
    )
    parser.add_argument(
        "--resume-from-checkpoint", type=Path, help="Resume a saved Trainer checkpoint"
    )
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument(
        "--freeze-layers",
        type=int,
        default=0,
        help="Freeze embeddings and the first N encoder layers",
    )
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--tokenization", choices=["head", "head_tail"], default="head")
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or not 8 <= args.max_length <= 512:
        parser.error(
            "Epochs and batch size must be positive; max length must be 8..512."
        )
    if (
        args.resume_from_checkpoint
        and not (args.resume_from_checkpoint / "trainer_state.json").is_file()
    ):
        parser.error("Resume checkpoint has no trainer_state.json")
    if (
        args.output_dir.exists()
        and any(args.output_dir.iterdir())
        and not args.resume_from_checkpoint
    ):
        parser.error(
            "Output directory is nonempty. Choose a new directory to preserve the existing model."
        )
    train_df, val_df = (
        read_labeled_csv(args.train_csv),
        read_labeled_csv(args.validation_csv),
    )
    if set(train_df.code.str.strip()) & set(val_df.code.str.strip()):
        raise ValueError("Training and validation contain overlapping code.")
    from datasets import Dataset
    from sklearn.metrics import accuracy_score, f1_score
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    set_seed(42)
    initialization_source = str(
        args.resume_from_checkpoint
        or args.initial_checkpoint
        or "microsoft/codebert-base"
    )
    tokenizer = AutoTokenizer.from_pretrained(initialization_source)
    model = AutoModelForSequenceClassification.from_pretrained(
        initialization_source,
        num_labels=2,
        id2label=LABELS,
        label2id={v: k for k, v in LABELS.items()},
    )

    if args.freeze_layers:
        if not 0 <= args.freeze_layers <= len(model.roberta.encoder.layer):
            parser.error("Freeze layer count is out of range")
        for parameter in model.roberta.embeddings.parameters():
            parameter.requires_grad = False
        for layer in model.roberta.encoder.layer[: args.freeze_layers]:
            for parameter in layer.parameters():
                parameter.requires_grad = False
    from model_utils import encode_texts

    def dataset(df):
        ds = Dataset.from_dict(
            {
                "text": [
                    add_language_prefix(r.language, r.code) for r in df.itertuples()
                ],
                "labels": df.label.tolist(),
            }
        )
        return ds.map(
            lambda batch: encode_texts(
                tokenizer, batch["text"], args.max_length, args.tokenization
            ),
            batched=True,
            remove_columns=["text"],
        )

    training_args = TrainingArguments(
        output_dir=str(args.output_dir / "runs"),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        save_total_limit=1,
        report_to="none",
        seed=42,
        data_seed=42,
        logging_steps=10,
        dataloader_pin_memory=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset(train_df),
        eval_dataset=dataset(val_df),
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=lambda p: {
            "accuracy": accuracy_score(p.label_ids, p.predictions.argmax(-1)),
            "macro_f1": f1_score(
                p.label_ids, p.predictions.argmax(-1), average="macro", zero_division=0
            ),
        },
    )
    print(f"Training device: {training_args.device}", flush=True)
    trainer.train(
        resume_from_checkpoint=str(args.resume_from_checkpoint)
        if args.resume_from_checkpoint
        else None
    )
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(args.output_dir)
    metadata = {
        "base_model": "microsoft/codebert-base",
        "initial_checkpoint": str(args.initial_checkpoint)
        if args.initial_checkpoint
        else None,
        "freeze_layers": args.freeze_layers,
        "learning_rate": args.learning_rate,
        "tokenization": args.tokenization,
        "seed": 42,
        "train_rows": len(train_df),
        "validation_rows": len(val_df),
        "epochs_requested": args.epochs,
        "steps_completed": trainer.state.global_step,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_validation_macro_f1": trainer.state.best_metric,
        "device": str(training_args.device),
        "max_length": args.max_length,
        "smoke_run": args.max_steps > 0,
        "train_csv_sha256": hashlib.sha256(args.train_csv.read_bytes()).hexdigest(),
        "validation_csv_sha256": hashlib.sha256(
            args.validation_csv.read_bytes()
        ).hexdigest(),
        "training_code_hashes": [
            hashlib.sha256(c.strip().encode()).hexdigest() for c in train_df.code
        ],
        "validation_code_hashes": [
            hashlib.sha256(c.strip().encode()).hexdigest() for c in val_df.code
        ],
    }
    if (
        args.initial_checkpoint
        and (args.initial_checkpoint / "training_metadata.json").exists()
    ):
        prior = json.loads(
            (args.initial_checkpoint / "training_metadata.json").read_text()
        )
        metadata["prior_supervised_training_rows"] = prior.get("train_rows", 0)
        metadata["training_code_hashes"] = sorted(
            set(metadata["training_code_hashes"])
            | set(prior.get("training_code_hashes", []))
        )
        metadata["validation_code_hashes"] = sorted(
            set(metadata["validation_code_hashes"])
            | set(prior.get("validation_code_hashes", []))
        )
    (args.output_dir / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2)
    )
    print(f"Trained checkpoint saved to {args.output_dir}")


if __name__ == "__main__":
    main()
