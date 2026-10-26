# Evaluation

See README.md for the complete training and evaluation workflow.

Run `.venv/bin/python evaluate_model.py` after data preparation and training.
Defaults: `checkpoints/`, `data/test.csv`, output `results/`.
Use `--checkpoint`, `--test-csv`, `--output-dir`, and `--batch-size` to customize.
MODEL_CHECKPOINT, TEST_CSV and EVALUATION_OUTPUT_DIR are also supported.

Evaluation reports accuracy; per-class precision, recall, F1 and support;
macro precision, recall and F1; weighted averages; and a confusion matrix.
Both classes are required. Fractional/missing labels are rejected. Test snippets
matching training or validation code are rejected to prevent direct leakage.
Training metadata, test data hash and checkpoint path accompany JSON metrics.
