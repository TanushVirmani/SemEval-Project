# Accuracy improvement experiments

This is the **academic/demo track**. It adds CodeMirage training data, so it is
not eligible as an official SemEval submission. The competition-compatible
CodeBERT checkpoint and original results are retained separately.

## What changed

- Broader language and production-code coverage using CodeMirage, rather than
  only the algorithmic SemEval training domain.
- Class and language balancing, normalization of lexical features, and separate
  structural classifiers with ensembles selected using validation macro F1.
- Full-file and function/contiguous-fragment training views. Views inherit
  their parent labels; supplied source code is parsed, never executed.
- Conservative grouping across identifier names, literals, comments and
  whitespace, alongside exact-code exclusions across fitting and evaluation.
- A CodeBERT transfer experiment, initialized from the original trained model,
  with eight lower layers frozen, first-plus-last token handling, 4,000 fitting
  examples and 1,000 validation examples.
- Validation-selected decision thresholds. Displayed confidence values are
  uncalibrated decision scores, not probabilities of authorship being true.

## Evaluation discipline

The 1,000-example public SemEval sample was already inspected during the initial
43.1% evaluation. Subsequent results on it are **development rechecks**, not fresh
final tests. Model-selection decisions in `select_improved_model.py` use only
shared validation macro F1. The fresh final holdout contains previously
uninspected CodeMirage parent files, with matching normalized fitting, tuning
and previously inspected views excluded.

Views from one parent are correlated. Evaluation therefore reports both view
metrics and parent-file metrics (averaging decision scores across a parent's
views). These datasets measure different distributions: results on a broader
holdout must not be described as an improvement on the original SemEval sample.
The public SemEval majority-class accuracy baseline is 77.7%; accuracy alone is
insufficient to establish a useful detector.

## Recorded exploratory experiments

- Original CodeBERT: 2,000 official training examples; 43.1% public-sample accuracy.
- Balanced 48,000-example CPU ensemble: 95.4% on an in-domain holdout, but 29.7%
  on the public sample. Preserved as a failed generalization experiment.
- Broader full-file ensemble: 91.9% on a 4,000-example class-balanced CodeMirage
  holdout, but 49.0% on the reused public sample.
- Fragment-aware ensemble: 66.3% on the reused public sample after a 0.55
  decision threshold chosen using validation data; 79.8% on its cleaned
  4,833-view holdout. These are separate benchmarks, not directly comparable
  accuracy gains.

The final selected model and final holdout metrics are recorded in
`model_selection.json` and `results/final-holdout/evaluation_metrics.json`.

## Reproduction

Use the project's `.venv` and installed dependencies. For the exact experimental
sequence, first prepare the official data and train the original CodeBERT model
as described in README.md, then run:

```sh
python improve_model.py
python download_broader_data.py
python improve_model.py --track broader --per-group 600 --output-dir checkpoints-broader
python improve_model.py --track broader --fragment-views --per-group 600 --output-dir checkpoints-fragments
python calibrate_detector.py checkpoints-fragments
```

The transfer fitting/validation CSVs are deterministic equal-class/language
subsamples of the fragment experiment (200 and 50 examples per class/language,
respectively, using SEED=20261026). See `prepare_transfer_holdout.py` for the fresh
holdout reservation. The transfer training command is:

```sh
python train_model.py --train-csv data/transfer/train.csv --validation-csv data/transfer/validation.csv --output-dir checkpoints-transfer --initial-checkpoint checkpoints --freeze-layers 8 --epochs 1 --batch-size 8 --max-length 128 --tokenization head_tail --learning-rate 0.00005
python select_improved_model.py
```

Existing model directories are protected from overwrite. Use fresh directories
or preserve old experiments before a deliberate rerun.

## Data provenance

- Official SemEval data: DaniilOr/SemEval-2026-Task13, configuration A, revision
  `df2aec18238a47fabfb22c79570db15fd1588aa3`.
- CodeMirage: HanxiGuo/CodeMirage, revision
  `174c25ffa9e9f218f8526d403b3aba9631dcfc43`, license CC BY-NC-ND 4.0.
  Source: https://huggingface.co/datasets/HanxiGuo/CodeMirage
- Original pretrained encoder: microsoft/codebert-base, revision
  `3b0952feddeffad0063f274080e3c23d75e7eb39`.

Training/evaluation code and summary metrics can be shared; downloaded datasets,
local weights, caches and raw prediction CSVs remain ignored by Git. Normalized
fingerprints are a conservative duplicate check, not proof that all repository
families or semantically related solutions are independent.
