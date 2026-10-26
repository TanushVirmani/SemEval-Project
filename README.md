# SemEval code authorship detector

Fine-tunes CodeBERT for SemEval-2026 Task 13 Subtask A (0 = human, 1 = machine).
The Gradio app also computes cyclomatic complexity using Lizard.

## Run locally (Python 3.11 recommended)

```sh
cd "/Users/fxtanush/Desktop/ADA Project/SemEval-Project"
source .venv/bin/activate
# On a fresh machine: python3 -m venv .venv, then activate it.
python -m pip install -r requirements.txt
# Exact versions from the verified local environment: requirements-lock.txt
python prepare_data.py
python train_model.py
python evaluate_model.py
python generate_report.py
python app.py
```

Open http://127.0.0.1:7860 after launching the app. Defaults resolve relative to
script locations. Set MODEL_CHECKPOINT to use a different trained checkpoint.
The detector requires trained weights; complexity analysis works without them.

## Data and initial experiment

Official source: https://huggingface.co/datasets/DaniilOr/SemEval-2026-Task13
(config A). Label mapping verified against the organizer repository:
https://github.com/mbzuai-nlp/SemEval-2026-Task13/blob/main/task_A/id_to_label.json

Defaults select 2,000 training and 500 validation examples with stratification
and seed 42, and use the 1,000-example public official test sample. The public
sample is not the full competition test set. Exact trimmed-code duplicates are
excluded across splits. Training uses two epochs and 256-token truncation. The dataset revision is pinned
in prepare_data.py and recorded in data/manifest.json.
This is a small initial experiment, not a full-data benchmark.

`data/manifest.json` records data hashes and counts. The checkpoint records
training settings and training/validation code hashes. Evaluation refuses test
snippets seen in either training or validation. Near-duplicate code and shared
programming problems are not detected by this exact-match check.

For full training, preferably on a CUDA GPU:

```sh
python prepare_data.py --train-size 0 --validation-size 0 --output-dir data/full
python train_model.py --train-csv data/full/train.csv --validation-csv data/full/validation.csv --output-dir checkpoints-full --epochs 3 --max-length 512
python evaluate_model.py --checkpoint checkpoints-full --test-csv data/full/test.csv --output-dir results/full
```

Training refuses to overwrite a nonempty checkpoint directory. Use a new
`--output-dir` for a new experiment. `--max-steps` is for pipeline smoke runs;
such runs are explicitly marked in the results. CUDA, Apple MPS and CPU are
supported, in that order of availability.

## Results

`results/` contains `evaluation_metrics.txt`, `evaluation_metrics.json`,
`evaluation_predictions.csv` and `confusion_matrix.png`. `Project_Report.pdf`
is regenerated from those real metrics. `generate_report_legacy.py` preserves
the original report source for reference; it is not the current report.
The root `test.csv` is an illustrative three-row example, not evaluation data.

Probabilities are uncalibrated estimates, not proof of authorship. Cyclomatic
complexity is independent of authorship and does not measure Big-O complexity.

## Verification

```sh
python -m unittest discover -s tests -v
python -m pip check
```

Model weights, datasets, caches and environment files stay local and are ignored
by Git. No changes are automatically pushed to GitHub.

## Resume after interruption

Training saves a recoverable checkpoint after each epoch. For this initial run:

```sh
python train_model.py --resume-from-checkpoint checkpoints/runs/checkpoint-250
```

Use the same data and training arguments as the interrupted run. This restores
model weights, optimizer, learning-rate scheduler and training progress. A partial
epoch since the last saved checkpoint must be repeated.

## Recorded result (initial 2,000-example model)

Validation accuracy: **95.8%**; validation macro F1: **0.9579**.
Public test sample (1,000 examples): accuracy **43.1%**, macro precision
**0.5831**, macro recall **0.5923**, macro F1 **0.4302**. The majority-class
accuracy baseline is **77.7%**. These results indicate poor test generalization;
the detector is experimental and should not be presented as reliable.

The training subset is predominantly Python (1,825 of 2,000 snippets), while
the public test sample includes eight languages. This distribution shift may
contribute to the validation/test gap; it is not established as the sole cause.
Future model selection and improvements should use validation data, followed
by evaluation on a fresh held-out test set. Do not tune thresholds on this test
sample to inflate the reported result.
