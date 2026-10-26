"""Choose a decision threshold using only the saved validation split."""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from improve_model import EfficientClassifier, scores


def choose_threshold(y, probabilities):
    candidates = []
    for threshold in np.linspace(0.2, 0.8, 25):
        # scores() uses a 0.5 cutoff; shift only for evaluating the threshold.
        result = scores(y, np.asarray(probabilities) - threshold + 0.5)
        candidates.append({"threshold": float(threshold), **result})
    return max(
        candidates,
        key=lambda x: (
            x["macro_f1"],
            x["balanced_accuracy"],
            -abs(x["threshold"] - 0.5),
        ),
    ), candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    df = pd.read_csv(args.checkpoint / "validation.csv")
    meta_path = args.checkpoint / "training_metadata.json"
    metadata = json.loads(meta_path.read_text())
    hashes = set(df.code.map(lambda c: hashlib.sha256(c.strip().encode()).hexdigest()))
    if hashes != set(metadata["validation_code_hashes"]):
        raise ValueError(
            "Threshold tuning must use exactly the recorded validation split."
        )
    if hashes & set(metadata["training_code_hashes"]):
        raise ValueError("Validation overlaps training.")
    model = EfficientClassifier(args.checkpoint)
    probabilities = model.predict_machine_probability(df.code)
    best, candidates = choose_threshold(df.label, probabilities)
    metadata.update(
        decision_threshold=best["threshold"],
        threshold_selection="validation macro F1",
        threshold_validation_metrics=best,
    )
    meta_path.write_text(json.dumps(metadata, indent=2))
    (args.checkpoint / "threshold_selection.json").write_text(
        json.dumps({"selected": best, "candidates": candidates}, indent=2)
    )
    print(json.dumps(best), flush=True)


if __name__ == "__main__":
    main()
