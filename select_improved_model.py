"""Lock the improved model using shared validation data, before final holdout evaluation."""

import gc
import hashlib
import json
import pandas as pd
from calibrate_detector import choose_threshold
from detector import Detector
from improve_model import scores
from model_utils import BASE_DIR

if __name__ == "__main__":
    validation = BASE_DIR / "data/transfer/validation.csv"
    df = pd.read_csv(validation)
    candidates = {}
    names = ["checkpoints-fragments", "checkpoints-transfer"]
    if (BASE_DIR / "checkpoints-external/training_metadata.json").exists():
        names.append("checkpoints-external")
    for name in names:
        checkpoint = BASE_DIR / name
        model = Detector(checkpoint)
        if name == "checkpoints-transfer":
            model.metadata["decision_threshold"] = 0.5
        p = []
        for start in range(0, len(df), 32):
            batch = df.iloc[start : start + 32]
            p.extend(model.probabilities(batch.code, batch.language)[:, 1].tolist())
        if name == "checkpoints-transfer":
            # These are raw softmax values: this model has no threshold shift yet.
            best, grid = choose_threshold(df.label, p)
            meta_path = checkpoint / "training_metadata.json"
            meta = json.loads(meta_path.read_text())
            meta.update(
                decision_threshold=best["threshold"],
                threshold_selection="shared validation macro F1",
                threshold_validation_metrics=best,
            )
            meta_path.write_text(json.dumps(meta, indent=2))
            candidates[name] = best
            (checkpoint / "threshold_selection.json").write_text(
                json.dumps({"selected": best, "candidates": grid}, indent=2)
            )
        else:
            candidates[name] = scores(df.label, p)
            candidates[name]["threshold"] = model.metadata.get(
                "decision_threshold", 0.5
            )
        print(name, json.dumps(candidates[name]), flush=True)
        del model
        gc.collect()
    selected = max(
        candidates,
        key=lambda name: (
            candidates[name]["macro_f1"],
            candidates[name]["balanced_accuracy"],
        ),
    )
    record = {
        "checkpoint": selected,
        "selection": "shared validation macro F1",
        "validation_sha256": hashlib.sha256(validation.read_bytes()).hexdigest(),
        "validation_rows": len(df),
        "candidates": candidates,
        "fresh_holdout_used_for_selection": False,
        "public_test_used_for_this_selection": False,
        "track": "academic-demo-with-external-data",
    }
    (BASE_DIR / "model_selection.json").write_text(json.dumps(record, indent=2))
    (BASE_DIR / "active_model.json").write_text(
        json.dumps(
            {
                "checkpoint": selected,
                "backend": "efficient" if "fragments" in selected else "codebert",
                "track": "academic-demo-with-external-data",
            },
            indent=2,
        )
    )
    print("ACTIVE", selected, flush=True)
