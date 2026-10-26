"""One inference interface for local CodeBERT and CPU-efficient checkpoints."""

import json
from pathlib import Path
import numpy as np
from model_utils import (
    DEFAULT_CHECKPOINT,
    add_language_prefix,
    encode_texts,
    load_classifier,
    validate_checkpoint,
)


def metadata_for(path):
    path = Path(path).expanduser().resolve()
    file = path / "training_metadata.json"
    if not file.is_file():
        raise ValueError(
            f"No trained detector at {path}. See README.md for training instructions."
        )
    return json.loads(file.read_text())


def validate_detector(path):
    path = Path(path).expanduser().resolve()
    metadata = metadata_for(path)
    if metadata.get("backend") == "efficient":
        if not (path / "model.joblib").is_file():
            raise ValueError("Efficient classifier weights are missing.")
        if metadata.get("train_rows", 0) < 1 or not metadata.get(
            "training_code_hashes"
        ):
            raise ValueError("Efficient classifier has incomplete training metadata.")
        return path
    return validate_checkpoint(path)


def decision_scores(machine_probability, threshold=0.5):
    """Shift odds so the validation-selected decision boundary maps to 0.5.

    Returned values are decision scores, not calibrated authorship likelihoods.
    """
    p = np.asarray(machine_probability, dtype=float)
    if not 0 < threshold < 1:
        raise ValueError("Decision threshold must lie strictly between 0 and 1.")
    numerator = p * (1 - threshold)
    adjusted = numerator / (numerator + (1 - p) * threshold)
    return np.column_stack([1 - adjusted, adjusted])


class Detector:
    def __init__(self, path=DEFAULT_CHECKPOINT):
        self.path = validate_detector(path)
        self.metadata = metadata_for(self.path)
        self.backend = self.metadata.get("backend", "codebert")
        if self.backend == "efficient":
            from improve_model import EfficientClassifier

            self.model = EfficientClassifier(self.path)
        else:
            self.model, self.tokenizer, self.device = load_classifier(self.path)

    def probabilities(self, codes, languages):
        codes = list(codes)
        languages = list(languages)
        if len(codes) != len(languages):
            raise ValueError("Every code snippet needs a language value.")
        if not codes:
            return np.empty((0, 2))
        if self.backend == "efficient":
            p = self.model.predict_machine_probability(codes)
            return decision_scores(p, self.metadata.get("decision_threshold", 0.5))
        import torch

        encoded = encode_texts(
            self.tokenizer,
            [
                add_language_prefix(lang, code)
                if self.metadata.get("add_language_prefix", True)
                else str(code).strip()
                for lang, code in zip(languages, codes)
            ],
            self.metadata["max_length"],
            self.metadata.get("tokenization", "head"),
        )
        inputs = self.tokenizer.pad(encoded, return_tensors="pt")
        with torch.inference_mode():
            logits = self.model(
                **{k: v.to(self.device) for k, v in inputs.items()}
            ).logits
            return torch.softmax(logits, dim=-1).cpu().numpy()
