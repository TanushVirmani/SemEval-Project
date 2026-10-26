"""Shared preprocessing and strict loading of trained local classifiers."""

import json
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
os.environ.setdefault("HF_HOME", str(BASE_DIR / ".cache/huggingface"))
os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / ".cache/matplotlib"))
LABELS = {0: "Human-written", 1: "Machine-generated"}


def default_checkpoint():
    explicit = os.environ.get("MODEL_CHECKPOINT")
    if explicit:
        return Path(explicit)
    active = BASE_DIR / "active_model.json"
    if active.exists():
        return BASE_DIR / json.loads(active.read_text())["checkpoint"]
    return BASE_DIR / "checkpoints"


DEFAULT_CHECKPOINT = default_checkpoint()


def add_language_prefix(language, code):
    prefixes = {
        "python": "# language: python\n",
        "c++": "// language: cpp\n",
        "cpp": "// language: cpp\n",
        "java": "// language: java\n",
    }
    return prefixes.get(str(language).strip().lower(), "") + str(code).strip()


def device_for_torch():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def validate_checkpoint(path):
    path = Path(path).expanduser().resolve()
    if not (path / "config.json").is_file() or not any(
        (path / name).is_file()
        for name in [
            "model.safetensors",
            "pytorch_model.bin",
            "model.safetensors.index.json",
            "pytorch_model.bin.index.json",
        ]
    ):
        raise ValueError(f"No trained classifier at {path}. Run train_model.py first.")
    config = json.loads((path / "config.json").read_text())
    if config.get("id2label") != {str(k): v for k, v in LABELS.items()}:
        raise ValueError(
            "Checkpoint label mapping must be 0=Human-written, 1=Machine-generated."
        )
    if not (path / "training_metadata.json").is_file():
        raise ValueError(
            "Checkpoint lacks training_metadata.json; use train_model.py to save a verified training run."
        )
    metadata = json.loads((path / "training_metadata.json").read_text())
    required = {
        "max_length",
        "training_code_hashes",
        "validation_code_hashes",
        "smoke_run",
    }
    if metadata.get("training_origin") != "third_party":
        required.add("steps_completed")
    elif not (
        metadata.get("training_overlap_unknown")
        and metadata.get("source_model_card")
        and metadata.get("source_revision")
    ):
        raise ValueError(
            "External checkpoint must document its source and unknown training overlap."
        )
    if required - metadata.keys():
        raise ValueError(
            "Training metadata is incomplete; regenerate the checkpoint with train_model.py."
        )
    if (
        not isinstance(metadata["max_length"], int)
        or not 8 <= metadata["max_length"] <= 512
    ):
        raise ValueError("Invalid tokenizer maximum length in training metadata.")
    if (
        metadata.get("training_origin") != "third_party"
        and metadata["steps_completed"] < 1
    ):
        raise ValueError("Checkpoint has no completed training steps.")
    return path


def load_classifier(path=DEFAULT_CHECKPOINT):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    path = validate_checkpoint(path)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model, info = AutoModelForSequenceClassification.from_pretrained(
        path, local_files_only=True, output_loading_info=True
    )
    if info["missing_keys"] or info.get("mismatched_keys") or info["unexpected_keys"]:
        raise ValueError(f"Incomplete or incompatible checkpoint: {info}")
    device = device_for_torch()
    model.to(device).eval()
    return model, tokenizer, device


def read_labeled_csv(path):
    import pandas as pd

    df = pd.read_csv(path, dtype={"code": str, "language": str})
    missing = {"code", "label", "language"} - set(df.columns)
    if missing:
        raise ValueError(f"Missing CSV columns: {sorted(missing)}")
    if df.empty or df[["code", "label", "language"]].isna().any().any():
        raise ValueError(
            "Dataset must be nonempty with no missing code, label, or language."
        )
    if (
        df.code.astype(str).str.strip().eq("").any()
        or df.language.astype(str).str.strip().eq("").any()
    ):
        raise ValueError("Code and language must not be blank.")
    labels = pd.to_numeric(df.label, errors="raise")
    if not labels.isin([0, 1]).all() or set(labels) != {0, 1}:
        raise ValueError(
            "Dataset must contain both classes with labels exactly 0 or 1."
        )
    df["label"] = labels.astype(int)
    return df


def encode_texts(tokenizer, texts, max_length, strategy="head"):
    """Use identical first-only or first+last tokenization for fitting and inference."""
    if strategy == "head":
        return tokenizer(list(texts), truncation=True, max_length=max_length)
    available = max_length - tokenizer.num_special_tokens_to_add(pair=False)
    if available < 2:
        raise ValueError("Token limit is too small")
    result = {"input_ids": [], "attention_mask": []}
    for ids in tokenizer(list(texts), add_special_tokens=False, truncation=False)[
        "input_ids"
    ]:
        if len(ids) > available:
            first = (available + 1) // 2
            ids = ids[:first] + ids[-(available - first) :]
        prepared = tokenizer.prepare_for_model(
            ids, return_attention_mask=True, return_token_type_ids=False
        )
        result["input_ids"].append(prepared["input_ids"])
        result["attention_mask"].append(prepared["attention_mask"])
    return result
