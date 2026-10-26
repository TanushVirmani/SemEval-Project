"""Import a third-party trained detector for the academic comparison only."""

import argparse
import hashlib
import json
from pathlib import Path
import model_utils
from huggingface_hub import HfApi, snapshot_download

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="azherali/CodeGenDetect-CodeBert")
    parser.add_argument(
        "--output-dir", type=Path, default=model_utils.BASE_DIR / "checkpoints-external"
    )
    args = parser.parse_args()
    if (
        args.output_dir.exists()
        and (args.output_dir / "training_metadata.json").exists()
    ):
        parser.error("Imported detector already exists")
    info = HfApi().model_info(args.model_id)
    print("Downloading", args.model_id, info.sha, flush=True)
    snapshot_download(
        args.model_id,
        revision=info.sha,
        local_dir=args.output_dir,
        allow_patterns=["*.json", "*.txt", "*.safetensors", "README.md"],
        ignore_patterns=["optimizer*"],
    )
    weights = args.output_dir / "model.safetensors"
    if not weights.exists():
        raise ValueError("Only safetensors checkpoints are accepted.")
    config_path = args.output_dir / "config.json"
    config = json.loads(config_path.read_text())
    # Author model card explicitly states label 0=human, label 1=machine.
    config.update(
        id2label={str(k): v for k, v in model_utils.LABELS.items()},
        label2id={v: k for k, v in model_utils.LABELS.items()},
    )
    config_path.write_text(json.dumps(config, indent=2))
    metadata = {
        "backend": "codebert",
        "training_origin": "third_party",
        "base_model": args.model_id,
        "source_revision": info.sha,
        "source_model_card": f"https://huggingface.co/{args.model_id}",
        "training_overlap_unknown": True,
        "training_rows": None,
        "train_rows": None,
        "validation_rows": None,
        "max_length": 512,
        "tokenization": "head",
        "add_language_prefix": False,
        "training_code_hashes": [],
        "validation_code_hashes": [],
        "smoke_run": False,
        "competition_compliant": False,
        "track": "academic-third-party-comparison",
        "model_sha256": hashlib.file_digest(open(weights, "rb"), "sha256").hexdigest(),
        "provenance_note": "Author does not identify the exact training dataset; independent evaluation overlap cannot be verified.",
    }
    (args.output_dir / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2)
    )
    print(
        "Imported trained safetensors detector; training overlap remains unknown.",
        flush=True,
    )
