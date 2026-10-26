"""Download official Subtask A data; keep train/validation/test separate."""

import argparse
import hashlib
import json
from pathlib import Path

from model_utils import BASE_DIR

DATASET = "DaniilOr/SemEval-2026-Task13"
DATASET_REVISION = "df2aec18238a47fabfb22c79570db15fd1588aa3"


def code_hash(code):
    return hashlib.sha256(str(code).strip().encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-size", type=int, default=2000, help="0 uses the full training split"
    )
    parser.add_argument("--validation-size", type=int, default=500)
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR / "data")
    args = parser.parse_args()
    if args.train_size < 0 or args.validation_size < 0:
        parser.error("Sizes cannot be negative")
    from datasets import load_dataset
    from sklearn.model_selection import train_test_split

    from model_utils import read_labeled_csv

    args.output_dir.mkdir(parents=True, exist_ok=True)
    seen = set()
    manifest = {
        "dataset": DATASET,
        "revision": DATASET_REVISION,
        "config": "A",
        "seed": 42,
        "splits": {},
    }
    # Reserve test examples first so no identical snippet can enter training.
    for split, limit in [
        ("test", 0),
        ("validation", args.validation_size),
        ("train", args.train_size),
    ]:
        df = load_dataset(
            DATASET, "A", split=split, revision=DATASET_REVISION
        ).to_pandas()
        df = df[["code", "label", "language"]].dropna().copy()
        df = df[df.code.astype(str).str.strip().ne("")]
        df["_hash"] = df.code.map(code_hash)
        before = len(df)
        df = df.drop_duplicates("_hash")
        df = df[~df["_hash"].isin(seen)]
        seen.update(df["_hash"])
        if limit and limit < len(df):
            df, _ = train_test_split(
                df, train_size=limit, stratify=df.label, random_state=42
            )
        df = df.drop(columns="_hash").reset_index(drop=True)
        path = args.output_dir / f"{split}.csv"
        df.to_csv(path, index=False)
        read_labeled_csv(path)
        manifest["splits"][split] = {
            "rows": len(df),
            "source_rows": before,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "labels": df.label.value_counts().to_dict(),
        }
        print(f"{split}: {len(df)} rows -> {path}", flush=True)
    manifest["test_note"] = (
        "Public official test sample only, not the full competition test set."
    )
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
