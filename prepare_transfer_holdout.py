"""Reserve remaining CodeMirage test parent files, excluding all inspected holdout parents."""

import json
from pathlib import Path
import pandas as pd
from improve_model import normalized_fingerprint, fingerprint, equal_class_sample
from fragment_views import fragments
from model_utils import BASE_DIR

if __name__ == "__main__":
    root = BASE_DIR / "data/codemirage"
    df = pd.read_csv(Path((root / "test.path").read_text().strip())).dropna(
        subset=["code", "source", "language"]
    )
    df["label"] = df.source.ne("Human").astype(int)
    df["generator"] = df.source
    df["language"] = df.language.replace({"CPP": "C++", "CSharp": "C#"})
    df["_hash"] = df.code.map(fingerprint)
    df["_group"] = df.code.map(normalized_fingerprint)
    old = pd.read_csv(BASE_DIR / "checkpoints-broader/holdout.csv")
    old_groups = set(old.code.map(normalized_fingerprint))
    pool = df[~df._group.isin(old_groups)].drop_duplicates("_hash")
    previous_fragment_parents = equal_class_sample(pool, 75)
    forbidden = old_groups | set(previous_fragment_parents._group)
    public = pd.read_csv(BASE_DIR / "data/test.csv")
    forbidden.update(public.code.map(normalized_fingerprint))
    forbidden.update(
        pd.read_csv(BASE_DIR / "checkpoints-fragments/holdout.csv").code.map(
            normalized_fingerprint
        )
    )
    parents = equal_class_sample(
        df[~df._group.isin(forbidden)].drop_duplicates("_hash"), 25
    )
    fitting = set()
    for file in [
        "data/transfer/train.csv",
        "data/transfer/validation.csv",
        "data/train.csv",
        "data/validation.csv",
        "checkpoints-fragments/train.csv",
        "checkpoints-fragments/validation.csv",
    ]:
        fitting.update(pd.read_csv(BASE_DIR / file).code.map(normalized_fingerprint))
    seen = set()
    rows = []
    for row in parents.to_dict("records"):
        for index, code in enumerate(
            [row["code"]] + fragments(row["code"], row["language"])
        ):
            group = normalized_fingerprint(code)
            if group in fitting or group in forbidden or group in seen:
                continue
            seen.add(group)
            rows.append(
                {
                    "code": code,
                    "label": row["label"],
                    "language": row["language"],
                    "generator": row["generator"],
                    "parent_hash": row["_hash"],
                    "view": "full" if index == 0 else "fragment",
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(BASE_DIR / "data/transfer/fresh_holdout.csv", index=False)
    parents[["code", "label", "language", "generator", "_hash"]].to_csv(
        BASE_DIR / "data/transfer/fresh_parents.csv", index=False
    )
    info = {
        "candidate_parents": len(parents),
        "represented_parents": result.parent_hash.nunique(),
        "views": len(result),
        "labels": result.label.value_counts().to_dict(),
        "policy": "Previously uninspected parent files; normalized code groups excluded from fitting, tuning and prior public/holdout views.",
    }
    (BASE_DIR / "data/transfer/fresh_holdout_manifest.json").write_text(
        json.dumps(info, indent=2)
    )
    print(json.dumps(info), flush=True)
