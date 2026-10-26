"""Train CPU-efficient authorship models, selecting only on validation data."""

import argparse
import hashlib
import json
import re
from pathlib import Path

import model_utils
import joblib
import numpy as np
import pandas as pd
from datasets import Dataset
from scipy.sparse import hstack, csr_matrix
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    balanced_accuracy_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

BASE = model_utils.BASE_DIR
SEED = 20261026


def fingerprint(code):
    return hashlib.sha256(str(code).strip().encode()).hexdigest()


def normalize(code):
    code = re.sub(
        r'"""[\s\S]*?"""|\x27\x27\x27[\s\S]*?\x27\x27\x27|/\*[\s\S]*?\*/|//[^\n]*|\#[^\n]*',
        " ",
        code,
    )
    code = re.sub(r'"(?:\\.|[^"\\])*"|\x27(?:\\.|[^\x27\\])*\x27', " STR ", code)
    code = re.sub(r"\b\d+(?:\.\d+)?\b", " NUM ", code)
    return code


def structural_features(codes):
    rows = []
    for code in codes:
        lines = code.splitlines() or [""]
        nonempty = [x for x in lines if x.strip()] or [""]
        n = max(len(code), 1)
        nlines = len(lines)
        tokens = re.findall(r"[A-Za-z_$][\w$]*", code)
        lengths = np.array([len(x) for x in nonempty])
        names = np.array([len(x) for x in tokens] or [0])
        counts = [
            code.count(x) / n
            for x in [
                "\n",
                "\t",
                " ",
                ";",
                ":",
                "{",
                "}",
                "(",
                ")",
                "[",
                "]",
                "=",
                ",",
                ".",
                "_",
                '"',
                "'",
                "#",
                "//",
                "/*",
                "->",
                "==",
                "!=",
                "<=",
                ">=",
                "&&",
                "||",
                "++",
                "+=",
            ]
        ]
        words = [
            "if",
            "else",
            "elif",
            "for",
            "while",
            "return",
            "class",
            "def",
            "public",
            "private",
            "static",
            "void",
            "int",
            "float",
            "double",
            "string",
            "String",
            "bool",
            "true",
            "false",
            "None",
            "null",
            "import",
            "include",
            "try",
            "except",
            "catch",
            "throw",
            "print",
            "cout",
            "System",
            "self",
            "this",
            "const",
            "let",
            "var",
            "function",
            "async",
            "await",
        ]
        freq = {w: tokens.count(w) / max(len(tokens), 1) for w in words}
        row = [
            np.log1p(n),
            np.log1p(nlines),
            len(nonempty) / nlines,
            lengths.mean(),
            lengths.std(),
            np.median(lengths),
            np.quantile(lengths, 0.9),
            lengths.max(),
            names.mean(),
            names.std(),
            np.quantile(names, 0.9),
            len(set(tokens)) / max(len(tokens), 1),
            sum(x.lstrip().startswith(("#", "//", "/*", "*")) for x in nonempty)
            / len(nonempty),
            sum(bool(re.match(r"\s+", x)) for x in nonempty) / len(nonempty),
            sum(x.isupper() for x in tokens) / max(len(tokens), 1),
            sum("_" in x for x in tokens) / max(len(tokens), 1),
            sum(bool(re.search(r"[a-z][A-Z]", x)) for x in tokens)
            / max(len(tokens), 1),
        ]
        rows.append(row + counts + list(freq.values()))
    return np.asarray(rows, dtype=np.float32)


def scores(y, p):
    pred = (np.asarray(p) >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "report": classification_report(y, pred, output_dict=True, zero_division=0),
    }


def balanced_sample(df, per_group):
    return (
        pd.concat(
            [
                g.sample(min(per_group, len(g)), random_state=SEED)
                for _, g in df.groupby(["language", "label"])
            ]
        )
        .sample(frac=1, random_state=SEED)
        .reset_index(drop=True)
    )


def load_arrow(split):
    paths = list(
        (BASE / ".cache/huggingface/datasets").rglob(
            f"sem_eval-2026-task13-{split}.arrow"
        )
    )
    if len(paths) != 1:
        raise ValueError(
            f"Expected one cached official {split} split; run prepare_data.py first."
        )
    df = (
        Dataset.from_file(str(paths[0]))
        .to_pandas()
        .dropna(subset=["code", "label", "language"])
    )
    df = df[df.code.str.strip().ne("")].copy()
    df["_hash"] = df.code.map(fingerprint)
    # Exclude conflicting labels and duplicate snippets.
    conflicts = df.groupby("_hash").label.nunique()
    return df[~df._hash.isin(conflicts[conflicts > 1].index)].drop_duplicates("_hash")


def normalized_fingerprint(code):
    """Conservative duplicate grouping across identifiers, literals and whitespace."""
    text = normalize(str(code))
    keywords = set(
        "if else elif for while return class def public private protected static void int float double string String bool true false None null import include try except catch throw const let var function async await new switch case break continue package from with as in is not and or lambda yield".split()
    )
    tokens = re.findall(r"[A-Za-z_$][\w$]*|[^\w\s]", text)
    tokens = [
        t if t in keywords or not re.match(r"[A-Za-z_$]", t) else "ID" for t in tokens
    ]
    return hashlib.sha256(" ".join(tokens).encode()).hexdigest()


def equal_class_sample(df, limit):
    groups = []
    for _, g in df.groupby("language"):
        n = min(limit, *(len(g[g.label == label]) for label in [0, 1]))
        if n:
            groups.extend(
                g[g.label == label].sample(n, random_state=SEED) for label in [0, 1]
            )
    return pd.concat(groups).sample(frac=1, random_state=SEED).reset_index(drop=True)


def prepare_broader_data(public, per_group):
    root = BASE / "data/codemirage"
    provenance = json.loads((root / "provenance.json").read_text())
    prohibited = set(public.code.map(normalized_fingerprint))
    frames = {}
    for split in ["train", "test"]:
        path = Path((root / f"{split}.path").read_text().strip())
        df = pd.read_csv(path).dropna(subset=["code", "source", "language"])
        df["label"] = df.source.ne("Human").astype(int)
        df["generator"] = df.source
        df["language"] = df.language.replace({"CPP": "C++", "CSharp": "C#"})
        df["_hash"] = df.code.map(fingerprint)
        df["_group"] = df.code.map(normalized_fingerprint)
        conflicts = df.groupby("_hash").label.nunique()
        df = df[~df._hash.isin(conflicts[conflicts > 1].index)]
        df = df[~df._group.isin(prohibited)].drop_duplicates("_hash")
        frames[split] = df
        print(
            f"CodeMirage {split}: {len(df)} usable after duplicate/conflict exclusion",
            flush=True,
        )
    # Entire provided test split is excluded from fitting and tuning by normalized code group.
    test_groups = set(frames["test"]._group)
    pool = frames["train"][~frames["train"]._group.isin(test_groups)].copy()
    tune_mask = pool._group.map(lambda x: int(x[:8], 16) % 5 == 0)
    broad_train = equal_class_sample(pool[~tune_mask], per_group)
    broad_tune = equal_class_sample(pool[tune_mask], 200)
    holdout = equal_class_sample(frames["test"], 200)
    official_train = load_arrow("train")
    official_val = load_arrow("validation")
    # Do not use any previously inspected official validation rows for selection again.
    old_hashes = set(pd.read_csv(BASE / "data/validation.csv").code.map(fingerprint))
    prior_dir = BASE / "checkpoints-efficient"
    for name in ["validation.csv", "holdout.csv"]:
        if (prior_dir / name).exists():
            old_hashes.update(pd.read_csv(prior_dir / name).code.map(fingerprint))
    official_val = official_val[~official_val._hash.isin(old_hashes)]
    official_tune = equal_class_sample(official_val, 100)
    reserved = (
        set(holdout._group)
        | set(broad_tune._group)
        | set(official_tune.code.map(normalized_fingerprint))
        | prohibited
    )
    official_train = equal_class_sample(official_train, per_group)
    official_train["_group"] = official_train.code.map(normalized_fingerprint)
    official_train = official_train[~official_train._group.isin(reserved)]
    train = pd.concat([broad_train, official_train], ignore_index=True)
    train = train[~train._group.isin(reserved)].drop_duplicates("_hash")
    tune = pd.concat([broad_tune, official_tune], ignore_index=True)
    train["_dataset"] = (
        np.where(train.source.notna(), "CodeMirage", "SemEval")
        if "source" in train
        else "mixed"
    )
    assert not set(train._group) & set(tune._group)
    assert not set(train._group) & set(holdout._group)
    assert not set(tune._group) & set(holdout._group)
    return train, tune, holdout, provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-group", type=int, default=8000)
    parser.add_argument(
        "--output-dir", type=Path, default=BASE / "checkpoints-efficient"
    )
    parser.add_argument(
        "--fragment-views",
        action="store_true",
        help="Augment training and validation with function fragments",
    )
    parser.add_argument("--track", choices=["official", "broader"], default="official")
    args = parser.parse_args()
    if (args.output_dir / "model.joblib").exists():
        parser.error("Model already exists; choose a new output directory.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    provenance = None
    if args.track == "broader":
        public = pd.read_csv(BASE / "data/test.csv")
        if args.fragment_views:
            parent = BASE / "checkpoints-broader"
            train = pd.read_csv(parent / "train.csv")
            tune = pd.read_csv(parent / "validation.csv")
            holdout = pd.read_csv(parent / "holdout.csv")
            for frame in [train, tune, holdout]:
                frame["_hash"] = frame.code.map(fingerprint)
                frame["_group"] = frame.code.map(normalized_fingerprint)
            provenance = json.loads(
                (BASE / "data/codemirage/provenance.json").read_text()
            )
        else:
            train, tune, holdout, provenance = prepare_broader_data(
                public, args.per_group
            )
    else:
        train = load_arrow("train")
        val = load_arrow("validation")
        public = pd.read_csv(BASE / "data/test.csv")
        excluded = set(public.code.map(fingerprint))
        # Do not reuse validation snippets inspected in the earlier CodeBERT experiment.
        excluded.update(pd.read_csv(BASE / "data/validation.csv").code.map(fingerprint))
        val = val[~val._hash.isin(excluded)]
        val = balanced_sample(val, 1000)
        tune, holdout = train_test_split(
            val,
            test_size=0.5,
            stratify=val.language + "_" + val.label.astype(str),
            random_state=SEED,
        )
        excluded.update(val._hash)
        train = balanced_sample(train[~train._hash.isin(excluded)], args.per_group)
        # Reserve all official validation snippets from fitting, even if unselected.
    if args.fragment_views:
        from fragment_views import fragments

        # Preserve the previous untouched holdout as an audit artifact; it is now inspected.
        used_holdout = pd.read_csv(BASE / "checkpoints-broader/holdout.csv")
        old_groups = set(used_holdout.code.map(normalized_fingerprint))
        raw_test = pd.read_csv(
            Path((BASE / "data/codemirage/test.path").read_text().strip())
        ).dropna(subset=["code", "source", "language"])
        raw_test["label"] = raw_test.source.ne("Human").astype(int)
        raw_test["generator"] = raw_test.source
        raw_test["language"] = raw_test.language.replace({"CPP": "C++", "CSharp": "C#"})
        raw_test["_group"] = raw_test.code.map(normalized_fingerprint)
        raw_test["_hash"] = raw_test.code.map(fingerprint)
        raw_test = raw_test[~raw_test._group.isin(old_groups)].drop_duplicates("_hash")
        holdout = equal_class_sample(raw_test, 75)

        def expand(df):
            rows = []
            for row in df.to_dict("records"):
                for code in [row["code"]] + fragments(row["code"], row["language"]):
                    new = dict(row)
                    new["code"] = code
                    new["_hash"] = fingerprint(code)
                    new["_group"] = normalized_fingerprint(code)
                    rows.append(new)
            return pd.DataFrame(rows).drop_duplicates("_hash")

        holdout = expand(holdout)
        holdout = holdout[
            ~holdout._group.isin(set(public.code.map(normalized_fingerprint)))
        ]
        tune = expand(tune)
        reserved = (
            set(holdout._group)
            | set(tune._group)
            | set(public.code.map(normalized_fingerprint))
        )
        train = expand(train)
        train = train[~train._group.isin(reserved)]
        tune = tune[~tune._group.isin(set(holdout._group))]
        # Equalize classes within each language after view expansion.
        train = equal_class_sample(train, args.per_group * 4)
        tune = equal_class_sample(tune, 800)
        holdout = equal_class_sample(holdout, 250)
        assert not set(train._group) & set(holdout._group)
        assert not set(train._group) & set(tune._group)
        assert not set(tune._group) & set(holdout._group)
    print(
        f"Train {len(train)}, tune {len(tune)}, fresh balanced holdout {len(holdout)}",
        flush=True,
    )
    for name, df in [("train", train), ("validation", tune), ("holdout", holdout)]:
        df[["code", "label", "language", "generator"]].to_csv(
            args.output_dir / f"{name}.csv", index=False
        )
    y = train.label.to_numpy()
    yt = tune.label.to_numpy()
    print("Extracting structural features", flush=True)
    x = structural_features(train.code)
    xt = structural_features(tune.code)
    candidates = {}
    fitted = {}
    for name, est in [
        (
            "structure_boost",
            HistGradientBoostingClassifier(
                max_iter=200,
                max_leaf_nodes=15,
                l2_regularization=10,
                learning_rate=0.08,
                random_state=SEED,
            ),
        ),
        (
            "structure_forest",
            ExtraTreesClassifier(
                n_estimators=200,
                max_features=0.7,
                min_samples_leaf=5,
                n_jobs=2,
                random_state=SEED,
            ),
        ),
    ]:
        est.fit(x, y)
        p = est.predict_proba(xt)[:, 1]
        candidates[name] = p
        fitted[name] = est
        print(name, json.dumps(scores(yt, p)), flush=True)
    print("Extracting lexical and character features", flush=True)
    word = TfidfVectorizer(
        token_pattern=r"[A-Za-z_$][A-Za-z_0-9$]*|[^\w\s]",
        ngram_range=(1, 2),
        max_features=80000,
        min_df=3,
        sublinear_tf=True,
        dtype=np.float32,
        lowercase=False,
    )
    char = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        max_features=100000,
        min_df=4,
        sublinear_tf=True,
        dtype=np.float32,
    )
    texts = train.code.map(normalize).tolist()
    ttexts = tune.code.map(normalize).tolist()
    wx = word.fit_transform(texts)
    cx = char.fit_transform(texts)
    wt = word.transform(ttexts)
    ct = char.transform(ttexts)
    scaler = StandardScaler().fit(x)
    features = hstack([wx, cx, csr_matrix(scaler.transform(x) * 0.03)], format="csr")
    tfeatures = hstack([wt, ct, csr_matrix(scaler.transform(xt) * 0.03)], format="csr")
    for c in [0.1, 1.0]:
        name = f"lexical_C{c}"
        est = LogisticRegression(
            C=c, solver="liblinear", max_iter=300, random_state=SEED
        )
        est.fit(features, y)
        p = est.predict_proba(tfeatures)[:, 1]
        candidates[name] = p
        fitted[name] = est
        print(name, json.dumps(scores(yt, p)), flush=True)
    # Fixed ensemble family chosen only by validation macro F1.
    for name in ["lexical_C0.1", "lexical_C1.0"]:
        candidates[name + "_ensemble"] = (
            candidates[name]
            + candidates["structure_boost"]
            + candidates["structure_forest"]
        ) / 3
    selected = max(
        candidates,
        key=lambda k: (
            scores(yt, candidates[k])["macro_f1"],
            scores(yt, candidates[k])["accuracy"],
        ),
    )
    print("SELECTED", selected, flush=True)
    bundle = {
        "selected": selected,
        "estimators": fitted,
        "word": word,
        "char": char,
        "scaler": scaler,
        "version": 1,
    }
    joblib.dump(bundle, args.output_dir / "model.joblib", compress=3)
    metadata = {
        "backend": "efficient",
        "track": args.track,
        "fragment_views": args.fragment_views,
        "external_data_provenance": provenance,
        "competition_compliant": args.track == "official",
        "seed": SEED,
        "train_rows": len(train),
        "validation_rows": len(tune),
        "holdout_rows": len(holdout),
        "selected": selected,
        "validation_candidates": {k: scores(yt, p) for k, p in candidates.items()},
        "training_code_hashes": train._hash.tolist(),
        "validation_code_hashes": tune._hash.tolist(),
        "dataset_revision": "df2aec18238a47fabfb22c79570db15fd1588aa3",
        "selection_metric": "validation macro F1",
        "public_test_used_for_selection": False,
    }
    (args.output_dir / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2)
    )
    # Evaluate only after selection is locked and the model has been saved.
    model = EfficientClassifier(args.output_dir)
    result = {
        "selected": selected,
        "validation": scores(yt, candidates[selected]),
        "fresh_balanced_holdout": scores(
            holdout.label, model.predict_machine_probability(holdout.code)
        ),
        "public_test_recheck": scores(
            public.label, model.predict_machine_probability(public.code)
        ),
        "note": (
            "Academic/demo track with CodeMirage training data; fresh holdout is class-balanced CodeMirage test data. Public SemEval sample was previously inspected. Not competition compliant."
            if args.track == "broader"
            else "Public test sample was inspected in a previous experiment; fresh holdout is in-domain only."
        ),
    }
    (args.output_dir / "experiment_results.json").write_text(
        json.dumps(result, indent=2)
    )
    print("FINAL", json.dumps(result), flush=True)


class EfficientClassifier:
    def __init__(self, path):
        self.path = Path(path)
        self.bundle = joblib.load(self.path / "model.joblib")
        self.metadata = json.loads((self.path / "training_metadata.json").read_text())

    def predict_machine_probability(self, codes):
        codes = list(codes)
        b = self.bundle
        name = b["selected"]
        x = structural_features(codes)
        if name.startswith("structure_"):
            return b["estimators"][name].predict_proba(x)[:, 1]
        text = [normalize(c) for c in codes]
        features = hstack(
            [
                b["word"].transform(text),
                b["char"].transform(text),
                csr_matrix(b["scaler"].transform(x) * 0.03),
            ],
            format="csr",
        )
        base = name.removesuffix("_ensemble")
        p = b["estimators"][base].predict_proba(features)[:, 1]
        if name.endswith("_ensemble"):
            p = (
                p
                + b["estimators"]["structure_boost"].predict_proba(x)[:, 1]
                + b["estimators"]["structure_forest"].predict_proba(x)[:, 1]
            ) / 3
        return p


if __name__ == "__main__":
    main()
