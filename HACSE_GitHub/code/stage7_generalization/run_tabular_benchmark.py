from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


LABELS = ["normal", "logical", "structural"]
VARIANTS = {"V": ("V__",), "V+O": ("V__", "O__"), "V+O+R": ("V__", "O__", "R__")}
N_ESTIMATORS = 600


def make_model(kind: str, seed: int):
    if kind == "rf":
        return RandomForestClassifier(n_estimators=N_ESTIMATORS, class_weight="balanced_subsample", random_state=seed, n_jobs=-1)
    if kind == "et":
        return ExtraTreesClassifier(n_estimators=N_ESTIMATORS, class_weight="balanced", random_state=seed, n_jobs=-1)
    if kind == "lr":
        return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000, random_state=seed))
    raise ValueError(kind)


def metrics(y, pred):
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=LABELS, average="macro", zero_division=0)),
        **{f"{c}_f1": float(f1_score(y, pred, labels=[c], average="macro", zero_division=0)) for c in LABELS},
    }


def feature_columns(df: pd.DataFrame, prefixes: tuple[str, ...]) -> list[str]:
    return [c for c in df.columns if c.startswith(prefixes)]


def select_model(xtr, ytr, xval, yval, seed):
    best = None
    for kind in ("rf", "et", "lr"):
        model = make_model(kind, seed)
        model.fit(xtr, ytr)
        score = balanced_accuracy_score(yval, model.predict(xval))
        candidate = (float(score), kind)
        if best is None or candidate > best:
            best = candidate
    return best[1], best[0]


def split_three_way(y: np.ndarray, seed: int):
    idx = np.arange(len(y))
    train, tmp = train_test_split(idx, test_size=0.4, stratify=y, random_state=seed)
    val, test = train_test_split(tmp, test_size=0.5, stratify=y[tmp], random_state=seed + 1000)
    return train, val, test


def main() -> None:
    ap = argparse.ArgumentParser(description="Run target-refit or source-to-target V/O/R experiments on validated common-schema features.")
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--protocol", choices=["target-refit", "source-to-target"], required=True)
    ap.add_argument("--source-dataset", default="mvtec_loco")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    ap.add_argument("--n-estimators", type=int, default=600, help="Tree count; reduce only for smoke tests.")
    args = ap.parse_args()

    global N_ESTIMATORS
    N_ESTIMATORS = args.n_estimators

    manifest = pd.read_csv(args.manifest, encoding="utf-8-sig")
    features = pd.read_csv(args.features, encoding="utf-8-sig")
    strict = manifest[manifest["include_strict"].astype(str).isin(["1", "true", "True"])]
    df = strict.merge(features, on="sample_id", how="inner", validate="one_to_one")
    if len(df) != len(strict):
        raise RuntimeError("Feature coverage is incomplete. Run validate_feature_bundle.py first.")
    rows, predictions = [], []

    for variant, prefixes in VARIANTS.items():
        cols = feature_columns(df, prefixes)
        if not cols:
            raise RuntimeError(f"No columns found for {variant}; expected prefixes {prefixes}")
        x = df[cols].to_numpy(dtype=np.float32)
        y = df["gt_group"].to_numpy(dtype=str)
        if args.protocol == "target-refit":
            targets = sorted(df["dataset"].unique())
            for target in targets:
                pos = np.where(df["dataset"].to_numpy() == target)[0]
                yt = y[pos]
                if any(np.sum(yt == c) < 5 for c in LABELS):
                    continue
                for seed in args.seeds:
                    tr, va, te = split_three_way(yt, seed)
                    kind, val_bacc = select_model(x[pos][tr], yt[tr], x[pos][va], yt[va], seed)
                    model = make_model(kind, seed)
                    model.fit(np.vstack([x[pos][tr], x[pos][va]]), np.concatenate([yt[tr], yt[va]]))
                    pred = model.predict(x[pos][te])
                    rows.append({"protocol": args.protocol, "source_dataset": target, "target_dataset": target, "variant": variant, "seed": seed, "selected_model": kind, "validation_bacc": val_bacc, "n_test": len(te), **metrics(yt[te], pred)})
                    for local_i, p in zip(te, pred):
                        gi = pos[local_i]
                        predictions.append({"sample_id": df.iloc[gi]["sample_id"], "dataset": target, "method": f"{variant}_target_refit_seed{seed}", "pred_group": p, "gt_group": y[gi]})
        else:
            src = np.where(df["dataset"].to_numpy() == args.source_dataset)[0]
            ys = y[src]
            if any(np.sum(ys == c) < 5 for c in LABELS):
                raise RuntimeError("Source dataset does not contain enough samples from all three classes.")
            targets = sorted(d for d in df["dataset"].unique() if d != args.source_dataset)
            for seed in args.seeds:
                tr, va = train_test_split(np.arange(len(src)), test_size=0.25, stratify=ys, random_state=seed)
                kind, val_bacc = select_model(x[src][tr], ys[tr], x[src][va], ys[va], seed)
                model = make_model(kind, seed)
                model.fit(x[src], ys)
                for target in targets:
                    te = np.where(df["dataset"].to_numpy() == target)[0]
                    pred = model.predict(x[te])
                    rows.append({"protocol": args.protocol, "source_dataset": args.source_dataset, "target_dataset": target, "variant": variant, "seed": seed, "selected_model": kind, "validation_bacc": val_bacc, "n_test": len(te), **metrics(y[te], pred)})
                    for gi, p in zip(te, pred):
                        predictions.append({"sample_id": df.iloc[gi]["sample_id"], "dataset": target, "method": f"{variant}_transfer_seed{seed}", "pred_group": p, "gt_group": y[gi]})

    args.out_dir.mkdir(parents=True, exist_ok=True)
    detail = pd.DataFrame(rows)
    detail.to_csv(args.out_dir / "results_by_seed.csv", index=False, encoding="utf-8-sig")
    if not detail.empty:
        metrics_cols = ["accuracy", "balanced_accuracy", "macro_f1", "normal_f1", "logical_f1", "structural_f1"]
        summary = detail.groupby(["protocol", "source_dataset", "target_dataset", "variant"], as_index=False)[metrics_cols].agg(["mean", "std"])
        summary.columns = ["_".join(str(x) for x in col if x).rstrip("_") for col in summary.columns]
        summary.to_csv(args.out_dir / "results_mean_std.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(predictions).to_csv(args.out_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    report = {"status": "complete", "protocol": args.protocol, "feature_schema": {k: feature_columns(df, v) for k, v in VARIANTS.items()}, "scientific_scope": "target-refit measures dataset portability; source-to-target measures direct transfer. Do not conflate the two."}
    (args.out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "complete", "runs": len(rows), "predictions": len(predictions)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
