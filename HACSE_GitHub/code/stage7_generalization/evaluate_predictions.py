from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, precision_recall_fscore_support


CLASS_ORDER = ["normal", "logical", "structural"]


def metric_dict(y: np.ndarray, pred: np.ndarray) -> dict[str, float | int | bool | None]:
    present = [c for c in CLASS_ORDER if np.any(y == c)]
    complete = len(present) == len(CLASS_ORDER)
    _, _, f1, support = precision_recall_fscore_support(y, pred, labels=present, average=None, zero_division=0)
    return {
        "n": int(len(y)),
        "complete_three_class": complete,
        "n_observed_classes": len(present),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy_observed": float(balanced_accuracy_score(y, pred)),
        "macro_f1_observed": float(np.mean(f1)),
        "macro_f1_three_class": float(np.mean(precision_recall_fscore_support(y, pred, labels=CLASS_ORDER, average=None, zero_division=0)[2])) if complete else None,
        "min_class_support": int(np.min(support)),
    }


def stratified_bootstrap(y: np.ndarray, pred: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    per_class = {c: np.where(y == c)[0] for c in CLASS_ORDER if np.any(y == c)}
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(v, size=len(v), replace=True) for v in per_class.values()])
        vals.append(metric_dict(y[idx], pred[idx])["macro_f1_observed"])
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate long-form predictions with class-coverage-aware three-class metrics.")
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--predictions", type=Path, required=True, help="CSV columns: sample_id, method, pred_group")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--reference-method", default="ODRC")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--allow-partial", action="store_true")
    args = ap.parse_args()

    manifest = pd.read_csv(args.manifest, encoding="utf-8-sig", dtype=str)
    pred = pd.read_csv(args.predictions, encoding="utf-8-sig", dtype=str)
    required_m = {"sample_id", "dataset", "gt_group", "include_strict"}
    required_p = {"sample_id", "method", "pred_group"}
    if missing := required_m - set(manifest.columns):
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")
    if missing := required_p - set(pred.columns):
        raise ValueError(f"Predictions missing columns: {sorted(missing)}")
    if pred.duplicated(["sample_id", "method"]).any():
        raise ValueError("Duplicate (sample_id, method) predictions found.")
    bad = sorted((set(pred["pred_group"].dropna()) | set(manifest["gt_group"].dropna())) - set(CLASS_ORDER) - {"mixed", "unknown"})
    if bad:
        raise ValueError(f"Unknown labels: {bad}")

    strict = manifest[manifest["include_strict"].astype(str).isin(["1", "true", "True"])].copy()
    merged = strict.merge(pred, on="sample_id", how="left", validate="one_to_many")
    coverage_rows, result_rows, class_rows, confusion_rows = [], [], [], []
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for dataset, ds in strict.groupby("dataset", sort=True):
        expected = len(ds)
        for method in sorted(pred["method"].unique()):
            sub = merged[(merged["dataset"] == dataset) & (merged["method"] == method)].dropna(subset=["pred_group"])
            coverage = len(sub) / expected if expected else 0.0
            coverage_rows.append({"dataset": dataset, "method": method, "expected": expected, "predicted": len(sub), "coverage": coverage})
            if coverage < 1.0 and not args.allow_partial:
                continue
            if sub.empty:
                continue
            y = sub["gt_group"].to_numpy(dtype=str)
            yp = sub["pred_group"].to_numpy(dtype=str)
            metrics = metric_dict(y, yp)
            lo, hi = stratified_bootstrap(y, yp, args.bootstrap, seed=20260913)
            result_rows.append({"dataset": dataset, "method": method, "coverage": coverage, **metrics, "macro_f1_ci_low": lo, "macro_f1_ci_high": hi})
            precision, recall, f1, support = precision_recall_fscore_support(y, yp, labels=CLASS_ORDER, average=None, zero_division=0)
            for i, cls in enumerate(CLASS_ORDER):
                class_rows.append({"dataset": dataset, "method": method, "class": cls, "precision": precision[i], "recall": recall[i], "f1": f1[i], "support": int(support[i])})
            cm = confusion_matrix(y, yp, labels=CLASS_ORDER)
            for i, gt in enumerate(CLASS_ORDER):
                for j, pr in enumerate(CLASS_ORDER):
                    confusion_rows.append({"dataset": dataset, "method": method, "gt_group": gt, "pred_group": pr, "count": int(cm[i, j])})

    pairwise_rows = []
    for dataset, ds in merged.dropna(subset=["method", "pred_group"]).groupby("dataset"):
        wide = ds.pivot(index="sample_id", columns="method", values="pred_group")
        truth = ds.drop_duplicates("sample_id").set_index("sample_id")["gt_group"]
        if args.reference_method not in wide.columns:
            continue
        for method in sorted(c for c in wide.columns if c != args.reference_method):
            common = wide[[args.reference_method, method]].dropna().index
            if len(common) == 0:
                continue
            y = truth.loc[common].to_numpy(dtype=str)
            a = wide.loc[common, args.reference_method].to_numpy(dtype=str)
            b = wide.loc[common, method].to_numpy(dtype=str)
            pairwise_rows.append({
                "dataset": dataset,
                "reference": args.reference_method,
                "method": method,
                "n_common": len(common),
                "delta_accuracy_method_minus_reference": float(accuracy_score(y, b) - accuracy_score(y, a)),
                "delta_macro_f1_observed_method_minus_reference": float(metric_dict(y, b)["macro_f1_observed"] - metric_dict(y, a)["macro_f1_observed"]),
                "reference_only_correct": int(np.sum((a == y) & (b != y))),
                "method_only_correct": int(np.sum((b == y) & (a != y))),
            })

    outputs = {
        "coverage.csv": coverage_rows,
        "summary.csv": result_rows,
        "class_metrics.csv": class_rows,
        "confusion_matrices.csv": confusion_rows,
        "pairwise_vs_reference.csv": pairwise_rows,
    }
    for name, rows in outputs.items():
        pd.DataFrame(rows).to_csv(args.out_dir / name, index=False, encoding="utf-8-sig")
    report = {
        "status": "complete",
        "class_order": CLASS_ORDER,
        "strict_rule": "Rows with include_strict=1 only.",
        "metric_rule": "Three-class Macro-F1 is emitted only when all three GT classes are present; otherwise observed-class metrics are used.",
        "coverage_gate": "Incomplete method/dataset pairs are excluded unless --allow-partial is supplied.",
        "datasets_evaluated": sorted({r["dataset"] for r in result_rows}),
        "methods_evaluated": sorted({r["method"] for r in result_rows}),
    }
    (args.out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
