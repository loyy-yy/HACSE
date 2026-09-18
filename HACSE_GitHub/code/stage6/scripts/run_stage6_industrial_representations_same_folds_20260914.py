#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SALAD/CSAD evidence baselines on the canonical Stage6 outer folds."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = HERE / "run_stage6_strong_baselines_common_protocol.py"
ASSIGN_FILE = ROOT / "stage6" / "outputs" / "stage6_nested5fold_ablation" / "outer_fold_assignments.csv"
OUT_DIR = ROOT / "stage6" / "outputs" / "stage6_industrial_representations_20260914"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


src = load_module(SOURCE, "stage6_common_baselines_source_20260914")
CLASS_ORDER = list(src.CLASS_ORDER)
METHODS = ["SALAD-only", "CSAD-only", "SALAD+CSAD"]


def metric_values(y_true, y_pred):
    out = {
        "acc": float(accuracy_score(y_true, y_pred)),
        "bacc": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=CLASS_ORDER, average="macro", zero_division=0)),
    }
    for cls in CLASS_ORDER:
        out[f"{cls}_f1"] = float(f1_score(y_true, y_pred, labels=[cls], average="macro", zero_division=0))
    return out


def main():
    df, _, _, _, _, y, _, sample_keys, _ = src.load_benchmark()
    salad, csad, both = src.load_aligned_22d(sample_keys)
    feature_map = {
        "SALAD-only": salad,
        "CSAD-only": csad,
        "SALAD+CSAD": both,
    }
    assignments = pd.read_csv(ASSIGN_FILE, encoding="utf-8-sig")
    folds = assignments["outer_test_fold"].to_numpy(dtype=int)
    if not (len(df) == len(y) == len(folds) == 1568):
        raise RuntimeError("Canonical 1568-row alignment failed")

    fold_rows, pred_rows = [], []
    for fold in range(1, 6):
        train = np.where(folds != fold)[0]
        test = np.where(folds == fold)[0]
        for method_idx, method in enumerate(METHODS):
            p = src.generic_outer_probe(
                feature_map[method], y, train, test, 961000 + 100 * fold + method_idx
            )
            yp = src.pred_from_prob(p)
            fold_rows.append({"outer_fold": fold, "method": method, **metric_values(y[test], yp)})
            for pos, idx in enumerate(test):
                pred_rows.append({
                    "sample_index": int(idx),
                    "sample_key": str(sample_keys[idx]),
                    "category": str(df.iloc[idx]["category"]),
                    "label": str(y[idx]),
                    "outer_fold": fold,
                    "method": method,
                    "prediction": str(yp[pos]),
                    "prob_normal": float(p[pos, 0]),
                    "prob_logical": float(p[pos, 1]),
                    "prob_structural": float(p[pos, 2]),
                })

    fold_df = pd.DataFrame(fold_rows)
    summary_rows = []
    metric_cols = ["acc", "bacc", "macro_f1", "normal_f1", "logical_f1", "structural_f1"]
    for method in METHODS:
        group = fold_df[fold_df["method"] == method]
        row = {"method": method, "n_folds": int(len(group))}
        for metric in metric_cols:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1))
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    fold_df.to_csv(OUT_DIR / "fold_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(pred_rows).to_csv(OUT_DIR / "oof_predictions.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(OUT_DIR / "summary.csv", index=False, encoding="utf-8-sig")
    config = {
        "date": "2026-09-14",
        "n_samples": 1568,
        "outer_folds": 5,
        "outer_assignments": str(ASSIGN_FILE),
        "probe": "StandardScaler + balanced LogisticRegression(C=1)",
        "representations": {"SALAD-only": 4, "CSAD-only": 18, "SALAD+CSAD": 22},
        "note": "All numbers are recomputed per-image evidence results; no paper-reported values are used.",
    }
    (OUT_DIR / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
