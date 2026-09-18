#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Frozen generic representation probes on canonical Stage6 outer folds."""

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
OUT_DIR = ROOT / "stage6" / "outputs" / "stage6_foundation_representations_20260914"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


src = load_module(SOURCE, "stage6_foundation_source_20260914")
CLASS_ORDER = list(src.CLASS_ORDER)
METHODS = ["ResNet18 frozen", "CLIP-ViT-B/32 frozen", "DINOv2-S/14 frozen"]


def score(y_true, y_pred):
    out = {
        "acc": float(accuracy_score(y_true, y_pred)),
        "bacc": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=CLASS_ORDER, average="macro", zero_division=0)),
    }
    for cls in CLASS_ORDER:
        out[f"{cls}_f1"] = float(f1_score(y_true, y_pred, labels=[cls], average="macro", zero_division=0))
    return out


def load_cached(path, sample_keys):
    z = np.load(path, allow_pickle=True)
    x = np.asarray(z["embeddings"], dtype=np.float32)
    keys = np.asarray(z["sample_keys"], dtype=str)
    z.close()
    if not np.array_equal(keys, np.asarray(sample_keys, dtype=str)):
        raise RuntimeError(f"Cache key mismatch: {path}")
    return x


def main():
    df, _, _, _, D, y, _, sample_keys, _ = src.load_benchmark()
    R18 = load_cached(src.RESNET_CACHE, sample_keys)
    CLIP = load_cached(src.CLIP_CACHE, sample_keys)
    features = {
        "ResNet18 frozen": R18,
        "CLIP-ViT-B/32 frozen": CLIP,
        "DINOv2-S/14 frozen": D,
    }
    folds = pd.read_csv(ASSIGN_FILE, encoding="utf-8-sig")["outer_test_fold"].to_numpy(dtype=int)
    fold_rows, pred_rows = [], []
    for fold in range(1, 6):
        tr, te = np.where(folds != fold)[0], np.where(folds == fold)[0]
        for method_idx, method in enumerate(METHODS):
            p = src.generic_outer_probe(features[method], y, tr, te, 971000 + 100 * fold + method_idx)
            yp = src.pred_from_prob(p)
            fold_rows.append({"outer_fold": fold, "method": method, **score(y[te], yp)})
            for pos, idx in enumerate(te):
                pred_rows.append({
                    "sample_index": int(idx), "sample_key": str(sample_keys[idx]),
                    "category": str(df.iloc[idx]["category"]), "label": str(y[idx]),
                    "outer_fold": fold, "method": method, "prediction": str(yp[pos]),
                    "prob_normal": float(p[pos, 0]), "prob_logical": float(p[pos, 1]),
                    "prob_structural": float(p[pos, 2]),
                })
    fold_df = pd.DataFrame(fold_rows)
    rows = []
    metrics = ["acc", "bacc", "macro_f1", "normal_f1", "logical_f1", "structural_f1"]
    for method in METHODS:
        g = fold_df[fold_df["method"] == method]
        row = {"method": method, "n_folds": int(len(g))}
        for metric in metrics:
            row[f"{metric}_mean"] = float(g[metric].mean())
            row[f"{metric}_std"] = float(g[metric].std(ddof=1))
        rows.append(row)
    summary = pd.DataFrame(rows)
    fold_df.to_csv(OUT_DIR / "fold_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(pred_rows).to_csv(OUT_DIR / "oof_predictions.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_DIR / "summary.csv", index=False, encoding="utf-8-sig")
    (OUT_DIR / "config.json").write_text(json.dumps({
        "date": "2026-09-14", "n_samples": 1568, "outer_folds": 5,
        "outer_assignments": str(ASSIGN_FILE),
        "probe": "StandardScaler + balanced LogisticRegression(C=1)",
        "representations": {name: int(features[name].shape[1]) for name in METHODS},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
