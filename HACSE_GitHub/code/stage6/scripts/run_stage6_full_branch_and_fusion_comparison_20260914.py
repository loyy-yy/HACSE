#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Unified five-fold branch ablation and fusion-strategy comparison.

Uses the already frozen Stage6 sample cohort, outer-fold assignments, V/O
representations, and frozen DINOv2-S/14 features.  Every learned probability
fusion head is trained only from inner out-of-fold probabilities.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = HERE / "audit_stage6_dino_global_v1_ceiling_dev.py"
OUT_DIR = ROOT / "stage6" / "outputs" / "stage6_full_branch_fusion_20260914"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


src = import_module(SOURCE, "stage6_dino_source_20260914")
base = src.base
v6 = src.v6
CLASS_ORDER = list(src.CLASS_ORDER)

METHODS = [
    "V",
    "O",
    "D",
    "V+O",
    "V+D",
    "O+D",
    "V+O+D",
    "Probability average",
    "Feature concatenation",
]


def metrics(y_true, y_pred):
    result = {
        "acc": float(accuracy_score(y_true, y_pred)),
        "bacc": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=CLASS_ORDER, average="macro", zero_division=0)
        ),
    }
    for cls in CLASS_ORDER:
        result[f"{cls}_f1"] = float(
            f1_score(y_true, y_pred, labels=[cls], average="macro", zero_division=0)
        )
    return result


def pred(proba):
    return np.asarray(CLASS_ORDER, dtype=object)[np.argmax(proba, axis=1)]


def fusion_factory(seed):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=4000,
            random_state=seed,
        ),
    )


def aligned(model, x):
    return src.aligned_proba_local(model, x)


def bounded_base_model(kind, seed):
    model = base.model_factory(kind, seed)
    if "n_jobs" in model.get_params(deep=False):
        model.set_params(n_jobs=4)
    return model


def fit_probability_fusion(train_parts, test_parts, y_train, seed):
    x_train = np.concatenate(train_parts, axis=1).astype(np.float32)
    x_test = np.concatenate(test_parts, axis=1).astype(np.float32)
    model = fusion_factory(seed)
    model.fit(x_train, y_train)
    return aligned(model, x_test)


def strict_oof_vo(V, O, y, categories, outer_train, outer_fold, v_kind, o_kind):
    strata = base.make_strata(outer_train, categories, y)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=181000 + outer_fold)
    pv = np.zeros((len(outer_train), 3), dtype=np.float64)
    po = np.zeros((len(outer_train), 3), dtype=np.float64)
    for inner_fold, (a, b) in enumerate(skf.split(outer_train, strata)):
        itr, iva = outer_train[a], outer_train[b]
        mv = bounded_base_model(v_kind, 182000 + outer_fold * 100 + inner_fold)
        mo = bounded_base_model(o_kind, 183000 + outer_fold * 100 + inner_fold)
        mv.fit(V[itr], y[itr])
        mo.fit(O[itr], y[itr])
        pv[b] = base.aligned_proba(mv, V[iva])
        po[b] = base.aligned_proba(mo, O[iva])
    return pv, po


def fit_predict_vo_outer(V, O, y, outer_train, outer_test, outer_fold, v_kind, o_kind):
    mv = bounded_base_model(v_kind, 191000 + outer_fold)
    mo = bounded_base_model(o_kind, 192000 + outer_fold)
    mv.fit(V[outer_train], y[outer_train])
    mo.fit(O[outer_train], y[outer_train])
    return (
        base.aligned_proba(mv, V[outer_test]),
        base.aligned_proba(mo, O[outer_test]),
    )


def summarize(fold_df):
    metric_cols = [
        "acc", "bacc", "macro_f1", "normal_f1", "logical_f1", "structural_f1"
    ]
    rows = []
    for method in METHODS:
        group = fold_df[fold_df["method"] == method]
        row = {"method": method, "n_folds": int(len(group))}
        for col in metric_cols:
            row[f"{col}_mean"] = float(group[col].mean())
            row[f"{col}_std"] = float(group[col].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    required = [
        src.FEATURE_FILE,
        src.ASSIGN_FILE,
        src.SETTING_FILE,
        src.DINO_CACHE,
        base.O_FILE,
        base.O_RAW,
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)

    df = pd.read_excel(src.FEATURE_FILE, sheet_name="merged_data")
    df = df.loc[:, ~df.columns.duplicated()].copy().reset_index(drop=True)
    feat = pd.read_excel(src.FEATURE_FILE, sheet_name="features")
    vcols = list(dict.fromkeys(feat["feature_cols"].dropna().astype(str).tolist()))
    if len(vcols) != 186:
        raise RuntimeError(f"Expected V=186D, got {len(vcols)}")
    for col in vcols:
        df[col] = pd.to_numeric(df[col], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        ).fillna(0.0)

    V = df[vcols].to_numpy(dtype=np.float32)
    y = df.apply(base.infer_gt_group, axis=1).to_numpy(dtype=object)
    categories = df["category"].astype(str).reset_index(drop=True)
    sample_keys = (
        df["merge_key"].astype(str).to_numpy(dtype=object)
        if "merge_key" in df.columns
        else np.asarray([str(i) for i in range(len(df))], dtype=object)
    )
    _, _, O = base.load_o()
    recs = base.load_raw()

    cache = np.load(src.DINO_CACHE, allow_pickle=True)
    D = np.asarray(cache["embeddings"], dtype=np.float32)
    cached_keys = np.asarray(cache["sample_keys"], dtype=object)
    if len(cached_keys) != len(sample_keys):
        raise RuntimeError("DINO cache length mismatch")
    if not np.array_equal(cached_keys.astype(str), sample_keys.astype(str)):
        mismatch = int(np.where(cached_keys.astype(str) != sample_keys.astype(str))[0][0])
        raise RuntimeError(f"DINO/workbook key mismatch at row {mismatch}")

    assignments = pd.read_csv(src.ASSIGN_FILE, encoding="utf-8-sig")
    fold_assignment = assignments["outer_test_fold"].to_numpy(dtype=int)
    settings = pd.read_csv(src.SETTING_FILE, encoding="utf-8-sig")
    if not (len(df) == len(V) == len(O) == len(D) == len(recs) == len(fold_assignment) == 1568):
        raise RuntimeError("Expected aligned 1568-row cohort")

    fold_rows = []
    prediction_rows = []

    for outer_fold in range(1, 6):
        print(f"OUTER FOLD {outer_fold}/5", flush=True)
        outer_test = np.where(fold_assignment == outer_fold)[0]
        outer_train = np.where(fold_assignment != outer_fold)[0]
        setting = settings[settings["outer_fold"] == outer_fold].iloc[0]
        v_kind, o_kind = str(setting["v_model"]), str(setting["o_model"])

        pv_oof, po_oof = strict_oof_vo(
            V, O, y, categories, outer_train, outer_fold, v_kind, o_kind
        )
        pd_oof = src.strict_dino_oof(D, y, categories, outer_train, outer_fold)
        pv_te, po_te = fit_predict_vo_outer(
            V, O, y, outer_train, outer_test, outer_fold, v_kind, o_kind
        )
        pd_te = src.fit_predict_dino_outer(D, y, outer_train, outer_test, outer_fold)
        y_train, y_test = y[outer_train], y[outer_test]

        probabilities = {
            "V": pv_te,
            "O": po_te,
            "D": pd_te,
            "V+O": fit_probability_fusion(
                [pv_oof, po_oof], [pv_te, po_te], y_train, 931000 + outer_fold
            ),
            "V+D": fit_probability_fusion(
                [pv_oof, pd_oof], [pv_te, pd_te], y_train, 932000 + outer_fold
            ),
            "O+D": fit_probability_fusion(
                [po_oof, pd_oof], [po_te, pd_te], y_train, 933000 + outer_fold
            ),
            "V+O+D": fit_probability_fusion(
                [pv_oof, po_oof, pd_oof], [pv_te, po_te, pd_te], y_train, 934000 + outer_fold
            ),
            "Probability average": (pv_te + po_te + pd_te) / 3.0,
        }

        concat_model = fusion_factory(935000 + outer_fold)
        concat_model.fit(
            np.concatenate([V[outer_train], O[outer_train], D[outer_train]], axis=1),
            y_train,
        )
        probabilities["Feature concatenation"] = aligned(
            concat_model,
            np.concatenate([V[outer_test], O[outer_test], D[outer_test]], axis=1),
        )

        for method in METHODS:
            p = probabilities[method]
            yp = pred(p)
            fold_rows.append({"outer_fold": outer_fold, "method": method, **metrics(y_test, yp)})
            for local_pos, row_index in enumerate(outer_test):
                prediction_rows.append(
                    {
                        "sample_index": int(row_index),
                        "sample_key": str(sample_keys[row_index]),
                        "category": str(categories.iloc[row_index]),
                        "label": str(y_test[local_pos]),
                        "outer_fold": outer_fold,
                        "method": method,
                        "prediction": str(yp[local_pos]),
                        "prob_normal": float(p[local_pos, 0]),
                        "prob_logical": float(p[local_pos, 1]),
                        "prob_structural": float(p[local_pos, 2]),
                    }
                )

    fold_df = pd.DataFrame(fold_rows)
    pred_df = pd.DataFrame(prediction_rows)
    summary_df = summarize(fold_df)
    fold_df.to_csv(OUT_DIR / "fold_metrics.csv", index=False, encoding="utf-8-sig")
    pred_df.to_csv(OUT_DIR / "oof_predictions.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(OUT_DIR / "summary.csv", index=False, encoding="utf-8-sig")

    vod = fold_df[fold_df["method"] == "V+O+D"].set_index("outer_fold")
    delta_rows = []
    for comparator in ["V", "O", "D", "V+O", "V+D", "O+D", "Probability average", "Feature concatenation"]:
        comp = fold_df[fold_df["method"] == comparator].set_index("outer_fold")
        for metric in ["bacc", "macro_f1", "structural_f1"]:
            delta = vod[metric] - comp[metric]
            delta_rows.append(
                {
                    "reference": "V+O+D",
                    "comparator": comparator,
                    "metric": metric,
                    "mean_delta": float(delta.mean()),
                    "positive_folds": int((delta > 0).sum()),
                    "fold_deltas": json.dumps([float(x) for x in delta.to_list()]),
                }
            )
    pd.DataFrame(delta_rows).to_csv(
        OUT_DIR / "paired_deltas.csv", index=False, encoding="utf-8-sig"
    )

    config = {
        "date": "2026-09-14",
        "task": "LOCO supervised N/L/S diagnosis",
        "n_samples": 1568,
        "outer_folds": 5,
        "outer_assignments": str(src.ASSIGN_FILE),
        "inner_folds": 5,
        "class_order": CLASS_ORDER,
        "V_dim": int(V.shape[1]),
        "O_dim": int(O.shape[1]),
        "D_dim": int(D.shape[1]),
        "D_model": src.DINO_MODEL,
        "fusion": "StandardScaler + balanced LogisticRegression(C=1)",
        "probability_fusion_training": "inner OOF only",
        "feature_concatenation": "[V;O;D] -> StandardScaler + balanced LogisticRegression(C=1)",
        "methods": METHODS,
    }
    (OUT_DIR / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summary_df.to_string(index=False), flush=True)
    print(f"Saved: {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
