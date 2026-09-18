#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Stage6 ODRC Fusion V2 — DEVELOPMENT ONLY
========================================

Goal
----
Redesign R/C fusion after the first strict Nested-5-fold experiment showed:
- O gives a strong and stable gain;
- R is approximately neutral;
- C can over-correct and hurt generalization.

This V2 does NOT change the evidence sources O/R/C.
It changes only how R/C are allowed to modify the strong V+O prediction.

Key design
----------
1) V+O is the stable base classifier.
2) R is a structural specialist used through an uncertainty/disagreement gate.
   It updates ONLY structural mass, with a train-OOF calibrated binary head.
3) C is a constraint specialist used only for Logical-vs-Structural ambiguity.
   It PRESERVES Normal probability and only redistributes anomaly mass L/S.
4) Both R and C have train-OOF shrinkage factors selected from a tiny fixed grid
   that includes zero, so weak evidence can be suppressed.
5) All calibration predictions used for downstream training are cross-fitted.

This script intentionally reuses the already-observed Nested-CV folds as
DEVELOPMENT folds. Therefore its scores are NOT final confirmatory results.

Prerequisite
------------
Keep this file in the same folder as:
    run_stage6_nested5fold_ablation_all1568.py

Run
---
cd code/stage6
python scripts\evaluate_stage6_odrc_safe_fusion_v2_dev.py

Outputs
-------
stage6/outputs/stage6_odrc_safe_fusion_v2_dev/
    v2_dev_by_fold.csv
    v2_dev_mean_std.csv
    v2_dev_deltas.csv
    v2_dev_settings.csv
    v2_dev_summary.json
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# -----------------------------------------------------------------------------
# Load the previous strict Nested-CV implementation as a utility module.
# -----------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
BASE_SCRIPT = HERE / "run_stage6_nested5fold_ablation_all1568.py"

if not BASE_SCRIPT.exists():
    raise FileNotFoundError(
        f"Missing prerequisite script:\n{BASE_SCRIPT}\n"
        "Place run_stage6_nested5fold_ablation_all1568.py in the same scripts folder."
    )

spec = importlib.util.spec_from_file_location("nested_base", BASE_SCRIPT)
base = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(base)

ROOT = Path(__file__).resolve().parents[2]

FEATURE_FILE = ROOT / "full_loco_salad_csad_fusion_analysis.xlsx"
ASSIGN_FILE = (
    ROOT / "stage6" / "outputs" / "stage6_nested5fold_ablation"
    / "outer_fold_assignments.csv"
)
SETTING_FILE = (
    ROOT / "stage6" / "outputs" / "stage6_nested5fold_ablation"
    / "nested5fold_selected_settings.csv"
)

OUT_DIR = ROOT / "stage6" / "outputs" / "stage6_odrc_safe_fusion_v2_dev"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FOLD = OUT_DIR / "v2_dev_by_fold.csv"
OUT_MEAN = OUT_DIR / "v2_dev_mean_std.csv"
OUT_DELTA = OUT_DIR / "v2_dev_deltas.csv"
OUT_SETTINGS = OUT_DIR / "v2_dev_settings.csv"
OUT_SUMMARY = OUT_DIR / "v2_dev_summary.json"

CLASS_ORDER = base.CLASS_ORDER
INNER_FOLDS = 5
R_TOPK = 64
C_TOPK = 32

# Strong regularization for the residual calibrators.
CALIB_C = 0.25

# Tiny fixed shrinkage grid. Zero is deliberately allowed.
LAMBDA_GRID = [0.0, 0.25, 0.50, 0.75, 1.0]

# Stable, low-dimensional C summaries only.
C_SMALL = [
    "c_rel_mean_abs_z",
    "c_rel_median_abs_z",
    "c_rel_p90_abs_z",
    "c_rel_max_abs_z",
    "c_rel_frac_gt2",
    "c_rel_frac_gt3",
    "c_rel_top3_mean",
]


# -----------------------------------------------------------------------------
# Probability helpers
# -----------------------------------------------------------------------------

EPS = 1e-6


def clip_prob(p):
    p = np.asarray(p, dtype=float)
    return np.clip(p, EPS, 1.0 - EPS)


def entropy3(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1.0)
    return -(p * np.log(p)).sum(axis=1) / math.log(3.0)


def l1_disagreement(p1, p2):
    return np.abs(p1 - p2).sum(axis=1) / 2.0


def preserve_ns_ratio_with_structural(p_base, s_new):
    """
    Replace P(structural) with s_new and preserve the N:L ratio
    of the base classifier in the remaining mass.
    """
    p_base = np.asarray(p_base, dtype=float)
    s_new = np.clip(np.asarray(s_new, dtype=float), EPS, 1.0 - EPS)

    nl = p_base[:, 0] + p_base[:, 1]
    n_ratio = np.divide(
        p_base[:, 0],
        nl,
        out=np.full(len(p_base), 0.5, dtype=float),
        where=nl > EPS,
    )

    rem = 1.0 - s_new

    out = np.column_stack(
        [
            rem * n_ratio,
            rem * (1.0 - n_ratio),
            s_new,
        ]
    )

    out /= out.sum(axis=1, keepdims=True)
    return out


def preserve_normal_redistribute_ls(p_base, structural_given_anomaly):
    """
    Keep P(normal) exactly unchanged.
    Reallocate only L/S inside the anomaly mass.
    """
    p_base = np.asarray(p_base, dtype=float)
    r = np.clip(np.asarray(structural_given_anomaly, dtype=float), EPS, 1.0 - EPS)

    pn = np.clip(p_base[:, 0], 0.0, 1.0)
    anomaly_mass = 1.0 - pn

    out = np.column_stack(
        [
            pn,
            anomaly_mass * (1.0 - r),
            anomaly_mass * r,
        ]
    )

    out /= out.sum(axis=1, keepdims=True)
    return out


def blend_rows(p_old, p_new, weight):
    w = np.asarray(weight, dtype=float).reshape(-1, 1)
    w = np.clip(w, 0.0, 1.0)

    out = (1.0 - w) * p_old + w * p_new
    out /= out.sum(axis=1, keepdims=True)

    return out


def pred_from_prob(p):
    return np.asarray(CLASS_ORDER, dtype=object)[np.argmax(p, axis=1)]


def objective(y, p):
    pred = pred_from_prob(p)
    bacc = balanced_accuracy_score(y, pred)
    macro = f1_score(
        y,
        pred,
        labels=CLASS_ORDER,
        average="macro",
        zero_division=0,
    )
    # Equal weight: balanced recognition + class-balanced F1.
    return 0.5 * bacc + 0.5 * macro


def choose_lambda(y, p_old, p_new, gate):
    best = None

    for lam in LAMBDA_GRID:
        p = blend_rows(p_old, p_new, lam * gate)
        score = objective(y, p)

        # Tie-break toward smaller lambda -> lower intervention.
        key = (score, -lam)

        if best is None or key > best[0]:
            best = (key, lam)

    return float(best[1])


# -----------------------------------------------------------------------------
# Small calibrators
# -----------------------------------------------------------------------------

def binary_calibrator(seed):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=CALIB_C,
            max_iter=3000,
            class_weight="balanced",
            random_state=seed,
        ),
    )


def r_features(p_vo, p_v, p_o, q_r):
    """
    R is allowed to intervene mainly when VO is uncertain or V/O disagree.
    """
    p_vo = np.asarray(p_vo, dtype=float)
    q_r = clip_prob(q_r)

    s = clip_prob(p_vo[:, 2])
    ent = entropy3(p_vo)
    dis = l1_disagreement(p_v, p_o)

    # Relation-specialist evidence relative to the current structural belief.
    delta_s = q_r - s

    return np.column_stack(
        [
            np.log(s / (1.0 - s)),
            np.log(q_r / (1.0 - q_r)),
            delta_s,
            ent,
            dis,
            ent * delta_s,
            dis * delta_s,
        ]
    ).astype(np.float32)


def c_small_matrix(c_df, idx):
    cols = [c for c in C_SMALL if c in c_df.columns]

    blocks = []

    for c in cols:
        x = c_df.loc[idx, c].to_numpy(dtype=float)

        if "frac_" in c:
            z = np.clip(x, 0.0, 1.0)
        else:
            z = np.log1p(np.clip(x, 0.0, None))

        blocks.append(z.reshape(-1, 1))

    if not blocks:
        return np.zeros((len(idx), 0), dtype=np.float32)

    return np.concatenate(blocks, axis=1).astype(np.float32)


def c_features(p_r, q_c, c_small):
    """
    C works only on Logical-vs-Structural allocation.
    """
    p_r = np.asarray(p_r, dtype=float)
    q_c = clip_prob(q_c)

    anomaly = np.clip(p_r[:, 1] + p_r[:, 2], EPS, 1.0)
    r = np.clip(p_r[:, 2] / anomaly, EPS, 1.0 - EPS)

    ls_ambiguity = 4.0 * r * (1.0 - r)

    core = np.column_stack(
        [
            np.log(r / (1.0 - r)),
            np.log(q_c / (1.0 - q_c)),
            q_c - r,
            anomaly,
            ls_ambiguity,
            anomaly * ls_ambiguity,
        ]
    ).astype(np.float32)

    if c_small.shape[1] == 0:
        return core

    return np.concatenate(
        [
            core,
            c_small,
            ls_ambiguity[:, None] * c_small,
        ],
        axis=1,
    ).astype(np.float32)


# -----------------------------------------------------------------------------
# Cross-fitting helpers
# -----------------------------------------------------------------------------

def crossfit_multiclass_logreg(meta, y, strata, seed):
    """
    Cross-fit the fixed V+O fusion head to produce OOF probabilities.
    """
    skf = StratifiedKFold(
        n_splits=INNER_FOLDS,
        shuffle=True,
        random_state=seed,
    )

    out = np.zeros((len(y), 3), dtype=float)

    for fold, (tr, va) in enumerate(skf.split(np.arange(len(y)), strata)):
        model = base.fusion_factory(seed + fold)
        model.fit(meta[tr], y[tr])
        out[va] = base.aligned_proba(model, meta[va])

    return out


def crossfit_binary(features, y_binary, strata, seed):
    skf = StratifiedKFold(
        n_splits=INNER_FOLDS,
        shuffle=True,
        random_state=seed,
    )

    out = np.zeros(len(y_binary), dtype=float)

    for fold, (tr, va) in enumerate(skf.split(np.arange(len(y_binary)), strata)):
        model = binary_calibrator(seed + fold)
        model.fit(features[tr], y_binary[tr])

        p = model.predict_proba(features[va])
        classes = list(model.classes_)

        if 1 in classes:
            out[va] = p[:, classes.index(1)]
        else:
            out[va] = 0.0

    return out


def crossfit_binary_on_anomaly(features, y, anomaly_mask, seed):
    """
    Cross-fit only among logical/structural samples.
    Returns P(structural | anomaly) for ALL rows by fitting each fold's
    anomaly-only calibrator and predicting its held-out anomaly subset.
    Non-anomaly rows are filled later by the final model, because only
    anomaly rows are needed for lambda selection.
    """
    idx = np.where(anomaly_mask)[0]
    ya = (y[idx] == "structural").astype(int)

    skf = StratifiedKFold(
        n_splits=INNER_FOLDS,
        shuffle=True,
        random_state=seed,
    )

    out = np.full(len(y), 0.5, dtype=float)

    for fold, (trp, vap) in enumerate(skf.split(idx, ya)):
        tr = idx[trp]
        va = idx[vap]

        model = binary_calibrator(seed + fold)
        model.fit(features[tr], (y[tr] == "structural").astype(int))

        p = model.predict_proba(features[va])
        classes = list(model.classes_)

        out[va] = (
            p[:, classes.index(1)]
            if 1 in classes
            else 0.0
        )

    return out


# -----------------------------------------------------------------------------
# Strict OOF base signals
# -----------------------------------------------------------------------------

def strict_oof_base_signals(
    V,
    O,
    recs,
    y,
    categories,
    outer_train,
    outer_fold,
    v_kind,
    o_kind,
    r_kind,
    c_kind,
):
    """
    Produce strict OOF pV, pO, qR, qC and compact C summaries for outer_train.
    R palette and C prototype are rebuilt using each inner-train NORMAL subset.
    """

    strata = base.make_strata(
        outer_train,
        categories,
        y,
    )

    skf = StratifiedKFold(
        n_splits=INNER_FOLDS,
        shuffle=True,
        random_state=base.INNER_RANDOM_STATE + outer_fold,
    )

    yb = (y == "structural").astype(int)

    n = len(outer_train)

    p_v = np.zeros((n, 3), dtype=float)
    p_o = np.zeros((n, 3), dtype=float)
    q_r = np.zeros(n, dtype=float)
    q_c = np.zeros(n, dtype=float)
    c_small = np.zeros((n, len(C_SMALL)), dtype=float)

    for fold, (a, b) in enumerate(skf.split(outer_train, strata)):
        itr = outer_train[a]
        iva = outer_train[b]

        mv = base.model_factory(
            v_kind,
            outer_fold * 1000 + fold,
        )

        mo = base.model_factory(
            o_kind,
            outer_fold * 1000 + fold + 20,
        )

        mv.fit(V[itr], y[itr])
        mo.fit(O[itr], y[itr])

        p_v[b] = base.aligned_proba(mv, V[iva])
        p_o[b] = base.aligned_proba(mo, O[iva])

        train_normal = [
            int(i)
            for i in itr
            if y[i] == "normal"
        ]

        palettes = {}

        for cat in sorted(categories.unique()):
            palettes[cat] = (
                []
                if cat == "pushpins"
                else base.learn_palette(
                    recs,
                    train_normal,
                    cat,
                )
            )

        r_df, r_cols, R = base.build_r(
            recs,
            palettes,
        )

        c_df, c_cols, C = base.build_c(
            r_df,
            r_cols,
            categories,
            train_normal,
        )

        r_sel = base.train_rank(
            R[itr],
            yb[itr],
            R_TOPK,
        )

        c_sel = base.train_rank(
            C[itr],
            yb[itr],
            C_TOPK,
        )

        mr = base.model_factory(
            r_kind,
            outer_fold * 1000 + fold + 40,
        )

        mc = base.model_factory(
            c_kind,
            outer_fold * 1000 + fold + 60,
        )

        mr.fit(
            R[itr][:, r_sel],
            yb[itr],
        )

        mc.fit(
            C[itr][:, c_sel],
            yb[itr],
        )

        q_r[b] = base.structural_proba(
            mr,
            R[iva][:, r_sel],
        )

        q_c[b] = base.structural_proba(
            mc,
            C[iva][:, c_sel],
        )

        for j, c in enumerate(C_SMALL):
            if c in c_df.columns:
                x = c_df.loc[iva, c].to_numpy(dtype=float)
                if "frac_" in c:
                    x = np.clip(x, 0.0, 1.0)
                else:
                    x = np.log1p(np.clip(x, 0.0, None))
                c_small[b, j] = x

    return p_v, p_o, q_r, q_c, c_small


def outer_test_base_signals(
    V,
    O,
    recs,
    y,
    categories,
    outer_train,
    outer_test,
    outer_fold,
    v_kind,
    o_kind,
    r_kind,
    c_kind,
):
    """
    Refit all base components on the full outer-training partition and
    produce test signals.
    """

    train_normal = [
        int(i)
        for i in outer_train
        if y[i] == "normal"
    ]

    palettes = {}

    for cat in sorted(categories.unique()):
        palettes[cat] = (
            []
            if cat == "pushpins"
            else base.learn_palette(
                recs,
                train_normal,
                cat,
            )
        )

    r_df, r_cols, R = base.build_r(
        recs,
        palettes,
    )

    c_df, c_cols, C = base.build_c(
        r_df,
        r_cols,
        categories,
        train_normal,
    )

    mv = base.model_factory(
        v_kind,
        80000 + outer_fold * 100 + 1,
    )

    mo = base.model_factory(
        o_kind,
        80000 + outer_fold * 100 + 2,
    )

    mv.fit(V[outer_train], y[outer_train])
    mo.fit(O[outer_train], y[outer_train])

    p_v = base.aligned_proba(mv, V[outer_test])
    p_o = base.aligned_proba(mo, O[outer_test])

    yb = (y == "structural").astype(int)

    r_sel = base.train_rank(
        R[outer_train],
        yb[outer_train],
        R_TOPK,
    )

    c_sel = base.train_rank(
        C[outer_train],
        yb[outer_train],
        C_TOPK,
    )

    mr = base.model_factory(
        r_kind,
        80000 + outer_fold * 100 + 3,
    )

    mc = base.model_factory(
        c_kind,
        80000 + outer_fold * 100 + 4,
    )

    mr.fit(
        R[outer_train][:, r_sel],
        yb[outer_train],
    )

    mc.fit(
        C[outer_train][:, c_sel],
        yb[outer_train],
    )

    q_r = base.structural_proba(
        mr,
        R[outer_test][:, r_sel],
    )

    q_c = base.structural_proba(
        mc,
        C[outer_test][:, c_sel],
    )

    c_small = np.zeros(
        (len(outer_test), len(C_SMALL)),
        dtype=float,
    )

    for j, c in enumerate(C_SMALL):
        if c in c_df.columns:
            x = c_df.loc[
                outer_test,
                c,
            ].to_numpy(dtype=float)

            if "frac_" in c:
                x = np.clip(x, 0.0, 1.0)
            else:
                x = np.log1p(np.clip(x, 0.0, None))

            c_small[:, j] = x

    return p_v, p_o, q_r, q_c, c_small


# -----------------------------------------------------------------------------
# Main development evaluation
# -----------------------------------------------------------------------------

def main():
    print("=" * 120)
    print("ODRC SAFE FUSION V2 — DEVELOPMENT ON ALREADY-OBSERVED NESTED FOLDS")
    print("=" * 120)

    for p in [
        FEATURE_FILE,
        base.O_FILE,
        base.O_RAW,
        ASSIGN_FILE,
        SETTING_FILE,
    ]:
        if not p.exists():
            raise FileNotFoundError(p)

    # V
    df = pd.read_excel(
        FEATURE_FILE,
        sheet_name="merged_data",
    )
    df = df.loc[:, ~df.columns.duplicated()].copy()

    feat = pd.read_excel(
        FEATURE_FILE,
        sheet_name="features",
    )

    v_cols = list(
        dict.fromkeys(
            feat["feature_cols"]
            .dropna()
            .astype(str)
            .tolist()
        )
    )

    for c in v_cols:
        df[c] = (
            pd.to_numeric(df[c], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )

    V = df[v_cols].to_numpy(dtype=np.float32)
    y = df.apply(base.infer_gt_group, axis=1).to_numpy(dtype=object)
    categories = df["category"].astype(str).reset_index(drop=True)

    # O/raw
    o_df, o_cols, O = base.load_o()
    recs = base.load_raw()

    assignments = pd.read_csv(
        ASSIGN_FILE,
        encoding="utf-8-sig",
    )

    settings = pd.read_csv(
        SETTING_FILE,
        encoding="utf-8-sig",
    )

    all_rows = []
    setting_rows = []
    delta_rows = []

    for outer_fold in range(1, 6):
        print()
        print("-" * 120)
        print(f"DEVELOPMENT FOLD {outer_fold}/5")

        outer_test = np.where(
            assignments["outer_test_fold"].to_numpy(dtype=int)
            == outer_fold
        )[0]

        outer_train = np.where(
            assignments["outer_test_fold"].to_numpy(dtype=int)
            != outer_fold
        )[0]

        s = settings[
            settings["outer_fold"] == outer_fold
        ].iloc[0]

        v_kind = str(s["v_model"])
        o_kind = str(s["o_model"])
        r_kind = str(s["r_model"])
        c_kind = str(s["c_model"])

        # -------------------------------------------------------------
        # Strict OOF base signals on development-train.
        # -------------------------------------------------------------
        pv_oof, po_oof, qr_oof, qc_oof, csmall_oof = strict_oof_base_signals(
            V,
            O,
            recs,
            y,
            categories,
            outer_train,
            outer_fold,
            v_kind,
            o_kind,
            r_kind,
            c_kind,
        )

        ytr = y[outer_train]
        strata_tr = base.make_strata(
            outer_train,
            categories,
            y,
        )

        # -------------------------------------------------------------
        # V + O: stable base
        # -------------------------------------------------------------
        meta_vo_oof = base.meta_vo(
            pv_oof,
            po_oof,
        )

        pvo_oof = crossfit_multiclass_logreg(
            meta_vo_oof,
            ytr,
            strata_tr,
            seed=91000 + outer_fold,
        )

        fusion_vo = base.fusion_factory(
            92000 + outer_fold
        )

        fusion_vo.fit(
            meta_vo_oof,
            ytr,
        )

        # -------------------------------------------------------------
        # R: structural residual calibrator
        # -------------------------------------------------------------
        XR = r_features(
            pvo_oof,
            pv_oof,
            po_oof,
            qr_oof,
        )

        yR = (
            ytr == "structural"
        ).astype(int)

        qR_cal_oof = crossfit_binary(
            XR,
            yR,
            strata_tr,
            seed=93000 + outer_fold,
        )

        pR_raw_oof = preserve_ns_ratio_with_structural(
            pvo_oof,
            qR_cal_oof,
        )

        gate_R_oof = np.clip(
            0.5 * entropy3(pvo_oof)
            + 0.5 * l1_disagreement(pv_oof, po_oof),
            0.0,
            1.0,
        )

        lambda_R = choose_lambda(
            ytr,
            pvo_oof,
            pR_raw_oof,
            gate_R_oof,
        )

        pR_oof = blend_rows(
            pvo_oof,
            pR_raw_oof,
            lambda_R * gate_R_oof,
        )

        r_cal = binary_calibrator(
            94000 + outer_fold
        )

        r_cal.fit(
            XR,
            yR,
        )

        # -------------------------------------------------------------
        # C: conditional L-vs-S constraint calibrator
        # -------------------------------------------------------------
        XC = c_features(
            pR_oof,
            qc_oof,
            csmall_oof,
        )

        anomaly_mask = (
            ytr != "normal"
        )

        qC_cond_oof = crossfit_binary_on_anomaly(
            XC,
            ytr,
            anomaly_mask,
            seed=95000 + outer_fold,
        )

        pC_raw_oof = preserve_normal_redistribute_ls(
            pR_oof,
            qC_cond_oof,
        )

        anomaly_mass_oof = (
            pR_oof[:, 1] + pR_oof[:, 2]
        )

        r_ls_oof = np.divide(
            pR_oof[:, 2],
            np.clip(anomaly_mass_oof, EPS, None),
        )

        gate_C_oof = np.clip(
            anomaly_mass_oof
            * 4.0
            * r_ls_oof
            * (1.0 - r_ls_oof),
            0.0,
            1.0,
        )

        lambda_C = choose_lambda(
            ytr,
            pR_oof,
            pC_raw_oof,
            gate_C_oof,
        )

        c_cal = binary_calibrator(
            96000 + outer_fold
        )

        idx_a = np.where(
            anomaly_mask
        )[0]

        c_cal.fit(
            XC[idx_a],
            (
                ytr[idx_a]
                == "structural"
            ).astype(int),
        )

        # -------------------------------------------------------------
        # Full outer-development-test signals
        # -------------------------------------------------------------
        pv_te, po_te, qr_te, qc_te, csmall_te = outer_test_base_signals(
            V,
            O,
            recs,
            y,
            categories,
            outer_train,
            outer_test,
            outer_fold,
            v_kind,
            o_kind,
            r_kind,
            c_kind,
        )

        pvo_te = base.aligned_proba(
            fusion_vo,
            base.meta_vo(
                pv_te,
                po_te,
            ),
        )

        # R test
        XR_te = r_features(
            pvo_te,
            pv_te,
            po_te,
            qr_te,
        )

        qR_te = r_cal.predict_proba(
            XR_te
        )
        r_classes = list(
            r_cal[-1].classes_
            if hasattr(r_cal, "__getitem__")
            else r_cal.classes_
        )

        # Pipeline exposes classes_ directly in current sklearn.
        r_classes = list(r_cal.classes_)
        qR_te = qR_te[:, r_classes.index(1)]

        pR_raw_te = preserve_ns_ratio_with_structural(
            pvo_te,
            qR_te,
        )

        gate_R_te = np.clip(
            0.5 * entropy3(pvo_te)
            + 0.5 * l1_disagreement(pv_te, po_te),
            0.0,
            1.0,
        )

        pR_te = blend_rows(
            pvo_te,
            pR_raw_te,
            lambda_R * gate_R_te,
        )

        # C test
        XC_te = c_features(
            pR_te,
            qc_te,
            csmall_te,
        )

        qC_te_all = c_cal.predict_proba(
            XC_te
        )

        c_classes = list(
            c_cal.classes_
        )

        qC_te = qC_te_all[
            :,
            c_classes.index(1),
        ]

        pC_raw_te = preserve_normal_redistribute_ls(
            pR_te,
            qC_te,
        )

        anomaly_mass_te = (
            pR_te[:, 1] + pR_te[:, 2]
        )

        r_ls_te = np.divide(
            pR_te[:, 2],
            np.clip(
                anomaly_mass_te,
                EPS,
                None,
            ),
        )

        gate_C_te = np.clip(
            anomaly_mass_te
            * 4.0
            * r_ls_te
            * (1.0 - r_ls_te),
            0.0,
            1.0,
        )

        pC_te = blend_rows(
            pR_te,
            pC_raw_te,
            lambda_C * gate_C_te,
        )

        # V baseline for reference
        mv = base.model_factory(
            v_kind,
            80000 + outer_fold * 100 + 1,
        )
        mv.fit(
            V[outer_train],
            y[outer_train],
        )
        pV_te_ref = base.aligned_proba(
            mv,
            V[outer_test],
        )

        variants = {
            "V": pV_te_ref,
            "V+O": pvo_te,
            "V+O+R": pR_te,
            "V+O+R+C": pC_te,
        }

        fold_metrics = {}

        for name, prob in variants.items():
            m = base.metrics(
                y[outer_test],
                pred_from_prob(prob),
            )

            fold_metrics[name] = m

            all_rows.append({
                "fold": outer_fold,
                "variant": name,
                **m,
            })

            print(
                f"{name:<9} "
                f"Acc={m['acc']:.4f}  "
                f"BAcc={m['bacc']:.4f}  "
                f"MacroF1={m['macro_f1']:.4f}  "
                f"N/L/S="
                f"{m['normal_f1']:.4f}/"
                f"{m['logical_f1']:.4f}/"
                f"{m['structural_f1']:.4f}"
            )

        print(
            f"lambda_R={lambda_R:.2f} | "
            f"lambda_C={lambda_C:.2f}"
        )

        setting_rows.append({
            "fold": outer_fold,
            "v_model": v_kind,
            "o_model": o_kind,
            "r_model": r_kind,
            "c_model": c_kind,
            "lambda_R": lambda_R,
            "lambda_C": lambda_C,
            "r_gate_mean_test": float(gate_R_te.mean()),
            "c_gate_mean_test": float(gate_C_te.mean()),
        })

        for comp, a, b in [
            ("O_vs_V", "V+O", "V"),
            ("R_vs_VO", "V+O+R", "V+O"),
            ("C_vs_VOR", "V+O+R+C", "V+O+R"),
            ("Full_vs_V", "V+O+R+C", "V"),
        ]:
            ma = fold_metrics[a]
            mb = fold_metrics[b]

            delta_rows.append({
                "fold": outer_fold,
                "comparison": comp,
                "delta_acc": ma["acc"] - mb["acc"],
                "delta_bacc": ma["bacc"] - mb["bacc"],
                "delta_macro_f1": ma["macro_f1"] - mb["macro_f1"],
                "delta_normal_f1": ma["normal_f1"] - mb["normal_f1"],
                "delta_logical_f1": ma["logical_f1"] - mb["logical_f1"],
                "delta_structural_f1": ma["structural_f1"] - mb["structural_f1"],
            })

    result = pd.DataFrame(all_rows)
    delta_df = pd.DataFrame(delta_rows)
    setting_df = pd.DataFrame(setting_rows)

    order = [
        "V",
        "V+O",
        "V+O+R",
        "V+O+R+C",
    ]

    metric_cols = [
        "acc",
        "bacc",
        "macro_f1",
        "normal_f1",
        "logical_f1",
        "structural_f1",
    ]

    mean_rows = []

    for name in order:
        g = result[
            result["variant"] == name
        ]

        row = {
            "variant": name,
            "n_folds": len(g),
        }

        for c in metric_cols:
            row[f"{c}_mean"] = float(g[c].mean())
            row[f"{c}_std"] = float(g[c].std(ddof=0))

        mean_rows.append(row)

    mean_df = pd.DataFrame(mean_rows)

    result.to_csv(
        OUT_FOLD,
        index=False,
        encoding="utf-8-sig",
    )

    mean_df.to_csv(
        OUT_MEAN,
        index=False,
        encoding="utf-8-sig",
    )

    delta_df.to_csv(
        OUT_DELTA,
        index=False,
        encoding="utf-8-sig",
    )

    setting_df.to_csv(
        OUT_SETTINGS,
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "status": "DEVELOPMENT_ONLY",
        "architecture": {
            "base": "V+O unified logistic stacking",
            "R": (
                "uncertainty/disagreement-gated structural calibration; "
                "normal-logical ratio preserved"
            ),
            "C": (
                "logical-vs-structural conditional calibration; "
                "normal probability preserved"
            ),
            "lambda_grid": LAMBDA_GRID,
            "calibrator_C": CALIB_C,
        },
        "important": (
            "These folds were already observed in the previous Nested-CV run. "
            "Use only for redesign screening, not as final confirmation."
        ),
        "mean_results": mean_df.to_dict(orient="records"),
    }

    OUT_SUMMARY.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 120)
    print("SAFE FUSION V2 — DEVELOPMENT MEAN ± STD")
    print("=" * 120)

    for _, r in mean_df.iterrows():
        print(
            f"{r['variant']:<9} "
            f"Acc={r['acc_mean']:.4f}±{r['acc_std']:.4f}  "
            f"BAcc={r['bacc_mean']:.4f}±{r['bacc_std']:.4f}  "
            f"MacroF1={r['macro_f1_mean']:.4f}±{r['macro_f1_std']:.4f}  "
            f"N/L/S="
            f"{r['normal_f1_mean']:.4f}/"
            f"{r['logical_f1_mean']:.4f}/"
            f"{r['structural_f1_mean']:.4f}"
        )

    print()
    print("PAIRED MEAN DELTAS")
    print("-" * 120)

    for comp in [
        "O_vs_V",
        "R_vs_VO",
        "C_vs_VOR",
        "Full_vs_V",
    ]:
        g = delta_df[
            delta_df["comparison"] == comp
        ]

        print(
            f"{comp:<12} "
            f"Acc={g['delta_acc'].mean():+.4f}  "
            f"BAcc={g['delta_bacc'].mean():+.4f}  "
            f"MacroF1={g['delta_macro_f1'].mean():+.4f}  "
            f"N/L/S="
            f"{g['delta_normal_f1'].mean():+.4f}/"
            f"{g['delta_logical_f1'].mean():+.4f}/"
            f"{g['delta_structural_f1'].mean():+.4f}"
        )

    print()
    print("Gate/shrinkage settings by development fold:")
    print(setting_df.to_string(index=False))
    print()
    print(
        "DECISION RULE: Prefer V2 only if one fixed design gives "
        "positive mean BAcc, Macro-F1 and Structural-F1 increments at R and C "
        "without relying on per-fold architecture changes."
    )
    print("=" * 120)


if __name__ == "__main__":
    main()
