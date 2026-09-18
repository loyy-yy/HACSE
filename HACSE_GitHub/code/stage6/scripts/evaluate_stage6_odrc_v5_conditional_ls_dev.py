#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Stage6 ODRC V5 — Conditional Logical/Structural Fusion (DEVELOPMENT ONLY)
========================================================================

Motivation
----------
Observed development results show:
- V+O is the stable three-class backbone.
- Previous R usage as "structural-vs-rest" is weak/unstable.
- Previous C direct/gated corrections either hurt or collapse to abstention.

This V5 changes the TASK of R/C rather than continuing to tune gates:

    Stage 1: V+O handles N / L / S globally.
    Stage 2: R is trained ONLY on anomaly samples to distinguish L vs S.
    Stage 3: C is trained ONLY on anomaly samples to calibrate the L/S decision
             using normal-relation violation evidence.

At inference:
    P(N) is kept EXACTLY from V+O.
    R/C only redistribute the anomaly mass P(L)+P(S).

This matches the intended semantics:
    O : object/component evidence
    R : relationships among abnormal components
    C : whether those relationships violate normal constraints

Ablation:
    V
    V+O
    V+O+R
    V+O+R+C

Strict development protocol
---------------------------
- Uses the already-observed outer 5 folds ONLY for development screening.
- V/O families reuse the frozen settings from the prior nested-CV run.
- Inside each development outer-train:
    * p(V+O) is cross-fitted.
    * R/C relation evidence is rebuilt in every inner fold from INNER-TRAIN
      normal samples only.
    * R/C specialists are trained on INNER-TRAIN anomaly samples only.
    * feature ranking uses INNER-TRAIN anomaly samples only.
- No test-fold tuning.
- R and C fusion heads are fixed strongly-regularized Logistic Regression.
- No alpha/beta/gamma/tau search and no per-fold shrinkage choice.

IMPORTANT
---------
These outer folds have already been observed in earlier development.
This script is NOT a fresh final confirmation.

Prerequisites in the same scripts folder
----------------------------------------
    run_stage6_nested5fold_ablation_all1568.py
    evaluate_stage6_odrc_safe_fusion_v2_dev.py

Run
---
cd code/stage6
python scripts\evaluate_stage6_odrc_v5_conditional_ls_dev.py
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# =============================================================================
# Import existing strict utilities
# =============================================================================

HERE = Path(__file__).resolve().parent
V2_SCRIPT = HERE / "evaluate_stage6_odrc_safe_fusion_v2_dev.py"

if not V2_SCRIPT.exists():
    raise FileNotFoundError(
        f"Missing prerequisite:\n{V2_SCRIPT}"
    )

spec = importlib.util.spec_from_file_location("v2", V2_SCRIPT)
v2 = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(v2)

base = v2.base

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

OUT_DIR = ROOT / "stage6" / "outputs" / "stage6_odrc_v5_conditional_ls_dev"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FOLD = OUT_DIR / "v5_dev_by_fold.csv"
OUT_MEAN = OUT_DIR / "v5_dev_mean_std.csv"
OUT_DELTA = OUT_DIR / "v5_dev_deltas.csv"
OUT_SETTINGS = OUT_DIR / "v5_dev_settings.csv"
OUT_SUMMARY = OUT_DIR / "v5_dev_summary.json"

CLASS_ORDER = base.CLASS_ORDER

INNER_FOLDS = 5

# Fixed compact dimensions; no development search.
R_TOPK_LS = 48
C_TOPK_LS = 24

# Strong regularization for the anomaly-only specialists and stackers.
SPECIALIST_C = 0.20
STACK_C = 0.15

EPS = 1e-6

C_SMALL = [
    "c_rel_mean_abs_z",
    "c_rel_median_abs_z",
    "c_rel_p90_abs_z",
    "c_rel_max_abs_z",
    "c_rel_frac_gt1",
    "c_rel_frac_gt2",
    "c_rel_frac_gt3",
    "c_rel_top3_mean",
    "c_rel_top5_mean",
]


# =============================================================================
# Generic helpers
# =============================================================================

def logit(p):
    p = np.clip(
        np.asarray(p, dtype=float),
        EPS,
        1.0 - EPS,
    )
    return np.log(p / (1.0 - p))


def pred_from_prob(p):
    return np.asarray(
        CLASS_ORDER,
        dtype=object,
    )[np.argmax(p, axis=1)]


def ls_ratio_from_threeclass(p):
    p = np.asarray(p, dtype=float)
    a = np.clip(
        p[:, 1] + p[:, 2],
        EPS,
        None,
    )
    return np.clip(
        p[:, 2] / a,
        EPS,
        1.0 - EPS,
    )


def preserve_normal_set_ls(p_base, q_struct_given_anomaly):
    """
    Preserve P(N) exactly; redistribute only anomaly mass between L and S.
    """
    p_base = np.asarray(p_base, dtype=float)
    q = np.clip(
        np.asarray(q_struct_given_anomaly, dtype=float),
        EPS,
        1.0 - EPS,
    )

    pn = np.clip(
        p_base[:, 0],
        0.0,
        1.0,
    )

    anomaly = 1.0 - pn

    out = np.column_stack(
        [
            pn,
            anomaly * (1.0 - q),
            anomaly * q,
        ]
    )

    out /= out.sum(axis=1, keepdims=True)
    return out


def specialist_factory(seed):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=SPECIALIST_C,
            max_iter=3000,
            class_weight="balanced",
            random_state=seed,
        ),
    )


def stack_factory(seed):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=STACK_C,
            max_iter=3000,
            class_weight="balanced",
            random_state=seed,
        ),
    )


def binary_proba(model, X):
    p = model.predict_proba(X)
    classes = list(model.classes_)
    if 1 not in classes:
        return np.zeros(len(X), dtype=float)
    return p[:, classes.index(1)]


def anomaly_rank(X, y, train_idx, k):
    """
    Feature ranking ONLY on logical/structural samples.
    logical=0, structural=1
    """
    an = np.asarray(
        [
            i for i in train_idx
            if y[i] in {"logical", "structural"}
        ],
        dtype=int,
    )

    if len(an) == 0:
        raise RuntimeError("No anomaly samples for L/S ranking")

    yb = (
        y[an] == "structural"
    ).astype(int)

    with np.errstate(divide="ignore", invalid="ignore"):
        scores, _ = f_classif(
            X[an],
            yb,
        )

    scores = np.nan_to_num(
        scores,
        nan=-1e9,
        posinf=1e9,
        neginf=-1e9,
    )

    return np.argsort(-scores)[:min(k, X.shape[1])]


def c_small_matrix(c_df, idx):
    out = np.zeros(
        (len(idx), len(C_SMALL)),
        dtype=np.float32,
    )

    for j, c in enumerate(C_SMALL):
        if c not in c_df.columns:
            continue

        x = c_df.loc[
            idx,
            c,
        ].to_numpy(dtype=float)

        if "frac_" in c:
            x = np.clip(x, 0.0, 1.0)
        else:
            x = np.log1p(
                np.clip(x, 0.0, None)
            )

        out[:, j] = x.astype(np.float32)

    return out


# =============================================================================
# R/C conditional meta features
# =============================================================================

def r_stack_features(p_vo, q_r):
    """
    R contributes to the conditional L-vs-S decision.
    """
    base_ls = ls_ratio_from_threeclass(p_vo)

    ambiguity = (
        4.0
        * base_ls
        * (1.0 - base_ls)
    )

    anomaly_mass = (
        p_vo[:, 1]
        + p_vo[:, 2]
    )

    return np.column_stack(
        [
            logit(base_ls),
            logit(q_r),
            q_r - base_ls,
            ambiguity,
            anomaly_mass,
        ]
    ).astype(np.float32)


def c_stack_features(p_r, q_c, c_small):
    """
    C calibrates the already relation-aware L/S decision.
    It sees:
      - current relation-aware L/S belief
      - C specialist belief
      - compact normal-constraint deviation statistics
    """
    r_ls = ls_ratio_from_threeclass(p_r)

    ambiguity = (
        4.0
        * r_ls
        * (1.0 - r_ls)
    )

    anomaly_mass = (
        p_r[:, 1]
        + p_r[:, 2]
    )

    core = np.column_stack(
        [
            logit(r_ls),
            logit(q_c),
            q_c - r_ls,
            ambiguity,
            anomaly_mass,
        ]
    ).astype(np.float32)

    return np.concatenate(
        [
            core,
            c_small,
        ],
        axis=1,
    ).astype(np.float32)


# =============================================================================
# Strict OOF conditional specialist signals
# =============================================================================

def strict_oof_conditional_signals(
    V,
    O,
    recs,
    y,
    categories,
    outer_train,
    outer_fold,
    v_kind,
    o_kind,
):
    """
    Returns on outer_train, all cross-fitted:
      pV, pO, pVO, qR_LS, qC_LS, C-small

    qR_LS and qC_LS are trained ONLY on inner-train anomalies.
    R/C transformations are built from inner-train NORMAL samples only.
    """

    strata = base.make_strata(
        outer_train,
        categories,
        y,
    )

    skf = StratifiedKFold(
        n_splits=INNER_FOLDS,
        shuffle=True,
        random_state=151000 + outer_fold,
    )

    n = len(outer_train)

    pv = np.zeros((n, 3), dtype=float)
    po = np.zeros((n, 3), dtype=float)
    qr = np.zeros(n, dtype=float)
    qc = np.zeros(n, dtype=float)
    cs = np.zeros((n, len(C_SMALL)), dtype=np.float32)

    for fold, (a, b) in enumerate(
        skf.split(outer_train, strata)
    ):
        itr = outer_train[a]
        iva = outer_train[b]

        # -------------------------------------------------------------
        # V / O base predictions
        # -------------------------------------------------------------
        mv = base.model_factory(
            v_kind,
            152000 + outer_fold * 100 + fold,
        )

        mo = base.model_factory(
            o_kind,
            153000 + outer_fold * 100 + fold,
        )

        mv.fit(
            V[itr],
            y[itr],
        )

        mo.fit(
            O[itr],
            y[itr],
        )

        pv[b] = base.aligned_proba(
            mv,
            V[iva],
        )

        po[b] = base.aligned_proba(
            mo,
            O[iva],
        )

        # -------------------------------------------------------------
        # Build R/C from INNER-TRAIN normal only
        # -------------------------------------------------------------
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

        # -------------------------------------------------------------
        # L/S-only specialists
        # -------------------------------------------------------------
        itr_an = np.asarray(
            [
                i for i in itr
                if y[i] in {"logical", "structural"}
            ],
            dtype=int,
        )

        if len(itr_an) == 0:
            raise RuntimeError(
                f"Fold {outer_fold}-{fold}: no anomaly samples"
            )

        y_ls = (
            y[itr_an] == "structural"
        ).astype(int)

        r_sel = anomaly_rank(
            R,
            y,
            itr,
            R_TOPK_LS,
        )

        c_sel = anomaly_rank(
            C,
            y,
            itr,
            C_TOPK_LS,
        )

        mr = specialist_factory(
            154000 + outer_fold * 100 + fold,
        )

        mc = specialist_factory(
            155000 + outer_fold * 100 + fold,
        )

        mr.fit(
            R[itr_an][:, r_sel],
            y_ls,
        )

        mc.fit(
            C[itr_an][:, c_sel],
            y_ls,
        )

        qr[b] = binary_proba(
            mr,
            R[iva][:, r_sel],
        )

        qc[b] = binary_proba(
            mc,
            C[iva][:, c_sel],
        )

        cs[b] = c_small_matrix(
            c_df,
            iva,
        )

    # Cross-fitted V+O stack
    meta_vo = base.meta_vo(
        pv,
        po,
    )

    ytr = y[
        outer_train
    ]

    pvo = v2.crossfit_multiclass_logreg(
        meta_vo,
        ytr,
        strata,
        seed=156000 + outer_fold,
    )

    return (
        pv,
        po,
        pvo,
        qr,
        qc,
        cs,
    )


# =============================================================================
# Cross-fit anomaly-only stackers
# =============================================================================

def crossfit_anomaly_stack(
    X,
    y,
    seed,
):
    """
    Fit/predict only on anomaly samples; return q for all rows.
    Normal rows are assigned the base-neutral value 0.5 because these
    q-values are used only to redistribute anomaly mass.
    """
    q = np.full(
        len(y),
        0.5,
        dtype=float,
    )

    idx = np.where(
        np.isin(
            y,
            ["logical", "structural"],
        )
    )[0]

    ya = (
        y[idx] == "structural"
    ).astype(int)

    skf = StratifiedKFold(
        n_splits=INNER_FOLDS,
        shuffle=True,
        random_state=seed,
    )

    for fold, (a, b) in enumerate(
        skf.split(idx, ya)
    ):
        tr = idx[a]
        va = idx[b]

        m = stack_factory(
            seed + fold,
        )

        m.fit(
            X[tr],
            (
                y[tr] == "structural"
            ).astype(int),
        )

        q[va] = binary_proba(
            m,
            X[va],
        )

    return q


# =============================================================================
# Full outer-train refit and outer-test signals
# =============================================================================

def outer_test_signals(
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
):
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
        161000 + outer_fold,
    )

    mo = base.model_factory(
        o_kind,
        162000 + outer_fold,
    )

    mv.fit(
        V[outer_train],
        y[outer_train],
    )

    mo.fit(
        O[outer_train],
        y[outer_train],
    )

    pv = base.aligned_proba(
        mv,
        V[outer_test],
    )

    po = base.aligned_proba(
        mo,
        O[outer_test],
    )

    # R/C L/S specialists
    tr_an = np.asarray(
        [
            i for i in outer_train
            if y[i] in {"logical", "structural"}
        ],
        dtype=int,
    )

    y_ls = (
        y[tr_an] == "structural"
    ).astype(int)

    r_sel = anomaly_rank(
        R,
        y,
        outer_train,
        R_TOPK_LS,
    )

    c_sel = anomaly_rank(
        C,
        y,
        outer_train,
        C_TOPK_LS,
    )

    mr = specialist_factory(
        163000 + outer_fold,
    )

    mc = specialist_factory(
        164000 + outer_fold,
    )

    mr.fit(
        R[tr_an][:, r_sel],
        y_ls,
    )

    mc.fit(
        C[tr_an][:, c_sel],
        y_ls,
    )

    qr = binary_proba(
        mr,
        R[outer_test][:, r_sel],
    )

    qc = binary_proba(
        mc,
        C[outer_test][:, c_sel],
    )

    cs = c_small_matrix(
        c_df,
        outer_test,
    )

    return (
        pv,
        po,
        qr,
        qc,
        cs,
    )


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 120)
    print("ODRC V5 — CONDITIONAL LOGICAL/STRUCTURAL FUSION — DEVELOPMENT ONLY")
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

    # -------------------------------------------------------------------------
    # Load V
    # -------------------------------------------------------------------------
    df = pd.read_excel(
        FEATURE_FILE,
        sheet_name="merged_data",
    )

    df = df.loc[
        :,
        ~df.columns.duplicated(),
    ].copy()

    feat = pd.read_excel(
        FEATURE_FILE,
        sheet_name="features",
    )

    v_cols = list(
        dict.fromkeys(
            feat[
                "feature_cols"
            ]
            .dropna()
            .astype(str)
            .tolist()
        )
    )

    if len(v_cols) != 186:
        raise RuntimeError(
            f"Expected V dim 186, got {len(v_cols)}"
        )

    for c in v_cols:
        df[c] = (
            pd.to_numeric(
                df[c],
                errors="coerce",
            )
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .fillna(0.0)
        )

    V = df[
        v_cols
    ].to_numpy(
        dtype=np.float32
    )

    y = df.apply(
        base.infer_gt_group,
        axis=1,
    ).to_numpy(
        dtype=object
    )

    categories = (
        df["category"]
        .astype(str)
        .reset_index(drop=True)
    )

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

    rows = []
    delta_rows = []
    setting_rows = []

    # =========================================================================
    # Development outer folds
    # =========================================================================
    for outer_fold in range(1, 6):
        print()
        print("-" * 120)
        print(
            f"DEVELOPMENT FOLD {outer_fold}/5"
        )

        outer_test = np.where(
            assignments[
                "outer_test_fold"
            ].to_numpy(dtype=int)
            == outer_fold
        )[0]

        outer_train = np.where(
            assignments[
                "outer_test_fold"
            ].to_numpy(dtype=int)
            != outer_fold
        )[0]

        s = settings[
            settings["outer_fold"]
            == outer_fold
        ].iloc[0]

        v_kind = str(
            s["v_model"]
        )

        o_kind = str(
            s["o_model"]
        )

        # ---------------------------------------------------------------------
        # OOF conditional signals
        # ---------------------------------------------------------------------
        (
            pv_oof,
            po_oof,
            pvo_oof,
            qr_oof,
            qc_oof,
            cs_oof,
        ) = strict_oof_conditional_signals(
            V,
            O,
            recs,
            y,
            categories,
            outer_train,
            outer_fold,
            v_kind,
            o_kind,
        )

        ytr = y[
            outer_train
        ]

        # ---------------------------------------------------------------------
        # R stacker: anomaly-only
        # ---------------------------------------------------------------------
        XR_oof = r_stack_features(
            pvo_oof,
            qr_oof,
        )

        qR_stack_oof = crossfit_anomaly_stack(
            XR_oof,
            ytr,
            seed=171000 + outer_fold,
        )

        pR_oof = preserve_normal_set_ls(
            pvo_oof,
            qR_stack_oof,
        )

        idx_an = np.where(
            np.isin(
                ytr,
                ["logical", "structural"],
            )
        )[0]

        r_stack = stack_factory(
            172000 + outer_fold
        )

        r_stack.fit(
            XR_oof[idx_an],
            (
                ytr[idx_an]
                == "structural"
            ).astype(int),
        )

        # ---------------------------------------------------------------------
        # C stacker: anomaly-only calibration after R
        # ---------------------------------------------------------------------
        XC_oof = c_stack_features(
            pR_oof,
            qc_oof,
            cs_oof,
        )

        qC_stack_oof = crossfit_anomaly_stack(
            XC_oof,
            ytr,
            seed=173000 + outer_fold,
        )

        pC_oof = preserve_normal_set_ls(
            pR_oof,
            qC_stack_oof,
        )

        c_stack = stack_factory(
            174000 + outer_fold
        )

        c_stack.fit(
            XC_oof[idx_an],
            (
                ytr[idx_an]
                == "structural"
            ).astype(int),
        )

        # ---------------------------------------------------------------------
        # Full outer-test base signals
        # ---------------------------------------------------------------------
        (
            pv_te,
            po_te,
            qr_te,
            qc_te,
            cs_te,
        ) = outer_test_signals(
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
        )

        # V+O final fit
        meta_vo_oof = base.meta_vo(
            pv_oof,
            po_oof,
        )

        fusion_vo = base.fusion_factory(
            175000 + outer_fold
        )

        fusion_vo.fit(
            meta_vo_oof,
            ytr,
        )

        pvo_te = base.aligned_proba(
            fusion_vo,
            base.meta_vo(
                pv_te,
                po_te,
            ),
        )

        # R conditional
        XR_te = r_stack_features(
            pvo_te,
            qr_te,
        )

        qR_te = binary_proba(
            r_stack,
            XR_te,
        )

        pR_te = preserve_normal_set_ls(
            pvo_te,
            qR_te,
        )

        # C conditional
        XC_te = c_stack_features(
            pR_te,
            qc_te,
            cs_te,
        )

        qC_te = binary_proba(
            c_stack,
            XC_te,
        )

        pC_te = preserve_normal_set_ls(
            pR_te,
            qC_te,
        )

        # V baseline reference
        mv = base.model_factory(
            v_kind,
            176000 + outer_fold,
        )

        mv.fit(
            V[outer_train],
            y[outer_train],
        )

        pV_te = base.aligned_proba(
            mv,
            V[outer_test],
        )

        variants = {
            "V": pV_te,
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

            fold_metrics[
                name
            ] = m

            rows.append({
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

        setting_rows.append({
            "fold": outer_fold,
            "v_model": v_kind,
            "o_model": o_kind,
            "r_topk_ls": R_TOPK_LS,
            "c_topk_ls": C_TOPK_LS,
            "specialist_C": SPECIALIST_C,
            "stack_C": STACK_C,
            "r_mean_q_test": float(
                qR_te.mean()
            ),
            "c_mean_q_test": float(
                qC_te.mean()
            ),
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

    # =========================================================================
    # Summary
    # =========================================================================
    result = pd.DataFrame(
        rows
    )

    delta_df = pd.DataFrame(
        delta_rows
    )

    settings_df = pd.DataFrame(
        setting_rows
    )

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

    for variant in order:
        g = result[
            result["variant"]
            == variant
        ]

        row = {
            "variant": variant,
            "n_folds": len(g),
        }

        for c in metric_cols:
            row[
                f"{c}_mean"
            ] = float(
                g[c].mean()
            )

            row[
                f"{c}_std"
            ] = float(
                g[c].std(ddof=0)
            )

        mean_rows.append(
            row
        )

    mean_df = pd.DataFrame(
        mean_rows
    )

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

    settings_df.to_csv(
        OUT_SETTINGS,
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "status": "DEVELOPMENT_ONLY",
        "design": {
            "V+O": "stable three-class low-dimensional stack",
            "R": (
                "anomaly-only logical-vs-structural relation specialist "
                "+ regularized conditional stack"
            ),
            "C": (
                "anomaly-only logical-vs-structural constraint specialist "
                "+ regularized post-R calibration"
            ),
            "normal_probability_preserved_by_R": True,
            "normal_probability_preserved_by_C": True,
            "r_topk_ls": R_TOPK_LS,
            "c_topk_ls": C_TOPK_LS,
            "specialist_C": SPECIALIST_C,
            "stack_C": STACK_C,
            "hyperparameter_search": False,
        },
        "important": (
            "These folds were already observed. Use only to determine whether "
            "the task decomposition is viable; do not report as fresh final evidence."
        ),
        "mean_results": mean_df.to_dict(
            orient="records"
        ),
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
    print("ODRC V5 — DEVELOPMENT MEAN ± STD")
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
            delta_df["comparison"]
            == comp
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
    print("Fixed settings:")
    print(
        settings_df.to_string(
            index=False
        )
    )

    print()
    print(
        "PASS target: R_vs_VO and C_vs_VOR should both be positive in mean "
        "BAcc and Macro-F1, with no major Structural-F1 regression."
    )
    print("=" * 120)


if __name__ == "__main__":
    main()
