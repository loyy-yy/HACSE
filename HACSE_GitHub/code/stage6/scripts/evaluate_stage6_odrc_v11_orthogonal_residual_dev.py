#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Stage6 ODRC V11 — Orthogonal Stage-wise Residual Learning (DEVELOPMENT ONLY)
============================================================================

Goal
----
Test one final principled hypothesis for monotonic ablation:

    V < V+O < V+O+R < V+O+R+C

Instead of asking R/C to re-learn the whole Logical-vs-Structural task,
V11 forces each stage to learn ONLY information not already explained by
the previous stage.

R stage
-------
1) Start from the strong V+O posterior.
2) Let q_R be the category-conditioned R specialist score.
3) Regress logit(q_R) from V/O context using TRAIN data only.
4) Keep the orthogonal residual:

       R_perp = logit(q_R) - E[logit(q_R) | V,O,category]

5) Learn the residual target:

       y_structural - P_VO(structural | anomaly)

   using only R_perp-derived features.

C stage
-------
1) Start from the already R-corrected conditional L/S belief.
2) Let q_C be the category-conditioned C specialist score.
3) Regress logit(q_C) from V/O/R context.
4) Keep:

       C_perp = logit(q_C) - E[logit(q_C) | V,O,R,category]

5) Learn only the remaining residual:

       y_structural - P_VOR(structural | anomaly)

Safety
------
- R/C are trained only on Logical/Structural samples.
- P(Normal) from the V+O backbone is never changed.
- If V+O predicts Normal, R/C do not override that Normal decision.
- R/C palettes/prototypes are still rebuilt from training-normal samples only.
- All orthogonalizers and residual models are cross-fitted for TRAIN-OOF
  evaluation.
- Stage shrinkage alpha is selected from TRAIN-OOF only.
- alpha=0 is allowed; if a stage has no incremental information, it abstains.

Scientific boundary
-------------------
The five outer folds have already been observed in earlier development.
This script is DEVELOPMENT ONLY. If V11 cannot produce stable R and C
incremental gains, stop pursuing monotonic fusion on these data.

Prerequisites in the same scripts folder
----------------------------------------
    run_stage6_nested5fold_ablation_all1568.py
    evaluate_stage6_odrc_safe_fusion_v2_dev.py
    evaluate_stage6_odrc_v5_conditional_ls_dev.py
    evaluate_stage6_odrc_v6_category_conditioned_dev.py

Run
---
cd code/stage6
python scripts\evaluate_stage6_odrc_v11_orthogonal_residual_dev.py
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# =============================================================================
# Import strict evidence builders from V6
# =============================================================================

HERE = Path(__file__).resolve().parent
V6_SCRIPT = HERE / "evaluate_stage6_odrc_v6_category_conditioned_dev.py"

if not V6_SCRIPT.exists():
    raise FileNotFoundError(f"Missing prerequisite:\n{V6_SCRIPT}")

spec = importlib.util.spec_from_file_location("v6", V6_SCRIPT)
v6 = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(v6)

v5 = v6.v5
v2 = v6.v2
base = v6.base

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

OUT_DIR = ROOT / "stage6" / "outputs" / "stage6_odrc_v11_orthogonal_residual_dev"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FOLD = OUT_DIR / "v11_dev_by_fold.csv"
OUT_MEAN = OUT_DIR / "v11_dev_mean_std.csv"
OUT_DELTA = OUT_DIR / "v11_dev_deltas.csv"
OUT_SETTINGS = OUT_DIR / "v11_dev_settings.csv"
OUT_DIAG = OUT_DIR / "v11_orthogonal_diagnostics.csv"
OUT_SUMMARY = OUT_DIR / "v11_dev_summary.json"

CLASS_ORDER = base.CLASS_ORDER
EPS = 1e-5

INNER_FOLDS = 5

# Fixed regularization: no grid search over model complexity.
ORTH_RIDGE_ALPHA = 10.0
RESID_RIDGE_ALPHA = 20.0

# Only the stage shrinkage is selected from OOF training predictions.
# Tie-breaking always prefers the smaller alpha.
ALPHA_GRID = [0.0, 0.25, 0.50, 0.75, 1.00]


# =============================================================================
# Basic helpers
# =============================================================================

def logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def ls_ratio(p):
    p = np.asarray(p, dtype=float)
    den = np.clip(p[:, 1] + p[:, 2], EPS, None)
    return np.clip(p[:, 2] / den, EPS, 1.0 - EPS)


def pred_from_prob(p):
    return np.asarray(CLASS_ORDER, dtype=object)[np.argmax(p, axis=1)]


def entropy_binary(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)
    return -(
        p * np.log(p)
        + (1.0 - p) * np.log(1.0 - p)
    ) / math.log(2.0)


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

    return 0.5 * bacc + 0.5 * macro


def category_onehot(categories, original_idx, all_cats):
    vals = categories.iloc[original_idx].astype(str).to_numpy()
    lookup = {c: j for j, c in enumerate(all_cats)}

    out = np.zeros(
        (len(original_idx), len(all_cats)),
        dtype=np.float32,
    )

    for i, c in enumerate(vals):
        out[i, lookup[c]] = 1.0

    return out


def ridge_factory(alpha):
    return make_pipeline(
        StandardScaler(),
        Ridge(alpha=alpha),
    )


# =============================================================================
# Context and orthogonal residual features
# =============================================================================

def vo_context(
    p_vo,
    p_v,
    p_o,
    categories,
    original_idx,
    all_cats,
):
    qvo = ls_ratio(p_vo)
    qv = ls_ratio(p_v)
    qo = ls_ratio(p_o)

    anomaly_mass = p_vo[:, 1] + p_vo[:, 2]
    ambiguity = 4.0 * qvo * (1.0 - qvo)
    disagreement = np.abs(qv - qo)

    core = np.column_stack(
        [
            logit(qvo),
            qvo,
            entropy_binary(qvo),
            ambiguity,
            anomaly_mass,
            qv,
            qo,
            disagreement,
            np.abs(qv - qvo),
            np.abs(qo - qvo),
        ]
    ).astype(np.float32)

    return np.concatenate(
        [
            core,
            category_onehot(
                categories,
                original_idx,
                all_cats,
            ),
        ],
        axis=1,
    ).astype(np.float32)


def vor_context(
    p_vo,
    p_v,
    p_o,
    q_r_current,
    q_r_raw,
    categories,
    original_idx,
    all_cats,
):
    base_ctx = vo_context(
        p_vo,
        p_v,
        p_o,
        categories,
        original_idx,
        all_cats,
    )

    qr_cur = np.clip(q_r_current, EPS, 1.0 - EPS)
    qr_raw = np.clip(q_r_raw, EPS, 1.0 - EPS)

    extra = np.column_stack(
        [
            logit(qr_cur),
            qr_cur,
            entropy_binary(qr_cur),
            logit(qr_raw),
            qr_raw,
            qr_raw - qr_cur,
            np.abs(qr_raw - qr_cur),
            np.abs(qr_cur - ls_ratio(p_vo)),
        ]
    ).astype(np.float32)

    return np.concatenate(
        [base_ctx, extra],
        axis=1,
    ).astype(np.float32)


def residual_features(perp, q_base):
    """
    The downstream learner receives only the orthogonal evidence residual
    and interactions with the current-stage uncertainty.

    It does NOT receive the original R/C score or raw V/O features again.
    """
    perp = np.asarray(perp, dtype=float)
    qb = np.clip(np.asarray(q_base, dtype=float), EPS, 1.0 - EPS)

    ambiguity = 4.0 * qb * (1.0 - qb)
    uncertainty = entropy_binary(qb)

    return np.column_stack(
        [
            perp,
            np.abs(perp),
            np.sign(perp) * np.sqrt(np.abs(perp) + EPS),
            perp * ambiguity,
            perp * uncertainty,
        ]
    ).astype(np.float32)


# =============================================================================
# Cross-fitted stage learner
# =============================================================================

def make_anomaly_strata(
    y_local,
    categories,
    original_idx,
    anomaly_local_idx,
):
    vals = []

    for li in anomaly_local_idx:
        oi = int(original_idx[li])
        vals.append(
            f"{categories.iloc[oi]}||{y_local[li]}"
        )

    return np.asarray(vals, dtype=object)


def crossfit_orthogonal_stage(
    q_base,
    q_raw_signal,
    context,
    y_local,
    categories,
    original_idx,
    seed,
):
    """
    Returns cross-fitted:
      delta_oof : predicted probability residual
      perp_oof  : orthogonalized evidence residual

    Only true L/S samples participate in fitting.
    """
    n = len(y_local)

    delta_oof = np.zeros(n, dtype=float)
    perp_oof = np.zeros(n, dtype=float)

    an = np.where(
        np.isin(
            y_local,
            ["logical", "structural"],
        )
    )[0]

    if len(an) == 0:
        raise RuntimeError("No anomaly samples for residual stage.")

    strata = make_anomaly_strata(
        y_local,
        categories,
        original_idx,
        an,
    )

    # Fallback to binary labels if any category/class stratum is too small.
    _, counts = np.unique(
        strata,
        return_counts=True,
    )

    if counts.min() < INNER_FOLDS:
        strata = y_local[an].astype(str)

    _, counts = np.unique(
        strata,
        return_counts=True,
    )

    n_splits = min(
        INNER_FOLDS,
        int(counts.min()),
    )

    if n_splits < 2:
        raise RuntimeError(
            "Not enough anomaly samples for cross-fitted residual learning."
        )

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )

    raw_logit = logit(q_raw_signal)

    for fold, (a, b) in enumerate(
        skf.split(an, strata)
    ):
        tr = an[a]
        va = an[b]

        # -------------------------------------------------------------
        # Orthogonalizer: raw evidence explained by previous-stage context.
        # -------------------------------------------------------------
        orth = ridge_factory(
            ORTH_RIDGE_ALPHA
        )

        orth.fit(
            context[tr],
            raw_logit[tr],
        )

        perp_tr = (
            raw_logit[tr]
            - orth.predict(
                context[tr]
            )
        )

        perp_va = (
            raw_logit[va]
            - orth.predict(
                context[va]
            )
        )

        # -------------------------------------------------------------
        # Residual target: what the previous stage has not explained.
        # -------------------------------------------------------------
        target_tr = (
            (y_local[tr] == "structural").astype(float)
            - q_base[tr]
        )

        Xr_tr = residual_features(
            perp_tr,
            q_base[tr],
        )

        Xr_va = residual_features(
            perp_va,
            q_base[va],
        )

        resid = ridge_factory(
            RESID_RIDGE_ALPHA
        )

        resid.fit(
            Xr_tr,
            target_tr,
        )

        delta_oof[va] = resid.predict(
            Xr_va
        )

        perp_oof[va] = perp_va

    return (
        delta_oof,
        perp_oof,
    )


def fit_orthogonal_stage_full(
    q_base,
    q_raw_signal,
    context,
    y_local,
):
    an = np.where(
        np.isin(
            y_local,
            ["logical", "structural"],
        )
    )[0]

    raw_logit = logit(
        q_raw_signal
    )

    orth = ridge_factory(
        ORTH_RIDGE_ALPHA
    )

    orth.fit(
        context[an],
        raw_logit[an],
    )

    perp = (
        raw_logit[an]
        - orth.predict(
            context[an]
        )
    )

    target = (
        (y_local[an] == "structural").astype(float)
        - q_base[an]
    )

    resid = ridge_factory(
        RESID_RIDGE_ALPHA
    )

    resid.fit(
        residual_features(
            perp,
            q_base[an],
        ),
        target,
    )

    return (
        orth,
        resid,
    )


def predict_orthogonal_stage(
    q_base,
    q_raw_signal,
    context,
    orth,
    resid,
):
    raw_logit = logit(
        q_raw_signal
    )

    perp = (
        raw_logit
        - orth.predict(
            context
        )
    )

    delta = resid.predict(
        residual_features(
            perp,
            q_base,
        )
    )

    return (
        delta,
        perp,
    )


# =============================================================================
# Safe application + alpha selection
# =============================================================================

def apply_stage(
    p_previous,
    q_base,
    delta,
    alpha,
):
    """
    Preserve the previous Normal decision.

    If previous full prediction is Normal:
        leave all three probabilities unchanged.

    Otherwise:
        keep P(N) fixed and redistribute anomaly mass using
        q_new = q_base + alpha * delta.
    """
    p_previous = np.asarray(
        p_previous,
        dtype=float,
    )

    out = p_previous.copy()

    apply = (
        np.argmax(
            p_previous,
            axis=1,
        )
        != 0
    )

    q_new = np.clip(
        np.asarray(
            q_base,
            dtype=float,
        )
        + alpha
        * np.asarray(
            delta,
            dtype=float,
        ),
        EPS,
        1.0 - EPS,
    )

    pn = p_previous[
        apply,
        0,
    ]

    anomaly_mass = (
        1.0 - pn
    )

    out[
        apply,
        1,
    ] = (
        anomaly_mass
        * (
            1.0
            - q_new[
                apply
            ]
        )
    )

    out[
        apply,
        2,
    ] = (
        anomaly_mass
        * q_new[
            apply
        ]
    )

    out /= out.sum(
        axis=1,
        keepdims=True,
    )

    return out


def choose_alpha_oof(
    y_local,
    p_previous,
    q_base,
    delta,
):
    rows = []

    for alpha in ALPHA_GRID:
        p = apply_stage(
            p_previous,
            q_base,
            delta,
            alpha,
        )

        rows.append(
            {
                "alpha": float(alpha),
                "objective": float(
                    objective(
                        y_local,
                        p,
                    )
                ),
            }
        )

    # Prefer smaller alpha on ties.
    best = sorted(
        rows,
        key=lambda z: (
            -z["objective"],
            z["alpha"],
        ),
    )[0]

    return (
        float(best["alpha"]),
        rows,
    )


# =============================================================================
# Diagnostics
# =============================================================================

def safe_corr(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if (
        np.std(a) < 1e-12
        or np.std(b) < 1e-12
    ):
        return np.nan

    return float(
        np.corrcoef(
            a,
            b,
        )[0, 1]
    )


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 120)
    print("ODRC V11 — ORTHOGONAL STAGE-WISE RESIDUAL LEARNING — DEVELOPMENT ONLY")
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
    # Load V / labels / O
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
            feat["feature_cols"]
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
        .reset_index(
            drop=True
        )
    )

    all_cats = sorted(
        categories.unique()
    )

    _, _, O = base.load_o()
    recs = base.load_raw()

    assignments = pd.read_csv(
        ASSIGN_FILE,
        encoding="utf-8-sig",
    )

    settings = pd.read_csv(
        SETTING_FILE,
        encoding="utf-8-sig",
    )

    metric_rows = []
    delta_rows = []
    setting_rows = []
    diag_rows = []

    # =========================================================================
    # Development folds
    # =========================================================================
    for outer_fold in range(
        1,
        6,
    ):
        print()
        print("-" * 120)
        print(
            f"DEVELOPMENT FOLD {outer_fold}/5"
        )

        outer_test = np.where(
            assignments[
                "outer_test_fold"
            ].to_numpy(
                dtype=int
            )
            == outer_fold
        )[0]

        outer_train = np.where(
            assignments[
                "outer_test_fold"
            ].to_numpy(
                dtype=int
            )
            != outer_fold
        )[0]

        s = settings[
            settings[
                "outer_fold"
            ]
            == outer_fold
        ].iloc[0]

        v_kind = str(
            s["v_model"]
        )

        o_kind = str(
            s["o_model"]
        )

        # ---------------------------------------------------------------------
        # Strict OOF category-conditioned evidence
        # ---------------------------------------------------------------------
        (
            pv_oof,
            po_oof,
            pvo_oof,
            qr_oof,
            qc_oof,
            cs_oof,
            _,
        ) = v6.strict_oof_category_signals(
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
        # R: orthogonal evidence residual against V/O context
        # ---------------------------------------------------------------------
        qvo_oof = ls_ratio(
            pvo_oof
        )

        ctx_r_oof = vo_context(
            pvo_oof,
            pv_oof,
            po_oof,
            categories,
            outer_train,
            all_cats,
        )

        (
            delta_r_oof,
            r_perp_oof,
        ) = crossfit_orthogonal_stage(
            qvo_oof,
            qr_oof,
            ctx_r_oof,
            ytr,
            categories,
            outer_train,
            seed=401000 + outer_fold,
        )

        (
            alpha_R,
            alpha_R_rows,
        ) = choose_alpha_oof(
            ytr,
            pvo_oof,
            qvo_oof,
            delta_r_oof,
        )

        pR_oof = apply_stage(
            pvo_oof,
            qvo_oof,
            delta_r_oof,
            alpha_R,
        )

        qR_current_oof = ls_ratio(
            pR_oof
        )

        (
            orth_R,
            resid_R,
        ) = fit_orthogonal_stage_full(
            qvo_oof,
            qr_oof,
            ctx_r_oof,
            ytr,
        )

        # ---------------------------------------------------------------------
        # C: orthogonal residual against V/O/R context
        # ---------------------------------------------------------------------
        ctx_c_oof = vor_context(
            pvo_oof,
            pv_oof,
            po_oof,
            qR_current_oof,
            qr_oof,
            categories,
            outer_train,
            all_cats,
        )

        (
            delta_c_oof,
            c_perp_oof,
        ) = crossfit_orthogonal_stage(
            qR_current_oof,
            qc_oof,
            ctx_c_oof,
            ytr,
            categories,
            outer_train,
            seed=402000 + outer_fold,
        )

        (
            alpha_C,
            alpha_C_rows,
        ) = choose_alpha_oof(
            ytr,
            pR_oof,
            qR_current_oof,
            delta_c_oof,
        )

        pC_oof = apply_stage(
            pR_oof,
            qR_current_oof,
            delta_c_oof,
            alpha_C,
        )

        (
            orth_C,
            resid_C,
        ) = fit_orthogonal_stage_full(
            qR_current_oof,
            qc_oof,
            ctx_c_oof,
            ytr,
        )

        # ---------------------------------------------------------------------
        # Outer development-test evidence
        # ---------------------------------------------------------------------
        (
            pv_te,
            po_te,
            qr_te,
            qc_te,
            cs_te,
            _,
        ) = v6.outer_test_category_signals(
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

        # V+O refit
        fusion_vo = base.fusion_factory(
            403000 + outer_fold
        )

        fusion_vo.fit(
            base.meta_vo(
                pv_oof,
                po_oof,
            ),
            ytr,
        )

        pvo_te = base.aligned_proba(
            fusion_vo,
            base.meta_vo(
                pv_te,
                po_te,
            ),
        )

        # R test
        qvo_te = ls_ratio(
            pvo_te
        )

        ctx_r_te = vo_context(
            pvo_te,
            pv_te,
            po_te,
            categories,
            outer_test,
            all_cats,
        )

        (
            delta_r_te,
            r_perp_te,
        ) = predict_orthogonal_stage(
            qvo_te,
            qr_te,
            ctx_r_te,
            orth_R,
            resid_R,
        )

        pR_te = apply_stage(
            pvo_te,
            qvo_te,
            delta_r_te,
            alpha_R,
        )

        qR_current_te = ls_ratio(
            pR_te
        )

        # C test
        ctx_c_te = vor_context(
            pvo_te,
            pv_te,
            po_te,
            qR_current_te,
            qr_te,
            categories,
            outer_test,
            all_cats,
        )

        (
            delta_c_te,
            c_perp_te,
        ) = predict_orthogonal_stage(
            qR_current_te,
            qc_te,
            ctx_c_te,
            orth_C,
            resid_C,
        )

        pC_te = apply_stage(
            pR_te,
            qR_current_te,
            delta_c_te,
            alpha_C,
        )

        # V reference
        mv = base.model_factory(
            v_kind,
            404000 + outer_fold,
        )

        mv.fit(
            V[
                outer_train
            ],
            y[
                outer_train
            ],
        )

        pV_te = base.aligned_proba(
            mv,
            V[
                outer_test
            ],
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
                y[
                    outer_test
                ],
                pred_from_prob(
                    prob
                ),
            )

            fold_metrics[
                name
            ] = m

            metric_rows.append(
                {
                    "fold": outer_fold,
                    "variant": name,
                    **m,
                }
            )

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

        an_local = np.where(
            np.isin(
                ytr,
                ["logical", "structural"],
            )
        )[0]

        target_r = (
            (ytr[an_local] == "structural").astype(float)
            - qvo_oof[an_local]
        )

        target_c = (
            (ytr[an_local] == "structural").astype(float)
            - qR_current_oof[an_local]
        )

        diag_rows.append(
            {
                "fold": outer_fold,
                "stage": "R",
                "alpha_selected": alpha_R,
                "perp_std_train_oof": float(
                    np.std(
                        r_perp_oof[
                            an_local
                        ]
                    )
                ),
                "delta_std_train_oof": float(
                    np.std(
                        delta_r_oof[
                            an_local
                        ]
                    )
                ),
                "perp_target_corr_train_oof": safe_corr(
                    r_perp_oof[
                        an_local
                    ],
                    target_r,
                ),
                "delta_target_corr_train_oof": safe_corr(
                    delta_r_oof[
                        an_local
                    ],
                    target_r,
                ),
                "perp_std_test": float(
                    np.std(
                        r_perp_te
                    )
                ),
                "delta_std_test": float(
                    np.std(
                        delta_r_te
                    )
                ),
            }
        )

        diag_rows.append(
            {
                "fold": outer_fold,
                "stage": "C",
                "alpha_selected": alpha_C,
                "perp_std_train_oof": float(
                    np.std(
                        c_perp_oof[
                            an_local
                        ]
                    )
                ),
                "delta_std_train_oof": float(
                    np.std(
                        delta_c_oof[
                            an_local
                        ]
                    )
                ),
                "perp_target_corr_train_oof": safe_corr(
                    c_perp_oof[
                        an_local
                    ],
                    target_c,
                ),
                "delta_target_corr_train_oof": safe_corr(
                    delta_c_oof[
                        an_local
                    ],
                    target_c,
                ),
                "perp_std_test": float(
                    np.std(
                        c_perp_te
                    )
                ),
                "delta_std_test": float(
                    np.std(
                        delta_c_te
                    )
                ),
            }
        )

        setting_rows.append(
            {
                "fold": outer_fold,
                "v_model": v_kind,
                "o_model": o_kind,
                "orth_ridge_alpha": ORTH_RIDGE_ALPHA,
                "resid_ridge_alpha": RESID_RIDGE_ALPHA,
                "alpha_R": alpha_R,
                "alpha_C": alpha_C,
                "R_oof_objective_alpha0": next(
                    z["objective"]
                    for z in alpha_R_rows
                    if z["alpha"] == 0.0
                ),
                "R_oof_objective_selected": max(
                    z["objective"]
                    for z in alpha_R_rows
                    if z["alpha"] == alpha_R
                ),
                "C_oof_objective_alpha0": next(
                    z["objective"]
                    for z in alpha_C_rows
                    if z["alpha"] == 0.0
                ),
                "C_oof_objective_selected": max(
                    z["objective"]
                    for z in alpha_C_rows
                    if z["alpha"] == alpha_C
                ),
            }
        )

        for comp, a, b in [
            (
                "O_vs_V",
                "V+O",
                "V",
            ),
            (
                "R_vs_VO",
                "V+O+R",
                "V+O",
            ),
            (
                "C_vs_VOR",
                "V+O+R+C",
                "V+O+R",
            ),
            (
                "Full_vs_V",
                "V+O+R+C",
                "V",
            ),
        ]:
            ma = fold_metrics[
                a
            ]

            mb = fold_metrics[
                b
            ]

            delta_rows.append(
                {
                    "fold": outer_fold,
                    "comparison": comp,
                    "delta_acc": ma["acc"] - mb["acc"],
                    "delta_bacc": ma["bacc"] - mb["bacc"],
                    "delta_macro_f1": ma["macro_f1"] - mb["macro_f1"],
                    "delta_normal_f1": ma["normal_f1"] - mb["normal_f1"],
                    "delta_logical_f1": ma["logical_f1"] - mb["logical_f1"],
                    "delta_structural_f1": ma["structural_f1"] - mb["structural_f1"],
                }
            )

        print(
            f"Selected alpha_R={alpha_R:.2f}, alpha_C={alpha_C:.2f}"
        )

    # =========================================================================
    # Summaries
    # =========================================================================
    result = pd.DataFrame(
        metric_rows
    )

    delta_df = pd.DataFrame(
        delta_rows
    )

    settings_df = pd.DataFrame(
        setting_rows
    )

    diag_df = pd.DataFrame(
        diag_rows
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
            result[
                "variant"
            ]
            == variant
        ]

        row = {
            "variant": variant,
            "n_folds": int(
                len(g)
            ),
        }

        for c in metric_cols:
            row[
                f"{c}_mean"
            ] = float(
                g[
                    c
                ].mean()
            )

            row[
                f"{c}_std"
            ] = float(
                g[
                    c
                ].std(
                    ddof=0
                )
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

    diag_df.to_csv(
        OUT_DIAG,
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "status": "DEVELOPMENT_ONLY",
        "design": {
            "VO": "stable three-class V/O stack",
            "R": (
                "category-conditioned R score -> orthogonalize against V/O "
                "context -> stage-wise residual correction"
            ),
            "C": (
                "category-conditioned C score -> orthogonalize against V/O/R "
                "context -> stage-wise residual correction"
            ),
            "orthogonalizer": {
                "model": "StandardScaler + Ridge",
                "alpha": ORTH_RIDGE_ALPHA,
            },
            "residual_learner": {
                "model": "StandardScaler + Ridge",
                "alpha": RESID_RIDGE_ALPHA,
            },
            "stage_alpha_grid_train_oof_only": ALPHA_GRID,
            "normal_decision_preserved": True,
        },
        "scientific_boundary": (
            "The outer folds have already been observed in previous development. "
            "This is the final planned monotonic-fusion development hypothesis."
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
    print("ODRC V11 — DEVELOPMENT MEAN ± STD")
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
            delta_df[
                "comparison"
            ]
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
    print("ORTHOGONAL RESIDUAL SETTINGS")
    print("-" * 120)
    print(
        settings_df.to_string(
            index=False
        )
    )

    print()
    print("ORTHOGONAL RESIDUAL DIAGNOSTICS")
    print("-" * 120)
    print(
        diag_df.to_string(
            index=False
        )
    )

    print()
    print(
        "PASS target: mean R_vs_VO and C_vs_VOR both positive in BAcc and "
        "Macro-F1, preferably Structural-F1 non-negative. If either stage "
        "selects alpha=0 in most folds or remains non-positive, do not continue "
        "tuning these already-observed folds."
    )
    print("=" * 120)


if __name__ == "__main__":
    main()
