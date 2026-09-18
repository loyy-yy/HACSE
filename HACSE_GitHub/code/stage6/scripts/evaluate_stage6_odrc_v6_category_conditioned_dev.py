#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Stage6 ODRC V6 — Category-Conditioned R/C Experts (DEVELOPMENT ONLY)
===================================================================

Why V6
------
V5 showed:
    O_vs_V      : clearly positive
    R_vs_VO     : small positive
    C_vs_VOR    : small negative

The remaining issue is not the three-class backbone. It is that a single
global R/C Logical-vs-Structural specialist must learn very different
relationship semantics across the five MVTec LOCO categories.

V6 redesign
-----------
Keep V+O exactly as the stable backbone.

R:
    train CATEGORY-CONDITIONED Logical-vs-Structural relation experts.
    Each expert sees only anomaly samples of its category.

C:
    train CATEGORY-CONDITIONED Logical-vs-Structural constraint experts.
    C features are still derived only from train-normal relation prototypes.

Fallback:
    if a category has too few L/S training samples, use a global anomaly-only
    fallback expert trained on the same inner-training partition.

Fusion:
    P(Normal) from V+O is preserved exactly.
    R only redistributes P(L)+P(S).
    C calibrates the R-aware L/S allocation and again preserves P(Normal).

Strictness
----------
- Existing already-observed outer folds are used for DEVELOPMENT only.
- Inside every inner fold:
    * R semantic palette is learned from INNER-TRAIN normals only.
    * C normal prototype is learned from INNER-TRAIN normals only.
    * category-specific feature ranking is learned from INNER-TRAIN anomalies only.
    * inner validation is never used to fit R/C transformations or experts.
- Fixed model class, regularization and dimensions.
- No lambda/tau/gate search.
- No per-fold architecture selection.

Prerequisites in the same scripts directory
-------------------------------------------
    run_stage6_nested5fold_ablation_all1568.py
    evaluate_stage6_odrc_safe_fusion_v2_dev.py
    evaluate_stage6_odrc_v5_conditional_ls_dev.py

Run
---
cd code/stage6
python scripts\evaluate_stage6_odrc_v6_category_conditioned_dev.py
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.feature_selection import f_classif
from sklearn.model_selection import StratifiedKFold


# =============================================================================
# Import V5 utilities
# =============================================================================

HERE = Path(__file__).resolve().parent
V5_SCRIPT = HERE / "evaluate_stage6_odrc_v5_conditional_ls_dev.py"

if not V5_SCRIPT.exists():
    raise FileNotFoundError(
        f"Missing prerequisite:\n{V5_SCRIPT}"
    )

spec = importlib.util.spec_from_file_location("v5", V5_SCRIPT)
v5 = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(v5)

base = v5.base
v2 = v5.v2

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

OUT_DIR = ROOT / "stage6" / "outputs" / "stage6_odrc_v6_category_conditioned_dev"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FOLD = OUT_DIR / "v6_dev_by_fold.csv"
OUT_MEAN = OUT_DIR / "v6_dev_mean_std.csv"
OUT_DELTA = OUT_DIR / "v6_dev_deltas.csv"
OUT_SETTINGS = OUT_DIR / "v6_dev_settings.csv"
OUT_SUMMARY = OUT_DIR / "v6_dev_summary.json"

CLASS_ORDER = base.CLASS_ORDER
INNER_FOLDS = 5

# Fixed compact dimensions, deliberately smaller than V5 global specialists.
R_TOPK_CAT = 20
C_TOPK_CAT = 12

# Global fallback compact dimensions.
R_TOPK_GLOBAL = 48
C_TOPK_GLOBAL = 24

# Minimum per-class examples required for a category-specific expert.
MIN_PER_LS_CLASS = 8

# Reuse strong regularization from V5.
SPECIALIST_C = v5.SPECIALIST_C
STACK_C = v5.STACK_C

C_SMALL = v5.C_SMALL


# =============================================================================
# Ranking / specialist helpers
# =============================================================================

def rank_on_indices(X, y_binary, idx, k):
    idx = np.asarray(idx, dtype=int)

    if len(idx) == 0:
        raise RuntimeError("Cannot rank features on empty index set")

    with np.errstate(divide="ignore", invalid="ignore"):
        scores, _ = f_classif(
            X[idx],
            y_binary,
        )

    scores = np.nan_to_num(
        scores,
        nan=-1e9,
        posinf=1e9,
        neginf=-1e9,
    )

    return np.argsort(-scores)[:min(k, X.shape[1])]


def train_global_ls_expert(X, y, train_idx, k, seed):
    an = np.asarray(
        [
            i for i in train_idx
            if y[i] in {"logical", "structural"}
        ],
        dtype=int,
    )

    yb = (
        y[an] == "structural"
    ).astype(int)

    sel = rank_on_indices(
        X,
        yb,
        an,
        k,
    )

    model = v5.specialist_factory(seed)

    model.fit(
        X[an][:, sel],
        yb,
    )

    return model, sel


def train_category_ls_experts(
    X,
    y,
    categories,
    train_idx,
    k_cat,
    k_global,
    seed,
):
    """
    Returns:
        global_model, global_sel,
        experts: cat -> (model, sel) or None
        stats
    """
    global_model, global_sel = train_global_ls_expert(
        X,
        y,
        train_idx,
        k_global,
        seed,
    )

    experts = {}
    stats = {}

    for ci, cat in enumerate(sorted(categories.unique())):
        idx = np.asarray(
            [
                i for i in train_idx
                if categories.iloc[i] == cat
                and y[i] in {"logical", "structural"}
            ],
            dtype=int,
        )

        n_l = int(
            np.sum(y[idx] == "logical")
        )

        n_s = int(
            np.sum(y[idx] == "structural")
        )

        use_cat = (
            n_l >= MIN_PER_LS_CLASS
            and n_s >= MIN_PER_LS_CLASS
        )

        stats[cat] = {
            "n_logical": n_l,
            "n_structural": n_s,
            "category_expert": bool(use_cat),
        }

        if not use_cat:
            experts[cat] = None
            continue

        yb = (
            y[idx] == "structural"
        ).astype(int)

        sel = rank_on_indices(
            X,
            yb,
            idx,
            k_cat,
        )

        model = v5.specialist_factory(
            seed + 100 + ci
        )

        model.fit(
            X[idx][:, sel],
            yb,
        )

        experts[cat] = (
            model,
            sel,
        )

    return (
        global_model,
        global_sel,
        experts,
        stats,
    )


def predict_category_experts(
    X,
    categories,
    pred_idx,
    global_model,
    global_sel,
    experts,
):
    pred_idx = np.asarray(
        pred_idx,
        dtype=int,
    )

    out = np.zeros(
        len(pred_idx),
        dtype=float,
    )

    for cat in sorted(categories.unique()):
        pos = np.where(
            categories.iloc[pred_idx].to_numpy()
            == cat
        )[0]

        if len(pos) == 0:
            continue

        original_idx = pred_idx[pos]

        item = experts.get(cat)

        if item is None:
            out[pos] = v5.binary_proba(
                global_model,
                X[original_idx][:, global_sel],
            )
        else:
            model, sel = item

            out[pos] = v5.binary_proba(
                model,
                X[original_idx][:, sel],
            )

    return out


# =============================================================================
# Strict OOF category-conditioned R/C evidence
# =============================================================================

def strict_oof_category_signals(
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
    strata = base.make_strata(
        outer_train,
        categories,
        y,
    )

    skf = StratifiedKFold(
        n_splits=INNER_FOLDS,
        shuffle=True,
        random_state=181000 + outer_fold,
    )

    n = len(outer_train)

    pv = np.zeros(
        (n, 3),
        dtype=float,
    )

    po = np.zeros(
        (n, 3),
        dtype=float,
    )

    qr = np.zeros(
        n,
        dtype=float,
    )

    qc = np.zeros(
        n,
        dtype=float,
    )

    cs = np.zeros(
        (n, len(C_SMALL)),
        dtype=np.float32,
    )

    cat_expert_count_r = []
    cat_expert_count_c = []

    for fold, (a, b) in enumerate(
        skf.split(
            outer_train,
            strata,
        )
    ):
        itr = outer_train[a]
        iva = outer_train[b]

        # -------------------------------------------------------------
        # V/O OOF
        # -------------------------------------------------------------
        mv = base.model_factory(
            v_kind,
            182000 + outer_fold * 100 + fold,
        )

        mo = base.model_factory(
            o_kind,
            183000 + outer_fold * 100 + fold,
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
        # Inner-train-normal R/C transformation
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
        # Category-conditioned R experts
        # -------------------------------------------------------------
        (
            rg,
            rgs,
            rex,
            rstats,
        ) = train_category_ls_experts(
            R,
            y,
            categories,
            itr,
            R_TOPK_CAT,
            R_TOPK_GLOBAL,
            184000 + outer_fold * 100 + fold,
        )

        qr[b] = predict_category_experts(
            R,
            categories,
            iva,
            rg,
            rgs,
            rex,
        )

        cat_expert_count_r.append(
            sum(
                int(z["category_expert"])
                for z in rstats.values()
            )
        )

        # -------------------------------------------------------------
        # Category-conditioned C experts
        # -------------------------------------------------------------
        (
            cg,
            cgs,
            cex,
            cstats,
        ) = train_category_ls_experts(
            C,
            y,
            categories,
            itr,
            C_TOPK_CAT,
            C_TOPK_GLOBAL,
            185000 + outer_fold * 100 + fold,
        )

        qc[b] = predict_category_experts(
            C,
            categories,
            iva,
            cg,
            cgs,
            cex,
        )

        cat_expert_count_c.append(
            sum(
                int(z["category_expert"])
                for z in cstats.values()
            )
        )

        cs[b] = v5.c_small_matrix(
            c_df,
            iva,
        )

    # -------------------------------------------------------------
    # Cross-fitted V+O
    # -------------------------------------------------------------
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
        seed=186000 + outer_fold,
    )

    diagnostics = {
        "r_category_experts_mean_inner": float(
            np.mean(cat_expert_count_r)
        ),
        "c_category_experts_mean_inner": float(
            np.mean(cat_expert_count_c)
        ),
    }

    return (
        pv,
        po,
        pvo,
        qr,
        qc,
        cs,
        diagnostics,
    )


# =============================================================================
# Outer-test category-conditioned evidence
# =============================================================================

def outer_test_category_signals(
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

    # V/O
    mv = base.model_factory(
        v_kind,
        191000 + outer_fold,
    )

    mo = base.model_factory(
        o_kind,
        192000 + outer_fold,
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

    # R category experts
    (
        rg,
        rgs,
        rex,
        rstats,
    ) = train_category_ls_experts(
        R,
        y,
        categories,
        outer_train,
        R_TOPK_CAT,
        R_TOPK_GLOBAL,
        193000 + outer_fold,
    )

    qr = predict_category_experts(
        R,
        categories,
        outer_test,
        rg,
        rgs,
        rex,
    )

    # C category experts
    (
        cg,
        cgs,
        cex,
        cstats,
    ) = train_category_ls_experts(
        C,
        y,
        categories,
        outer_train,
        C_TOPK_CAT,
        C_TOPK_GLOBAL,
        194000 + outer_fold,
    )

    qc = predict_category_experts(
        C,
        categories,
        outer_test,
        cg,
        cgs,
        cex,
    )

    cs = v5.c_small_matrix(
        c_df,
        outer_test,
    )

    diagnostics = {
        "r_category_experts_outer": int(
            sum(
                int(z["category_expert"])
                for z in rstats.values()
            )
        ),
        "c_category_experts_outer": int(
            sum(
                int(z["category_expert"])
                for z in cstats.values()
            )
        ),
    }

    return (
        pv,
        po,
        qr,
        qc,
        cs,
        diagnostics,
    )


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 120)
    print("ODRC V6 — CATEGORY-CONDITIONED R/C EXPERTS — DEVELOPMENT ONLY")
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
        # Strict OOF
        # ---------------------------------------------------------------------
        (
            pv_oof,
            po_oof,
            pvo_oof,
            qr_oof,
            qc_oof,
            cs_oof,
            diag_inner,
        ) = strict_oof_category_signals(
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

        idx_an = np.where(
            np.isin(
                ytr,
                ["logical", "structural"],
            )
        )[0]

        # ---------------------------------------------------------------------
        # R conditional stack
        # ---------------------------------------------------------------------
        XR_oof = v5.r_stack_features(
            pvo_oof,
            qr_oof,
        )

        qR_stack_oof = v5.crossfit_anomaly_stack(
            XR_oof,
            ytr,
            seed=201000 + outer_fold,
        )

        pR_oof = v5.preserve_normal_set_ls(
            pvo_oof,
            qR_stack_oof,
        )

        r_stack = v5.stack_factory(
            202000 + outer_fold
        )

        r_stack.fit(
            XR_oof[idx_an],
            (
                ytr[idx_an]
                == "structural"
            ).astype(int),
        )

        # ---------------------------------------------------------------------
        # C conditional stack
        # ---------------------------------------------------------------------
        XC_oof = v5.c_stack_features(
            pR_oof,
            qc_oof,
            cs_oof,
        )

        qC_stack_oof = v5.crossfit_anomaly_stack(
            XC_oof,
            ytr,
            seed=203000 + outer_fold,
        )

        pC_oof = v5.preserve_normal_set_ls(
            pR_oof,
            qC_stack_oof,
        )

        c_stack = v5.stack_factory(
            204000 + outer_fold
        )

        c_stack.fit(
            XC_oof[idx_an],
            (
                ytr[idx_an]
                == "structural"
            ).astype(int),
        )

        # ---------------------------------------------------------------------
        # Outer test signals
        # ---------------------------------------------------------------------
        (
            pv_te,
            po_te,
            qr_te,
            qc_te,
            cs_te,
            diag_outer,
        ) = outer_test_category_signals(
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

        # Final V+O fit
        fusion_vo = base.fusion_factory(
            205000 + outer_fold
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

        # R
        XR_te = v5.r_stack_features(
            pvo_te,
            qr_te,
        )

        qR_te = v5.binary_proba(
            r_stack,
            XR_te,
        )

        pR_te = v5.preserve_normal_set_ls(
            pvo_te,
            qR_te,
        )

        # C
        XC_te = v5.c_stack_features(
            pR_te,
            qc_te,
            cs_te,
        )

        qC_te = v5.binary_proba(
            c_stack,
            XC_te,
        )

        pC_te = v5.preserve_normal_set_ls(
            pR_te,
            qC_te,
        )

        # V reference
        mv = base.model_factory(
            v_kind,
            206000 + outer_fold,
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
                v5.pred_from_prob(prob),
            )

            fold_metrics[name] = m

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
            "r_topk_cat": R_TOPK_CAT,
            "c_topk_cat": C_TOPK_CAT,
            "r_topk_global": R_TOPK_GLOBAL,
            "c_topk_global": C_TOPK_GLOBAL,
            "min_per_ls_class": MIN_PER_LS_CLASS,
            "r_category_experts_mean_inner": diag_inner[
                "r_category_experts_mean_inner"
            ],
            "c_category_experts_mean_inner": diag_inner[
                "c_category_experts_mean_inner"
            ],
            "r_category_experts_outer": diag_outer[
                "r_category_experts_outer"
            ],
            "c_category_experts_outer": diag_outer[
                "c_category_experts_outer"
            ],
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
    result = pd.DataFrame(rows)
    delta_df = pd.DataFrame(delta_rows)
    settings_df = pd.DataFrame(setting_rows)

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
            result["variant"] == variant
        ]

        row = {
            "variant": variant,
            "n_folds": len(g),
        }

        for c in metric_cols:
            row[f"{c}_mean"] = float(
                g[c].mean()
            )

            row[f"{c}_std"] = float(
                g[c].std(ddof=0)
            )

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

    settings_df.to_csv(
        OUT_SETTINGS,
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "status": "DEVELOPMENT_ONLY",
        "design": {
            "VO": "stable three-class low-dimensional stack",
            "R": "category-conditioned anomaly-only L/S relation expert",
            "C": "category-conditioned anomaly-only L/S constraint expert",
            "normal_probability_preserved_by_R": True,
            "normal_probability_preserved_by_C": True,
            "r_topk_cat": R_TOPK_CAT,
            "c_topk_cat": C_TOPK_CAT,
            "r_topk_global": R_TOPK_GLOBAL,
            "c_topk_global": C_TOPK_GLOBAL,
            "min_per_ls_class": MIN_PER_LS_CLASS,
            "hyperparameter_search": False,
        },
        "important": (
            "The outer folds were already observed in prior development. "
            "Use these results only to judge whether category-conditioned "
            "R/C evidence is viable."
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
    print("ODRC V6 — DEVELOPMENT MEAN ± STD")
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
    print("CATEGORY EXPERT DIAGNOSTICS")
    print("-" * 120)
    print(
        settings_df.to_string(
            index=False
        )
    )

    print()
    print(
        "PASS target: fixed V6 should give positive mean R_vs_VO and "
        "C_vs_VOR in BAcc/Macro-F1, with Structural-F1 non-negative."
    )
    print("=" * 120)


if __name__ == "__main__":
    main()
