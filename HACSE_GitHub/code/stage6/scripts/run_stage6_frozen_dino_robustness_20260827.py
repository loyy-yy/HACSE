#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
Frozen V+O+DINO — 20260827 Robustness Confirmation
===================================================

SCIENTIFIC LABEL
----------------
Post-development robustness confirmation on a different 5-fold partition
of the SAME 1568 MVTec LOCO AD samples.

This is NOT untouched independent validation.

Frozen before this run
----------------------
V model family      : RandomForest ("rf")
O model family      : ExtraTrees ("et")
DINO representation : frozen whole-image DINOv2-S/14 cache
DINO probe          : StandardScaler + balanced LogisticRegression(C=1)
Fusion              : StandardScaler + balanced LogisticRegression(C=1)
No threshold/gate/category routing
No R/C/TAFA
No hyperparameter tuning from this confirmation

Partition
---------
StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=20260827
)
Strata = category || N/L/S label.

Compared variants
-----------------
    V
    V+O
    V+DINO
    V+O+DINO        <-- FROZEN PRIMARY METHOD

Success criteria
----------------
The purpose is confirmation, not new model selection.

We will report:
- mean±std Acc/BAcc/Macro-F1/N-L-S F1
- paired fold deltas of V+O+DINO vs V+O
- number of positive/zero/negative folds
- pooled category metrics
- rescue/harm transitions
- exact two-sided McNemar p-value

Interpretation:
- If V+O+DINO remains positive on >=4/5 folds for Macro-F1 and BAcc,
  with positive mean Macro-F1 and Structural-F1 deltas, the development
  conclusion is considered robust under repartitioning.
- Do NOT tune the frozen method after seeing this result.

Run
---
cd code/stage6
python scripts\run_stage6_frozen_dino_robustness_20260827.py
"""

from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# =============================================================================
# Import frozen project implementation
# =============================================================================

HERE = Path(__file__).resolve().parent
V11_SCRIPT = HERE / "evaluate_stage6_odrc_v11_orthogonal_residual_dev.py"

if not V11_SCRIPT.exists():
    raise FileNotFoundError(f"Missing prerequisite:\n{V11_SCRIPT}")


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


v11 = import_module(V11_SCRIPT, "v11_frozen_dino_confirm")
v6 = v11.v6
base = v11.base


# =============================================================================
# Frozen paths/settings
# =============================================================================

ROOT = Path(__file__).resolve().parents[2]

FEATURE_FILE = ROOT / "full_loco_salad_csad_fusion_analysis.xlsx"

DINO_CACHE = (
    ROOT
    / "stage6"
    / "outputs"
    / "stage6_dino_global_v1_ceiling_dev"
    / "dino_global_v1_features.npz"
)

OUT_DIR = (
    ROOT
    / "stage6"
    / "outputs"
    / "stage6_frozen_dino_robustness_20260827"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_ASSIGN = OUT_DIR / "outer_fold_assignments_20260827.csv"
OUT_FOLD = OUT_DIR / "frozen_dino_by_fold.csv"
OUT_MEAN = OUT_DIR / "frozen_dino_mean_std.csv"
OUT_DELTA = OUT_DIR / "frozen_dino_deltas.csv"
OUT_CAT = OUT_DIR / "frozen_dino_category_pooled.csv"
OUT_PRED = OUT_DIR / "frozen_dino_oof_predictions.csv"
OUT_TRANS = OUT_DIR / "frozen_dino_transition_summary.csv"
OUT_JSON = OUT_DIR / "frozen_dino_summary.json"

N_EXPECTED = 1568
N_SPLITS = 5
OUTER_RANDOM_STATE = 20260827

V_MODEL = "rf"
O_MODEL = "et"

CLASS_ORDER = list(base.CLASS_ORDER)

VARIANTS = [
    "V",
    "V+O",
    "V+DINO",
    "V+O+DINO",
]


# =============================================================================
# Fixed helpers
# =============================================================================

def global_strata(categories, y):
    return np.asarray(
        [
            f"{categories.iloc[i]}||{y[i]}"
            for i in range(len(y))
        ],
        dtype=object,
    )


def pred_from_prob(p):
    return np.asarray(
        CLASS_ORDER,
        dtype=object,
    )[np.argmax(p, axis=1)]


def aligned_proba_local(model, X):
    p = model.predict_proba(X)

    out = np.zeros(
        (len(X), len(CLASS_ORDER)),
        dtype=np.float64,
    )

    for j, c in enumerate(model.classes_):
        if c in CLASS_ORDER:
            out[:, CLASS_ORDER.index(c)] = p[:, j]

    denom = out.sum(axis=1, keepdims=True)
    denom[denom <= 0] = 1.0
    return out / denom


def linear_factory(seed):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=4000,
            class_weight="balanced",
            C=1.0,
            random_state=seed,
        ),
    )


def meta_v_dino(pv, pdino):
    return np.concatenate(
        [pv, pdino],
        axis=1,
    ).astype(np.float32)


def meta_vo_dino(pv, po, pdino):
    return np.concatenate(
        [pv, po, pdino],
        axis=1,
    ).astype(np.float32)


def strict_dino_oof(D, y, categories, outer_train, fold):
    strata = base.make_strata(
        outer_train,
        categories,
        y,
    )

    skf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=881000 + fold,
    )

    out = np.zeros(
        (len(outer_train), len(CLASS_ORDER)),
        dtype=np.float64,
    )

    for inner_fold, (a, b) in enumerate(
        skf.split(outer_train, strata),
        start=1,
    ):
        itr = outer_train[a]
        iva = outer_train[b]

        model = linear_factory(
            882000 + fold * 100 + inner_fold
        )

        model.fit(
            D[itr],
            y[itr],
        )

        out[b] = aligned_proba_local(
            model,
            D[iva],
        )

    return out


def fit_predict_dino_outer(D, y, outer_train, outer_test, fold):
    model = linear_factory(
        883000 + fold
    )

    model.fit(
        D[outer_train],
        y[outer_train],
    )

    return aligned_proba_local(
        model,
        D[outer_test],
    )


def summarize(result):
    metrics = [
        "acc",
        "bacc",
        "macro_f1",
        "normal_f1",
        "logical_f1",
        "structural_f1",
    ]

    rows = []

    for name in VARIANTS:
        g = result[result["variant"] == name]

        row = {
            "variant": name,
            "n_folds": int(len(g)),
        }

        for m in metrics:
            row[f"{m}_mean"] = float(g[m].mean())
            row[f"{m}_std"] = float(g[m].std(ddof=0))

        rows.append(row)

    return pd.DataFrame(rows)


def sign_counts(values, tol=1e-12):
    x = np.asarray(values, dtype=float)

    return {
        "positive": int((x > tol).sum()),
        "zero": int((np.abs(x) <= tol).sum()),
        "negative": int((x < -tol).sum()),
    }


# =============================================================================
# Main
# =============================================================================

def main():
    t0 = time.perf_counter()

    print("=" * 160)
    print("FROZEN V+O+DINO — 20260827 ROBUSTNESS CONFIRMATION")
    print("=" * 160)
    print(
        "Scientific label: repartition robustness confirmation on the SAME "
        "1568 benchmark samples; NOT untouched independent validation."
    )
    print(
        f"Frozen model families: V={V_MODEL}, O={O_MODEL}; "
        f"outer random_state={OUTER_RANDOM_STATE}"
    )

    for p in [
        FEATURE_FILE,
        DINO_CACHE,
        base.O_FILE,
        base.O_RAW,
    ]:
        if not p.exists():
            raise FileNotFoundError(p)

    # -------------------------------------------------------------------------
    # Benchmark
    # -------------------------------------------------------------------------
    df = pd.read_excel(
        FEATURE_FILE,
        sheet_name="merged_data",
    )

    df = df.loc[
        :,
        ~df.columns.duplicated(),
    ].copy().reset_index(drop=True)

    if len(df) != N_EXPECTED:
        raise RuntimeError(
            f"Expected {N_EXPECTED} rows, got {len(df)}"
        )

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

    V = df[v_cols].to_numpy(
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

    _, _, O = base.load_o()
    recs = base.load_raw()

    # -------------------------------------------------------------------------
    # Frozen whole-image DINO cache
    # -------------------------------------------------------------------------
    z = np.load(
        DINO_CACHE,
        allow_pickle=True,
    )

    D = z["embeddings"].astype(np.float32)
    sample_keys = z["sample_keys"].astype(str).tolist()
    dino_model = str(z["model_name"].item())

    if D.shape[0] != N_EXPECTED:
        raise RuntimeError(
            f"DINO rows {D.shape[0]} != {N_EXPECTED}"
        )

    print(
        "DINO model:",
        dino_model,
        "| feature shape:",
        D.shape,
    )

    # -------------------------------------------------------------------------
    # Reproduce the fixed 20260827 repartition WITHOUT reading prior results.
    # -------------------------------------------------------------------------
    strata = global_strata(
        categories,
        y,
    )

    skf = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=OUTER_RANDOM_STATE,
    )

    dummy = np.zeros(
        len(y),
        dtype=np.uint8,
    )

    folds = np.zeros(
        len(y),
        dtype=int,
    )

    split_list = []

    for fold, (tr, te) in enumerate(
        skf.split(dummy, strata),
        start=1,
    ):
        tr = tr.astype(int)
        te = te.astype(int)

        folds[te] = fold
        split_list.append(
            (fold, tr, te)
        )

    assign_df = pd.DataFrame(
        {
            "row_index": np.arange(
                len(y),
                dtype=int,
            ),
            "sample_key": sample_keys,
            "category": categories,
            "label": y,
            "outer_test_fold": folds,
        }
    )

    assign_df.to_csv(
        OUT_ASSIGN,
        index=False,
        encoding="utf-8-sig",
    )

    # -------------------------------------------------------------------------
    # OOF holders across confirmation folds
    # -------------------------------------------------------------------------
    oof_prob = {
        name: np.full(
            (
                len(y),
                len(CLASS_ORDER),
            ),
            np.nan,
            dtype=np.float64,
        )
        for name in VARIANTS
    }

    fold_rows = []
    delta_rows = []

    # =========================================================================
    # Confirmation outer folds
    # =========================================================================
    for fold, outer_train, outer_test in split_list:
        print()
        print("-" * 160)
        print(
            f"ROBUSTNESS FOLD {fold}/{N_SPLITS} "
            f"(train={len(outer_train)}, test={len(outer_test)})"
        )

        # ---------------------------------------------------------------------
        # Strict V/O OOF under fixed model families
        # ---------------------------------------------------------------------
        (
            pv_oof,
            po_oof,
            _pvo_unused,
            _qr,
            _qc,
            _cs,
            _,
        ) = v6.strict_oof_category_signals(
            V,
            O,
            recs,
            y,
            categories,
            outer_train,
            800 + fold,
            V_MODEL,
            O_MODEL,
        )

        ytr = y[outer_train]

        # Strict DINO OOF
        pdino_oof = strict_dino_oof(
            D,
            y,
            categories,
            outer_train,
            fold,
        )

        # ---------------------------------------------------------------------
        # Fixed V/O outer test signals
        # ---------------------------------------------------------------------
        (
            pv_te,
            po_te,
            _qr_te,
            _qc_te,
            _cs_te,
            _,
        ) = v6.outer_test_category_signals(
            V,
            O,
            recs,
            y,
            categories,
            outer_train,
            outer_test,
            800 + fold,
            V_MODEL,
            O_MODEL,
        )

        # Frozen DINO outer-test probe
        pdino_te = fit_predict_dino_outer(
            D,
            y,
            outer_train,
            outer_test,
            fold,
        )

        # ---------------------------------------------------------------------
        # V baseline
        # ---------------------------------------------------------------------
        mv = base.model_factory(
            V_MODEL,
            884000 + fold,
        )

        mv.fit(
            V[outer_train],
            y[outer_train],
        )

        pV_te = base.aligned_proba(
            mv,
            V[outer_test],
        )

        # ---------------------------------------------------------------------
        # V+O frozen stack
        # ---------------------------------------------------------------------
        f_vo = base.fusion_factory(
            885000 + fold
        )

        f_vo.fit(
            base.meta_vo(
                pv_oof,
                po_oof,
            ),
            ytr,
        )

        pVO_te = base.aligned_proba(
            f_vo,
            base.meta_vo(
                pv_te,
                po_te,
            ),
        )

        # ---------------------------------------------------------------------
        # V+DINO fixed stack
        # ---------------------------------------------------------------------
        f_vd = linear_factory(
            886000 + fold
        )

        f_vd.fit(
            meta_v_dino(
                pv_oof,
                pdino_oof,
            ),
            ytr,
        )

        pVD_te = aligned_proba_local(
            f_vd,
            meta_v_dino(
                pv_te,
                pdino_te,
            ),
        )

        # ---------------------------------------------------------------------
        # Frozen primary V+O+DINO
        # ---------------------------------------------------------------------
        f_vod = linear_factory(
            887000 + fold
        )

        f_vod.fit(
            meta_vo_dino(
                pv_oof,
                po_oof,
                pdino_oof,
            ),
            ytr,
        )

        pVOD_te = aligned_proba_local(
            f_vod,
            meta_vo_dino(
                pv_te,
                po_te,
                pdino_te,
            ),
        )

        probs = {
            "V": pV_te,
            "V+O": pVO_te,
            "V+DINO": pVD_te,
            "V+O+DINO": pVOD_te,
        }

        fold_metrics = {}

        for name in VARIANTS:
            p = probs[name]

            m = base.metrics(
                y[outer_test],
                pred_from_prob(p),
            )

            fold_metrics[name] = m

            fold_rows.append(
                {
                    "fold": fold,
                    "variant": name,
                    **m,
                }
            )

            oof_prob[name][outer_test] = p

            print(
                f"{name:<12} "
                f"Acc={m['acc']:.4f} "
                f"BAcc={m['bacc']:.4f} "
                f"MacroF1={m['macro_f1']:.4f} "
                f"N/L/S={m['normal_f1']:.4f}/"
                f"{m['logical_f1']:.4f}/"
                f"{m['structural_f1']:.4f}"
            )

        b = fold_metrics["V+O"]
        z = fold_metrics["V+O+DINO"]

        delta_rows.append(
            {
                "fold": fold,
                "delta_acc": z["acc"] - b["acc"],
                "delta_bacc": z["bacc"] - b["bacc"],
                "delta_macro_f1": z["macro_f1"] - b["macro_f1"],
                "delta_normal_f1": z["normal_f1"] - b["normal_f1"],
                "delta_logical_f1": z["logical_f1"] - b["logical_f1"],
                "delta_structural_f1": (
                    z["structural_f1"]
                    - b["structural_f1"]
                ),
            }
        )

    # =========================================================================
    # Integrity
    # =========================================================================
    for name in VARIANTS:
        if np.isnan(oof_prob[name]).any():
            raise RuntimeError(
                f"Missing OOF confirmation predictions for {name}"
            )

    # =========================================================================
    # Aggregate
    # =========================================================================
    fold_df = pd.DataFrame(fold_rows)
    mean_df = summarize(fold_df)
    delta_df = pd.DataFrame(delta_rows)

    # -------------------------------------------------------------------------
    # Per-sample pooled predictions
    # -------------------------------------------------------------------------
    pred_df = assign_df.copy()

    for name in VARIANTS:
        pred_df[
            "pred_" + name.replace("+", "_")
        ] = pred_from_prob(
            oof_prob[name]
        )

    old_pred = pred_from_prob(
        oof_prob["V+O"]
    )

    new_pred = pred_from_prob(
        oof_prob["V+O+DINO"]
    )

    gt = y

    changed = old_pred != new_pred
    rescue = (
        changed
        & (old_pred != gt)
        & (new_pred == gt)
    )
    harm = (
        changed
        & (old_pred == gt)
        & (new_pred != gt)
    )
    wrong2wrong = (
        changed
        & (old_pred != gt)
        & (new_pred != gt)
    )

    transitions = {
        "changed": int(changed.sum()),
        "rescues": int(rescue.sum()),
        "harms": int(harm.sum()),
        "wrong_to_wrong": int(wrong2wrong.sum()),
        "net_correct": int(
            rescue.sum()
            - harm.sum()
        ),
    }

    # Exact paired McNemar test:
    # discordant pairs are b=rescues, c=harms.
    discordant = (
        transitions["rescues"]
        + transitions["harms"]
    )

    if discordant > 0:
        mcnemar_p = float(
            binomtest(
                min(
                    transitions["rescues"],
                    transitions["harms"],
                ),
                n=discordant,
                p=0.5,
                alternative="two-sided",
            ).pvalue
        )
    else:
        mcnemar_p = 1.0

    transitions["mcnemar_exact_two_sided_p"] = mcnemar_p

    # -------------------------------------------------------------------------
    # Category pooled
    # -------------------------------------------------------------------------
    cat_rows = []

    for name in [
        "V+O",
        "V+DINO",
        "V+O+DINO",
    ]:
        pcol = (
            "pred_"
            + name.replace("+", "_")
        )

        for cat in sorted(
            categories.unique()
        ):
            g = pred_df[
                pred_df["category"] == cat
            ]

            m = base.metrics(
                g["label"].to_numpy(
                    dtype=object
                ),
                g[pcol].to_numpy(
                    dtype=object
                ),
            )

            cat_rows.append(
                {
                    "variant": name,
                    "category": cat,
                    "n": int(len(g)),
                    **m,
                }
            )

    cat_df = pd.DataFrame(cat_rows)

    # -------------------------------------------------------------------------
    # Stability counts
    # -------------------------------------------------------------------------
    stability = {
        "acc": sign_counts(
            delta_df["delta_acc"]
        ),
        "bacc": sign_counts(
            delta_df["delta_bacc"]
        ),
        "macro_f1": sign_counts(
            delta_df["delta_macro_f1"]
        ),
        "structural_f1": sign_counts(
            delta_df["delta_structural_f1"]
        ),
    }

    confirm_pass = bool(
        stability["macro_f1"]["positive"] >= 4
        and stability["bacc"]["positive"] >= 4
        and float(
            delta_df["delta_macro_f1"].mean()
        ) > 0
        and float(
            delta_df["delta_structural_f1"].mean()
        ) > 0
    )

    decision = (
        "FROZEN_DINO_ROBUSTNESS_CONFIRMED"
        if confirm_pass
        else "FROZEN_DINO_ROBUSTNESS_NOT_CONFIRMED"
    )

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------
    fold_df.to_csv(
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

    cat_df.to_csv(
        OUT_CAT,
        index=False,
        encoding="utf-8-sig",
    )

    pred_df.to_csv(
        OUT_PRED,
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        [transitions]
    ).to_csv(
        OUT_TRANS,
        index=False,
        encoding="utf-8-sig",
    )

    report = {
        "status": "POST_DEVELOPMENT_ROBUSTNESS_CONFIRMATION",
        "scientific_label": (
            "Different 5-fold partition on the same 1568 benchmark samples; "
            "not untouched independent validation."
        ),
        "decision": decision,
        "elapsed_seconds": float(
            time.perf_counter() - t0
        ),
        "outer_random_state": OUTER_RANDOM_STATE,
        "frozen": {
            "V_model": V_MODEL,
            "O_model": O_MODEL,
            "DINO_model": dino_model,
            "DINO_feature_shape": list(D.shape),
            "DINO_probe": (
                "StandardScaler + balanced LogisticRegression(C=1)"
            ),
            "fusion": (
                "StandardScaler + balanced LogisticRegression(C=1)"
            ),
            "threshold_or_gate": None,
            "TAFA_used": False,
            "R_used": False,
            "C_used": False,
        },
        "mean_results": mean_df.to_dict(
            orient="records"
        ),
        "mean_deltas_VOD_vs_VO": {
            "acc": float(
                delta_df["delta_acc"].mean()
            ),
            "bacc": float(
                delta_df["delta_bacc"].mean()
            ),
            "macro_f1": float(
                delta_df["delta_macro_f1"].mean()
            ),
            "normal_f1": float(
                delta_df["delta_normal_f1"].mean()
            ),
            "logical_f1": float(
                delta_df["delta_logical_f1"].mean()
            ),
            "structural_f1": float(
                delta_df["delta_structural_f1"].mean()
            ),
        },
        "fold_stability_VOD_vs_VO": stability,
        "transitions_VOD_vs_VO": transitions,
        "confirmation_rule": {
            "macro_f1_positive_folds_at_least": 4,
            "bacc_positive_folds_at_least": 4,
            "mean_macro_f1_delta_positive": True,
            "mean_structural_f1_delta_positive": True,
        },
        "no_post_confirmation_tuning": True,
    }

    OUT_JSON.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # -------------------------------------------------------------------------
    # Console
    # -------------------------------------------------------------------------
    print()
    print("=" * 160)
    print("FROZEN DINO ROBUSTNESS CONFIRMATION — MEAN ± STD")
    print("=" * 160)
    print(
        mean_df.to_string(
            index=False
        )
    )

    print()
    print("V+O+DINO vs V+O — PAIRED ROBUSTNESS-FOLD DELTAS")
    print("-" * 160)
    print(
        delta_df.to_string(
            index=False
        )
    )

    print()
    print("FOLD STABILITY")
    print("-" * 160)
    print(
        json.dumps(
            stability,
            ensure_ascii=False,
            indent=2,
        )
    )

    print()
    print("CATEGORY POOLED")
    print("-" * 160)
    print(
        cat_df.to_string(
            index=False
        )
    )

    print()
    print("TRANSITION / McNEMAR")
    print("-" * 160)
    print(
        json.dumps(
            transitions,
            ensure_ascii=False,
            indent=2,
        )
    )

    print()
    print("Decision:")
    print(decision)

    print()
    print("Outputs:")
    print(OUT_DIR)
    print("=" * 160)


if __name__ == "__main__":
    main()
