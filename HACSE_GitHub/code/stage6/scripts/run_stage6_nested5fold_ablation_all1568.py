#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Stage6 Final 5-Fold Nested-CV Ablation — Unified LogReg
=======================================================

Purpose
-------
Use ALL 1568 strict MVTec LOCO three-class samples in the final evaluation.
Every sample is used as OUTER-TEST exactly once.

Final ablation:
    V
    V + O
    V + O + R
    V + O + R + C

Protocol
--------
OUTER 5-fold stratified CV:
    ~80% outer-train
    ~20% outer-test

INNER 5-fold CV inside each outer-train:
    - select base model families for V / O / R / C
    - generate OOF meta-features for the fixed unified fusion head
    - no outer-test sample is used for any fitting or selection

R / C provenance:
    - R semantic palette is learned from NORMAL samples of the relevant
      training partition only
    - C robust normal-relation prototype is learned from NORMAL samples
      of the relevant training partition only
    - for INNER OOF generation, R/C are rebuilt from each inner-train
      normal subset, preventing prototype leakage into inner validation
    - for OUTER test inference, R/C are rebuilt from the full outer-train
      normal subset only

Fusion:
    V+O       : fixed Logistic Regression over low-dimensional meta-features
    V+O+R     : fixed Logistic Regression over low-dimensional meta-features
    V+O+R+C   : fixed Logistic Regression over low-dimensional meta-features

No alpha/beta/gamma/tau residual chain.
No MLP-vs-LogReg selection.
No direct high-dimensional concatenation.

Important
---------
This is a NEW evaluation protocol. Once you inspect its OUTER-test results,
do NOT tune the framework and rerun it as if it were still confirmatory.

Inputs
------
code/full_loco_salad_csad_fusion_analysis.xlsx
code/stage6/outputs/stage6_o_v2/stage6_o_v2_features.csv
code/stage6/outputs/stage6_o_v2/stage6_o_v2_raw_components.jsonl

Outputs
-------
stage6/outputs/stage6_nested5fold_ablation/
    outer_fold_assignments.csv
    nested5fold_ablation_by_fold.csv
    nested5fold_ablation_mean_std.csv
    nested5fold_ablation_oof_predictions.csv
    nested5fold_selected_settings.csv
    nested5fold_paired_deltas.csv
    nested5fold_summary.json
"""

from __future__ import annotations

import json
import math
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
    GradientBoostingClassifier,
)
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# =============================================================================
# Paths / frozen design
# =============================================================================

ROOT = Path(__file__).resolve().parents[2]

FEATURE_FILE = ROOT / "full_loco_salad_csad_fusion_analysis.xlsx"
O_FILE = (
    ROOT / "stage6" / "outputs" / "stage6_o_v2"
    / "stage6_o_v2_features.csv"
)
O_RAW = (
    ROOT / "stage6" / "outputs" / "stage6_o_v2"
    / "stage6_o_v2_raw_components.jsonl"
)

OUT_DIR = ROOT / "stage6" / "outputs" / "stage6_nested5fold_ablation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_ASSIGN = OUT_DIR / "outer_fold_assignments.csv"
OUT_FOLD = OUT_DIR / "nested5fold_ablation_by_fold.csv"
OUT_MEAN = OUT_DIR / "nested5fold_ablation_mean_std.csv"
OUT_OOF = OUT_DIR / "nested5fold_ablation_oof_predictions.csv"
OUT_SETTINGS = OUT_DIR / "nested5fold_selected_settings.csv"
OUT_DELTA = OUT_DIR / "nested5fold_paired_deltas.csv"
OUT_SUMMARY = OUT_DIR / "nested5fold_summary.json"

CLASS_ORDER = ["normal", "logical", "structural"]

OUTER_FOLDS = 5
INNER_FOLDS = 5
OUTER_RANDOM_STATE = 20260826
INNER_RANDOM_STATE = 20260827

R_TOPK = 64
C_TOPK = 32

PALETTE_MIN_PREVALENCE = 0.80
MAX_PALETTE_COLORS = 8
MAX_SEMANTIC_SLOTS = 8

Z_CLIP = 12.0
MIN_SCALE = 1e-6

MULTI_MODELS = ["et", "rf", "log", "gb", "mlp"]
BIN_MODELS = ["et", "rf", "log", "gb"]

C_AGG_CANDIDATES = [
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

def safe_str(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def infer_gt_group(row):
    if "gt_group" in row.index and pd.notna(row.get("gt_group")):
        return safe_str(row.get("gt_group"))

    defect = safe_str(row.get("defect_class", "")).lower()

    if defect == "good":
        return "normal"
    if "logical" in defect:
        return "logical"
    if "structural" in defect:
        return "structural"

    return defect if defect else "unknown"


def sf(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if math.isfinite(v):
            return v
    except Exception:
        pass
    return default


def metrics(y_true, y_pred):
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "bacc": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=CLASS_ORDER,
                average="macro",
                zero_division=0,
            )
        ),
        "normal_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=["normal"],
                average="macro",
                zero_division=0,
            )
        ),
        "logical_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=["logical"],
                average="macro",
                zero_division=0,
            )
        ),
        "structural_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=["structural"],
                average="macro",
                zero_division=0,
            )
        ),
    }


def model_factory(kind: str, seed: int):
    if kind == "et":
        return ExtraTreesClassifier(
            n_estimators=500,
            random_state=seed,
            class_weight="balanced",
            min_samples_leaf=1,
            n_jobs=-1,
        )

    if kind == "rf":
        return RandomForestClassifier(
            n_estimators=400,
            random_state=seed,
            class_weight="balanced",
            min_samples_leaf=1,
            n_jobs=-1,
        )

    if kind == "log":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                random_state=seed,
            ),
        )

    if kind == "gb":
        return GradientBoostingClassifier(
            n_estimators=120,
            learning_rate=0.05,
            max_depth=2,
            random_state=seed,
        )

    if kind == "mlp":
        return make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(64, 24),
                alpha=1e-3,
                learning_rate_init=7e-4,
                max_iter=500,
                random_state=seed,
            ),
        )

    raise ValueError(kind)


def fusion_factory(seed: int):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            C=1.0,
            random_state=seed,
        ),
    )


def aligned_proba(model, X):
    p = model.predict_proba(X)
    out = np.zeros((len(X), 3), dtype=np.float64)

    for j, c in enumerate(model.classes_):
        if c in CLASS_ORDER:
            out[:, CLASS_ORDER.index(c)] = p[:, j]

    s = out.sum(axis=1, keepdims=True)
    s[s <= 0] = 1.0

    return out / s


def structural_proba(model, X):
    p = model.predict_proba(X)
    classes = list(model.classes_)

    if 1 not in classes:
        return np.zeros(len(X), dtype=np.float64)

    return p[:, classes.index(1)]


def train_rank(X, y_binary, k):
    with np.errstate(divide="ignore", invalid="ignore"):
        score, _ = f_classif(X, y_binary)

    score = np.nan_to_num(
        score,
        nan=-1e9,
        posinf=1e9,
        neginf=-1e9,
    )

    return np.argsort(-score)[:min(k, X.shape[1])]


def make_strata(indices, categories, y):
    return (
        categories.iloc[indices].astype(str).to_numpy()
        + "||"
        + y[indices].astype(str)
    )


# =============================================================================
# Load fixed O evidence and raw component evidence
# =============================================================================

def load_o():
    o = pd.read_csv(O_FILE, encoding="utf-8-sig")
    o = o.loc[:, ~o.columns.duplicated()].copy()
    o = o.sort_values("row_index").reset_index(drop=True)

    if len(o) != 1568:
        raise RuntimeError(f"O expected 1568 rows, got {len(o)}")

    if o["row_index"].astype(int).tolist() != list(range(1568)):
        raise RuntimeError("O row_index mismatch")

    id_cols = {
        "row_index",
        "category",
        "router_mode",
        "map_width",
        "map_height",
        "map_unique_values",
    }

    cols = [c for c in o.columns if c not in id_cols]

    for c in cols:
        o[c] = (
            pd.to_numeric(o[c], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )

    X = o[cols].to_numpy(dtype=np.float32)

    return o, cols, X


def load_raw():
    recs = []

    with O_RAW.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                recs.append(json.loads(line))

    recs = sorted(recs, key=lambda z: int(z["row_index"]))

    if len(recs) != 1568:
        raise RuntimeError(f"O raw expected 1568, got {len(recs)}")

    if [int(z["row_index"]) for z in recs] != list(range(1568)):
        raise RuntimeError("O raw row_index mismatch")

    return recs


# =============================================================================
# R: semantic-aligned relation evidence
# =============================================================================

def image_color_ids(rec):
    ids = set()

    for z in rec.get("components", {}).get("colors_by_area", []):
        cid = int(z.get("color_id", -1))
        frac = sf(z.get("area_frac"))

        if cid >= 0 and frac > 0:
            ids.add(cid)

    return ids


def learn_palette(recs, train_normal_idx, cat):
    ids = [
        i
        for i in train_normal_idx
        if str(recs[i]["category"]) == cat
    ]

    if not ids:
        raise RuntimeError(
            f"No train-normal samples for palette: {cat}"
        )

    presence = Counter()
    area_sum = Counter()

    for i in ids:
        presence.update(image_color_ids(recs[i]))

        for z in recs[i].get("components", {}).get("colors_by_area", []):
            cid = int(z.get("color_id", -1))
            frac = sf(z.get("area_frac"))

            if cid >= 0 and frac > 0:
                area_sum[cid] += frac

    cand = []

    for cid, count in presence.items():
        prevalence = count / len(ids)

        if prevalence >= PALETTE_MIN_PREVALENCE:
            mean_area = area_sum[cid] / max(1, count)
            cand.append(
                (cid, prevalence, mean_area)
            )

    cand.sort(
        key=lambda x: (-x[1], -x[2], x[0])
    )

    return [
        x[0]
        for x in cand[:MAX_PALETTE_COLORS]
    ]


def region_by_color(components):
    grouped = {}

    for r in components.get("regions", []):
        cid = int(r.get("color_id", -1))

        if cid >= 0:
            grouped.setdefault(cid, []).append(r)

    nodes = {}

    for cid, items in grouped.items():
        areas = np.asarray(
            [
                max(0.0, sf(r.get("area_frac")))
                for r in items
            ],
            dtype=float,
        )

        total = float(areas.sum())

        if total <= 1e-12:
            continue

        w = areas / total

        cxs = np.asarray(
            [sf(r.get("cx_norm")) for r in items]
        )

        cys = np.asarray(
            [sf(r.get("cy_norm")) for r in items]
        )

        x0s, x1s, y0s, y1s = [], [], [], []

        for r in items:
            cx = sf(r.get("cx_norm"))
            cy = sf(r.get("cy_norm"))
            bw = max(0.0, sf(r.get("bbox_w_norm")))
            bh = max(0.0, sf(r.get("bbox_h_norm")))

            x0s.append(cx - bw / 2.0)
            x1s.append(cx + bw / 2.0)
            y0s.append(cy - bh / 2.0)
            y1s.append(cy + bh / 2.0)

        nodes[cid] = {
            "present": 1.0,
            "area": total,
            "cx": float((w * cxs).sum()),
            "cy": float((w * cys).sum()),
            "bw": float(max(x1s) - min(x0s)),
            "bh": float(max(y1s) - min(y0s)),
            "aspect": float(
                (
                    w
                    * np.asarray(
                        [sf(r.get("aspect_ratio")) for r in items]
                    )
                ).sum()
            ),
        }

    return nodes


def bbox_iou(a, b):
    ax0 = a["cx"] - a["bw"] / 2.0
    ax1 = a["cx"] + a["bw"] / 2.0
    ay0 = a["cy"] - a["bh"] / 2.0
    ay1 = a["cy"] + a["bh"] / 2.0

    bx0 = b["cx"] - b["bw"] / 2.0
    bx1 = b["cx"] + b["bw"] / 2.0
    by0 = b["cy"] - b["bh"] / 2.0
    by1 = b["cy"] + b["bh"] / 2.0

    iw = max(
        0.0,
        min(ax1, bx1) - max(ax0, bx0),
    )

    ih = max(
        0.0,
        min(ay1, by1) - max(ay0, by0),
    )

    inter = iw * ih

    union = (
        a["bw"] * a["bh"]
        + b["bw"] * b["bh"]
        - inter
    )

    return inter / union if union > 1e-12 else 0.0


def pair_rel(a, b):
    dx = b["cx"] - a["cx"]
    dy = b["cy"] - a["cy"]

    return {
        "dx": dx,
        "dy": dy,
        "abs_dx": abs(dx),
        "abs_dy": abs(dy),
        "dist": float(math.sqrt(dx * dx + dy * dy)),
        "iou": bbox_iou(a, b),
        "same_row": float(
            abs(dy)
            <= 0.25 * max(a["bh"], b["bh"], 1e-6)
        ),
        "same_col": float(
            abs(dx)
            <= 0.25 * max(a["bw"], b["bw"], 1e-6)
        ),
        "lr_sign": float(np.sign(dx)),
        "ab_sign": float(np.sign(dy)),
        "log_area_ratio": float(
            math.log(
                max(b["area"], 1e-8)
                / max(a["area"], 1e-8)
            )
        ),
        "aspect_diff": abs(
            b["aspect"] - a["aspect"]
        ),
    }


def semantic_rel_features(components, palette):
    nodes = region_by_color(components)

    feats = {}
    slot_nodes = []

    for s in range(MAX_SEMANTIC_SLOTS):
        if s < len(palette) and palette[s] in nodes:
            n = nodes[palette[s]]
        else:
            n = {
                "present": 0.0,
                "area": 0.0,
                "cx": 0.0,
                "cy": 0.0,
                "bw": 0.0,
                "bh": 0.0,
                "aspect": 0.0,
            }

        slot_nodes.append(n)

        p = f"r4_node{s}"

        for k in [
            "present",
            "area",
            "cx",
            "cy",
            "bw",
            "bh",
            "aspect",
        ]:
            feats[f"{p}_{k}"] = n[k]

    valid_pairs = []
    pair_id = 0

    for i in range(MAX_SEMANTIC_SLOTS):
        for j in range(i + 1, MAX_SEMANTIC_SLOTS):
            p = f"r4_pair{pair_id:02d}"

            valid = float(
                i < len(palette)
                and j < len(palette)
                and slot_nodes[i]["present"] > 0
                and slot_nodes[j]["present"] > 0
            )

            feats[f"{p}_valid"] = valid

            if valid:
                rr = pair_rel(
                    slot_nodes[i],
                    slot_nodes[j],
                )
                valid_pairs.append(rr)

            else:
                rr = {
                    "dx": 0.0,
                    "dy": 0.0,
                    "abs_dx": 0.0,
                    "abs_dy": 0.0,
                    "dist": 0.0,
                    "iou": 0.0,
                    "same_row": 0.0,
                    "same_col": 0.0,
                    "lr_sign": 0.0,
                    "ab_sign": 0.0,
                    "log_area_ratio": 0.0,
                    "aspect_diff": 0.0,
                }

            for k, v in rr.items():
                feats[f"{p}_{k}"] = v

            pair_id += 1

    feats["r4_present_ratio"] = (
        sum(
            slot_nodes[i]["present"] > 0
            for i in range(len(palette))
        )
        / len(palette)
        if palette
        else 0.0
    )

    for name in [
        "abs_dx",
        "abs_dy",
        "dist",
        "iou",
        "same_row",
        "same_col",
        "aspect_diff",
    ]:
        vals = np.asarray(
            [z[name] for z in valid_pairs],
            dtype=float,
        )

        if len(vals) == 0:
            vals = np.asarray([0.0])

        feats[f"r4_{name}_mean"] = float(vals.mean())
        feats[f"r4_{name}_std"] = float(vals.std())
        feats[f"r4_{name}_max"] = float(vals.max())

    return feats


def pushpin_rel_features(components):
    slots = components.get("slots", [])

    by = {
        (int(s["row"]), int(s["col"])): s
        for s in slots
    }

    feats = {}

    def val(s, key):
        return sf(s.get(key))

    k = 0

    for r in range(3):
        for c in range(4):
            a = by.get((r, c), {})
            b = by.get((r, c + 1), {})

            feats[f"r4_grid_h{k:02d}_domdiff"] = abs(
                val(a, "dominant_fraction")
                - val(b, "dominant_fraction")
            )

            feats[f"r4_grid_h{k:02d}_entdiff"] = abs(
                val(a, "color_entropy")
                - val(b, "color_entropy")
            )

            feats[f"r4_grid_h{k:02d}_ncdiff"] = abs(
                val(a, "n_colors")
                - val(b, "n_colors")
            )

            k += 1

    k = 0

    for r in range(2):
        for c in range(5):
            a = by.get((r, c), {})
            b = by.get((r + 1, c), {})

            feats[f"r4_grid_v{k:02d}_domdiff"] = abs(
                val(a, "dominant_fraction")
                - val(b, "dominant_fraction")
            )

            feats[f"r4_grid_v{k:02d}_entdiff"] = abs(
                val(a, "color_entropy")
                - val(b, "color_entropy")
            )

            feats[f"r4_grid_v{k:02d}_ncdiff"] = abs(
                val(a, "n_colors")
                - val(b, "n_colors")
            )

            k += 1

    for r in range(3):
        dom = [
            val(
                by.get((r, c), {}),
                "dominant_fraction",
            )
            for c in range(5)
        ]

        ent = [
            val(
                by.get((r, c), {}),
                "color_entropy",
            )
            for c in range(5)
        ]

        feats[f"r4_grid_row{r}_dom_std"] = float(
            np.std(dom)
        )

        feats[f"r4_grid_row{r}_ent_std"] = float(
            np.std(ent)
        )

    for c in range(5):
        dom = [
            val(
                by.get((r, c), {}),
                "dominant_fraction",
            )
            for r in range(3)
        ]

        ent = [
            val(
                by.get((r, c), {}),
                "color_entropy",
            )
            for r in range(3)
        ]

        feats[f"r4_grid_col{c}_dom_std"] = float(
            np.std(dom)
        )

        feats[f"r4_grid_col{c}_ent_std"] = float(
            np.std(ent)
        )

    return feats


def build_r(recs, palettes):
    rows = []

    for rec in recs:
        cat = str(rec["category"])

        row = {
            "row_index": int(rec["row_index"]),
            "category": cat,
        }

        if cat == "pushpins":
            row.update(
                pushpin_rel_features(
                    rec.get("components", {})
                )
            )

        else:
            row.update(
                semantic_rel_features(
                    rec.get("components", {}),
                    palettes[cat],
                )
            )

        rows.append(row)

    r = (
        pd.DataFrame(rows)
        .sort_values("row_index")
        .reset_index(drop=True)
    )

    cols = [
        c
        for c in r.columns
        if c not in {"row_index", "category"}
    ]

    r[cols] = (
        r[cols]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    return (
        r,
        cols,
        r[cols].to_numpy(dtype=np.float32),
    )


# =============================================================================
# C: robust normal-relation violation evidence
# =============================================================================

def is_c_source(c):
    if c.startswith("r4_node"):
        return False

    return (
        c.startswith("r4_pair")
        or c.startswith("r4_grid_")
        or c.startswith("r4_abs_")
        or c.startswith("r4_dist_")
        or c.startswith("r4_iou_")
        or c.startswith("r4_same_")
        or c.startswith("r4_aspect_")
        or c == "r4_present_ratio"
    )


def robust_center_scale(x):
    x = np.asarray(x, dtype=float)

    center = float(np.median(x))

    mad = 1.4826 * float(
        np.median(
            np.abs(x - center)
        )
    )

    q25, q75 = np.percentile(
        x,
        [25, 75],
    )

    iqr = float(
        (q75 - q25) / 1.349
    )

    std = float(np.std(x))

    if mad > MIN_SCALE:
        scale = mad
    elif iqr > MIN_SCALE:
        scale = iqr
    elif std > MIN_SCALE:
        scale = std
    else:
        scale = 0.05

    return center, scale


def build_c(
    r,
    r_cols,
    categories,
    train_normal_idx,
):
    src = [
        c
        for c in r_cols
        if is_c_source(c)
    ]

    out = pd.DataFrame({
        "row_index": r["row_index"].astype(int),
        "category": r["category"].astype(str),
    })

    for cat in sorted(categories.unique()):
        cat_idx = np.where(
            categories.to_numpy() == cat
        )[0]

        tn = [
            i
            for i in train_normal_idx
            if categories.iloc[i] == cat
        ]

        if not tn:
            raise RuntimeError(
                f"No train-normal for C: {cat}"
            )

        zcols = []

        for col in src:
            x_cat = r.loc[
                cat_idx,
                col,
            ].to_numpy(dtype=float)

            if not np.any(
                np.abs(x_cat) > 1e-12
            ):
                continue

            center, scale = robust_center_scale(
                r.loc[
                    tn,
                    col,
                ].to_numpy(dtype=float)
            )

            signed = np.clip(
                (x_cat - center) / scale,
                -Z_CLIP,
                Z_CLIP,
            )

            absz = np.abs(signed)

            an = f"c_absz__{col}"
            sn = f"c_signedz__{col}"

            if an not in out.columns:
                out[an] = 0.0
                out[sn] = 0.0

            out.loc[
                cat_idx,
                an,
            ] = absz

            out.loc[
                cat_idx,
                sn,
            ] = signed

            zcols.append(absz)

        if zcols:
            z = np.column_stack(zcols)

            aggs = {
                "c_rel_mean_abs_z": z.mean(axis=1),
                "c_rel_median_abs_z": np.median(z, axis=1),
                "c_rel_p90_abs_z": np.percentile(z, 90, axis=1),
                "c_rel_max_abs_z": z.max(axis=1),
                "c_rel_frac_gt1": (z > 1).mean(axis=1),
                "c_rel_frac_gt2": (z > 2).mean(axis=1),
                "c_rel_frac_gt3": (z > 3).mean(axis=1),
                "c_rel_top3_mean": np.sort(
                    z,
                    axis=1,
                )[
                    :,
                    -min(3, z.shape[1]):,
                ].mean(axis=1),
                "c_rel_top5_mean": np.sort(
                    z,
                    axis=1,
                )[
                    :,
                    -min(5, z.shape[1]):,
                ].mean(axis=1),
            }

            for name, vals in aggs.items():
                if name not in out.columns:
                    out[name] = 0.0

                out.loc[
                    cat_idx,
                    name,
                ] = vals

    cols = [
        c
        for c in out.columns
        if c not in {"row_index", "category"}
    ]

    out[cols] = (
        out[cols]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    return (
        out,
        cols,
        out[cols].to_numpy(dtype=np.float32),
    )


# =============================================================================
# Low-dimensional meta features
# =============================================================================

def meta_vo(pv, po):
    return np.concatenate(
        [
            pv,
            po,
            pv - po,
            np.max(pv, axis=1, keepdims=True),
            np.max(po, axis=1, keepdims=True),
        ],
        axis=1,
    ).astype(np.float32)


def meta_vor(pv, po, qr):
    return np.concatenate(
        [
            pv,
            po,
            qr.reshape(-1, 1),
            pv - po,
            np.max(pv, axis=1, keepdims=True),
            np.max(po, axis=1, keepdims=True),
        ],
        axis=1,
    ).astype(np.float32)


def meta_vorc(
    pv,
    po,
    qr,
    qc,
    c_df,
    indices,
):
    blocks = [
        pv,
        po,
        qr.reshape(-1, 1),
        qc.reshape(-1, 1),
        pv - po,
        np.max(pv, axis=1, keepdims=True),
        np.max(po, axis=1, keepdims=True),
    ]

    agg_cols = [
        c
        for c in C_AGG_CANDIDATES
        if c in c_df.columns
    ]

    if agg_cols:
        blocks.append(
            c_df.loc[
                indices,
                agg_cols,
            ].to_numpy(dtype=np.float32)
        )

    return np.concatenate(
        blocks,
        axis=1,
    ).astype(np.float32)


# =============================================================================
# Inner-CV family selection
# =============================================================================

def select_multiclass_family(
    X,
    y,
    train_idx,
    categories,
    seed,
):
    strata = make_strata(
        train_idx,
        categories,
        y,
    )

    skf = StratifiedKFold(
        n_splits=INNER_FOLDS,
        shuffle=True,
        random_state=seed,
    )

    scores = {
        kind: []
        for kind in MULTI_MODELS
    }

    for fold, (a, b) in enumerate(
        skf.split(train_idx, strata)
    ):
        itr = train_idx[a]
        iva = train_idx[b]

        for kind in MULTI_MODELS:
            model = model_factory(
                kind,
                seed * 100 + fold,
            )

            model.fit(
                X[itr],
                y[itr],
            )

            pred = model.predict(
                X[iva]
            )

            scores[kind].append(
                (
                    balanced_accuracy_score(
                        y[iva],
                        pred,
                    ),
                    f1_score(
                        y[iva],
                        pred,
                        labels=CLASS_ORDER,
                        average="macro",
                        zero_division=0,
                    ),
                )
            )

    means = {
        kind: (
            float(np.mean([x[0] for x in vals])),
            float(np.mean([x[1] for x in vals])),
        )
        for kind, vals in scores.items()
    }

    best = max(
        means,
        key=lambda k: means[k],
    )

    return best, means


def select_binary_family_strict(
    recs,
    y,
    categories,
    outer_train_idx,
    seed,
    target,
):
    """
    Strict selection for R or C.

    For each inner fold:
    - learn R palette from inner-train NORMAL only
    - build R for all samples under that inner-train palette
    - if target == C, learn C prototype from inner-train NORMAL only
    - rank features on inner-train only
    - evaluate family on inner-validation

    Thus inner-validation normals do not participate in palette/prototype fitting.
    """

    strata = make_strata(
        outer_train_idx,
        categories,
        y,
    )

    skf = StratifiedKFold(
        n_splits=INNER_FOLDS,
        shuffle=True,
        random_state=seed,
    )

    scores = {
        kind: []
        for kind in BIN_MODELS
    }

    yb = (y == "structural").astype(int)

    for fold, (a, b) in enumerate(
        skf.split(outer_train_idx, strata)
    ):
        itr = outer_train_idx[a]
        iva = outer_train_idx[b]

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
                else learn_palette(
                    recs,
                    train_normal,
                    cat,
                )
            )

        r_df, r_cols, R = build_r(
            recs,
            palettes,
        )

        if target == "R":
            X = R
            topk = R_TOPK

        elif target == "C":
            c_df, c_cols, C = build_c(
                r_df,
                r_cols,
                categories,
                train_normal,
            )

            X = C
            topk = C_TOPK

        else:
            raise ValueError(target)

        sel = train_rank(
            X[itr],
            yb[itr],
            topk,
        )

        for kind in BIN_MODELS:
            model = model_factory(
                kind,
                seed * 100 + fold,
            )

            model.fit(
                X[itr][:, sel],
                yb[itr],
            )

            pred = model.predict(
                X[iva][:, sel]
            )

            scores[kind].append(
                (
                    balanced_accuracy_score(
                        yb[iva],
                        pred,
                    ),
                    f1_score(
                        yb[iva],
                        pred,
                        zero_division=0,
                    ),
                )
            )

    means = {
        kind: (
            float(np.mean([x[0] for x in vals])),
            float(np.mean([x[1] for x in vals])),
        )
        for kind, vals in scores.items()
    }

    best = max(
        means,
        key=lambda k: means[k],
    )

    return best, means


# =============================================================================
# Strict inner-OOF meta training
# =============================================================================

def build_strict_oof_meta(
    V,
    O,
    recs,
    y,
    categories,
    outer_train_idx,
    outer_seed,
    v_kind,
    o_kind,
    r_kind,
    c_kind,
    variant,
):
    strata = make_strata(
        outer_train_idx,
        categories,
        y,
    )

    skf = StratifiedKFold(
        n_splits=INNER_FOLDS,
        shuffle=True,
        random_state=INNER_RANDOM_STATE + outer_seed,
    )

    yb = (y == "structural").astype(int)

    meta = None

    for fold, (a, b) in enumerate(
        skf.split(outer_train_idx, strata)
    ):
        itr = outer_train_idx[a]
        iva = outer_train_idx[b]

        # V/O
        mv = model_factory(
            v_kind,
            outer_seed * 1000 + fold,
        )

        mo = model_factory(
            o_kind,
            outer_seed * 1000 + fold + 20,
        )

        mv.fit(
            V[itr],
            y[itr],
        )

        mo.fit(
            O[itr],
            y[itr],
        )

        pv = aligned_proba(
            mv,
            V[iva],
        )

        po = aligned_proba(
            mo,
            O[iva],
        )

        if variant == "V+O":
            fold_meta = meta_vo(
                pv,
                po,
            )

        else:
            # Rebuild R / C using INNER-TRAIN normals only.
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
                    else learn_palette(
                        recs,
                        train_normal,
                        cat,
                    )
                )

            r_df, r_cols, R = build_r(
                recs,
                palettes,
            )

            r_sel = train_rank(
                R[itr],
                yb[itr],
                R_TOPK,
            )

            mr = model_factory(
                r_kind,
                outer_seed * 1000 + fold + 40,
            )

            mr.fit(
                R[itr][:, r_sel],
                yb[itr],
            )

            qr = structural_proba(
                mr,
                R[iva][:, r_sel],
            )

            if variant == "V+O+R":
                fold_meta = meta_vor(
                    pv,
                    po,
                    qr,
                )

            elif variant == "V+O+R+C":
                c_df, c_cols, C = build_c(
                    r_df,
                    r_cols,
                    categories,
                    train_normal,
                )

                c_sel = train_rank(
                    C[itr],
                    yb[itr],
                    C_TOPK,
                )

                mc = model_factory(
                    c_kind,
                    outer_seed * 1000 + fold + 60,
                )

                mc.fit(
                    C[itr][:, c_sel],
                    yb[itr],
                )

                qc = structural_proba(
                    mc,
                    C[iva][:, c_sel],
                )

                fold_meta = meta_vorc(
                    pv,
                    po,
                    qr,
                    qc,
                    c_df,
                    iva,
                )

            else:
                raise ValueError(variant)

        if meta is None:
            meta = np.zeros(
                (
                    len(outer_train_idx),
                    fold_meta.shape[1],
                ),
                dtype=np.float32,
            )

        meta[b] = fold_meta

    return meta


# =============================================================================
# Outer fold evaluation
# =============================================================================

def summarize_by_fold(result):
    metric_cols = [
        "acc",
        "bacc",
        "macro_f1",
        "normal_f1",
        "logical_f1",
        "structural_f1",
    ]

    order = [
        "V",
        "V+O",
        "V+O+R",
        "V+O+R+C",
    ]

    rows = []

    for variant in order:
        g = result[
            result["variant"] == variant
        ]

        row = {
            "variant": variant,
            "n_folds": int(len(g)),
        }

        for c in metric_cols:
            row[f"{c}_mean"] = float(
                g[c].mean()
            )

            row[f"{c}_std"] = float(
                g[c].std(ddof=0)
            )

        rows.append(row)

    return pd.DataFrame(rows)


def main():
    print("=" * 120)
    print("STAGE6 FINAL 5-FOLD NESTED-CV ABLATION — ALL 1568 SAMPLES")
    print("=" * 120)

    if OUT_SUMMARY.exists():
        raise RuntimeError(
            "Nested-CV summary already exists.\n"
            "Do not repeatedly tune/rerun after inspecting outer-test results.\n"
            f"{OUT_SUMMARY}"
        )

    for p in [
        FEATURE_FILE,
        O_FILE,
        O_RAW,
    ]:
        if not p.exists():
            raise FileNotFoundError(p)

    # -------------------------------------------------------------------------
    # V
    # -------------------------------------------------------------------------
    df = pd.read_excel(
        FEATURE_FILE,
        sheet_name="merged_data",
    )

    df = df.loc[
        :,
        ~df.columns.duplicated(),
    ].copy()

    if len(df) != 1568:
        raise RuntimeError(
            f"Expected 1568 samples, got {len(df)}"
        )

    feat_sheet = pd.read_excel(
        FEATURE_FILE,
        sheet_name="features",
    )

    v_cols = list(
        dict.fromkeys(
            feat_sheet["feature_cols"]
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
        infer_gt_group,
        axis=1,
    ).to_numpy(
        dtype=object
    )

    categories = (
        df["category"]
        .astype(str)
        .reset_index(drop=True)
    )

    # -------------------------------------------------------------------------
    # O
    # -------------------------------------------------------------------------
    o_df, o_cols, O = load_o()

    if not np.array_equal(
        categories.to_numpy(),
        o_df["category"].astype(str).to_numpy(),
    ):
        raise RuntimeError(
            "V/O category alignment mismatch"
        )

    recs = load_raw()

    # -------------------------------------------------------------------------
    # Outer stratification
    # -------------------------------------------------------------------------
    all_idx = np.arange(len(df))

    outer_strata = (
        categories.astype(str).to_numpy()
        + "||"
        + y.astype(str)
    )

    stratum_counts = pd.Series(
        outer_strata
    ).value_counts()

    if stratum_counts.min() < OUTER_FOLDS:
        raise RuntimeError(
            "At least one category×class stratum has fewer than 5 samples; "
            "cannot perform the requested strict category×class stratified 5-fold."
        )

    outer = StratifiedKFold(
        n_splits=OUTER_FOLDS,
        shuffle=True,
        random_state=OUTER_RANDOM_STATE,
    )

    fold_assignment = np.full(
        len(df),
        -1,
        dtype=int,
    )

    result_rows = []
    setting_rows = []
    delta_rows = []

    oof_pred = pd.DataFrame({
        "row_index": np.arange(len(df)),
        "category": categories,
        "gt_group": y,
        "outer_fold": -1,
        "pred_V": "",
        "pred_VO": "",
        "pred_VOR": "",
        "pred_VORC": "",
    })

    # =========================================================================
    # OUTER loop
    # =========================================================================
    for outer_fold, (tr_pos, te_pos) in enumerate(
        outer.split(all_idx, outer_strata),
        start=1,
    ):
        outer_train = all_idx[tr_pos]
        outer_test = all_idx[te_pos]

        fold_assignment[
            outer_test
        ] = outer_fold

        print()
        print("-" * 120)
        print(
            f"OUTER FOLD {outer_fold}/5 | "
            f"train={len(outer_train)} | "
            f"test={len(outer_test)}"
        )

        # ---------------------------------------------------------------------
        # 1) Select base families using INNER CV only.
        # ---------------------------------------------------------------------
        v_kind, v_cv = select_multiclass_family(
            V,
            y,
            outer_train,
            categories,
            seed=INNER_RANDOM_STATE + outer_fold * 10 + 1,
        )

        o_kind, o_cv = select_multiclass_family(
            O,
            y,
            outer_train,
            categories,
            seed=INNER_RANDOM_STATE + outer_fold * 10 + 2,
        )

        r_kind, r_cv = select_binary_family_strict(
            recs,
            y,
            categories,
            outer_train,
            seed=INNER_RANDOM_STATE + outer_fold * 10 + 3,
            target="R",
        )

        c_kind, c_cv = select_binary_family_strict(
            recs,
            y,
            categories,
            outer_train,
            seed=INNER_RANDOM_STATE + outer_fold * 10 + 4,
            target="C",
        )

        print(
            f"Selected V/O/R/C = "
            f"{v_kind}/{o_kind}/{r_kind}/{c_kind}"
        )

        # ---------------------------------------------------------------------
        # 2) Strict inner-OOF meta features for each ablation.
        # ---------------------------------------------------------------------
        meta_train_vo = build_strict_oof_meta(
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
            "V+O",
        )

        meta_train_vor = build_strict_oof_meta(
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
            "V+O+R",
        )

        meta_train_vorc = build_strict_oof_meta(
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
            "V+O+R+C",
        )

        fusion_vo = fusion_factory(
            OUTER_RANDOM_STATE + outer_fold * 100 + 1
        )

        fusion_vor = fusion_factory(
            OUTER_RANDOM_STATE + outer_fold * 100 + 2
        )

        fusion_vorc = fusion_factory(
            OUTER_RANDOM_STATE + outer_fold * 100 + 3
        )

        fusion_vo.fit(
            meta_train_vo,
            y[outer_train],
        )

        fusion_vor.fit(
            meta_train_vor,
            y[outer_train],
        )

        fusion_vorc.fit(
            meta_train_vorc,
            y[outer_train],
        )

        # ---------------------------------------------------------------------
        # 3) Build OUTER-train-only R/C transformation.
        # ---------------------------------------------------------------------
        outer_train_normal = [
            int(i)
            for i in outer_train
            if y[i] == "normal"
        ]

        palettes = {}

        for cat in sorted(categories.unique()):
            palettes[cat] = (
                []
                if cat == "pushpins"
                else learn_palette(
                    recs,
                    outer_train_normal,
                    cat,
                )
            )

        r_df, r_cols, R_outer = build_r(
            recs,
            palettes,
        )

        c_df, c_cols, C_outer = build_c(
            r_df,
            r_cols,
            categories,
            outer_train_normal,
        )

        yb = (
            y == "structural"
        ).astype(int)

        # ---------------------------------------------------------------------
        # 4) Refit selected base models on ALL outer-train.
        # ---------------------------------------------------------------------
        mv = model_factory(
            v_kind,
            OUTER_RANDOM_STATE + outer_fold * 1000 + 10,
        )

        mo = model_factory(
            o_kind,
            OUTER_RANDOM_STATE + outer_fold * 1000 + 20,
        )

        mv.fit(
            V[outer_train],
            y[outer_train],
        )

        mo.fit(
            O[outer_train],
            y[outer_train],
        )

        r_sel = train_rank(
            R_outer[outer_train],
            yb[outer_train],
            R_TOPK,
        )

        c_sel = train_rank(
            C_outer[outer_train],
            yb[outer_train],
            C_TOPK,
        )

        mr = model_factory(
            r_kind,
            OUTER_RANDOM_STATE + outer_fold * 1000 + 30,
        )

        mc = model_factory(
            c_kind,
            OUTER_RANDOM_STATE + outer_fold * 1000 + 40,
        )

        mr.fit(
            R_outer[outer_train][:, r_sel],
            yb[outer_train],
        )

        mc.fit(
            C_outer[outer_train][:, c_sel],
            yb[outer_train],
        )

        # ---------------------------------------------------------------------
        # 5) OUTER TEST exactly once.
        # ---------------------------------------------------------------------
        pv_test = aligned_proba(
            mv,
            V[outer_test],
        )

        po_test = aligned_proba(
            mo,
            O[outer_test],
        )

        qr_test = structural_proba(
            mr,
            R_outer[outer_test][:, r_sel],
        )

        qc_test = structural_proba(
            mc,
            C_outer[outer_test][:, c_sel],
        )

        # V baseline
        pred_v = np.asarray(
            CLASS_ORDER,
            dtype=object,
        )[
            np.argmax(
                pv_test,
                axis=1,
            )
        ]

        # V+O
        test_meta_vo = meta_vo(
            pv_test,
            po_test,
        )

        pred_vo = fusion_vo.predict(
            test_meta_vo
        )

        # V+O+R
        test_meta_vor = meta_vor(
            pv_test,
            po_test,
            qr_test,
        )

        pred_vor = fusion_vor.predict(
            test_meta_vor
        )

        # V+O+R+C
        test_meta_vorc = meta_vorc(
            pv_test,
            po_test,
            qr_test,
            qc_test,
            c_df,
            outer_test,
        )

        pred_vorc = fusion_vorc.predict(
            test_meta_vorc
        )

        variants = {
            "V": pred_v,
            "V+O": pred_vo,
            "V+O+R": pred_vor,
            "V+O+R+C": pred_vorc,
        }

        fold_metrics = {}

        for variant, pred in variants.items():
            m = metrics(
                y[outer_test],
                pred,
            )

            fold_metrics[
                variant
            ] = m

            result_rows.append({
                "outer_fold": outer_fold,
                "variant": variant,
                "n_test": len(outer_test),
                **m,
            })

            print(
                f"{variant:<9} "
                f"Acc={m['acc']:.4f}  "
                f"BAcc={m['bacc']:.4f}  "
                f"MacroF1={m['macro_f1']:.4f}  "
                f"N/L/S="
                f"{m['normal_f1']:.4f}/"
                f"{m['logical_f1']:.4f}/"
                f"{m['structural_f1']:.4f}"
            )

        # Store per-sample outer OOF predictions.
        oof_pred.loc[
            outer_test,
            "outer_fold",
        ] = outer_fold

        oof_pred.loc[
            outer_test,
            "pred_V",
        ] = pred_v

        oof_pred.loc[
            outer_test,
            "pred_VO",
        ] = pred_vo

        oof_pred.loc[
            outer_test,
            "pred_VOR",
        ] = pred_vor

        oof_pred.loc[
            outer_test,
            "pred_VORC",
        ] = pred_vorc

        setting_rows.append({
            "outer_fold": outer_fold,
            "n_outer_train": len(outer_train),
            "n_outer_test": len(outer_test),
            "n_outer_train_normal": len(outer_train_normal),
            "v_model": v_kind,
            "o_model": o_kind,
            "r_model": r_kind,
            "c_model": c_kind,
            "r_topk": R_TOPK,
            "c_topk": C_TOPK,
            "fusion": "LogisticRegression(C=1.0)",
            "meta_dim_vo": int(test_meta_vo.shape[1]),
            "meta_dim_vor": int(test_meta_vor.shape[1]),
            "meta_dim_vorc": int(test_meta_vorc.shape[1]),
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
                "outer_fold": outer_fold,
                "comparison": comp,
                "delta_acc": ma["acc"] - mb["acc"],
                "delta_bacc": ma["bacc"] - mb["bacc"],
                "delta_macro_f1": ma["macro_f1"] - mb["macro_f1"],
                "delta_normal_f1": ma["normal_f1"] - mb["normal_f1"],
                "delta_logical_f1": ma["logical_f1"] - mb["logical_f1"],
                "delta_structural_f1": ma["structural_f1"] - mb["structural_f1"],
            })

    # =========================================================================
    # Final checks / summary
    # =========================================================================
    if np.any(
        fold_assignment < 1
    ):
        raise RuntimeError(
            "Some samples were never assigned to an outer test fold"
        )

    fold_counts = pd.Series(
        fold_assignment
    ).value_counts().sort_index()

    if fold_counts.sum() != 1568:
        raise RuntimeError(
            "Outer test coverage is not exactly 1568"
        )

    result = pd.DataFrame(
        result_rows
    )

    mean_df = summarize_by_fold(
        result
    )

    delta_df = pd.DataFrame(
        delta_rows
    )

    settings_df = pd.DataFrame(
        setting_rows
    )

    assign_df = pd.DataFrame({
        "row_index": np.arange(len(df)),
        "category": categories,
        "gt_group": y,
        "outer_test_fold": fold_assignment,
    })

    # Overall pooled OOF metrics: every sample exactly once as test.
    pooled = {}

    for variant, col in [
        ("V", "pred_V"),
        ("V+O", "pred_VO"),
        ("V+O+R", "pred_VOR"),
        ("V+O+R+C", "pred_VORC"),
    ]:
        pooled[variant] = metrics(
            oof_pred["gt_group"].to_numpy(dtype=object),
            oof_pred[col].to_numpy(dtype=object),
        )

    assign_df.to_csv(
        OUT_ASSIGN,
        index=False,
        encoding="utf-8-sig",
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

    oof_pred.to_csv(
        OUT_OOF,
        index=False,
        encoding="utf-8-sig",
    )

    settings_df.to_csv(
        OUT_SETTINGS,
        index=False,
        encoding="utf-8-sig",
    )

    delta_df.to_csv(
        OUT_DELTA,
        index=False,
        encoding="utf-8-sig",
    )

    report = {
        "status": "NESTED_5FOLD_ABLATION_COMPLETE",
        "samples": 1568,
        "outer_folds": OUTER_FOLDS,
        "inner_folds": INNER_FOLDS,
        "outer_random_state": OUTER_RANDOM_STATE,
        "inner_random_state": INNER_RANDOM_STATE,
        "variants": [
            "V",
            "V+O",
            "V+O+R",
            "V+O+R+C",
        ],
        "outer_test_coverage": {
            "every_sample_tested_exactly_once": True,
            "fold_counts": {
                str(k): int(v)
                for k, v in fold_counts.items()
            },
        },
        "mean_std_across_outer_folds": (
            mean_df.to_dict(
                orient="records"
            )
        ),
        "pooled_oof_metrics_all_1568": pooled,
        "protocol": {
            "outer_split": (
                "5-fold category×class stratified CV"
            ),
            "inner_selection": (
                "5-fold CV inside outer-train only"
            ),
            "r_palette": (
                "training-normal only; rebuilt inside inner folds"
            ),
            "c_prototype": (
                "training-normal only; rebuilt inside inner folds"
            ),
            "fusion_training": (
                "strict inner OOF meta-features"
            ),
            "fusion_head": (
                "fixed LogisticRegression(C=1.0)"
            ),
            "r_topk": R_TOPK,
            "c_topk": C_TOPK,
        },
        "scientific_boundary": {
            "all_1568_used_as_outer_test_once": True,
            "outer_test_used_for_selection": False,
            "gt_masks_as_predictive_input": False,
            "stage5_prediction_v3_used": False,
            "same_dataset_cross_validation": True,
            "external_independent_dataset_validation": False,
        },
    }

    OUT_SUMMARY.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 120)
    print("FINAL NESTED 5-FOLD ABLATION — OUTER-FOLD MEAN ± STD")
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
    print("=" * 120)
    print("POOLED OOF METRICS — ALL 1568 SAMPLES, EACH TESTED EXACTLY ONCE")
    print("=" * 120)

    for variant in [
        "V",
        "V+O",
        "V+O+R",
        "V+O+R+C",
    ]:
        m = pooled[variant]

        print(
            f"{variant:<9} "
            f"Acc={m['acc']:.4f}  "
            f"BAcc={m['bacc']:.4f}  "
            f"MacroF1={m['macro_f1']:.4f}  "
            f"N/L/S="
            f"{m['normal_f1']:.4f}/"
            f"{m['logical_f1']:.4f}/"
            f"{m['structural_f1']:.4f}"
        )

    print()
    print("PAIRED MEAN DELTAS ACROSS OUTER FOLDS")
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
    print(
        "DONE: all 1568 samples were used as OUTER TEST exactly once."
    )

    print(
        "Do not tune the framework on these outer-test results and rerun "
        "the same protocol as if it were still confirmatory."
    )

    print()
    print("Saved:")
    print(OUT_ASSIGN)
    print(OUT_FOLD)
    print(OUT_MEAN)
    print(OUT_OOF)
    print(OUT_SETTINGS)
    print(OUT_DELTA)
    print(OUT_SUMMARY)
    print("=" * 120)


if __name__ == "__main__":
    main()
