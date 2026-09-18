#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
Stage6 DINO-GLOBAL-V1 — Whole-image DINOv2 Representation Ceiling Audit
DEVELOPMENT ONLY

Question
--------
Does a frozen whole-image DINOv2 representation contain meaningful
conditional information beyond the canonical V+O_old system?

Compare
-------
    V
    V+O_old
    V+DINO
    V+O_old+DINO

Representation
--------------
Frozen DINOv2-S/14:
    vit_small_patch14_dinov2.lvd142m

For each whole image:
    normalized CLS token (384-D)
    normalized mean patch token (384-D)
Concatenate and L2-normalize -> 768-D.

No SAM instances.
No synthetic anomalies.
No relation features.
No threshold rescue.
No category-specific post-hoc routing.
No hyperparameter search.

Strict protocol
---------------
- Existing five outer folds are DEVELOPMENT ONLY.
- DINO feature extraction is split-independent and frozen.
- DINO classifier inner OOF:
      trained on inner-train real labels only
      predicts inner-val
- Outer DINO classifier:
      trained on outer-train real labels only
      predicts outer-test once
- Canonical V/O old models and fusion are reproduced exactly.
- Final augmented fusion is trained on strict OOF probabilities only.
- Outer-test labels never enter training/selection.
- stage6_frozen_odrc_robustness_20260827 is NOT read.

Predeclared ceiling gate vs canonical V+O_old:
    mean ΔMacro-F1      >= +0.008
    mean ΔStructural-F1 >= +0.010
    mean ΔBAcc          >= +0.005
    positive Macro-F1 folds >= 3/5
    positive BAcc folds     >= 3/5

If PASS:
    foundation representation has enough headroom to justify a
    task-aligned adapter/fine-tuning branch.
If FAIL:
    stop the current feature-increment search and reconsider the
    paper's core learning problem/protocol.

Run
---
cd code/stage6
python scripts\audit_stage6_dino_global_v1_ceiling_dev.py
"""

from __future__ import annotations

import importlib.util
import json
import pickle
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

from PIL import Image

import torch
import torch.nn.functional as F

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# =============================================================================
# Project imports
# =============================================================================

HERE = Path(__file__).resolve().parent
V11_SCRIPT = HERE / "evaluate_stage6_odrc_v11_orthogonal_residual_dev.py"

if not V11_SCRIPT.exists():
    raise FileNotFoundError(V11_SCRIPT)


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


v11 = import_module(V11_SCRIPT, "v11_dino_global")
v6 = v11.v6
base = v11.base


# =============================================================================
# Fixed paths / settings
# =============================================================================

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

INSTANCE_CACHE = (
    ROOT / "stage6" / "outputs"
    / "stage6_sirc_instance_cache_v2_efficient"
    / "sirc_all5_instance_cache.pkl"
)

FORBIDDEN_CONFIRM = (
    ROOT / "stage6" / "outputs"
    / "stage6_frozen_odrc_robustness_20260827"
)

OUT_DIR = (
    ROOT / "stage6" / "outputs"
    / "stage6_dino_global_v1_ceiling_dev"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

DINO_CACHE = OUT_DIR / "dino_global_v1_features.npz"

DINO_MODEL = "vit_small_patch14_dinov2.lvd142m"

EXPECTED_N = 1568
CLASS_ORDER = list(base.CLASS_ORDER)

INNER_FOLDS = 5
EXTRACT_BATCH = 32

VARIANTS = [
    "V",
    "V+O_old",
    "V+DINO",
    "V+O_old+DINO",
]


# =============================================================================
# Reproducibility
# =============================================================================

def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# Image paths
# =============================================================================

def load_image_paths():
    with INSTANCE_CACHE.open("rb") as f:
        obj = pickle.load(f)

    entries = sorted(
        list(obj["entries"]),
        key=lambda z: int(z["row_index"]),
    )

    if len(entries) != EXPECTED_N:
        raise RuntimeError(
            f"Expected {EXPECTED_N} cache entries, got {len(entries)}"
        )

    paths = []
    keys = []
    cats = []

    for i, e in enumerate(entries):
        if int(e["row_index"]) != i:
            raise RuntimeError(
                f"Cache row alignment failed at row {i}"
            )

        raw = str(e.get("image_path", "")).strip()
        if not raw:
            raise RuntimeError(
                f"Missing image_path for row {i}"
            )

        p = Path(raw)

        if not p.exists():
            raise FileNotFoundError(
                f"Image path does not exist for row {i}: {p}"
            )

        paths.append(p)
        keys.append(str(e["sample_key"]))
        cats.append(str(e["category"]))

    return paths, keys, cats


# =============================================================================
# DINO feature extraction
# =============================================================================

def load_dino():
    try:
        import timm
        from timm.data import (
            create_transform,
            resolve_model_data_config,
        )
    except Exception as exc:
        raise RuntimeError(
            "This audit requires timm. "
            "Install/activate the same environment used by previous DINO runs.\n"
            f"Original error: {exc}"
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("DINO model:", DINO_MODEL)
    print("DINO device:", device)

    try:
        model = timm.create_model(
            DINO_MODEL,
            pretrained=True,
            num_classes=0,
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to load pretrained DINOv2-S/14.\n"
            "The model was used previously in this project; if weights are no "
            "longer cached, allow timm/HuggingFace to download them once.\n"
            f"Original error: {exc}"
        )

    model.eval()
    model.to(device)

    cfg = resolve_model_data_config(model)

    transform = create_transform(
        **cfg,
        is_training=False,
    )

    return model, transform, device


@torch.no_grad()
def dino_batch_embedding(model, batch, device):
    batch = batch.to(
        device,
        non_blocking=True,
    )

    amp_enabled = (
        device.type == "cuda"
    )

    with torch.autocast(
        device_type=device.type,
        enabled=amp_enabled,
        dtype=(
            torch.float16
            if device.type == "cuda"
            else torch.float32
        ),
    ):
        feat = model.forward_features(batch)

        if isinstance(feat, dict):
            if (
                "x_norm_clstoken" in feat
                and "x_norm_patchtokens" in feat
            ):
                cls = feat["x_norm_clstoken"]
                patch = feat["x_norm_patchtokens"].mean(dim=1)

            elif "x_prenorm" in feat:
                tok = feat["x_prenorm"]
                cls = tok[:, 0]
                patch = tok[:, 1:].mean(dim=1)

            else:
                tensors = [
                    v
                    for v in feat.values()
                    if torch.is_tensor(v)
                ]

                if not tensors:
                    raise RuntimeError(
                        "DINO forward_features dict contains no tensor"
                    )

                tok = tensors[0]

                if tok.ndim == 3:
                    cls = tok[:, 0]
                    patch = tok[:, 1:].mean(dim=1)
                elif tok.ndim == 2:
                    cls = tok
                    patch = tok
                else:
                    raise RuntimeError(
                        f"Unsupported DINO tensor shape: {tok.shape}"
                    )

        elif torch.is_tensor(feat):
            if feat.ndim == 3:
                cls = feat[:, 0]
                patch = feat[:, 1:].mean(dim=1)
            elif feat.ndim == 2:
                cls = feat
                patch = feat
            else:
                raise RuntimeError(
                    f"Unsupported DINO output shape: {feat.shape}"
                )

        else:
            raise RuntimeError(
                "Unsupported DINO forward_features output type"
            )

        cls = F.normalize(
            cls.float(),
            dim=1,
        )

        patch = F.normalize(
            patch.float(),
            dim=1,
        )

        emb = torch.cat(
            [cls, patch],
            dim=1,
        )

        emb = F.normalize(
            emb,
            dim=1,
        )

    return (
        emb.detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )


def build_or_load_dino_cache(paths, sample_keys):
    if DINO_CACHE.exists():
        z = np.load(
            DINO_CACHE,
            allow_pickle=True,
        )

        model_name = str(
            z["model_name"].item()
        )

        if model_name != DINO_MODEL:
            raise RuntimeError(
                f"Existing cache model={model_name}, expected {DINO_MODEL}"
            )

        emb = z["embeddings"].astype(np.float32)
        cached_keys = z["sample_keys"].astype(str).tolist()

        if emb.shape[0] != EXPECTED_N:
            raise RuntimeError(
                f"Existing DINO cache rows={emb.shape[0]}"
            )

        if cached_keys != list(sample_keys):
            raise RuntimeError(
                "Existing DINO cache sample_key alignment mismatch"
            )

        print(
            "Loaded existing whole-image DINO cache:",
            DINO_CACHE,
            "| shape:",
            emb.shape,
        )

        return emb

    model, transform, device = load_dino()

    features = []

    print(
        f"Extracting whole-image DINO features for {len(paths)} images..."
    )

    start = time.perf_counter()

    for start_i in range(
        0,
        len(paths),
        EXTRACT_BATCH,
    ):
        batch_paths = paths[
            start_i:
            start_i + EXTRACT_BATCH
        ]

        tensors = []

        for p in batch_paths:
            with Image.open(p) as im:
                im = im.convert("RGB")
                tensors.append(
                    transform(im)
                )

        batch = torch.stack(
            tensors,
            dim=0,
        )

        feat = dino_batch_embedding(
            model,
            batch,
            device,
        )

        features.append(feat)

        done = min(
            start_i + EXTRACT_BATCH,
            len(paths),
        )

        if (
            done == len(paths)
            or done % 160 == 0
        ):
            elapsed = time.perf_counter() - start
            print(
                f"  DINO {done}/{len(paths)} "
                f"| elapsed {elapsed/60:.1f} min"
            )

    emb = np.concatenate(
        features,
        axis=0,
    )

    if emb.shape[0] != EXPECTED_N:
        raise RuntimeError(
            f"DINO extraction produced shape {emb.shape}"
        )

    np.savez_compressed(
        DINO_CACHE,
        model_name=np.asarray(DINO_MODEL),
        embeddings=emb,
        sample_keys=np.asarray(
            sample_keys,
            dtype=object,
        ),
    )

    print(
        "Saved DINO cache:",
        DINO_CACHE,
        "| shape:",
        emb.shape,
    )

    return emb


# =============================================================================
# Fixed DINO probe
# =============================================================================

def dino_probe_factory(seed):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=4000,
            class_weight="balanced",
            C=1.0,
            random_state=seed,
        ),
    )


def fusion_factory(seed):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            C=1.0,
            random_state=seed,
        ),
    )


def aligned_proba_local(model, X):
    p = model.predict_proba(X)

    out = np.zeros(
        (len(X), 3),
        dtype=np.float64,
    )

    for j, c in enumerate(model.classes_):
        if c in CLASS_ORDER:
            out[
                :,
                CLASS_ORDER.index(c),
            ] = p[:, j]

    s = out.sum(
        axis=1,
        keepdims=True,
    )

    s[
        s <= 0
    ] = 1.0

    return out / s


def strict_dino_oof(
    D,
    y,
    categories,
    outer_train,
    outer_fold,
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

    out = np.zeros(
        (len(outer_train), 3),
        dtype=np.float64,
    )

    for inner_fold, (a, b) in enumerate(
        skf.split(
            outer_train,
            strata,
        ),
        start=1,
    ):
        itr = outer_train[a]
        iva = outer_train[b]

        model = dino_probe_factory(
            910000
            + outer_fold * 100
            + inner_fold
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


def fit_predict_dino_outer(
    D,
    y,
    outer_train,
    outer_test,
    outer_fold,
):
    model = dino_probe_factory(
        920000 + outer_fold
    )

    model.fit(
        D[outer_train],
        y[outer_train],
    )

    return aligned_proba_local(
        model,
        D[outer_test],
    )


# =============================================================================
# Meta
# =============================================================================

def meta_v_dino(pv, pdino):
    return np.concatenate(
        [pv, pdino],
        axis=1,
    ).astype(np.float32)


def meta_v_o_dino(pv, po, pdino):
    return np.concatenate(
        [pv, po, pdino],
        axis=1,
    ).astype(np.float32)


def pred(p):
    return np.asarray(
        CLASS_ORDER,
        dtype=object,
    )[
        np.argmax(
            p,
            axis=1,
        )
    ]


def summarize(res):
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
        g = res[
            res["variant"] == name
        ]

        row = {
            "variant": name,
            "n_folds": int(len(g)),
        }

        for m in metrics:
            row[f"{m}_mean"] = float(
                g[m].mean()
            )

            row[f"{m}_std"] = float(
                g[m].std(ddof=0)
            )

        rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# Main
# =============================================================================

def main():
    t0 = time.perf_counter()

    print("=" * 160)
    print("DINO-GLOBAL-V1 WHOLE-IMAGE REPRESENTATION CEILING AUDIT — DEVELOPMENT ONLY")
    print("=" * 160)

    if FORBIDDEN_CONFIRM.exists():
        print(
            "Forbidden confirmation directory exists but is NOT read:",
            FORBIDDEN_CONFIRM,
        )

    for p in [
        FEATURE_FILE,
        ASSIGN_FILE,
        SETTING_FILE,
        INSTANCE_CACHE,
        base.O_FILE,
        base.O_RAW,
    ]:
        if not p.exists():
            raise FileNotFoundError(p)

    # -------------------------------------------------------------------------
    # Load canonical project data
    # -------------------------------------------------------------------------
    df = pd.read_excel(
        FEATURE_FILE,
        sheet_name="merged_data",
    )

    df = df.loc[
        :,
        ~df.columns.duplicated(),
    ].copy().reset_index(drop=True)

    feat = pd.read_excel(
        FEATURE_FILE,
        sheet_name="features",
    )

    vcols = list(
        dict.fromkeys(
            feat["feature_cols"]
            .dropna()
            .astype(str)
            .tolist()
        )
    )

    if len(vcols) != 186:
        raise RuntimeError(
            f"Expected V dim 186, got {len(vcols)}"
        )

    for c in vcols:
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
        vcols
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

    _, _, O_old = base.load_o()
    recs = base.load_raw()

    paths, sample_keys, cache_cats = load_image_paths()

    if not (
        len(df)
        == len(V)
        == len(O_old)
        == len(recs)
        == len(paths)
        == EXPECTED_N
    ):
        raise RuntimeError(
            "1568-row alignment failed"
        )

    for i in range(EXPECTED_N):
        if cache_cats[i] != str(
            categories.iloc[i]
        ):
            raise RuntimeError(
                f"Cache/workbook category mismatch row {i}"
            )

    # -------------------------------------------------------------------------
    # Extract/load whole-image DINO
    # -------------------------------------------------------------------------
    D = build_or_load_dino_cache(
        paths,
        sample_keys,
    )

    print(
        "Whole-image DINO feature shape:",
        D.shape,
    )

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
    pred_rows = []

    # =========================================================================
    # Outer development folds
    # =========================================================================
    for outer_fold in range(1, 6):
        print()
        print("-" * 160)
        print(
            f"OUTER DEVELOPMENT FOLD {outer_fold}/5"
        )

        fold_assignment = assignments[
            "outer_test_fold"
        ].to_numpy(
            dtype=int
        )

        outer_test = np.where(
            fold_assignment == outer_fold
        )[0]

        outer_train = np.where(
            fold_assignment != outer_fold
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
        # Canonical V/O OOF
        # ---------------------------------------------------------------------
        (
            pv_oof,
            po_oof,
            _pvo_oof_unused,
            _qr,
            _qc,
            _cs,
            _,
        ) = v6.strict_oof_category_signals(
            V,
            O_old,
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

        # Strict DINO OOF
        pdino_oof = strict_dino_oof(
            D,
            y,
            categories,
            outer_train,
            outer_fold,
        )

        # ---------------------------------------------------------------------
        # Outer test base signals
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
            O_old,
            recs,
            y,
            categories,
            outer_train,
            outer_test,
            outer_fold,
            v_kind,
            o_kind,
        )

        pdino_te = fit_predict_dino_outer(
            D,
            y,
            outer_train,
            outer_test,
            outer_fold,
        )

        # ---------------------------------------------------------------------
        # Exact canonical V+O_old
        # ---------------------------------------------------------------------
        f_old = base.fusion_factory(
            403000 + outer_fold
        )

        f_old.fit(
            base.meta_vo(
                pv_oof,
                po_oof,
            ),
            ytr,
        )

        p_old_te = base.aligned_proba(
            f_old,
            base.meta_vo(
                pv_te,
                po_te,
            ),
        )

        # ---------------------------------------------------------------------
        # V+DINO
        # ---------------------------------------------------------------------
        f_vd = fusion_factory(
            930000 + outer_fold
        )

        f_vd.fit(
            meta_v_dino(
                pv_oof,
                pdino_oof,
            ),
            ytr,
        )

        p_vd_te = aligned_proba_local(
            f_vd,
            meta_v_dino(
                pv_te,
                pdino_te,
            ),
        )

        # ---------------------------------------------------------------------
        # V+O_old+DINO
        # ---------------------------------------------------------------------
        f_aug = fusion_factory(
            940000 + outer_fold
        )

        f_aug.fit(
            meta_v_o_dino(
                pv_oof,
                po_oof,
                pdino_oof,
            ),
            ytr,
        )

        p_aug_te = aligned_proba_local(
            f_aug,
            meta_v_o_dino(
                pv_te,
                po_te,
                pdino_te,
            ),
        )

        # Canonical V
        mv = base.model_factory(
            v_kind,
            404000 + outer_fold,
        )

        mv.fit(
            V[outer_train],
            y[outer_train],
        )

        p_v_te = base.aligned_proba(
            mv,
            V[outer_test],
        )

        probs = {
            "V": p_v_te,
            "V+O_old": p_old_te,
            "V+DINO": p_vd_te,
            "V+O_old+DINO": p_aug_te,
        }

        fold_metrics = {}

        for name in VARIANTS:
            m = base.metrics(
                y[outer_test],
                pred(probs[name]),
            )

            fold_metrics[name] = m

            metric_rows.append(
                {
                    "fold": outer_fold,
                    "variant": name,
                    **m,
                }
            )

            print(
                f"{name:<16} "
                f"Acc={m['acc']:.4f} "
                f"BAcc={m['bacc']:.4f} "
                f"MacroF1={m['macro_f1']:.4f} "
                f"N/L/S={m['normal_f1']:.4f}/"
                f"{m['logical_f1']:.4f}/"
                f"{m['structural_f1']:.4f}"
            )

        old = fold_metrics[
            "V+O_old"
        ]

        for name in [
            "V+DINO",
            "V+O_old+DINO",
        ]:
            z = fold_metrics[name]

            delta_rows.append(
                {
                    "fold": outer_fold,
                    "comparison": f"{name}_vs_V+O_old",
                    "delta_acc": z["acc"] - old["acc"],
                    "delta_bacc": z["bacc"] - old["bacc"],
                    "delta_macro_f1": z["macro_f1"] - old["macro_f1"],
                    "delta_normal_f1": z["normal_f1"] - old["normal_f1"],
                    "delta_logical_f1": z["logical_f1"] - old["logical_f1"],
                    "delta_structural_f1": (
                        z["structural_f1"]
                        - old["structural_f1"]
                    ),
                }
            )

        pnames = {
            k: pred(v)
            for k, v in probs.items()
        }

        for local_i, global_i in enumerate(
            outer_test
        ):
            pred_rows.append(
                {
                    "fold": outer_fold,
                    "row_index": int(global_i),
                    "sample_key": sample_keys[
                        int(global_i)
                    ],
                    "category": str(
                        categories.iloc[
                            int(global_i)
                        ]
                    ),
                    "gt_group": str(
                        y[int(global_i)]
                    ),
                    "pred_V": str(
                        pnames["V"][
                            local_i
                        ]
                    ),
                    "pred_VO_old": str(
                        pnames["V+O_old"][
                            local_i
                        ]
                    ),
                    "pred_V_DINO": str(
                        pnames["V+DINO"][
                            local_i
                        ]
                    ),
                    "pred_VO_DINO": str(
                        pnames["V+O_old+DINO"][
                            local_i
                        ]
                    ),
                }
            )

    # =========================================================================
    # Aggregate
    # =========================================================================
    result = pd.DataFrame(
        metric_rows
    )

    mean_df = summarize(
        result
    )

    delta_df = pd.DataFrame(
        delta_rows
    )

    pred_df = pd.DataFrame(
        pred_rows
    )

    # Canonical check
    oldrow = mean_df[
        mean_df["variant"]
        == "V+O_old"
    ].iloc[0]

    expected_acc = 0.9049795486457336

    if abs(
        float(
            oldrow["acc_mean"]
        )
        - expected_acc
    ) > 1e-10:
        raise RuntimeError(
            "Canonical V+O_old reproduction failed: "
            f"{oldrow['acc_mean']}"
        )

    # Category pooled
    cat_rows = []

    for variant, col in [
        ("V+O_old", "pred_VO_old"),
        ("V+DINO", "pred_V_DINO"),
        ("V+O_old+DINO", "pred_VO_DINO"),
    ]:
        for cat in sorted(
            pred_df["category"].unique()
        ):
            g = pred_df[
                pred_df["category"]
                == cat
            ]

            m = base.metrics(
                g["gt_group"].to_numpy(
                    dtype=object
                ),
                g[col].to_numpy(
                    dtype=object
                ),
            )

            cat_rows.append(
                {
                    "variant": variant,
                    "category": cat,
                    "n": int(len(g)),
                    **m,
                }
            )

    cat_df = pd.DataFrame(
        cat_rows
    )

    # Gate on augmented system
    d = delta_df[
        delta_df["comparison"]
        == "V+O_old+DINO_vs_V+O_old"
    ].copy()

    gate = {
        "mean_delta_acc": float(
            d["delta_acc"].mean()
        ),
        "mean_delta_bacc": float(
            d["delta_bacc"].mean()
        ),
        "mean_delta_macro_f1": float(
            d["delta_macro_f1"].mean()
        ),
        "mean_delta_structural_f1": float(
            d["delta_structural_f1"].mean()
        ),
        "positive_acc_folds": int(
            (d["delta_acc"] > 0).sum()
        ),
        "positive_bacc_folds": int(
            (d["delta_bacc"] > 0).sum()
        ),
        "positive_macro_f1_folds": int(
            (d["delta_macro_f1"] > 0).sum()
        ),
        "positive_structural_f1_folds": int(
            (d["delta_structural_f1"] > 0).sum()
        ),
    }

    passed = bool(
        gate["mean_delta_macro_f1"] >= 0.008
        and gate["mean_delta_structural_f1"] >= 0.010
        and gate["mean_delta_bacc"] >= 0.005
        and gate["positive_macro_f1_folds"] >= 3
        and gate["positive_bacc_folds"] >= 3
    )

    decision = (
        "DINO_GLOBAL_V1_CEILING_PASS_FOUNDATION_ADAPTER_JUSTIFIED"
        if passed
        else "DINO_GLOBAL_V1_CEILING_FAIL_STOP_FEATURE_INCREMENT_SEARCH"
    )

    # Transitions
    oldp = pred_df[
        "pred_VO_old"
    ].to_numpy(
        dtype=object
    )

    newp = pred_df[
        "pred_VO_DINO"
    ].to_numpy(
        dtype=object
    )

    gt = pred_df[
        "gt_group"
    ].to_numpy(
        dtype=object
    )

    changed = oldp != newp

    rescue = (
        changed
        & (oldp != gt)
        & (newp == gt)
    )

    harm = (
        changed
        & (oldp == gt)
        & (newp != gt)
    )

    wrong2wrong = (
        changed
        & (oldp != gt)
        & (newp != gt)
    )

    transitions = {
        "changed": int(changed.sum()),
        "rescues": int(rescue.sum()),
        "harms": int(harm.sum()),
        "wrong_to_wrong": int(
            wrong2wrong.sum()
        ),
        "net_correct": int(
            rescue.sum()
            - harm.sum()
        ),
    }

    # Save
    result.to_csv(
        OUT_DIR / "dino_global_v1_by_fold.csv",
        index=False,
        encoding="utf-8-sig",
    )

    mean_df.to_csv(
        OUT_DIR / "dino_global_v1_mean_std.csv",
        index=False,
        encoding="utf-8-sig",
    )

    delta_df.to_csv(
        OUT_DIR / "dino_global_v1_deltas.csv",
        index=False,
        encoding="utf-8-sig",
    )

    cat_df.to_csv(
        OUT_DIR / "dino_global_v1_category_pooled.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pred_df.to_csv(
        OUT_DIR / "dino_global_v1_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    report = {
        "status": "DEVELOPMENT_ONLY",
        "decision": decision,
        "elapsed_seconds": float(
            time.perf_counter()
            - t0
        ),
        "dino_model": DINO_MODEL,
        "dino_feature_shape": list(
            D.shape
        ),
        "representation": (
            "L2-normalized concat of normalized CLS token "
            "and normalized mean patch token"
        ),
        "probe": (
            "StandardScaler + balanced LogisticRegression(C=1)"
        ),
        "outer_test_used_for_selection": False,
        "confirmation_20260827_read": False,
        "gate": gate,
        "transitions": transitions,
        "mean_results": mean_df.to_dict(
            orient="records"
        ),
    }

    (
        OUT_DIR
        / "dino_global_v1_summary.json"
    ).write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # Console
    print()
    print("=" * 160)
    print("DINO-GLOBAL-V1 DEVELOPMENT SUMMARY — MEAN ± STD")
    print("=" * 160)
    print(
        mean_df.to_string(
            index=False
        )
    )

    print()
    print("AUGMENTED DINO vs CANONICAL V+O_old")
    print("-" * 160)
    print(
        d[
            [
                "fold",
                "delta_acc",
                "delta_bacc",
                "delta_macro_f1",
                "delta_normal_f1",
                "delta_logical_f1",
                "delta_structural_f1",
            ]
        ].to_string(
            index=False
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
    print("TRANSITION SUMMARY")
    print("-" * 160)
    print(
        json.dumps(
            transitions,
            ensure_ascii=False,
            indent=2,
        )
    )

    print()
    print("REPRESENTATION CEILING GATE")
    print("-" * 160)

    for k, v in gate.items():
        print(
            f"{k}: {v}"
        )

    print()
    print("Decision:")
    print(
        decision
    )

    print()
    print("Outputs:")
    print(
        OUT_DIR
    )
    print("=" * 160)


if __name__ == "__main__":
    main()
