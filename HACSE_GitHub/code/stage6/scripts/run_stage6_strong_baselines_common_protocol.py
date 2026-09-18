#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
Stage6 Strong Baselines — Common Strict Protocol
=================================================

One common protocol for:
SALAD-only / CSAD-only / SALAD+CSAD / ResNet18 / CLIP ViT-B/32 /
DINOv2-only / V / V+O / V+DINO / V+O+DINO.

Protocol:
- same 1568 MVTec LOCO AD samples
- StratifiedKFold(5, shuffle=True, random_state=20260827)
- strata = category || N/L/S
- generic frozen-feature baselines use:
    StandardScaler + LogisticRegression(C=1, class_weight="balanced")
- no backbone fine-tuning
- no hyperparameter search
- V/O/DINO Stage6 variants reuse the frozen 20260827 implementation/seeds

Run:
    cd code/stage6
    python scripts\run_stage6_strong_baselines_common_protocol.py
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------
# Import frozen Stage6 code
# ---------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
CONFIRM_SCRIPT = HERE / "run_stage6_frozen_dino_robustness_20260827.py"

if not CONFIRM_SCRIPT.exists():
    raise FileNotFoundError(CONFIRM_SCRIPT)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


confirm = load_module(CONFIRM_SCRIPT, "stage6_confirm_common")
v6 = confirm.v6
base = confirm.base

CLASS_ORDER = list(confirm.CLASS_ORDER)
V_MODEL = confirm.V_MODEL
O_MODEL = confirm.O_MODEL


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
FEATURE_FILE = ROOT / "full_loco_salad_csad_fusion_analysis.xlsx"

ALIGNED_22D = (
    ROOT / "stage2_three_class" / "00_inputs" / "aligned_input_scores.csv"
)

DINO_CACHE = (
    ROOT / "stage6" / "outputs"
    / "stage6_dino_global_v1_ceiling_dev"
    / "dino_global_v1_features.npz"
)

PRIOR_CONFIRM_DIR = (
    ROOT / "stage6" / "outputs"
    / "stage6_frozen_dino_robustness_20260827"
)

OUT_DIR = (
    ROOT / "stage6" / "outputs"
    / "stage6_strong_baselines_common_protocol"
)
CACHE_DIR = OUT_DIR / "cache"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

RESNET_CACHE = CACHE_DIR / "resnet18_global_features.npz"
CLIP_CACHE = CACHE_DIR / "clip_vit_b32_features.npz"

EXPECTED_N = 1568
OUTER_RANDOM_STATE = 20260827
N_SPLITS = 5

SALAD_COLS = [
    "salad_score",
    "salad_score_img",
    "salad_score_mah",
    "salad_score_comp",
]

CSAD_COLS = [
    "local__patch_hist_score__scalar",
    "local__lgst_score__scalar",
    "local__diff_local__max",
    "local__diff_local__mean",
    "local__diff_local__q95",
    "local__diff_local__q99",
    "local__diff_local__top1pct_mean",
    "local__diff_local__top5pct_mean",
    "local__diff_global__max",
    "local__diff_global__mean",
    "local__diff_global__q95",
    "local__diff_global__q99",
    "local__diff_global__top1pct_mean",
    "local__diff_global__top5pct_mean",
    "local__segmap__seg_conf_mean",
    "local__segmap__seg_conf_std",
    "local__segmap__seg_entropy_mean",
    "local__segmap__seg_entropy_std",
]

assert len(SALAD_COLS) == 4
assert len(CSAD_COLS) == 18

EXTERNAL_VARIANTS = [
    "SALAD-only",
    "CSAD-only",
    "SALAD+CSAD",
    "ResNet18",
    "CLIP-ViT-B32",
    "DINOv2-only",
]

STAGE6_VARIANTS = [
    "V",
    "V+O",
    "V+DINO",
    "V+O+DINO",
]

VARIANTS = EXTERNAL_VARIANTS + STAGE6_VARIANTS

METRICS = [
    "acc",
    "bacc",
    "macro_f1",
    "normal_f1",
    "logical_f1",
    "structural_f1",
]


# ---------------------------------------------------------------------
# Generic utilities
# ---------------------------------------------------------------------

def linear_factory(seed: int):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=4000,
            class_weight="balanced",
            C=1.0,
            random_state=seed,
        ),
    )


def aligned_proba(model, X):
    p = model.predict_proba(X)
    out = np.zeros((len(X), len(CLASS_ORDER)), dtype=np.float64)
    for j, c in enumerate(model.classes_):
        if c in CLASS_ORDER:
            out[:, CLASS_ORDER.index(c)] = p[:, j]
    s = out.sum(axis=1, keepdims=True)
    s[s <= 0] = 1.0
    return out / s


def pred_from_prob(p):
    labels = np.asarray(CLASS_ORDER, dtype=object)
    return labels[np.argmax(p, axis=1)]


def safe_name(name: str):
    return name.replace("+", "_").replace("-", "_")


def normalize_key(value):
    text = str(value).strip().replace("\\", "/")
    text = re.sub(r"/+", "/", text)
    marker = "mvtec_loco_anomaly_detection/"
    low = text.lower()
    pos = low.find(marker)
    if pos >= 0:
        text = text[pos + len(marker):]
    return text.lstrip("./").lower()


def l2_normalize(X):
    X = np.asarray(X, dtype=np.float32)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n <= 1e-12] = 1.0
    return X / n


def generic_outer_probe(X, y, tr, te, seed):
    m = linear_factory(seed)
    m.fit(X[tr], y[tr])
    return aligned_proba(m, X[te])


# ---------------------------------------------------------------------
# Load benchmark
# ---------------------------------------------------------------------

def load_benchmark():
    df = pd.read_excel(FEATURE_FILE, sheet_name="merged_data")
    df = df.loc[:, ~df.columns.duplicated()].copy().reset_index(drop=True)

    if len(df) != EXPECTED_N:
        raise RuntimeError(f"Expected {EXPECTED_N} rows, got {len(df)}")

    feat = pd.read_excel(FEATURE_FILE, sheet_name="features")
    v_cols = list(dict.fromkeys(
        feat["feature_cols"].dropna().astype(str).tolist()
    ))

    if len(v_cols) != 186:
        raise RuntimeError(f"Expected 186 V features, got {len(v_cols)}")

    Vdf = (
        df.reindex(columns=v_cols)
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    V = Vdf.to_numpy(dtype=np.float32)

    y = df.apply(base.infer_gt_group, axis=1).to_numpy(dtype=object)
    categories = df["category"].astype(str).reset_index(drop=True)

    _, _, O = base.load_o()
    recs = base.load_raw()

    z = np.load(DINO_CACHE, allow_pickle=True)
    D = z["embeddings"].astype(np.float32)
    sample_keys = z["sample_keys"].astype(str).tolist()
    dino_model = str(z["model_name"].item())
    z.close()

    if D.shape[0] != EXPECTED_N:
        raise RuntimeError(f"DINO rows={D.shape[0]}")

    return df, V, O, recs, D, y, categories, sample_keys, dino_model


def load_aligned_22d(sample_keys):
    a = pd.read_csv(ALIGNED_22D)

    required = ["image_key", *SALAD_COLS, *CSAD_COLS]
    missing = [c for c in required if c not in a.columns]
    if missing:
        raise RuntimeError(f"22-D input missing columns: {missing}")

    if len(a) != EXPECTED_N:
        raise RuntimeError(f"22-D input rows={len(a)}")

    a = a.copy()
    a["_key"] = a["image_key"].map(normalize_key)

    if a["_key"].duplicated().any():
        raise RuntimeError("Duplicate image_key in aligned_input_scores.csv")

    lookup = {k: i for i, k in enumerate(a["_key"].tolist())}
    order = []
    missing_keys = []

    for key in sample_keys:
        nk = normalize_key(key)
        if nk not in lookup:
            missing_keys.append(nk)
        else:
            order.append(lookup[nk])

    if missing_keys:
        raise RuntimeError(
            f"{len(missing_keys)} keys fail 22-D alignment; "
            f"examples={missing_keys[:10]}"
        )

    a = a.iloc[order].reset_index(drop=True)

    def block(cols):
        return (
            a[cols]
            .apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .to_numpy(dtype=np.float32)
        )

    salad = block(SALAD_COLS)
    csad = block(CSAD_COLS)
    both = np.concatenate([salad, csad], axis=1).astype(np.float32)

    if salad.shape != (EXPECTED_N, 4):
        raise RuntimeError(salad.shape)
    if csad.shape != (EXPECTED_N, 18):
        raise RuntimeError(csad.shape)
    if both.shape != (EXPECTED_N, 22):
        raise RuntimeError(both.shape)

    return salad, csad, both


# ---------------------------------------------------------------------
# Image path resolution
# ---------------------------------------------------------------------

def dataset_root():
    candidates = [
        Path(os.environ["MVTEC_LOCO_ROOT"])
        if os.environ.get("MVTEC_LOCO_ROOT")
        else ROOT.parent / "datasets" / "mvtec_loco_anomaly_detection",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("MVTec LOCO root not found")


def resolve_images(df, sample_keys):
    path_col = None
    for c in ["image_path", "filepath", "file_path", "img_path", "path"]:
        if c in df.columns:
            path_col = c
            break

    root = dataset_root()
    out = []

    for i, key in enumerate(sample_keys):
        p = None

        if path_col is not None:
            raw = str(df.iloc[i][path_col]).strip()
            if raw and raw.lower() != "nan":
                q = Path(raw)
                if q.exists():
                    p = q

        if p is None:
            rel = Path(normalize_key(key).replace("/", os.sep))
            q = root / rel
            if q.exists():
                p = q

        if p is None:
            raise FileNotFoundError(f"Cannot resolve image: {key}")

        out.append(p)

    return out


# ---------------------------------------------------------------------
# ResNet18 extraction
# ---------------------------------------------------------------------

def resnet_weight():
    hf_home = Path(
        os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")
    )
    root = hf_home / "hub" / "models--timm--resnet18.a1_in1k" / "snapshots"
    found = sorted(root.glob("*/model.safetensors")) if root.exists() else []
    if found:
        return found[0]

    raise FileNotFoundError("Cached timm ResNet18 weights not found")


def extract_resnet18(image_paths, sample_keys):
    if RESNET_CACHE.exists():
        z = np.load(RESNET_CACHE, allow_pickle=True)
        X = z["embeddings"].astype(np.float32)
        keys = z["sample_keys"].astype(str).tolist()
        z.close()
        if keys != list(sample_keys):
            raise RuntimeError("ResNet18 cache key mismatch")
        print("Using ResNet18 cache:", RESNET_CACHE, X.shape)
        return X

    import torch
    import timm
    from safetensors.torch import load_file
    from timm.data import create_transform, resolve_model_data_config
    from torch.utils.data import Dataset, DataLoader

    class DS(Dataset):
        def __init__(self, paths, transform):
            self.paths = paths
            self.transform = transform

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, idx):
            with Image.open(self.paths[idx]) as im:
                x = self.transform(im.convert("RGB"))
            return x, idx

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight = resnet_weight()

    model = timm.create_model(
        "resnet18.a1_in1k",
        pretrained=False,
        num_classes=0,
        global_pool="avg",
    )

    state = load_file(str(weight))
    missing, unexpected = model.load_state_dict(state, strict=False)

    bad_missing = [x for x in missing if not x.startswith(("fc.", "classifier."))]
    bad_unexpected = [x for x in unexpected if not x.startswith(("fc.", "classifier."))]

    if bad_missing or bad_unexpected:
        raise RuntimeError(
            f"ResNet18 weight mismatch; "
            f"missing={bad_missing[:10]}, unexpected={bad_unexpected[:10]}"
        )

    model = model.to(device).eval()
    transform = create_transform(
        **resolve_model_data_config(model),
        is_training=False,
    )

    # Windows uses multiprocessing "spawn". Because DS is intentionally
    # defined locally inside this extraction function, worker processes cannot
    # pickle it. Use single-process loading here; with only 1568 images this is
    # negligible compared with GPU feature extraction and is maximally robust
    # on Windows/PyCharm/Anaconda.
    loader = DataLoader(
        DS(image_paths, transform),
        batch_size=64,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    feats = []
    indices = []

    print("Extracting ResNet18 embeddings...")

    with torch.inference_mode():
        for xb, idx in loader:
            xb = xb.to(device, non_blocking=True)
            f = model(xb)
            if f.ndim > 2:
                f = torch.flatten(f, 1)
            feats.append(f.float().cpu().numpy())
            indices.extend(idx.numpy().tolist())

    if indices != list(range(EXPECTED_N)):
        raise RuntimeError("ResNet18 extraction order mismatch")

    X = l2_normalize(np.concatenate(feats, axis=0))

    np.savez_compressed(
        RESNET_CACHE,
        embeddings=X,
        sample_keys=np.asarray(sample_keys, dtype=object),
        model_name=np.asarray("timm/resnet18.a1_in1k", dtype=object),
        weight_path=np.asarray(str(weight), dtype=object),
    )

    print("Saved ResNet18 cache:", RESNET_CACHE, X.shape)
    return X


# ---------------------------------------------------------------------
# CLIP extraction
# ---------------------------------------------------------------------

def clip_snapshot():
    hf_home = Path(
        os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")
    )
    root = hf_home / "hub" / "models--openai--clip-vit-base-patch32" / "snapshots"

    if not root.exists():
        raise FileNotFoundError(root)

    valid = []

    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue

        if not (p / "config.json").exists():
            continue

        if not (
            (p / "model.safetensors").exists()
            or (p / "pytorch_model.bin").exists()
        ):
            continue

        valid.append(p)

    if not valid:
        raise FileNotFoundError("No complete CLIP ViT-B/32 snapshot")

    valid.sort(
        key=lambda p: (
            not (
                (p / "preprocessor_config.json").exists()
                or (p / "processor_config.json").exists()
            ),
            str(p),
        )
    )

    return valid[0]


def extract_clip(image_paths, sample_keys):
    if CLIP_CACHE.exists():
        z = np.load(CLIP_CACHE, allow_pickle=True)
        X = z["embeddings"].astype(np.float32)
        keys = z["sample_keys"].astype(str).tolist()
        z.close()
        if keys != list(sample_keys):
            raise RuntimeError("CLIP cache key mismatch")
        print("Using CLIP cache:", CLIP_CACHE, X.shape)
        return X

    import torch
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    snap = clip_snapshot()

    processor = CLIPImageProcessor.from_pretrained(
        str(snap),
        local_files_only=True,
    )

    model = CLIPVisionModelWithProjection.from_pretrained(
        str(snap),
        local_files_only=True,
    ).to(device).eval()

    feats = []
    bs = 64

    print("Extracting CLIP ViT-B/32 embeddings...")

    with torch.inference_mode():
        for start in range(0, len(image_paths), bs):
            paths = image_paths[start:start + bs]
            images = []

            for p in paths:
                with Image.open(p) as im:
                    images.append(im.convert("RGB").copy())

            inp = processor(images=images, return_tensors="pt")
            px = inp["pixel_values"].to(device)
            out = model(pixel_values=px)

            feats.append(out.image_embeds.float().cpu().numpy())

            if start == 0 or (start + bs) % 512 == 0:
                print(f"  CLIP {min(start+bs, len(image_paths))}/{len(image_paths)}")

    X = l2_normalize(np.concatenate(feats, axis=0))

    np.savez_compressed(
        CLIP_CACHE,
        embeddings=X,
        sample_keys=np.asarray(sample_keys, dtype=object),
        model_name=np.asarray("openai/clip-vit-base-patch32", dtype=object),
        snapshot=np.asarray(str(snap), dtype=object),
    )

    print("Saved CLIP cache:", CLIP_CACHE, X.shape)
    return X


# ---------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------

def summarize(fold_df):
    rows = []
    for variant in VARIANTS:
        g = fold_df[fold_df["variant"] == variant]
        row = {"variant": variant, "n_folds": int(len(g))}
        for m in METRICS:
            row[m + "_mean"] = float(g[m].mean())
            row[m + "_std"] = float(g[m].std(ddof=0))
        rows.append(row)
    return pd.DataFrame(rows)


def pooled(pred_df):
    rows = []
    y = pred_df["label"].to_numpy(dtype=object)

    for variant in VARIANTS:
        pred = pred_df["pred_" + safe_name(variant)].to_numpy(dtype=object)
        rows.append({
            "variant": variant,
            "n": len(pred_df),
            **base.metrics(y, pred),
        })

    return pd.DataFrame(rows)


def category_pooled(pred_df):
    rows = []
    for cat, g in pred_df.groupby("category", sort=True):
        y = g["label"].to_numpy(dtype=object)
        for variant in VARIANTS:
            pred = g["pred_" + safe_name(variant)].to_numpy(dtype=object)
            rows.append({
                "category": cat,
                "variant": variant,
                "n": len(g),
                **base.metrics(y, pred),
            })
    return pd.DataFrame(rows)


def final_deltas(mean_df):
    final = mean_df[mean_df["variant"] == "V+O+DINO"].iloc[0]
    rows = []

    for variant in VARIANTS:
        if variant == "V+O+DINO":
            continue

        b = mean_df[mean_df["variant"] == variant].iloc[0]
        rows.append({
            "baseline": variant,
            "delta_acc": float(final["acc_mean"] - b["acc_mean"]),
            "delta_bacc": float(final["bacc_mean"] - b["bacc_mean"]),
            "delta_macro_f1": float(final["macro_f1_mean"] - b["macro_f1_mean"]),
            "delta_structural_f1": float(
                final["structural_f1_mean"] - b["structural_f1_mean"]
            ),
        })

    return pd.DataFrame(rows)


def reproduction_check(mean_df):
    candidates = [
        PRIOR_CONFIRM_DIR / "frozen_dino_mean_std.csv",
        PRIOR_CONFIRM_DIR / "frozen_dino_robustness_mean_std.csv",
    ]

    prior_path = next((p for p in candidates if p.exists()), None)

    out = {
        "checked": False,
        "prior_path": str(prior_path) if prior_path else None,
        "max_abs_diff": None,
        "pass": None,
        "per_variant": {},
    }

    if prior_path is None:
        return out

    old = pd.read_csv(prior_path)
    if "variant" not in old.columns:
        return out

    diffs = []

    for variant in STAGE6_VARIANTS:
        a = mean_df[mean_df["variant"] == variant]
        b = old[old["variant"] == variant]

        if len(a) != 1 or len(b) != 1:
            continue

        a = a.iloc[0]
        b = b.iloc[0]
        per = {}

        for m in METRICS:
            col = m + "_mean"
            if col not in b.index:
                continue

            d = abs(float(a[col]) - float(b[col]))
            per[col] = d
            diffs.append(d)

        out["per_variant"][variant] = per

    if diffs:
        out["checked"] = True
        out["max_abs_diff"] = float(max(diffs))
        out["pass"] = bool(max(diffs) <= 1e-10)

    return out


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    t0 = time.perf_counter()

    print("=" * 170)
    print("STAGE6 STRONG BASELINES — COMMON STRICT 20260827 PROTOCOL")
    print("=" * 170)

    for p in [FEATURE_FILE, ALIGNED_22D, DINO_CACHE, base.O_FILE, base.O_RAW]:
        if not p.exists():
            raise FileNotFoundError(p)

    df, V, O, recs, D, y, categories, sample_keys, dino_model = load_benchmark()
    salad, csad, salad_csad = load_aligned_22d(sample_keys)

    image_paths = resolve_images(df, sample_keys)
    R18 = extract_resnet18(image_paths, sample_keys)
    CLIP = extract_clip(image_paths, sample_keys)

    features = {
        "SALAD-only": salad,
        "CSAD-only": csad,
        "SALAD+CSAD": salad_csad,
        "ResNet18": R18,
        "CLIP-ViT-B32": CLIP,
        "DINOv2-only": D,
    }

    print("\nFeature shapes:")
    for k, X in features.items():
        print(f"  {k:<15} {X.shape}")
    print(f"  V               {V.shape}")
    print(f"  O               {O.shape}")

    strata = (
        categories.astype(str)
        + "||"
        + pd.Series(y).astype(str)
    ).to_numpy(dtype=object)

    skf = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=OUTER_RANDOM_STATE,
    )

    dummy = np.zeros(EXPECTED_N, dtype=np.uint8)

    fold_rows = []
    fold_assign = np.zeros(EXPECTED_N, dtype=int)

    pred_store = {
        v: np.empty(EXPECTED_N, dtype=object)
        for v in VARIANTS
    }

    prob_store = {
        v: np.full((EXPECTED_N, len(CLASS_ORDER)), np.nan, dtype=np.float64)
        for v in VARIANTS
    }

    ext_seed = {
        "SALAD-only": 930000,
        "CSAD-only": 931000,
        "SALAD+CSAD": 932000,
        "ResNet18": 933000,
        "CLIP-ViT-B32": 934000,
        "DINOv2-only": 935000,
    }

    for fold, (tr, te) in enumerate(skf.split(dummy, strata), start=1):
        tr = tr.astype(int)
        te = te.astype(int)
        fold_assign[te] = fold

        print("\n" + "-" * 170)
        print(f"OUTER FOLD {fold}/5 | train={len(tr)} | test={len(te)}")

        probs = {}

        # Fixed-feature direct linear probes.
        for variant in EXTERNAL_VARIANTS:
            probs[variant] = generic_outer_probe(
                features[variant],
                y,
                tr,
                te,
                ext_seed[variant] + fold,
            )

        # Frozen V/O OOF, exactly same helper and pseudo outer id.
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
            tr,
            800 + fold,
            V_MODEL,
            O_MODEL,
        )

        ytr = y[tr]

        # Frozen DINO OOF exactly as confirmation.
        pdino_oof = confirm.strict_dino_oof(
            D,
            y,
            categories,
            tr,
            fold,
        )

        # Frozen V/O outer test signals.
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
            tr,
            te,
            800 + fold,
            V_MODEL,
            O_MODEL,
        )

        # Frozen DINO outer probe.
        pdino_te = confirm.fit_predict_dino_outer(
            D,
            y,
            tr,
            te,
            fold,
        )

        # V direct.
        mv = base.model_factory(V_MODEL, 884000 + fold)
        mv.fit(V[tr], y[tr])
        probs["V"] = base.aligned_proba(mv, V[te])

        # V+O.
        f_vo = base.fusion_factory(885000 + fold)
        f_vo.fit(base.meta_vo(pv_oof, po_oof), ytr)
        probs["V+O"] = base.aligned_proba(
            f_vo,
            base.meta_vo(pv_te, po_te),
        )

        # V+DINO.
        f_vd = confirm.linear_factory(886000 + fold)
        f_vd.fit(confirm.meta_v_dino(pv_oof, pdino_oof), ytr)
        probs["V+DINO"] = confirm.aligned_proba_local(
            f_vd,
            confirm.meta_v_dino(pv_te, pdino_te),
        )

        # V+O+DINO.
        f_vod = confirm.linear_factory(887000 + fold)
        f_vod.fit(
            confirm.meta_vo_dino(pv_oof, po_oof, pdino_oof),
            ytr,
        )
        probs["V+O+DINO"] = confirm.aligned_proba_local(
            f_vod,
            confirm.meta_vo_dino(pv_te, po_te, pdino_te),
        )

        for variant in VARIANTS:
            p = probs[variant]
            pred = pred_from_prob(p)

            pred_store[variant][te] = pred
            prob_store[variant][te] = p

            m = base.metrics(y[te], pred)

            fold_rows.append({
                "outer_fold": fold,
                "variant": variant,
                **m,
            })

            print(
                f"  {variant:<15} "
                f"Acc={m['acc']:.4f} "
                f"BAcc={m['bacc']:.4f} "
                f"MacroF1={m['macro_f1']:.4f} "
                f"N/L/S={m['normal_f1']:.4f}/"
                f"{m['logical_f1']:.4f}/"
                f"{m['structural_f1']:.4f}"
            )

    fold_df = pd.DataFrame(fold_rows)
    mean_df = summarize(fold_df)

    pred_df = pd.DataFrame({
        "row_index": np.arange(EXPECTED_N, dtype=int),
        "sample_key": sample_keys,
        "category": categories.to_numpy(),
        "label": y,
        "outer_fold": fold_assign,
    })

    for variant in VARIANTS:
        s = safe_name(variant)
        pred_df["pred_" + s] = pred_store[variant]

        for ci, cls in enumerate(CLASS_ORDER):
            pred_df[f"prob_{s}_{cls}"] = prob_store[variant][:, ci]

    pooled_df = pooled(pred_df)
    cat_df = category_pooled(pred_df)
    delta_df = final_deltas(mean_df)
    repro = reproduction_check(mean_df)

    if repro["checked"] and not repro["pass"]:
        raise RuntimeError(
            "Frozen V/O/DINO reproduction failed: "
            f"max_abs_diff={repro['max_abs_diff']}"
        )

    fold_df.to_csv(
        OUT_DIR / "strong_baselines_by_fold.csv",
        index=False,
        encoding="utf-8-sig",
    )

    mean_df.to_csv(
        OUT_DIR / "strong_baselines_mean_std.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pooled_df.to_csv(
        OUT_DIR / "strong_baselines_pooled_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pred_df.to_csv(
        OUT_DIR / "strong_baselines_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    delta_df.to_csv(
        OUT_DIR / "strong_baselines_final_vs_baselines.csv",
        index=False,
        encoding="utf-8-sig",
    )

    cat_df.to_csv(
        OUT_DIR / "strong_baselines_category_pooled.csv",
        index=False,
        encoding="utf-8-sig",
    )

    (
        OUT_DIR / "strong_baselines_reproduction_check.json"
    ).write_text(
        json.dumps(repro, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    nonfinal = mean_df[mean_df["variant"] != "V+O+DINO"].copy()
    best = nonfinal.loc[nonfinal["macro_f1_mean"].idxmax()].to_dict()
    final = mean_df[mean_df["variant"] == "V+O+DINO"].iloc[0].to_dict()

    report = {
        "status": "STRONG_BASELINES_COMMON_PROTOCOL_COMPLETE",
        "protocol": {
            "n_samples": EXPECTED_N,
            "n_splits": N_SPLITS,
            "outer_random_state": OUTER_RANDOM_STATE,
            "strata": "category||N/L/S",
            "generic_probe": (
                "StandardScaler + balanced LogisticRegression(C=1,max_iter=4000)"
            ),
            "backbone_finetuning": False,
            "hyperparameter_search": False,
        },
        "encoders": {
            "ResNet18": "timm/resnet18.a1_in1k",
            "CLIP": "openai/clip-vit-base-patch32",
            "DINO": dino_model,
        },
        "best_nonfinal_by_macro_f1": best,
        "final": final,
        "reproduction": repro,
        "elapsed_seconds": float(time.perf_counter() - t0),
    }

    (
        OUT_DIR / "strong_baselines_summary.json"
    ).write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 170)
    print("STRONG BASELINES — MEAN ± STD")
    print("=" * 170)

    cols = [
        "variant",
        "n_folds",
        "acc_mean",
        "acc_std",
        "bacc_mean",
        "bacc_std",
        "macro_f1_mean",
        "macro_f1_std",
        "normal_f1_mean",
        "logical_f1_mean",
        "structural_f1_mean",
    ]
    print(mean_df[cols].to_string(index=False))

    print("\nFINAL V+O+DINO MINUS EACH BASELINE")
    print("-" * 170)
    print(delta_df.to_string(index=False))

    print("\nPOOLED METRICS")
    print("-" * 170)
    print(pooled_df.to_string(index=False))

    print("\nFROZEN STAGE6 REPRODUCTION CHECK")
    print("-" * 170)
    print(json.dumps(repro, ensure_ascii=False, indent=2))

    print("\nBEST NON-FINAL COMPARATOR BY MACRO-F1")
    print("-" * 170)
    print(json.dumps(best, ensure_ascii=False, indent=2))

    print("\nStatus:")
    print("STRONG_BASELINES_COMMON_PROTOCOL_COMPLETE")

    print("\nOutputs:")
    print(OUT_DIR)
    print("=" * 170)


if __name__ == "__main__":
    main()
