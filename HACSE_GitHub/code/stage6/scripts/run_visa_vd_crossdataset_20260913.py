#!/usr/bin/env python
"""Strict VisA cross-dataset validation for the Stage-6 V+D method.

This script deliberately has two gates:

1. ``audit`` (default) audits the VisA cohort, materialises a 40-image
   ``capsules`` pilot cohort, and checks whether the existing SALAD/CSAD
   installation can produce the *same* 93-D raw V evidence used on LOCO.
2. ``evaluate`` consumes already extracted 93-D raw V evidence and frozen
   768-D DINOv2-S/14 embeddings.  It performs five outer folds and inner OOF
   stacking without leakage and writes the four requested result artefacts.

The script never substitutes PatchCore or another anomaly representation for
V.  If the exact evidence contract is absent, it stops at the feasibility
gate and records the reason.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SEED = 20260827
N_SPLITS = 5
INNER_SPLITS = 5
EXPECTED_CATEGORIES = [
    "candle",
    "capsules",
    "cashew",
    "chewinggum",
    "fryum",
    "macaroni1",
    "macaroni2",
    "pcb1",
    "pcb2",
    "pcb3",
    "pcb4",
    "pipe_fryum",
]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

WORKSPACE = Path(__file__).resolve().parents[3]
DEFAULT_VISA_ROOT = Path(
    os.environ.get("VISA_ROOT", WORKSPACE / "datasets" / "ViSA_pytorch" / "1cls")
)
DEFAULT_RAW_TAR = Path(
    os.environ.get("VISA_RAW_TAR", WORKSPACE / "datasets" / "VisA_20220922.tar")
)
DEFAULT_OUTPUT = (
    WORKSPACE / "code" / "stage6" / "outputs" /
    "visa_vd_crossdataset_20260913"
)
FEATURE_MANIFEST = WORKSPACE / "code" / "full_loco_salad_csad_fusion_analysis.xlsx"
SALAD_ROOT = WORKSPACE / "复现" / "SALAD" / "00_源码与运行文件"
CSAD_ROOT = WORKSPACE / "复现" / "CSAD" / "00_源码与运行文件" / "CSAD"


def image_files(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_official_split(raw_tar: Path) -> tuple[pd.DataFrame, str]:
    with tarfile.open(raw_tar, "r") as tf:
        names = [n for n in tf.getnames() if n.endswith("split_csv/1cls.csv")]
        if len(names) != 1:
            raise RuntimeError(f"Expected one split_csv/1cls.csv, found {names}")
        payload = tf.extractfile(names[0])
        if payload is None:
            raise RuntimeError("Cannot read official VisA 1cls.csv")
        df = pd.read_csv(io.BytesIO(payload.read()))
    return df, names[0]


def processed_key(category: str, split: str, label: str, stem: str) -> str:
    label_dir = "good" if label == "normal" else "bad"
    return f"{category}/{split}/{label_dir}/{stem}.png"


def audit_dataset(visa_root: Path, raw_tar: Path, output: Path) -> tuple[pd.DataFrame, dict]:
    rows: list[dict] = []
    index_rows: list[dict] = []
    for category in EXPECTED_CATEGORIES:
        parts = {
            ("train", 0): visa_root / category / "train" / "good",
            ("test", 0): visa_root / category / "test" / "good",
            ("test", 1): visa_root / category / "test" / "bad",
        }
        counts = {}
        for (split, label), folder in parts.items():
            files = image_files(folder)
            counts[(split, label)] = len(files)
            for p in files:
                index_rows.append(
                    {
                        "sample_key": p.relative_to(visa_root).as_posix(),
                        "category": category,
                        "label": label,
                        "source_split": split,
                        "image_path": str(p.resolve()),
                    }
                )
        mask_count = len(image_files(visa_root / category / "ground_truth" / "bad"))
        rows.append(
            {
                "category": category,
                "train_normal": counts[("train", 0)],
                "test_normal": counts[("test", 0)],
                "test_anomaly": counts[("test", 1)],
                "anomaly_masks": mask_count,
                "total": sum(counts.values()),
                "mask_pairing_ok": mask_count == counts[("test", 1)],
            }
        )

    counts_df = pd.DataFrame(rows)
    index_df = pd.DataFrame(index_rows)
    split_df, split_member = load_official_split(raw_tar)
    split_df = split_df.copy()
    split_df["stem"] = split_df["image"].map(lambda x: Path(str(x)).stem)
    split_df["expected_identity"] = split_df.apply(
        lambda r: f"{r['object']}||{r['split']}||{r['label']}||{r['stem']}",
        axis=1,
    )
    split_df["expected_key"] = split_df.apply(
        lambda r: processed_key(str(r["object"]), str(r["split"]), str(r["label"]), str(r["stem"])),
        axis=1,
    )
    present = set(
        f"{r.category}||{r.source_split}||{'normal' if int(r.label) == 0 else 'anomaly'}||{Path(r.image_path).stem}"
        for r in index_df.itertuples(index=False)
    )
    missing_df = split_df.loc[
        ~split_df["expected_identity"].isin(present),
        ["object", "split", "label", "image", "mask", "expected_key"],
    ].copy()

    unreadable = []
    for p in index_df["image_path"].map(Path):
        try:
            with Image.open(p) as im:
                im.verify()
        except Exception as exc:  # pragma: no cover - machine/data dependent
            unreadable.append({"image_path": str(p), "error": repr(exc)})

    counts_df.to_csv(output / "data_audit.csv", index=False)
    index_df.to_csv(output / "cohort_index.csv", index=False)
    missing_df.to_csv(output / "missing_official_images.csv", index=False)
    pd.DataFrame(unreadable, columns=["image_path", "error"]).to_csv(
        output / "unreadable_images.csv", index=False
    )

    report = {
        "visa_root": str(visa_root.resolve()),
        "raw_tar": str(raw_tar.resolve()),
        "raw_tar_sha256": file_sha256(raw_tar),
        "official_split_member": split_member,
        "categories_expected": EXPECTED_CATEGORIES,
        "categories_found": sorted(p.name for p in visa_root.iterdir() if p.is_dir()),
        "processed_images": int(len(index_df)),
        "processed_normal": int((index_df["label"] == 0).sum()),
        "processed_anomaly": int((index_df["label"] == 1).sum()),
        "official_rows": int(len(split_df)),
        "missing_official_count": int(len(missing_df)),
        "missing_official": missing_df.to_dict(orient="records"),
        "unreadable_count": len(unreadable),
        "all_masks_paired": bool(counts_df["mask_pairing_ok"].all()),
    }
    return index_df, report


def build_pilot(index_df: pd.DataFrame, output: Path) -> pd.DataFrame:
    """Create a deterministic 40-image, one-category SALAD/CSAD pilot tree."""
    category = "capsules"
    c = index_df[index_df["category"] == category].copy()
    train_good = c[(c.source_split == "train") & (c.label == 0)].head(25)
    test_good = c[(c.source_split == "test") & (c.label == 0)].head(5)
    test_bad = c[(c.source_split == "test") & (c.label == 1)].head(10)

    # SALAD/CSAD require a normal validation directory.  Five of the selected
    # normal training images are held out for that role, leaving 20 train good.
    val_good = train_good.tail(5)
    train_good = train_good.head(20)
    selected = []
    assignments = [
        (train_good, "train", "good"),
        (val_good, "validation", "good"),
        (test_good, "test", "good"),
        (test_bad, "test", "bad"),
    ]
    pilot_root = output / "pilot_capsules_40" / category
    for frame, target_split, target_label in assignments:
        target_dir = pilot_root / target_split / target_label
        target_dir.mkdir(parents=True, exist_ok=True)
        for row in frame.itertuples(index=False):
            source = Path(row.image_path)
            target = target_dir / source.name
            shutil.copy2(source, target)
            selected.append(
                {
                    "sample_key": row.sample_key,
                    "category": category,
                    "binary_label": int(row.label),
                    "pilot_role": f"{target_split}/{target_label}",
                    "source_path": str(source),
                    "pilot_path": str(target.resolve()),
                }
            )
    pilot_df = pd.DataFrame(selected)
    if len(pilot_df) != 40:
        raise RuntimeError(f"Expected 40 pilot images, got {len(pilot_df)}")
    pilot_df.to_csv(output / "pilot_manifest.csv", index=False)
    return pilot_df


def exact_raw_v_columns() -> list[str]:
    feat = pd.read_excel(FEATURE_MANIFEST, sheet_name="features")
    all_cols = list(dict.fromkeys(feat["feature_cols"].dropna().astype(str)))
    raw = [c for c in all_cols if not c.startswith("cat_z_")]
    if len(all_cols) != 186 or len(raw) != 93:
        raise RuntimeError(
            f"Historical V contract changed: all={len(all_cols)}, raw={len(raw)}"
        )
    if any(f"cat_z_{c}" not in all_cols for c in raw):
        raise RuntimeError("Historical raw/cat_z V pairing is incomplete")
    return raw


def feasibility_report(output: Path, pilot_df: pd.DataFrame) -> dict:
    raw_cols = exact_raw_v_columns()
    salad_visa_weights = {
        c: all((SALAD_ROOT / "results" / c / name).exists() for name in [
            "teacher_final.pth",
            "student_final.pth",
            "autoencoder_final.pth",
            "comp_autoencoder_final.pth",
            "comp_unet_final.pth",
        ])
        for c in EXPECTED_CATEGORIES
    }
    csad_visa_weights = {
        c: (CSAD_ROOT / "ckpt" / "pytorch_models" / f"{c}.pth").exists()
        for c in EXPECTED_CATEGORIES
    }
    exact_resources = {
        "salad_teacher_medium": (SALAD_ROOT / "models" / "teacher_medium.pth").exists(),
        "sam_hq_vit_h": (SALAD_ROOT / "pretrained_models" / "sam_hq_vit_h.pth").exists(),
        "sam_vit_h": (SALAD_ROOT / "pretrained_models" / "sam_vit_h_4b8939.pth").exists(),
        "imagenet_penalty_dataset": (SALAD_ROOT / "data" / "imagenet" / "train").exists(),
    }
    code_checks = {
        "salad_train": (SALAD_ROOT / "train_salad.py").exists(),
        "salad_pseudo_labels": (SALAD_ROOT / "create_pseudo_labels.py").exists(),
        "salad_composition_model": (SALAD_ROOT / "train_composition_segmentation_model.py").exists(),
        "csad_train": (CSAD_ROOT / "CSAD.py").exists(),
    }
    pilot_evidence = output / "pilot_evidence_raw93.csv"
    evidence_valid = False
    evidence_error = "not generated"
    if pilot_evidence.exists():
        try:
            ev = pd.read_csv(pilot_evidence)
            missing = [c for c in ["sample_key", *raw_cols] if c not in ev.columns]
            evidence_valid = not missing and len(ev) == len(pilot_df)
            evidence_error = "" if evidence_valid else f"missing={missing}, rows={len(ev)}"
        except Exception as exc:  # pragma: no cover
            evidence_error = repr(exc)

    direct_ready = (
        all(code_checks.values())
        and all(exact_resources.values())
        and salad_visa_weights["capsules"]
        and csad_visa_weights["capsules"]
        and evidence_valid
    )
    blockers = []
    if not all(exact_resources.values()):
        blockers.append("Exact SALAD prerequisite assets are incomplete")
    if not any(salad_visa_weights.values()):
        blockers.append("No SALAD checkpoint exists for any VisA category")
    if not any(csad_visa_weights.values()):
        blockers.append("No CSAD checkpoint exists for any VisA category")
    if not evidence_valid:
        blockers.append("No verified 40-row, exact 93-D pilot V evidence table exists")

    report = {
        "experiment_name": "Cross-dataset validation / transferability evaluation",
        "scope": "VisA binary Normal vs Anomaly; V -> V+D; no O branch",
        "pilot_category": "capsules",
        "pilot_images": int(len(pilot_df)),
        "raw_v_dimension": len(raw_cols),
        "v_dimension_after_fold_contained_category_normalization": 2 * len(raw_cols),
        "raw_v_columns": raw_cols,
        "code_checks": code_checks,
        "exact_resource_checks": exact_resources,
        "salad_visa_checkpoint_checks": salad_visa_weights,
        "csad_visa_checkpoint_checks": csad_visa_weights,
        "pilot_evidence_path": str(pilot_evidence.resolve()),
        "pilot_evidence_valid": evidence_valid,
        "pilot_evidence_error": evidence_error,
        "feasibility_pass": direct_ready,
        "status": "PASS_EXACT_V_AVAILABLE" if direct_ready else "STOP_EXACT_V_NOT_YET_AVAILABLE",
        "blockers": blockers,
        "scientific_decision": (
            "Proceed to one-fold D/V/V+D smoke test"
            if direct_ready
            else "Do not run or report VisA V+D; do not substitute another anomaly feature for V"
        ),
    }
    return report


def category_stats_fit(x: np.ndarray, categories: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    stats = {}
    for category in sorted(set(categories.tolist())):
        xc = x[categories == category]
        mu = xc.mean(axis=0)
        sd = xc.std(axis=0)
        sd[sd < 1e-12] = 1.0
        stats[category] = (mu, sd)
    return stats


def category_stats_transform(
    x: np.ndarray,
    categories: np.ndarray,
    stats: dict[str, tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    z = np.empty_like(x, dtype=np.float32)
    for category in sorted(set(categories.tolist())):
        if category not in stats:
            raise RuntimeError(f"Category {category!r} absent from training statistics")
        mask = categories == category
        mu, sd = stats[category]
        z[mask] = (x[mask] - mu) / sd
    return np.concatenate([x.astype(np.float32), z], axis=1)


def normalize_dino(d: np.ndarray) -> np.ndarray:
    if d.ndim != 2 or d.shape[1] != 768:
        raise RuntimeError(f"DINO feature shape must be (N, 768), got {d.shape}")
    out = d.astype(np.float32).copy()
    for sl in (slice(0, 384), slice(384, 768)):
        norm = np.linalg.norm(out[:, sl], axis=1, keepdims=True)
        norm[norm < 1e-12] = 1.0
        out[:, sl] /= norm
    return out


def make_rf(seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=400,
        class_weight="balanced",
        min_samples_leaf=1,
        max_depth=None,
        random_state=seed,
        n_jobs=-1,
    )


def make_lr(seed: int):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            class_weight="balanced",
            random_state=seed,
            max_iter=2000,
        ),
    )


def binary_metrics(y: np.ndarray, pred: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "f1": f1_score(y, pred, pos_label=1),
        "auroc": roc_auc_score(y, prob),
    }


def align_inputs(raw_v_csv: Path, dino_npz: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    df = pd.read_csv(raw_v_csv)
    raw_cols = exact_raw_v_columns()
    required = ["sample_key", "category", "label", *raw_cols]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Raw V input violates exact 93-D contract; missing={missing}")
    if df["sample_key"].duplicated().any():
        raise RuntimeError("Duplicate sample_key in raw V table")
    x = (
        df[raw_cols].apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )
    if x.isna().any().any():
        raise RuntimeError("Raw V contains non-finite values")

    z = np.load(dino_npz, allow_pickle=True)
    keys = z["sample_keys"].astype(str)
    d = z["embeddings"].astype(np.float32)
    z.close()
    lookup = {k: i for i, k in enumerate(keys.tolist())}
    missing_keys = [k for k in df["sample_key"].astype(str) if k not in lookup]
    if missing_keys:
        raise RuntimeError(f"DINO alignment missing {len(missing_keys)} keys: {missing_keys[:5]}")
    order = [lookup[k] for k in df["sample_key"].astype(str)]
    return df.reset_index(drop=True), x.to_numpy(np.float32), normalize_dino(d[order])


def run_evaluation(raw_v_csv: Path, dino_npz: Path, output: Path) -> None:
    df, raw_v, dino = align_inputs(raw_v_csv, dino_npz)
    y = df["label"].astype(int).to_numpy()
    if set(np.unique(y)) != {0, 1}:
        raise RuntimeError(f"Expected binary labels {{0,1}}, got {np.unique(y)}")
    categories = df["category"].astype(str).to_numpy()
    strata = np.asarray([f"{c}||{label}" for c, label in zip(categories, y)])
    outer = StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED)

    methods = ["DINOv2-S/14", "V", "V+D"]
    n = len(df)
    pred_store = {m: np.full(n, -1, dtype=int) for m in methods}
    prob_store = {m: np.full(n, np.nan) for m in methods}
    fold_ids = np.zeros(n, dtype=int)
    fold_rows = []

    for fold, (tr, te) in enumerate(outer.split(np.zeros(n), strata), start=1):
        fold_ids[te] = fold
        inner = StratifiedKFold(INNER_SPLITS, shuffle=True, random_state=SEED + 1000 + fold)
        pv_oof = np.full((len(tr), 2), np.nan)
        pd_oof = np.full((len(tr), 2), np.nan)
        inner_strata = strata[tr]
        for inner_fold, (itr_rel, iva_rel) in enumerate(
            inner.split(np.zeros(len(tr)), inner_strata), start=1
        ):
            itr, iva = tr[itr_rel], tr[iva_rel]
            stats = category_stats_fit(raw_v[itr], categories[itr])
            xv_tr = category_stats_transform(raw_v[itr], categories[itr], stats)
            xv_va = category_stats_transform(raw_v[iva], categories[iva], stats)
            v_model = make_rf(SEED + fold * 100 + inner_fold)
            v_model.fit(xv_tr, y[itr])
            pv_oof[iva_rel] = v_model.predict_proba(xv_va)

            d_model = make_lr(SEED + fold * 100 + inner_fold)
            d_model.fit(dino[itr], y[itr])
            pd_oof[iva_rel] = d_model.predict_proba(dino[iva])

        if np.isnan(pv_oof).any() or np.isnan(pd_oof).any():
            raise RuntimeError(f"Incomplete inner OOF predictions in outer fold {fold}")
        fusion = make_lr(SEED + 5000 + fold)
        fusion.fit(np.concatenate([pv_oof, pd_oof], axis=1), y[tr])

        stats = category_stats_fit(raw_v[tr], categories[tr])
        xv_tr = category_stats_transform(raw_v[tr], categories[tr], stats)
        xv_te = category_stats_transform(raw_v[te], categories[te], stats)
        v_model = make_rf(SEED + 6000 + fold)
        v_model.fit(xv_tr, y[tr])
        pv = v_model.predict_proba(xv_te)

        d_model = make_lr(SEED + 7000 + fold)
        d_model.fit(dino[tr], y[tr])
        pd_prob = d_model.predict_proba(dino[te])
        pvd = fusion.predict_proba(np.concatenate([pv, pd_prob], axis=1))

        fold_probs = {
            "DINOv2-S/14": pd_prob[:, 1],
            "V": pv[:, 1],
            "V+D": pvd[:, 1],
        }
        for method, prob in fold_probs.items():
            pred = (prob >= 0.5).astype(int)
            pred_store[method][te] = pred
            prob_store[method][te] = prob
            fold_rows.append({"outer_fold": fold, "method": method, **binary_metrics(y[te], pred, prob)})

    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(output / "fold_metrics.csv", index=False)
    summary = (
        fold_df.groupby("method")[["accuracy", "balanced_accuracy", "f1", "auroc"]]
        .agg(["mean", "std"])
    )
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary = summary.reset_index()
    v = fold_df[fold_df.method == "V"].set_index("outer_fold")
    vd = fold_df[fold_df.method == "V+D"].set_index("outer_fold")
    summary["delta_f1_vs_v"] = np.nan
    summary["delta_bacc_vs_v"] = np.nan
    summary["positive_f1_folds_vs_v"] = np.nan
    mask = summary.method == "V+D"
    summary.loc[mask, "delta_f1_vs_v"] = vd.f1.mean() - v.f1.mean()
    summary.loc[mask, "delta_bacc_vs_v"] = vd.balanced_accuracy.mean() - v.balanced_accuracy.mean()
    summary.loc[mask, "positive_f1_folds_vs_v"] = int((vd.f1 > v.f1).sum())
    summary.to_csv(output / "summary.csv", index=False)

    oof = df[["sample_key", "category", "label"]].copy()
    oof["outer_fold"] = fold_ids
    for suffix, method in [("D", "DINOv2-S/14"), ("V", "V"), ("VD", "V+D")]:
        oof[f"pred_{suffix}"] = pred_store[method]
        oof[f"prob_{suffix}_normal"] = 1.0 - prob_store[method]
        oof[f"prob_{suffix}_anomaly"] = prob_store[method]
    oof.to_csv(output / "oof_predictions.csv", index=False)

    config = {
        "experiment_name": "Cross-dataset validation / transferability evaluation",
        "task": "VisA supervised binary diagnosis: Normal vs Anomaly",
        "methods": methods,
        "outer_folds": {"type": "StratifiedKFold", "n_splits": 5, "shuffle": True, "random_state": SEED, "stratum": "category||binary_label"},
        "inner_oof": {"n_splits": INNER_SPLITS, "purpose": "leakage-free V+D stacking"},
        "v": {"raw_dim": 93, "normalized_copy_dim": 93, "final_dim": 186, "normalization": "category-wise, fit on current training partition only", "classifier": "RandomForestClassifier(n_estimators=400,class_weight=balanced,min_samples_leaf=1,max_depth=None)"},
        "d": {"model": "DINOv2-S/14 frozen", "dim": 768, "construction": "L2(CLS-384) || L2(mean-patch-384)", "probe": "StandardScaler + balanced LogisticRegression(C=1)"},
        "fusion": {"input_dim": 4, "model": "StandardScaler + balanced LogisticRegression(C=1)", "training": "inner OOF probabilities only"},
        "raw_v_csv": str(raw_v_csv.resolve()),
        "dino_npz": str(dino_npz.resolve()),
        "n_samples": n,
    }
    (output / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["audit", "evaluate"], default="audit")
    p.add_argument("--visa-root", type=Path, default=DEFAULT_VISA_ROOT)
    p.add_argument("--raw-tar", type=Path, default=DEFAULT_RAW_TAR)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--raw-v-csv", type=Path)
    p.add_argument("--dino-npz", type=Path)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode == "audit":
        index_df, data_report = audit_dataset(args.visa_root, args.raw_tar, args.output)
        pilot_df = build_pilot(index_df, args.output)
        feasibility = feasibility_report(args.output, pilot_df)
        report = {"data_audit": data_report, "feasibility": feasibility}
        (args.output / "feasibility_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (args.output / "config.json").write_text(
            json.dumps(
                {
                    "experiment_name": "Cross-dataset validation / transferability evaluation",
                    "current_stage": "data audit + SALAD/CSAD feasibility gate",
                    "status": feasibility["status"],
                    "outer_protocol_reserved": {"n_splits": 5, "shuffle": True, "random_state": SEED, "stratum": "category||binary_label"},
                    "prohibited_substitution": "PatchCore or any non-identical anomaly representation must not be called V",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "processed_images": data_report["processed_images"],
                    "official_rows": data_report["official_rows"],
                    "missing_official_count": data_report["missing_official_count"],
                    "unreadable_count": data_report["unreadable_count"],
                    "all_masks_paired": data_report["all_masks_paired"],
                    "pilot_images": feasibility["pilot_images"],
                    "feasibility_status": feasibility["status"],
                    "blockers": feasibility["blockers"],
                    "report": str((args.output / "feasibility_report.json").resolve()),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    if args.raw_v_csv is None or args.dino_npz is None:
        raise SystemExit("evaluate mode requires --raw-v-csv and --dino-npz")
    run_evaluation(args.raw_v_csv, args.dino_npz, args.output)


if __name__ == "__main__":
    main()
