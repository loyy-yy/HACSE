#!/usr/bin/env python
"""Strong supervised RGB baselines under the frozen Stage-6 outer folds.

Models: ImageNet-1K pretrained ResNet50, ConvNeXt-Tiny, and ViT-B/16.  All
layers are fine-tuned for the supervised Normal/Logical/Structural task.
Outer-test data are never used for training, early stopping, or model choice.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


SEED = 20260827
CLASS_ORDER = ["normal", "logical", "structural"]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_ORDER)}
ROOT = Path(__file__).resolve().parents[2]
WORKBOOK = ROOT / "full_loco_salad_csad_fusion_analysis.xlsx"
FOLD_FILE = ROOT / "stage6" / "outputs" / "stage6_nested5fold_ablation" / "outer_fold_assignments.csv"
DATA_ROOTS = [
    Path(os.environ["MVTEC_LOCO_ROOT"])
    if os.environ.get("MVTEC_LOCO_ROOT")
    else ROOT.parent / "datasets" / "mvtec_loco_anomaly_detection",
]
WEIGHT_DIR = ROOT / "stage6" / "weights"
WEIGHTS = {
    "resnet50": WEIGHT_DIR / "resnet50-11ad3fa6.pth",
    "convnext_tiny": WEIGHT_DIR / "convnext_tiny-983f1562.pth",
    "vit_b_16": Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / "vit_b_16-c867db91.pth",
}
OUT_DIR = ROOT / "stage6" / "outputs" / "stage6_supervised_rgb_finetuned_20260914"
CACHE_DIR = OUT_DIR / "rgb_short_side_256_cache"


class ImageDataset(Dataset):
    def __init__(self, paths: list[Path], labels: np.ndarray, indices: np.ndarray, transform):
        self.paths = paths
        self.labels = labels
        self.indices = np.asarray(indices, dtype=int)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        global_index = int(self.indices[item])
        with Image.open(self.paths[global_index]) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, int(self.labels[global_index]), global_index


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_dataset_root() -> Path:
    for root in DATA_ROOTS:
        if root.exists():
            return root
    raise FileNotFoundError(f"MVTec LOCO root not found: {DATA_ROOTS}")


def load_cohort() -> tuple[pd.DataFrame, list[Path], np.ndarray, np.ndarray]:
    df = pd.read_excel(WORKBOOK, sheet_name="merged_data")
    folds = pd.read_csv(FOLD_FILE).sort_values("row_index").reset_index(drop=True)
    if len(df) != 1568 or len(folds) != 1568:
        raise RuntimeError(f"Expected 1568 rows, got workbook={len(df)}, folds={len(folds)}")
    if folds["row_index"].tolist() != list(range(1568)):
        raise RuntimeError("Fold assignment row_index is not exactly 0..1567")
    if not np.array_equal(df["category"].astype(str), folds["category"].astype(str)):
        raise RuntimeError("Category mismatch between workbook and fold assignment")
    if not np.array_equal(df["gt_group"].astype(str), folds["gt_group"].astype(str)):
        raise RuntimeError("Label mismatch between workbook and fold assignment")

    data_root = resolve_dataset_root()
    paths = [data_root / Path(str(key).replace("/", os.sep)) for key in df["merge_key"]]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} RGB images; examples={missing[:5]}")
    labels = df["gt_group"].map(CLASS_TO_ID)
    if labels.isna().any():
        raise RuntimeError(f"Unexpected labels: {sorted(df['gt_group'].astype(str).unique())}")
    return df, paths, labels.to_numpy(dtype=np.int64), folds["outer_test_fold"].to_numpy(dtype=int)


def materialize_rgb_cache(paths: list[Path], workers: int = 8) -> list[Path]:
    """Decode each source once and cache a high-quality short-side-256 JPEG."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    targets = [CACHE_DIR / f"{i:04d}.jpg" for i in range(len(paths))]

    def convert(pair: tuple[Path, Path]) -> None:
        source, target = pair
        if target.exists() and target.stat().st_size > 0:
            return
        with Image.open(source) as image:
            image = image.convert("RGB")
            width, height = image.size
            scale = 256.0 / min(width, height)
            resized = image.resize(
                (max(256, round(width * scale)), max(256, round(height * scale))),
                Image.Resampling.BILINEAR,
            )
            resized.save(target, format="JPEG", quality=95, subsampling=0)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        list(executor.map(convert, zip(paths, targets)))
    if not all(path.exists() and path.stat().st_size > 0 for path in targets):
        raise RuntimeError("RGB cache is incomplete")
    return targets


def train_transform():
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.80, 1.00), ratio=(0.90, 1.10)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def eval_transform():
    return transforms.Compose(
        [
            transforms.Resize(256, antialias=True),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def build_model(name: str) -> nn.Module:
    path = WEIGHTS[name]
    if not path.exists():
        raise FileNotFoundError(f"Missing pretrained weight: {path}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if name == "resnet50":
        model = models.resnet50(weights=None)
        model.load_state_dict(state, strict=True)
        model.fc = nn.Linear(model.fc.in_features, len(CLASS_ORDER))
    elif name == "convnext_tiny":
        model = models.convnext_tiny(weights=None)
        model.load_state_dict(state, strict=True)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, len(CLASS_ORDER))
    elif name == "vit_b_16":
        model = models.vit_b_16(weights=None)
        model.load_state_dict(state, strict=True)
        model.heads.head = nn.Linear(model.heads.head.in_features, len(CLASS_ORDER))
    else:  # pragma: no cover
        raise ValueError(name)
    return model


def metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    prediction = probabilities.argmax(axis=1)
    f1s = f1_score(y_true, prediction, labels=[0, 1, 2], average=None, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "macro_f1": float(f1_score(y_true, prediction, average="macro", zero_division=0)),
        "normal_f1": float(f1s[0]),
        "logical_f1": float(f1s[1]),
        "structural_f1": float(f1s[2]),
    }


@torch.inference_mode()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_indices, all_labels, all_probabilities = [], [], []
    for images, labels, indices in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(images)
        all_indices.append(indices.numpy())
        all_labels.append(labels.numpy())
        all_probabilities.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
    return np.concatenate(all_indices), np.concatenate(all_labels), np.concatenate(all_probabilities)


def train_one_fold(
    model_name: str,
    fold: int,
    paths: list[Path],
    labels: np.ndarray,
    categories: np.ndarray,
    outer_folds: np.ndarray,
    args: argparse.Namespace,
) -> None:
    pred_path = OUT_DIR / f"{model_name}_fold{fold}_predictions.csv"
    hist_path = OUT_DIR / f"{model_name}_fold{fold}_history.csv"
    if pred_path.exists() and hist_path.exists() and not args.force:
        print(f"SKIP complete {model_name} fold {fold}", flush=True)
        return

    fold_seed = SEED + {"resnet50": 10000, "convnext_tiny": 20000, "vit_b_16": 30000}[model_name] + fold
    seed_everything(fold_seed)
    outer_train = np.where(outer_folds != fold)[0]
    outer_test = np.where(outer_folds == fold)[0]
    strata = np.asarray([f"{categories[i]}||{labels[i]}" for i in outer_train])
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=args.val_fraction, random_state=fold_seed)
    train_rel, val_rel = next(splitter.split(np.zeros(len(outer_train)), strata))
    train_idx, val_idx = outer_train[train_rel], outer_train[val_rel]

    generator = torch.Generator().manual_seed(fold_seed)
    common = {"num_workers": args.workers, "pin_memory": True}
    if args.workers > 0:
        common["persistent_workers"] = True
    train_loader = DataLoader(
        ImageDataset(paths, labels, train_idx, train_transform()),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        **common,
    )
    val_loader = DataLoader(
        ImageDataset(paths, labels, val_idx, eval_transform()),
        batch_size=args.eval_batch_size,
        shuffle=False,
        **common,
    )
    test_loader = DataLoader(
        ImageDataset(paths, labels, outer_test, eval_transform()),
        batch_size=args.eval_batch_size,
        shuffle=False,
        **common,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_name).to(device)
    counts = np.bincount(labels[train_idx], minlength=3).astype(np.float64)
    class_weights = len(train_idx) / (len(CLASS_ORDER) * counts)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device),
        label_smoothing=args.label_smoothing,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_score = -np.inf
    best_epoch = 0
    best_state = None
    stale = 0
    history = []
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_n = 0
        for images, targets, _ in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach()) * len(images)
            total_n += len(images)
        scheduler.step()

        _, val_y, val_prob = predict(model, val_loader, device)
        val_metrics = metrics(val_y, val_prob)
        history.append(
            {
                "model": model_name,
                "outer_fold": fold,
                "epoch": epoch,
                "train_loss": total_loss / total_n,
                "learning_rate": optimizer.param_groups[0]["lr"],
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }
        )
        print(
            f"{model_name} fold={fold} epoch={epoch:02d} "
            f"loss={total_loss/total_n:.4f} val_macro_f1={val_metrics['macro_f1']:.4f}",
            flush=True,
        )
        if val_metrics["macro_f1"] > best_score + args.min_delta:
            best_score = val_metrics["macro_f1"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break

    if best_state is None:
        raise RuntimeError("No best model state was captured")
    model.load_state_dict(best_state)
    test_indices, test_y, test_prob = predict(model, test_loader, device)
    order = np.argsort(test_indices)
    test_indices, test_y, test_prob = test_indices[order], test_y[order], test_prob[order]
    out = pd.DataFrame(
        {
            "row_index": test_indices,
            "outer_fold": fold,
            "gt_group": [CLASS_ORDER[i] for i in test_y],
            "pred_group": [CLASS_ORDER[i] for i in test_prob.argmax(axis=1)],
            "prob_normal": test_prob[:, 0],
            "prob_logical": test_prob[:, 1],
            "prob_structural": test_prob[:, 2],
            "best_epoch": best_epoch,
            "best_val_macro_f1": best_score,
            "elapsed_seconds": time.time() - started,
        }
    )
    out.to_csv(pred_path, index=False)
    pd.DataFrame(history).to_csv(hist_path, index=False)
    print(f"DONE {model_name} fold {fold}: {metrics(test_y, test_prob)}", flush=True)
    del model, best_state
    torch.cuda.empty_cache()


def aggregate(df: pd.DataFrame, requested_models: list[str]) -> None:
    prediction_frames = []
    history_frames = []
    fold_rows = []
    for model_name in requested_models:
        for fold in range(1, 6):
            p = pd.read_csv(OUT_DIR / f"{model_name}_fold{fold}_predictions.csv")
            h = pd.read_csv(OUT_DIR / f"{model_name}_fold{fold}_history.csv")
            p.insert(0, "model", model_name)
            prediction_frames.append(p)
            history_frames.append(h)
            prob = p[["prob_normal", "prob_logical", "prob_structural"]].to_numpy()
            y = p["gt_group"].map(CLASS_TO_ID).to_numpy()
            fold_rows.append({"model": model_name, "outer_fold": fold, "n_test": len(p), **metrics(y, prob)})
    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions = predictions.merge(
        df.reset_index(names="row_index")[["row_index", "merge_key", "category"]],
        on="row_index",
        how="left",
        validate="many_to_one",
    )
    if predictions[["merge_key", "category"]].isna().any().any():
        raise RuntimeError("Prediction metadata merge failed")
    predictions.to_csv(OUT_DIR / "oof_predictions.csv", index=False)
    pd.concat(history_frames, ignore_index=True).to_csv(OUT_DIR / "training_history.csv", index=False)
    fold_metrics = pd.DataFrame(fold_rows)
    fold_metrics.to_csv(OUT_DIR / "fold_metrics.csv", index=False)
    summary = fold_metrics.groupby("model").agg(
        folds=("outer_fold", "count"),
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        balanced_accuracy_mean=("balanced_accuracy", "mean"),
        balanced_accuracy_std=("balanced_accuracy", "std"),
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_std=("macro_f1", "std"),
        normal_f1_mean=("normal_f1", "mean"),
        logical_f1_mean=("logical_f1", "mean"),
        structural_f1_mean=("structural_f1", "mean"),
    ).reset_index()
    summary.to_csv(OUT_DIR / "summary.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=list(WEIGHTS), default=list(WEIGHTS))
    parser.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df, paths, labels, outer_folds = load_cohort()
    categories = df["category"].astype(str).to_numpy()
    config = {
        "scientific_label": "ImageNet-pretrained, fully fine-tuned supervised RGB three-class baselines",
        "classes": CLASS_ORDER,
        "models": args.models,
        "outer_folds_source": str(FOLD_FILE.resolve()),
        "outer_fold_counts": pd.Series(outer_folds).value_counts().sort_index().to_dict(),
        "inner_validation": {"type": "StratifiedShuffleSplit", "stratum": "category||class", "fraction": args.val_fraction},
        "training": {
            "epochs_max": args.epochs,
            "early_stopping_patience": args.patience,
            "optimizer": "AdamW",
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "loss": "outer-inner-train class-weighted cross entropy",
            "label_smoothing": args.label_smoothing,
            "batch_size": args.batch_size,
            "image_size": 224,
            "shared_input_cache": "RGB JPEG quality 95, shorter side resized to 256 before augmentation",
            "all_layers_fine_tuned": True,
        },
        "augmentation_identical_across_models": [
            "RandomResizedCrop(224, scale=(0.80,1.00), ratio=(0.90,1.10))",
            "RandomHorizontalFlip(0.5)",
            "ColorJitter(0.15,0.15,0.10,0.02)",
            "ImageNet normalization",
        ],
        "weights": {name: str(path.resolve()) for name, path in WEIGHTS.items()},
        "seed_base": SEED,
        "n_samples": len(df),
    }
    (OUT_DIR / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    paths = materialize_rgb_cache(paths)
    for model_name in args.models:
        for fold in args.folds:
            if fold not in range(1, 6):
                raise ValueError(f"Invalid fold: {fold}")
            train_one_fold(model_name, fold, paths, labels, categories, outer_folds, args)
    if set(args.folds) == set(range(1, 6)):
        complete = all((OUT_DIR / f"{m}_fold{f}_predictions.csv").exists() for m in args.models for f in range(1, 6))
        if complete:
            aggregate(df, args.models)


if __name__ == "__main__":
    main()
