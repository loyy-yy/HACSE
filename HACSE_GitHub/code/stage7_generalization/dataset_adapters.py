from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
CLASS_ORDER = ("normal", "logical", "structural")


def image_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def sample_id(dataset: str, root: Path, image_path: Path) -> str:
    rel = image_path.resolve().relative_to(root.resolve()).as_posix().lower()
    digest = hashlib.sha256(f"{dataset}:{rel}".encode("utf-8")).hexdigest()[:20]
    return f"{dataset}:{digest}"


def record(
    dataset: str,
    root: Path,
    path: Path,
    category: str,
    split: str,
    subtype: str,
    group: str,
    label_source: str,
    condition: str = "default",
    mask_path: str = "",
    include_strict: bool = True,
    exclusion_reason: str = "",
) -> dict[str, Any]:
    return {
        "sample_id": sample_id(dataset, root, path),
        "dataset": dataset,
        "category": category,
        "condition": condition,
        "split": split,
        "anomaly_subtype": subtype,
        "gt_group": group,
        "image_path": str(path.resolve()),
        "mask_path": mask_path,
        "label_source": label_source,
        "include_strict": int(include_strict),
        "exclusion_reason": exclusion_reason,
    }


def build_mvtec_loco(dataset: str, root: Path, _: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    folder_to_group = {
        "good": "normal",
        "logical_anomalies": "logical",
        "structural_anomalies": "structural",
    }
    for category_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        test_dir = category_dir / "test"
        for folder, group in folder_to_group.items():
            for path in image_files(test_dir / folder):
                rows.append(record(dataset, root, path, category_dir.name, "test", folder, group, "official"))
    return rows


def build_mvtec_ad(dataset: str, root: Path, policy: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    logical = {k.lower(): {x.lower() for x in v} for k, v in policy["logical_defects"].items()}
    default_group = policy.get("default_anomaly_group", "structural")
    label_source = policy.get("policy_name", "semantic-regrouping")
    for category_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        test_dir = category_dir / "test"
        if not test_dir.exists():
            continue
        for defect_dir in sorted(p for p in test_dir.iterdir() if p.is_dir()):
            if defect_dir.name.lower() == "good":
                group = "normal"
            elif defect_dir.name.lower() in logical.get(category_dir.name.lower(), set()):
                group = "logical"
            else:
                group = default_group
            for path in image_files(defect_dir):
                rows.append(record(dataset, root, path, category_dir.name, "test", defect_dir.name, group, label_source))
    return rows


def _read_visa_id2class() -> dict[str, dict[int, str]]:
    # Frozen from the official amazon-science/spot-diff utils/id2class.py.
    raw = {
        "candle": {1: "Chunk Of Wax Missing", 2: "Damaged Corner of Packaging", 3: "weird candle wick", 4: "Different Colour Spot", 5: "Extra Wax in Candle", 6: "Foreign Particals on Candle", 7: "Wax Melded out of the candle", 8: "Other"},
        "capsules": {1: "Bubble", 2: "Discolor", 3: "Scratch", 4: "Leak", 5: "Misshape"},
        "fryum": {1: "Burnt", 2: "Corner or Edge Breakage", 3: "Different Colour Spot", 4: "Fryum Stuck Together", 5: "Middle Breakage", 6: "Similar Colour Spot", 7: "Small Scratches", 8: "Other"},
        "cashew": {1: "Burnt", 2: "Corner or Edge Breakage", 3: "Different Colour Spot", 4: "Middle Breakage", 5: "Small Holes", 6: "Small Scratches", 7: "Stuck Together", 8: "Same Colour Spot", 9: "Other"},
        "chewinggum": {1: "Chunk of gum missing", 2: "Corner Missing", 3: "Scratches", 4: "Similar Colour Spot", 5: "Small Cracks", 6: "Other"},
        "macaroni1": {1: "Chip Around Edge And Corner", 2: "Different Colour Spot", 3: "Middle Breakage", 4: "Similar Colour Spot", 5: "Small Cracks", 6: "Small Scratches", 7: "Other"},
        "macaroni2": {1: "Breakage down the middle", 2: "Color spot similar to the Object", 3: "Different Color spot", 4: "Small chip around edge", 5: "Small Cracks", 6: "Small Scratches", 7: "Other"},
        "pcb1": {1: "Bent", 2: "Melt", 3: "Scratch", 4: "Missing"},
        "pcb2": {1: "Bent", 2: "Melt", 3: "Scratch", 4: "Missing"},
        "pcb3": {1: "Bent", 2: "Melt", 3: "Scratch", 4: "Missing"},
        "pcb4": {1: "Burnt", 2: "Scratch", 3: "Missing", 4: "Damage", 5: "Extra", 6: "Wrong Place", 7: "Dirt"},
        "pipe_fryum": {1: "Burnt", 2: "Corner And Edge Breakage", 3: "Different Colour Spot", 4: "Middle Breakage", 5: "Similar Colour Spot", 6: "Small Scratches", 7: "Stuck Together", 8: "Small Cracks", 9: "Other"},
    }
    return raw


def _find_visa_mask(category_dir: Path, image_path: Path) -> Path | None:
    mask_root = category_dir / "Data" / "Masks" / "Anomaly"
    if not mask_root.exists():
        return None
    direct = mask_root / image_path.name
    if direct.exists():
        return direct
    candidates = list(mask_root.rglob(f"{image_path.stem}.*"))
    return sorted(candidates)[0] if candidates else None


def build_visa(dataset: str, root: Path, policy: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    id2class = _read_visa_id2class()
    logical_names = {k.lower(): {x.lower() for x in v} for k, v in policy["logical_mask_classes"].items()}
    label_source = policy.get("policy_name", "official-mask-class semantic regrouping")
    for category_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        cat = category_dir.name.lower().replace(" ", "_")
        image_root = category_dir / "Data" / "Images"
        for path in image_files(image_root / "Normal"):
            rows.append(record(dataset, root, path, cat, "all", "Normal", "normal", label_source))
        for path in image_files(image_root / "Anomaly"):
            mask = _find_visa_mask(category_dir, path)
            if mask is None:
                rows.append(record(dataset, root, path, cat, "all", "unknown", "unknown", label_source, include_strict=False, exclusion_reason="missing_mask"))
                continue
            ids = sorted(int(x) for x in np.unique(np.asarray(Image.open(mask))) if int(x) != 0)
            names = [id2class.get(cat, {}).get(i, f"UNKNOWN_ID_{i}") for i in ids]
            logical_flags = [name.lower() in logical_names.get(cat, set()) for name in names]
            if names and all(logical_flags):
                group, include, reason = "logical", True, ""
            elif any(logical_flags):
                group, include, reason = "mixed", False, "mixed_logical_structural_mask_classes"
            else:
                group, include, reason = "structural", True, ""
            rows.append(record(dataset, root, path, cat, "all", ";".join(names), group, label_source, mask_path=str(mask.resolve()), include_strict=include, exclusion_reason=reason))
    return rows


def build_vid_ad(dataset: str, root: Path, policy: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    label_source = policy.get("policy_name", "official logical-only labels")
    for category_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        name = category_dir.name
        condition = "Original"
        for suffix, value in (("_Cable_BG", "Cable_BG"), ("_Mesh_BG", "Mesh_BG"), ("_Blurry_CD", "Blurry_CD"), ("_Low-light_CD", "Low-light_CD")):
            if name.endswith(suffix):
                name, condition = name[: -len(suffix)], value
                break
        for path in image_files(category_dir / "test" / "good"):
            rows.append(record(dataset, root, path, name, "test", "good", "normal", label_source, condition=condition))
        logical_root = category_dir / "test" / "logical_anomalies"
        for path in image_files(logical_root):
            subtype = path.parent.name if path.parent != logical_root else "logical_anomalies"
            rows.append(record(dataset, root, path, name, "test", subtype, "logical", label_source, condition=condition))
    return rows


ADAPTERS = {
    "mvtec_loco": build_mvtec_loco,
    "mvtec_ad": build_mvtec_ad,
    "visa": build_visa,
    "vid_ad": build_vid_ad,
}


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["sample_id", "dataset", "gt_group", "image_path"]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
