from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
DEFAULT_SOURCE = HERE.parent / "stage6" / "outputs" / "stage6_v11_canonical_export" / "v11_canonical_predictions_all.csv"
DEFAULT_MANIFEST = HERE / "outputs" / "manifests" / "mvtec_loco_manifest.csv"


def canonical_key(value: str) -> str:
    s = str(value).replace("\\", "/").lower()
    parts = s.split("/")
    for i, part in enumerate(parts):
        if part in {"breakfast_box", "juice_bottle", "pushpins", "screw_bag", "splicing_connectors"}:
            return "/".join(parts[i:])
    return s


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert Stage6 V11 canonical LOCO predictions to the Stage7 long prediction schema.")
    ap.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--output", type=Path, default=HERE / "outputs" / "loco_canonical_predictions.csv")
    args = ap.parse_args()

    src = pd.read_csv(args.source, encoding="utf-8-sig")
    manifest = pd.read_csv(args.manifest, encoding="utf-8-sig")
    source_key_col = "sample_key" if "sample_key" in src else "image_path"
    src["join_key"] = src[source_key_col].map(canonical_key)
    manifest["join_key"] = manifest["image_path"].map(canonical_key)
    if src["join_key"].duplicated().any() or manifest["join_key"].duplicated().any():
        raise RuntimeError("Canonical LOCO join keys are not unique.")
    joined = manifest[["sample_id", "join_key"]].merge(src, on="join_key", how="left", validate="one_to_one")
    if len(joined) != len(manifest) or joined["pred_VOR"].isna().any():
        raise RuntimeError("Stage6 predictions do not cover the Stage7 LOCO manifest exactly.")
    mapping = {"V": "pred_V", "V+O": "pred_VO", "ODRC": "pred_VOR"}
    rows = []
    for method, col in mapping.items():
        rows.append(pd.DataFrame({"sample_id": joined["sample_id"], "method": method, "pred_group": joined[col]}))
    out = pd.concat(rows, ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"wrote {len(out)} predictions to {args.output}")


if __name__ == "__main__":
    main()
