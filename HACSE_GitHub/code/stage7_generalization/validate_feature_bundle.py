from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


BLOCK_PREFIXES = {"V": "V__", "O": "O__", "R": "R__"}
LEAKAGE_TOKENS = ("gt", "ground_truth", "label", "target", "defect_class", "anomaly_type", "test_stat", "prediction")


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate dataset-neutral V/O/R feature tables before generalization experiments.")
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    args = ap.parse_args()

    manifest = pd.read_csv(args.manifest, encoding="utf-8-sig")
    features = pd.read_csv(args.features, encoding="utf-8-sig")
    if "sample_id" not in features or "sample_id" not in manifest:
        raise ValueError("Both files must contain sample_id.")
    if features["sample_id"].duplicated().any():
        raise ValueError("Feature table has duplicate sample_id values.")
    strict_ids = set(manifest.loc[manifest["include_strict"].astype(str).isin(["1", "true", "True"]), "sample_id"].astype(str))
    feature_ids = set(features["sample_id"].astype(str))
    blocks = {name: [c for c in features.columns if c.startswith(prefix)] for name, prefix in BLOCK_PREFIXES.items()}
    feature_cols = [c for cols in blocks.values() for c in cols]
    suspicious = [c for c in feature_cols if any(t in c.lower() for t in LEAKAGE_TOKENS)]
    non_numeric = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(features[c])]
    non_finite = int(np.sum(~np.isfinite(features[feature_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)))) if feature_cols else 0
    report = {
        "status": "pass" if strict_ids == feature_ids and all(blocks.values()) and not suspicious and not non_numeric and non_finite == 0 else "fail",
        "manifest_strict_n": len(strict_ids),
        "feature_n": len(feature_ids),
        "missing_feature_rows": len(strict_ids - feature_ids),
        "extra_feature_rows": len(feature_ids - strict_ids),
        "block_dimensions": {k: len(v) for k, v in blocks.items()},
        "suspicious_feature_columns": suspicious,
        "non_numeric_feature_columns": non_numeric,
        "non_finite_values": non_finite,
        "scientific_requirement": "All datasets must use the same feature definitions and re-extract V/O/R from their own images. Cached LOCO features cannot be joined to external images.",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
