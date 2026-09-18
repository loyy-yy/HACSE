from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    df = pd.read_csv(args.manifest, encoding="utf-8-sig", dtype=str)
    df = df[df["include_strict"].isin(["1", "true", "True"])]
    rows = [{"sample_id": sid, "method": method, "pred_group": ""} for method in args.methods for sid in df["sample_id"]]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
