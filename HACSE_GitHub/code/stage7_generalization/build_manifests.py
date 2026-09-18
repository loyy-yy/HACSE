from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from dataset_adapters import ADAPTERS, CLASS_ORDER, write_csv


HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser(description="Build auditable multi-dataset normal/logical/structural manifests.")
    ap.add_argument("--config", type=Path, default=HERE / "dataset_config.json")
    ap.add_argument("--policies", type=Path, default=HERE / "label_policies.json")
    ap.add_argument("--skip-missing", action="store_true", help="Record missing datasets instead of failing.")
    args = ap.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    policies = json.loads(args.policies.read_text(encoding="utf-8"))
    out_dir = HERE / config.get("output_dir", "outputs/manifests")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    summary = {"schema_version": 1, "class_order": list(CLASS_ORDER), "datasets": {}}
    for item in config["datasets"]:
        if not item.get("enabled", True):
            continue
        name, adapter = item["name"], item["adapter"]
        root = Path(item["root"])
        if not root.exists():
            summary["datasets"][name] = {"status": "missing", "root": str(root)}
            if args.skip_missing:
                continue
            raise FileNotFoundError(f"Dataset root missing for {name}: {root}")
        policy = policies.get(adapter, {})
        rows = ADAPTERS[adapter](name, root, policy)
        if not rows:
            raise RuntimeError(f"Adapter {adapter} found no samples under {root}")
        write_csv(out_dir / f"{name}_manifest.csv", rows)
        strict = [r for r in rows if int(r["include_strict"]) == 1]
        counts = Counter(r["gt_group"] for r in strict)
        all_rows.extend(rows)
        summary["datasets"][name] = {
            "status": "ready",
            "root": str(root.resolve()),
            "n_all": len(rows),
            "n_strict": len(strict),
            "strict_class_counts": {k: counts.get(k, 0) for k in CLASS_ORDER},
            "complete_three_class": all(counts.get(k, 0) > 0 for k in CLASS_ORDER),
            "label_policy": policy.get("policy_name", "official"),
            "warning": policy.get("warning", ""),
        }
    write_csv(out_dir / "combined_manifest.csv", all_rows)
    (out_dir / "manifest_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
