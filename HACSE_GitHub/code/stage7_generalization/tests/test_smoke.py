from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dataset_adapters import build_mvtec_ad, build_mvtec_loco, build_vid_ad  # noqa: E402


def touch_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), "white").save(path)


def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        loco = base / "loco"
        for folder in ("good", "logical_anomalies", "structural_anomalies"):
            touch_image(loco / "thing" / "test" / folder / "000.png")
        rows = build_mvtec_loco("loco", loco, {})
        assert {r["gt_group"] for r in rows} == {"normal", "logical", "structural"}

        mvtec = base / "mvtec"
        for defect in ("good", "cable_swap", "cut_inner_insulation"):
            touch_image(mvtec / "cable" / "test" / defect / "000.png")
        rows = build_mvtec_ad("mvtec", mvtec, {"logical_defects": {"cable": ["cable_swap"]}, "default_anomaly_group": "structural"})
        assert [r["gt_group"] for r in rows] == ["logical", "structural", "normal"]

        vid = base / "vid"
        touch_image(vid / "Balls_Cable_BG" / "test" / "good" / "000.png")
        touch_image(vid / "Balls_Cable_BG" / "test" / "logical_anomalies" / "Dual-Aspects" / "001.png")
        rows = build_vid_ad("vid", vid, {})
        assert {r["condition"] for r in rows} == {"Cable_BG"}
        assert {r["gt_group"] for r in rows} == {"normal", "logical"}

        manifest = pd.DataFrame(build_mvtec_loco("loco", loco, {}))
        manifest_path = base / "manifest.csv"
        pred_path = base / "pred.csv"
        out = base / "eval"
        manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")
        pd.DataFrame({"sample_id": manifest.sample_id, "method": "ODRC", "pred_group": manifest.gt_group}).to_csv(pred_path, index=False, encoding="utf-8-sig")
        subprocess.run([sys.executable, str(ROOT / "evaluate_predictions.py"), "--manifest", str(manifest_path), "--predictions", str(pred_path), "--out-dir", str(out), "--bootstrap", "20"], check=True)

        # Common-schema feature validation and both benchmark protocols.
        records = []
        feats = []
        for dataset in ("source", "target"):
            for cls_i, cls in enumerate(("normal", "logical", "structural")):
                for i in range(8):
                    sid = f"{dataset}:{cls}:{i}"
                    records.append({"sample_id": sid, "dataset": dataset, "gt_group": cls, "include_strict": 1})
                    feats.append({"sample_id": sid, "V__signal": cls_i + i / 100, "O__signal": cls_i * 2 + i / 100, "R__signal": cls_i * 3 + i / 100})
        synthetic_manifest = base / "synthetic_manifest.csv"
        synthetic_features = base / "synthetic_features.csv"
        pd.DataFrame(records).to_csv(synthetic_manifest, index=False, encoding="utf-8-sig")
        pd.DataFrame(feats).to_csv(synthetic_features, index=False, encoding="utf-8-sig")
        subprocess.run([sys.executable, str(ROOT / "validate_feature_bundle.py"), "--manifest", str(synthetic_manifest), "--features", str(synthetic_features), "--report", str(base / "feature_report.json")], check=True)
        for protocol in ("target-refit", "source-to-target"):
            subprocess.run([sys.executable, str(ROOT / "run_tabular_benchmark.py"), "--manifest", str(synthetic_manifest), "--features", str(synthetic_features), "--out-dir", str(base / protocol), "--protocol", protocol, "--source-dataset", "source", "--seeds", "42", "--n-estimators", "10"], check=True)
            assert (base / protocol / "results_by_seed.csv").exists()
        result = pd.read_csv(out / "summary.csv")
        assert float(result.loc[0, "accuracy"]) == 1.0
        assert json.loads((out / "report.json").read_text(encoding="utf-8"))["status"] == "complete"
    print("stage7 smoke tests: PASS")


if __name__ == "__main__":
    main()
