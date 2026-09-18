#!/usr/bin/env python
"""Audit the incremental effect of O when moving from V+D to V+O+D."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score


ROOT = Path(__file__).resolve().parents[2]
INPUT = (
    ROOT
    / "stage6"
    / "outputs"
    / "stage6_full_branch_fusion_20260914"
    / "oof_predictions.csv"
)
OUT_DIR = (
    ROOT
    / "stage6"
    / "outputs"
    / "stage6_o_incremental_effect_20260917"
)
CLASS_ORDER = ["normal", "logical", "structural"]


def macro_f1(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float(
        f1_score(
            y_true,
            y_pred,
            labels=CLASS_ORDER,
            average="macro",
            zero_division=0,
        )
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(INPUT)
    pair = raw[raw["method"].isin(["V+D", "V+O+D"])].copy()

    meta_cols = ["sample_index", "sample_key", "category", "label", "outer_fold"]
    meta = pair[meta_cols].drop_duplicates("sample_index").set_index("sample_index")
    pred = pair.pivot(index="sample_index", columns="method", values="prediction").rename(
        columns={"V+D": "pred_vd", "V+O+D": "pred_vod"}
    )
    probability = pair.pivot(
        index="sample_index",
        columns="method",
        values=["prob_normal", "prob_logical", "prob_structural"],
    )
    probability.columns = [
        f"{prob}_{'vd' if method == 'V+D' else 'vod'}"
        for prob, method in probability.columns
    ]
    audit = meta.join(pred).join(probability)
    audit["vd_correct"] = audit["pred_vd"].eq(audit["label"])
    audit["vod_correct"] = audit["pred_vod"].eq(audit["label"])
    audit["status"] = np.select(
        [
            ~audit["vd_correct"] & audit["vod_correct"],
            audit["vd_correct"] & ~audit["vod_correct"],
            audit["pred_vd"].ne(audit["pred_vod"]),
        ],
        ["corrected", "regressed", "changed_wrong_to_wrong"],
        default="unchanged",
    )
    for method in ["vd", "vod"]:
        audit[f"true_probability_{method}"] = [
            audit.loc[index, f"prob_{label}_{method}"]
            for index, label in audit["label"].items()
        ]
    audit["delta_true_probability"] = (
        audit["true_probability_vod"] - audit["true_probability_vd"]
    )
    audit.reset_index().to_csv(OUT_DIR / "sample_transitions.csv", index=False)

    counts = []
    for group_name, group_col in [
        ("overall", None),
        ("outer_fold", "outer_fold"),
        ("category", "category"),
        ("true_label", "label"),
    ]:
        groups = [("all", audit)] if group_col is None else audit.groupby(group_col)
        for group_value, frame in groups:
            row = {
                "group_type": group_name,
                "group_value": group_value,
                "n": len(frame),
            }
            vc = frame["status"].value_counts()
            for status in ["corrected", "regressed", "changed_wrong_to_wrong", "unchanged"]:
                row[status] = int(vc.get(status, 0))
            row["net_corrections"] = row["corrected"] - row["regressed"]
            counts.append(row)
    pd.DataFrame(counts).to_csv(OUT_DIR / "transition_counts.csv", index=False)

    metric_rows = []
    metric_groups = [("overall", "all", audit)]
    metric_groups += [
        ("outer_fold", group_value, frame)
        for group_value, frame in audit.groupby("outer_fold")
    ]
    metric_groups += [
        ("category", group_value, frame)
        for group_value, frame in audit.groupby("category")
    ]
    for group_type, group_value, frame in metric_groups:
        vd_f1 = macro_f1(frame["label"], frame["pred_vd"])
        vod_f1 = macro_f1(frame["label"], frame["pred_vod"])
        vd_bacc = float(balanced_accuracy_score(frame["label"], frame["pred_vd"]))
        vod_bacc = float(balanced_accuracy_score(frame["label"], frame["pred_vod"]))
        metric_rows.append(
            {
                "group_type": group_type,
                "group_value": group_value,
                "n": len(frame),
                "vd_macro_f1": vd_f1,
                "vod_macro_f1": vod_f1,
                "delta_macro_f1": vod_f1 - vd_f1,
                "vd_bacc": vd_bacc,
                "vod_bacc": vod_bacc,
                "delta_bacc": vod_bacc - vd_bacc,
            }
        )
    metric_table = pd.DataFrame(metric_rows)
    metric_table.to_csv(OUT_DIR / "metric_deltas.csv", index=False)
    metric_table[metric_table["group_type"] == "category"].to_csv(
        OUT_DIR / "category_metric_deltas.csv", index=False
    )
    metric_table[metric_table["group_type"] == "outer_fold"].to_csv(
        OUT_DIR / "fold_metric_deltas.csv", index=False
    )

    changed = audit[audit["status"] != "unchanged"].reset_index()
    changed.to_csv(OUT_DIR / "changed_samples.csv", index=False)

    transition_table = (
        audit[audit["status"].isin(["corrected", "regressed"])]
        .groupby(["status", "category", "label", "pred_vd", "pred_vod"])
        .size()
        .rename("count")
        .reset_index()
        .sort_values(["status", "count"], ascending=[True, False])
    )
    transition_table.to_csv(OUT_DIR / "confusion_transitions.csv", index=False)

    print(pd.read_csv(OUT_DIR / "transition_counts.csv").to_string(index=False))
    print("\nCATEGORY METRICS")
    print(metric_table.to_string(index=False))
    print("\nCONFUSION TRANSITIONS")
    print(transition_table.to_string(index=False))
    print("\nCORRECTED SAMPLES")
    print(
        changed[changed["status"] == "corrected"][
            ["sample_key", "category", "label", "outer_fold", "pred_vd", "pred_vod"]
        ].to_string(index=False)
    )
    print("\nREGRESSED SAMPLES")
    print(
        changed[changed["status"] == "regressed"][
            ["sample_key", "category", "label", "outer_fold", "pred_vd", "pred_vod"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
