#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build compact manuscript-facing CSV/Markdown tables from completed runs."""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "stage6_expanded_comparison_20260914"
OUT.mkdir(parents=True, exist_ok=True)


def metric_row(group, method, row):
    return {
        "group": group,
        "method": method,
        "Acc_mean_pct": 100 * float(row["acc_mean"] if "acc_mean" in row else row["accuracy_mean"]),
        "BAcc_mean_pct": 100 * float(row["bacc_mean"] if "bacc_mean" in row else row["balanced_accuracy_mean"]),
        "Macro_F1_mean_pct": 100 * float(row["macro_f1_mean"]),
        "Macro_F1_std_pct": 100 * float(row["macro_f1_std"]),
        "Normal_F1_mean_pct": 100 * float(row["normal_f1_mean"]),
        "Logical_F1_mean_pct": 100 * float(row["logical_f1_mean"]),
        "Structural_F1_mean_pct": 100 * float(row["structural_f1_mean"]),
    }


def main():
    supervised = pd.read_csv(ROOT / "outputs/stage6_supervised_rgb_finetuned_20260914/summary.csv")
    foundation = pd.read_csv(ROOT / "outputs/stage6_foundation_representations_20260914/summary.csv")
    industrial = pd.read_csv(ROOT / "outputs/stage6_industrial_representations_20260914/summary.csv")
    branch = pd.read_csv(ROOT / "outputs/stage6_full_branch_fusion_20260914/summary.csv")

    rows = []
    supervised_names = {
        "resnet50": "ResNet50 fine-tuned",
        "convnext_tiny": "ConvNeXt-Tiny fine-tuned",
        "vit_b_16": "ViT-B/16 fine-tuned",
    }
    for _, row in supervised.iterrows():
        rows.append(metric_row("Generic supervised RGB", supervised_names[row["model"]], row))
    for _, row in foundation.iterrows():
        rows.append(metric_row("Frozen foundation representation", row["method"], row))
    for _, row in industrial.iterrows():
        rows.append(metric_row("Industrial anomaly representation", row["method"], row))
    for method in ["V", "V+O+D"]:
        row = branch[branch["method"] == method].iloc[0]
        rows.append(metric_row("Proposed", "HACSE" if method == "V+O+D" else "$V$", row))
    comparison = pd.DataFrame(rows)
    group_order = {
        "Generic supervised RGB": 0,
        "Frozen foundation representation": 1,
        "Industrial anomaly representation": 2,
        "Proposed": 3,
    }
    comparison["_order"] = comparison["group"].map(group_order)
    comparison = comparison.sort_values(["_order", "Macro_F1_mean_pct"], ascending=[True, False]).drop(columns="_order")
    comparison.to_csv(OUT / "table_main_comparison.csv", index=False, encoding="utf-8-sig", float_format="%.3f")

    ablation = branch[branch["method"].isin(["V", "O", "D", "V+O", "V+D", "O+D", "V+O+D"])].copy()
    ablation = ablation[["method", "acc_mean", "bacc_mean", "macro_f1_mean", "macro_f1_std", "structural_f1_mean"]]
    for col in ablation.columns[1:]:
        ablation[col] = 100 * ablation[col]
    ablation.to_csv(OUT / "table_full_branch_ablation.csv", index=False, encoding="utf-8-sig", float_format="%.3f")

    fusion = branch[branch["method"].isin(["Probability average", "Feature concatenation", "V+O+D"])].copy()
    fusion["method"] = fusion["method"].replace({"V+O+D": "Probability-level LR fusion (HACSE)"})
    fusion = fusion[["method", "acc_mean", "bacc_mean", "macro_f1_mean", "macro_f1_std", "structural_f1_mean"]]
    for col in fusion.columns[1:]:
        fusion[col] = 100 * fusion[col]
    fusion.to_csv(OUT / "table_fusion_strategy.csv", index=False, encoding="utf-8-sig", float_format="%.3f")

    status = pd.DataFrame([
        {"method": "ComAD", "status": "not included", "reason": "source exists, but no aligned per-image evidence has been generated; current environment also lacks pydensecrf"},
        {"method": "PSAD", "status": "not included", "reason": "no local implementation/checkpoint/per-image evidence found"},
        {"method": "LogicAL", "status": "not included", "reason": "no local implementation/checkpoint/per-image evidence found"},
    ])
    status.to_csv(OUT / "industrial_method_feasibility.csv", index=False, encoding="utf-8-sig")

    def md_table(frame, cols):
        f = frame[cols].copy()
        for col in cols:
            if col != cols[0] and pd.api.types.is_numeric_dtype(f[col]):
                f[col] = f[col].map(lambda x: f"{x:.2f}")
        return f.to_markdown(index=False)

    report = [
        "# Stage6 expanded comparative study (2026-09-14)",
        "",
        "All reported values were recomputed on the same fixed five outer folds. Values are mean percentages across folds.",
        "",
        "## Main comparison",
        "",
        md_table(comparison, ["method", "BAcc_mean_pct", "Macro_F1_mean_pct", "Structural_F1_mean_pct"]),
        "",
        "## Full branch ablation",
        "",
        md_table(ablation, ["method", "bacc_mean", "macro_f1_mean", "structural_f1_mean"]),
        "",
        "## Fusion strategy",
        "",
        md_table(fusion, ["method", "bacc_mean", "macro_f1_mean", "structural_f1_mean"]),
        "",
        "## Interpretation boundary",
        "",
        "- The supervised RGB models are ImageNet-pretrained and fully fine-tuned, with identical augmentation, class-balanced loss, inner validation, and early stopping.",
        "- Probability-level learned fusion uses inner-OOF branch probabilities only.",
        "- ComAD, PSAD, and LogicAL are not assigned literature values; they remain excluded until aligned per-image evidence can be generated locally.",
    ]
    (OUT / "RESULTS.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(comparison.to_string(index=False))
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
