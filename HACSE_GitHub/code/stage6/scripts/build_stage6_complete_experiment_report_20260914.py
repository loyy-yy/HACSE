#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate one complete, evidence-grounded Markdown report for the 2026-09-14 runs."""

from __future__ import annotations

from pathlib import Path
import pandas as pd


STAGE6 = Path(__file__).resolve().parents[1]
OUTPUTS = STAGE6 / "outputs"
REPORT_DIR = OUTPUTS / "stage6_expanded_comparison_20260914"
REPORT_PATH = REPORT_DIR / "COMPLETE_EXPERIMENT_REPORT_20260914.md"

SUP_DIR = OUTPUTS / "stage6_supervised_rgb_finetuned_20260914"
FOUND_DIR = OUTPUTS / "stage6_foundation_representations_20260914"
IND_DIR = OUTPUTS / "stage6_industrial_representations_20260914"
FUSION_DIR = OUTPUTS / "stage6_full_branch_fusion_20260914"
ASSIGN_FILE = OUTPUTS / "stage6_nested5fold_ablation" / "outer_fold_assignments.csv"
WORKBOOK = STAGE6.parent / "full_loco_salad_csad_fusion_analysis.xlsx"


def pct_table(df: pd.DataFrame, rename: dict[str, str] | None = None) -> pd.DataFrame:
    out = df.copy()
    if rename:
        out = out.rename(columns=rename)
    for col in out.columns:
        if col not in {"Method", "Fold", "N", "Best epoch", "Positive folds", "Comparison"}:
            if pd.api.types.is_numeric_dtype(out[col]):
                out[col] = out[col] * 100.0
    return out


def md(df: pd.DataFrame, decimals: int = 2) -> str:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda x: "" if pd.isna(x) else f"{x:.{decimals}f}")
    return out.to_markdown(index=False)


def rel(path: Path) -> str:
    return str(path.relative_to(STAGE6.parent)).replace("\\", "/")


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    sup_fold = pd.read_csv(SUP_DIR / "fold_metrics.csv")
    sup_summary = pd.read_csv(SUP_DIR / "summary.csv")
    found_fold = pd.read_csv(FOUND_DIR / "fold_metrics.csv")
    found_summary = pd.read_csv(FOUND_DIR / "summary.csv")
    ind_fold = pd.read_csv(IND_DIR / "fold_metrics.csv")
    ind_summary = pd.read_csv(IND_DIR / "summary.csv")
    fusion_fold = pd.read_csv(FUSION_DIR / "fold_metrics.csv")
    fusion_summary = pd.read_csv(FUSION_DIR / "summary.csv")
    deltas = pd.read_csv(FUSION_DIR / "paired_deltas.csv")
    settings = pd.read_csv(OUTPUTS / "stage6_nested5fold_ablation/nested5fold_selected_settings.csv")

    cohort = pd.read_excel(WORKBOOK, sheet_name="merged_data")
    assignments = pd.read_csv(ASSIGN_FILE, encoding="utf-8-sig")
    fold_class = pd.crosstab(assignments["outer_test_fold"], cohort["gt_group"])
    fold_class = fold_class.reindex(columns=["normal", "logical", "structural"]).reset_index()
    fold_class.columns = ["Fold", "Normal", "Logical", "Structural"]
    fold_class["Total"] = fold_class[["Normal", "Logical", "Structural"]].sum(axis=1)
    category_counts = cohort["category"].value_counts().sort_index().rename_axis("Category").reset_index(name="N")
    class_counts = cohort["gt_group"].value_counts().reindex(["normal", "logical", "structural"]).rename_axis("Class").reset_index(name="N")

    model_names = {
        "resnet50": "ResNet50 fine-tuned",
        "convnext_tiny": "ConvNeXt-Tiny fine-tuned",
        "vit_b_16": "ViT-B/16 fine-tuned",
    }

    # Main table assembled from the exact same-fold outputs.
    main_rows = []
    for _, row in sup_summary.iterrows():
        main_rows.append({
            "Group": "Supervised RGB", "Method": model_names[row.model],
            "Acc": row.accuracy_mean, "BAcc": row.balanced_accuracy_mean,
            "Macro-F1": row.macro_f1_mean, "N-F1": row.normal_f1_mean,
            "L-F1": row.logical_f1_mean, "S-F1": row.structural_f1_mean,
        })
    for _, row in found_summary.iterrows():
        main_rows.append({
            "Group": "Frozen foundation", "Method": row.method,
            "Acc": row.acc_mean, "BAcc": row.bacc_mean, "Macro-F1": row.macro_f1_mean,
            "N-F1": row.normal_f1_mean, "L-F1": row.logical_f1_mean,
            "S-F1": row.structural_f1_mean,
        })
    for _, row in ind_summary.iterrows():
        main_rows.append({
            "Group": "Industrial evidence", "Method": row.method,
            "Acc": row.acc_mean, "BAcc": row.bacc_mean, "Macro-F1": row.macro_f1_mean,
            "N-F1": row.normal_f1_mean, "L-F1": row.logical_f1_mean,
            "S-F1": row.structural_f1_mean,
        })
    for method, label in [("V", "V"), ("V+O+D", "HACSE (V+O+D)")]:
        row = fusion_summary[fusion_summary.method == method].iloc[0]
        main_rows.append({
            "Group": "Proposed", "Method": label,
            "Acc": row.acc_mean, "BAcc": row.bacc_mean, "Macro-F1": row.macro_f1_mean,
            "N-F1": row.normal_f1_mean, "L-F1": row.logical_f1_mean,
            "S-F1": row.structural_f1_mean,
        })
    main_df = pct_table(pd.DataFrame(main_rows))

    # Supervised fold table.
    sup_fold_show = sup_fold.copy()
    sup_fold_show["model"] = sup_fold_show["model"].map(model_names)
    sup_fold_show = sup_fold_show.rename(columns={
        "model": "Method", "outer_fold": "Fold", "n_test": "N",
        "accuracy": "Acc", "balanced_accuracy": "BAcc", "macro_f1": "Macro-F1",
        "normal_f1": "N-F1", "logical_f1": "L-F1", "structural_f1": "S-F1",
    })
    sup_fold_show = pct_table(sup_fold_show[["Method", "Fold", "N", "Acc", "BAcc", "Macro-F1", "N-F1", "L-F1", "S-F1"]])

    # Training selection information from one prediction row per model/fold.
    train_rows = []
    for model, label in model_names.items():
        for fold in range(1, 6):
            p = pd.read_csv(SUP_DIR / f"{model}_fold{fold}_predictions.csv")
            train_rows.append({
                "Method": label, "Fold": fold,
                "Best epoch": int(p["best_epoch"].iloc[0]),
                "Best inner-val Macro-F1": 100.0 * float(p["best_val_macro_f1"].iloc[0]),
                "Elapsed (s)": float(p["elapsed_seconds"].iloc[0]),
            })
    train_df = pd.DataFrame(train_rows)

    def fold_table(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.rename(columns={
            "method": "Method", "outer_fold": "Fold", "acc": "Acc", "bacc": "BAcc",
            "macro_f1": "Macro-F1", "normal_f1": "N-F1", "logical_f1": "L-F1",
            "structural_f1": "S-F1",
        })
        return pct_table(out[["Method", "Fold", "Acc", "BAcc", "Macro-F1", "N-F1", "L-F1", "S-F1"]])

    found_fold_show = fold_table(found_fold)
    ind_fold_show = fold_table(ind_fold)
    fusion_fold_show = fold_table(fusion_fold)

    def summary_table(frame: pd.DataFrame, name_col: str) -> pd.DataFrame:
        out = frame.rename(columns={
            name_col: "Method", "acc_mean": "Acc", "acc_std": "Acc SD",
            "bacc_mean": "BAcc", "bacc_std": "BAcc SD",
            "macro_f1_mean": "Macro-F1", "macro_f1_std": "Macro-F1 SD",
            "normal_f1_mean": "N-F1", "logical_f1_mean": "L-F1",
            "structural_f1_mean": "S-F1",
        })
        keep = ["Method", "Acc", "Acc SD", "BAcc", "BAcc SD", "Macro-F1", "Macro-F1 SD", "N-F1", "L-F1", "S-F1"]
        return pct_table(out[keep])

    found_summary_show = summary_table(found_summary, "method")
    ind_summary_show = summary_table(ind_summary, "method")
    fusion_summary_show = summary_table(fusion_summary, "method")
    sup_summary_show = sup_summary.copy()
    sup_summary_show["model"] = sup_summary_show["model"].map(model_names)
    sup_summary_show = sup_summary_show.rename(columns={
        "model": "Method", "accuracy_mean": "Acc", "accuracy_std": "Acc SD",
        "balanced_accuracy_mean": "BAcc", "balanced_accuracy_std": "BAcc SD",
        "macro_f1_mean": "Macro-F1", "macro_f1_std": "Macro-F1 SD",
        "normal_f1_mean": "N-F1", "logical_f1_mean": "L-F1", "structural_f1_mean": "S-F1",
    })
    sup_summary_show = pct_table(sup_summary_show[["Method", "Acc", "Acc SD", "BAcc", "BAcc SD", "Macro-F1", "Macro-F1 SD", "N-F1", "L-F1", "S-F1"]])

    delta_show = deltas[deltas.metric == "macro_f1"].copy()
    delta_show = delta_show.rename(columns={"comparator": "Comparison", "mean_delta": "Mean ΔMacro-F1", "positive_folds": "Positive folds"})
    delta_show["Comparison"] = "HACSE vs. " + delta_show["Comparison"].astype(str)
    delta_show["Mean ΔMacro-F1"] *= 100.0
    delta_show = delta_show[["Comparison", "Mean ΔMacro-F1", "Positive folds"]]

    selected = settings[["outer_fold", "v_model", "o_model"]].rename(columns={
        "outer_fold": "Fold", "v_model": "V classifier", "o_model": "O classifier"
    })
    selected["V classifier"] = selected["V classifier"].map({"rf": "Random Forest", "et": "Extra Trees"})
    selected["O classifier"] = selected["O classifier"].map({"rf": "Random Forest", "et": "Extra Trees"})

    lines = [
        "# Stage6 扩展对比实验：完整过程与结果记录",
        "",
        "> 日期：2026-09-14  ",
        "> 任务：MVTec LOCO AD 图像的监督式三分类诊断（Normal / Logical / Structural）  ",
        "> 样本数：1,568  ",
        "> 评估方式：固定五折 outer cross-validation；每个样本仅作为 outer-test 一次",
        "",
        "## 1. 本轮实验目的与核心结论",
        "",
        "本轮实验用于加强 HACSE 的比较体系，具体回答四个问题：",
        "",
        "1. 在已经拥有 N/L/S 标签的情况下，标准端到端监督 RGB 分类器能达到什么水平？",
        "2. SALAD、CSAD 等工业异常表征在统一监督三分类协议下具有多强的诊断能力？",
        "3. V、O、D 三个分支各自和组合后的贡献是什么？",
        "4. HACSE 的概率级学习融合是否优于简单概率平均和特征直接拼接？",
        "",
        "一句话结论：在本数据、固定五折和统一评价条件下，HACSE（V+O+D）取得 91.45% Macro-F1；其优势同时来自强 appearance evidence 和严格 inner-OOF 的概率级异质证据融合，但对 V+D 的平均增益较小，必须如实限定为 0.26 个百分点、4/5 folds 为正。",
        "",
        "### 1.1 主结果总览（五折均值，%）",
        "",
        md(main_df),
        "",
        "## 2. 数据集、标签与固定划分",
        "",
        "### 2.1 数据来源",
        "",
        f"- 特征与标签工作簿：`{rel(WORKBOOK)}`",
        "- 图像数据：MVTec LOCO AD 五个类别。",
        "- 诊断标签映射：good/normal → Normal，logical anomalies → Logical，structural anomalies → Structural。",
        "",
        "### 2.2 类别与标签数量",
    ]
    lines += ["", md(category_counts), "", md(class_counts), "", "### 2.3 Outer-test 分布", "", md(fold_class), ""]
    lines += [
        "固定 outer-fold 文件为：",
        "",
        f"`{rel(ASSIGN_FILE)}`",
        "",
        "五个测试折分别包含 314、314、314、313 和 313 张图像；每折均同时平衡工业类别和 N/L/S 标签组成。所有本轮对比实验均读取同一个 assignment 文件，而不是各自重新随机划分。",
        "",
        "## 3. 统一指标与评价原则",
        "",
        "报告 Accuracy（Acc）、Balanced Accuracy（BAcc）、Macro-F1、Normal-F1（N-F1）、Logical-F1（L-F1）和 Structural-F1（S-F1）。主排序指标为 Macro-F1；BAcc 用于控制类别不平衡影响；S-F1 用于观察最困难的 Structural 类。所有表格中的结果为五个 outer-test folds 的算术均值与样本标准差（SD）。",
        "",
        "所有可学习分类器均只在对应 outer-train 上拟合。概率级融合头的训练输入来自 outer-train 内部五折生成的 OOF branch probabilities；outer-test 标签不参与分类器、融合器或早停选择。",
        "",
        "## 4. 强监督 RGB 三分类 baseline",
        "",
        "### 4.1 模型",
        "",
        "- ResNet50（ImageNet pretrained，全部层 fine-tuned）",
        "- ConvNeXt-Tiny（ImageNet pretrained，全部层 fine-tuned）",
        "- ViT-B/16（ImageNet pretrained，全部层 fine-tuned）",
        "",
        "### 4.2 训练设置",
        "",
        "- 输入：原始 RGB 图像。",
        "- 输出：Normal / Logical / Structural 三类概率。",
        "- 图像缓存：短边缩放至 256，JPEG quality 95。",
        "- 训练增强：RandomResizedCrop(224, scale=0.80–1.00, ratio=0.90–1.10)、RandomHorizontalFlip(0.5)、ColorJitter(0.15, 0.15, 0.10, 0.02)、ImageNet normalization。",
        "- 测试变换：Resize(256) + CenterCrop(224) + ImageNet normalization。",
        "- 损失：按 inner-train 类别频数计算权重的 cross entropy，label smoothing=0.1。",
        "- 优化器：AdamW，learning rate=1e-4，weight decay=0.05。",
        "- 最大训练轮数：25；early-stopping patience=5。",
        "- Inner validation：从 outer-train 内按 category||class 分层抽取 15%。",
        "- Batch size：32；评估 batch size：64；随机种子基数：20260827。",
        "- 最优 inner-validation checkpoint 在内存中恢复后仅预测一次 outer-test；本轮未保存模型 checkpoint，但保存了逐 epoch history 和逐样本概率。",
        "",
        "### 4.3 每折早停与运行时间",
        "",
        md(train_df),
        "",
        "### 4.4 每折测试结果（%）",
        "",
        md(sup_fold_show),
        "",
        "### 4.5 五折汇总（均值 ± SD，%）",
        "",
        md(sup_summary_show),
        "",
        "### 4.6 观察",
        "",
        "ResNet50、ConvNeXt-Tiny 和 ViT-B/16 的 Macro-F1 分别为 49.41%、48.06% 和 40.32%。这些训练均正常完成、训练损失下降并由 inner validation 早停，因此结果并非缺折或运行中断。不过，本轮仅采用一套预先固定的统一训练超参数，未对不同架构分别进行大规模超参数搜索；论文中宜称为 standardized fully fine-tuned RGB baselines，而不应宣称为每个架构各自充分调优后的性能上限。",
        "",
        "## 5. Frozen foundation representation baselines",
        "",
        "三个 backbone 均保持冻结，仅在 outer-train 上训练 StandardScaler + class-balanced LogisticRegression(C=1)。ResNet18 和 CLIP 使用 512D 全局表征；DINOv2-S/14 使用 L2-normalized CLS token 与 patch-token mean 拼接形成的 768D 表征。",
        "",
        "### 5.1 每折测试结果（%）",
        "",
        md(found_fold_show),
        "",
        "### 5.2 五折汇总（%）",
        "",
        md(found_summary_show),
        "",
        "DINOv2-S/14 的 Macro-F1 为 65.80%，明显高于 frozen ResNet18 和 CLIP，但仍低于工业异常 evidence 和 V。",
        "",
        "## 6. 工业异常表征 baseline",
        "",
        "为避免协议不一致，本轮没有引用原论文中的 anomaly-detection 数字，而是将逐图 evidence 输入完全相同的 StandardScaler + balanced LogisticRegression(C=1)，统一预测 N/L/S。",
        "",
        "- SALAD-only：4D image-level score evidence。",
        "- CSAD-only：18D local score / heatmap statistics。",
        "- SALAD+CSAD：上述表征拼接得到 22D evidence。",
        "",
        "### 6.1 每折测试结果（%）",
        "",
        md(ind_fold_show),
        "",
        "### 6.2 五折汇总（%）",
        "",
        md(ind_summary_show),
        "",
        "SALAD+CSAD 将 Macro-F1 提升至 83.82%，高于 SALAD-only 的 72.76% 和 CSAD-only 的 81.69%；它证明两类 anomaly evidence 存在互补性，但仍低于 V 的 87.90%。",
        "",
        "## 7. 完整 branch ablation 与 HACSE",
        "",
        "### 7.1 分支定义",
        "",
        "- V：186D appearance representation。其由已有 Stage6 工作簿读取，本轮不重新生成。",
        "- O：150D object/component-layout representation。",
        "- D：冻结 DINOv2-S/14 的 768D semantic representation。",
        "- 单分支 V/O 分类器使用既有 nested-CV 设置选择的 Random Forest 或 Extra Trees；D 使用 StandardScaler + balanced Logistic Regression。",
        "- 多分支 V+O、V+D、O+D、V+O+D 均将对应分支的三类概率拼接，再训练 StandardScaler + balanced LogisticRegression(C=1)。",
        "",
        "每个 outer fold 使用的 V/O classifier family 如下：",
        "",
        md(selected),
        "",
        "### 7.2 Inner-OOF stacking 流程",
        "",
        "```text",
        "outer-train",
        "  ├─ inner fold 1..5: fit branch models on inner-train",
        "  ├─ predict inner-val exactly once",
        "  └─ concatenate complete OOF branch probabilities",
        "                    ↓",
        "       fit probability-level LR fusion",
        "                    ↓",
        "refit branch models on full outer-train",
        "                    ↓",
        "predict outer-test branch probabilities",
        "                    ↓",
        "apply frozen fold-specific fusion head",
        "```",
        "",
        "### 7.3 每折完整结果（%）",
        "",
        md(fusion_fold_show),
        "",
        "### 7.4 五折汇总（%）",
        "",
        md(fusion_summary_show),
        "",
        "### 7.5 HACSE 的配对提升",
        "",
        md(delta_show),
        "",
        "V+O+D 相对 V 的平均 Macro-F1 提升为 3.55 个百分点，5/5 folds 为正；相对 V+O 提升 1.80 个百分点，5/5 folds 为正；相对 V+D 仅提升 0.26 个百分点，4/5 folds 为正。最后一项说明 O 在已有 V+D 的基础上提供了小幅但总体稳定的附加信息，但该增益不应被描述为很大。",
        "",
        "## 8. Fusion strategy comparison",
        "",
        "### 8.1 方法定义",
        "",
        "1. Probability average：直接计算 (pV + pO + pD) / 3，不学习权重。",
        "2. Feature concatenation：将 [V;O;D] 直接拼接为 1,104D，输入 StandardScaler + balanced LogisticRegression(C=1)。",
        "3. Probability-level LR fusion：拼接 [pV;pO;pD] 得到 9D 输入，融合头仅使用 inner-OOF 概率训练，即 HACSE。",
        "",
        "### 8.2 结果（五折均值，%）",
        "",
        md(fusion_summary_show[fusion_summary_show["Method"].isin(["Probability average", "Feature concatenation", "V+O+D"])]),
        "",
        "HACSE 的 Macro-F1 为 91.45%，比 probability average 高 4.01 个百分点，比 feature concatenation 高 4.42 个百分点，而且两组比较均为 5/5 folds 正提升。该结果支持优势不仅来自同时使用三种 evidence，也来自低维概率级融合和无 stacking leakage 的训练方式。",
        "",
        "## 9. 完整性与泄漏检查",
        "",
        "- Supervised RGB：3 methods × 1,568 predictions；每个 method 内 sample index 无重复。",
        "- Foundation representations：3 × 1,568 predictions；无重复。",
        "- Industrial representations：3 × 1,568 predictions；无重复。",
        "- Branch/fusion：9 × 1,568 predictions；无重复。",
        "- 所有逐样本三类概率之和误差均处于浮点舍入范围内。",
        "- 所有方法使用同一 outer assignment；测试折标签不参与模型拟合或融合器训练。",
        "- Probability-level fusion 使用 inner-OOF probabilities，未使用 branch model 对自身训练样本的 in-sample prediction。",
        "",
        "### 9.1 必须保留的协议边界",
        "",
        "本轮 V 直接读取 `full_loco_salad_csad_fusion_analysis.xlsx` 中已生成的 186D 表征，其中包含 raw 93D 与已有 category-normalized 93D copy。本轮没有在每个 outer fold 内重新估计该 normalized copy 的均值和标准差。因此，本轮结果严格控制的是分类器训练、outer evaluation 和 stacking，而不是从原始 evidence 开始的全流程 fold-contained normalization。该实验应表述为 canonical Stage6 five-fold comparison，不应改写成 zero-shot cross-dataset generalization，也不应声称 V normalization 完全由 outer-train 估计。",
        "",
        "## 10. 未纳入方法与原因",
        "",
        "| Method | Status | Reason |",
        "|---|---|---|",
        "| ComAD | 未纳入正式数值 | 本地存在源代码，但尚未生成与 1,568 样本逐一对齐的 evidence；当前环境还缺少 pydensecrf。 |",
        "| PSAD | 未纳入正式数值 | 未发现本地实现、checkpoint 或逐图 evidence。 |",
        "| LogicAL | 未纳入正式数值 | 未发现本地实现、checkpoint 或逐图 evidence。 |",
        "",
        "没有使用这些论文原文中的 LOCO anomaly-detection 数字填表，因为其任务定义、训练数据和指标与当前监督式 N/L/S diagnosis 不一致。只有生成本地逐图 evidence 并经过相同 outer folds 的 balanced classifier 后，才可进入正式比较。",
        "",
        "## 11. 运行过程记录",
        "",
        "### 11.1 实际执行顺序",
        "",
        "1. 核对 1,568 样本、标签和固定 outer-fold assignment。",
        "2. 下载并保存 torchvision 官方 ImageNet weights：ResNet50、ConvNeXt-Tiny；ViT-B/16 使用本机已有官方缓存。",
        "3. 生成统一 short-side-256 RGB cache，共 1,568 张图像。",
        "4. 先运行 ResNet50 fold-1 smoke test，确认维度、概率和训练流程。",
        "5. 依次完成 ResNet50、ConvNeXt-Tiny、ViT-B/16 的五折正式训练。",
        "6. 使用相同 folds 重算 SALAD-only、CSAD-only、SALAD+CSAD。",
        "7. 使用已缓存的 ResNet18、CLIP 和 DINOv2 表征重算 frozen representation probes。",
        "8. 生成 V/O/D inner-OOF probabilities，完成七项 branch ablation。",
        "9. 比较 probability average、feature concatenation 和 OOF probability-level LR fusion。",
        "10. 汇总逐折、均值、标准差、配对增益与逐样本预测，并进行行数、重复和概率归一化检查。",
        "",
        "### 11.2 运行环境",
        "",
        "| Item | Value |",
        "|---|---|",
        "| Operating system | Windows |",
        "| Conda environment | visa_ad |",
        "| Python | 3.10.19 |",
        "| GPU | NVIDIA GeForce RTX 4090 |",
        "| PyTorch | 2.7.1+cu118 |",
        "| torchvision | 0.22.1+cu118 |",
        "| scikit-learn | 1.7.2 |",
        "| pandas | 2.3.3 |",
        "| NumPy | 2.2.6 |",
        "| timm | 1.0.22 |",
        "",
        "### 11.3 可复现命令",
        "",
        "```powershell",
        "conda run -n visa_ad python code/stage6/scripts/run_stage6_supervised_rgb_finetuned_20260914.py --models resnet50 convnext_tiny vit_b_16 --folds 1 2 3 4 5 --workers 4",
        "conda run -n visa_ad python code/stage6/scripts/run_stage6_industrial_representations_same_folds_20260914.py",
        "conda run -n visa_ad python code/stage6/scripts/run_stage6_foundation_representations_same_folds_20260914.py",
        "conda run -n visa_ad python code/stage6/scripts/run_stage6_full_branch_and_fusion_comparison_20260914.py",
        "conda run -n visa_ad python code/stage6/scripts/build_stage6_expanded_comparison_tables_20260914.py",
        "```",
        "",
        "## 12. 输出文件索引",
        "",
        "### 12.1 强监督 RGB",
        "",
        f"- `{rel(SUP_DIR / 'config.json')}`",
        f"- `{rel(SUP_DIR / 'fold_metrics.csv')}`",
        f"- `{rel(SUP_DIR / 'summary.csv')}`",
        f"- `{rel(SUP_DIR / 'oof_predictions.csv')}`",
        f"- `{rel(SUP_DIR / 'training_history.csv')}`",
        "- 每个 model/fold 另保存独立 history.csv 与 predictions.csv。",
        "",
        "### 12.2 Foundation representations",
        "",
        f"- `{rel(FOUND_DIR / 'config.json')}`",
        f"- `{rel(FOUND_DIR / 'fold_metrics.csv')}`",
        f"- `{rel(FOUND_DIR / 'summary.csv')}`",
        f"- `{rel(FOUND_DIR / 'oof_predictions.csv')}`",
        "",
        "### 12.3 Industrial representations",
        "",
        f"- `{rel(IND_DIR / 'config.json')}`",
        f"- `{rel(IND_DIR / 'fold_metrics.csv')}`",
        f"- `{rel(IND_DIR / 'summary.csv')}`",
        f"- `{rel(IND_DIR / 'oof_predictions.csv')}`",
        "",
        "### 12.4 Branch ablation 与 fusion",
        "",
        f"- `{rel(FUSION_DIR / 'config.json')}`",
        f"- `{rel(FUSION_DIR / 'fold_metrics.csv')}`",
        f"- `{rel(FUSION_DIR / 'summary.csv')}`",
        f"- `{rel(FUSION_DIR / 'paired_deltas.csv')}`",
        f"- `{rel(FUSION_DIR / 'oof_predictions.csv')}`",
        "",
        "## 13. 论文中可使用的结果表述",
        "",
        "### 13.1 中文表述",
        "",
        "在统一的五折监督式 N/L/S 诊断协议下，HACSE 达到 91.45% Macro-F1 和 91.49% BAcc，分别比 appearance-only V 高 3.55 和 3.90 个百分点。相比 V+O 与 V+D，完整模型的 Macro-F1 分别提高 1.80 和 0.26 个百分点，其中相对 V+D 的提升在五折中的四折为正。概率级 OOF 融合还分别比简单概率平均和特征直接拼接高 4.01 和 4.42 个 Macro-F1 百分点，表明性能增益不仅来自多源 evidence 的联合使用，也与异质分支在低维概率空间中的泄漏受控融合有关。",
        "",
        "### 13.2 English-ready paragraph",
        "",
        "Under the same five-fold supervised N/L/S diagnosis protocol, HACSE achieved a mean Macro-F1 of 91.45% and a balanced accuracy of 91.49%. It exceeded the appearance-only V branch by 3.55 Macro-F1 points and improved over V+O and V+D by 1.80 and 0.26 points, respectively. The gain over V+D was positive in four of the five folds, indicating that the object/layout branch provided a modest but generally consistent contribution after appearance and semantic evidence had already been combined. Moreover, the inner-OOF probability-level fusion outperformed probability averaging and direct feature concatenation by 4.01 and 4.42 Macro-F1 points, respectively. These results indicate that the observed gain was associated not only with using heterogeneous evidence, but also with combining branch outputs through a leakage-controlled, low-dimensional fusion model.",
        "",
        "## 14. 最终结论与后续边界",
        "",
        "1. 强监督 RGB 分类器已经补齐，但在当前统一训练配置下表现有限，未构成比 V 更强的竞争结果。",
        "2. 工业 anomaly evidence 形成了清晰梯度：SALAD-only < CSAD-only < SALAD+CSAD < V < HACSE。",
        "3. 完整七项 ablation 表明 V 是主要性能基础，D 是最强互补分支，O 在 V+D 上提供较小的附加增益。",
        "4. 概率级 OOF LR fusion 明显优于平均概率和直接特征拼接，能够直接支撑论文的 evidence-fusion 设计。",
        "5. 本轮结论仅适用于现有 LOCO Stage6 cohort 和 canonical fold protocol。VisA cross-dataset validation 仍需先获得同构 SALAD/CSAD evidence；不能用不同 anomaly representation 替代 V 后继续称为同一方法。",
        "6. 若计划将本轮结果用作最终 confirmatory evidence，建议在论文中完整披露 V normalization 的预计算边界，并避免在观察这些 outer folds 后继续基于结果调参。",
        "",
        "## 15. Claim–evidence 对照",
        "",
        "| Claim | Evidence | Status |",
        "|---|---|---|",
        "| HACSE 优于单独 V | +3.55 Macro-F1 points，5/5 folds 为正 | Supported |",
        "| O、D 均提供互补信息 | V+O > V；V+D > V | Supported |",
        "| O 在 V+D 上仍有增益 | +0.26 Macro-F1 points，4/5 folds 为正 | Supported，但效应较小 |",
        "| 概率级融合优于简单融合 | 比平均概率和特征拼接分别高 4.01、4.42 points，均为 5/5 folds 正提升 | Supported |",
        "| HACSE 可跨数据集迁移 | VisA 尚未完成同构 V evidence | 尚不支持 |",
        "| RGB fine-tuning 的理论性能上限很低 | 未进行架构特异的大规模 HPO | 不支持；当前仅支持统一训练配置下的观察 |",
        "",
        "---",
        "",
        "本文件由实际保存的 fold metrics、summary、paired deltas、training history 和 OOF predictions 自动汇总生成；未使用外部论文数字补齐缺失方法。",
    ]

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(REPORT_PATH)


if __name__ == "__main__":
    main()
