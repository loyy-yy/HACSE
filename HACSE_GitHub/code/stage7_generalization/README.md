# Stage7 跨数据集泛化对比实验

## 当前结论

现有 Stage6 的核心结果是 MVTec LOCO AD 上的三分类 `normal / logical / structural`。V11 canonical 主链为 `V -> V+O -> V+O+R`，但现有 186 维 V 特征和 O/R 特征缓存均与 LOCO 样本绑定，不能直接拼接到外部数据集后宣称完成泛化实验。

本目录提供一套独立、可审计的跨数据集协议。它不会修改 Stage6 的冻结文件，也不会在缺少外部特征时伪造结果。

## 推荐数据集矩阵

| 数据集 | 三分类来源 | 适合回答的问题 | 论文中的表述边界 |
| --- | --- | --- | --- |
| MVTec LOCO AD | 官方 normal/logical/structural | 域内主结果和消融 | 原生三分类基准 |
| MVTec AD semantic | normal 为官方 good；3 个缺陷族按 LogicAL 文献重分为 logical，其余为 structural | 跨对象、跨缺陷类型的补充验证 | 文献启发的语义重分组，不是官方三分类协议 |
| VisA semantic | 官方多类 mask；仅 PCB 的 Missing/Extra/Wrong Place 归为 logical；混合缺陷剔除 | 跨数据来源、复杂多对象场景 | 基于官方 mask 类别的作者重分组 |
| VID-AD | 官方 normal/logical，无 structural | 逻辑异常在背景、低照、模糊下的稳健性 | 部分类别覆盖，不能报告完整三分类 Macro-F1 |

MVTec AD 2、Real-IAD、普通 GoodsAD 主要提供 structural anomaly，不能独立支持完整三分类。它们可作为 structural-only 压力测试，但不应与原生三分类结果放在同一主表中。

## 公平实验设计

主表建议同时报告两种协议：

1. `target-refit`：每个数据集重新提取相同定义的 V/O/R 特征，在目标数据集内训练与测试。它检验方法是否能迁移到新数据集，但不是零样本域迁移。
2. `source-to-target`：只在 MVTec LOCO 上训练，冻结分类器后直接测试外部数据集。它才是严格的直接跨数据集泛化。

所有方法使用同一 manifest、同一标签映射、同一划分和同一图像集合。主指标为 Accuracy、Balanced Accuracy、Macro-F1 与三类 F1。每个数据集至少运行 5 个种子，并报告 mean±std；预测层面的比较用同一样本上的配对差异。类别缺失的数据集只报告 observed-class 指标。

建议比较以下方法：

- `V`：现有视觉证据基线。
- `V+O`：对象证据增量。
- `V+O+R`：当前 ODRC canonical 主方法。
- `Qwen2.5-VL direct`：统一三分类提示词的 VLM 基线。
- `WinCLIP/AnomalyCLIP three-prompt`：若使用自定义三类 prompt，必须标为 adapted baseline。
- `LogiCo`：其官方实现支持 MVTec LOCO、MVTec AD、VisA、Real-IAD，适合作为新的统一结构/逻辑强基线，但它的输出任务与本文三分类决策仍需用同一规则转换。

## 使用方法

先生成样本清单并运行烟雾测试：

```powershell
cd code/stage7_generalization
.\run_stage7.ps1 -SkipMissing
```

如果 Stage6 canonical 预测存在，该命令还会自动将 `V / V+O / ODRC` 接入统一评估器，并在 `outputs/loco_canonical_check` 下生成覆盖率、主指标、逐类指标、混淆矩阵和配对比较。

`dataset_config.json` 已指向本地 MVTec LOCO 和 MVTec AD。下载 VisA 与 VID-AD 后，只需修改其中的根目录再重跑。缺失数据集在 summary 中记录为 `missing`。

外部数据必须重新走相同的特征定义，输出一个 CSV：

```text
sample_id,V__...,O__...,R__...
```

先执行严格校验：

```powershell
conda run -n visa_ad python validate_feature_bundle.py `
  --manifest outputs/manifests/combined_manifest.csv `
  --features outputs/features/all_datasets_vor.csv `
  --report outputs/features/validation.json
```

目标数据集内重训与直接跨域实验分别运行：

```powershell
conda run -n visa_ad python run_tabular_benchmark.py --protocol target-refit `
  --manifest outputs/manifests/combined_manifest.csv `
  --features outputs/features/all_datasets_vor.csv `
  --out-dir outputs/target_refit

conda run -n visa_ad python run_tabular_benchmark.py --protocol source-to-target `
  --source-dataset mvtec_loco `
  --manifest outputs/manifests/combined_manifest.csv `
  --features outputs/features/all_datasets_vor.csv `
  --out-dir outputs/source_to_target
```

若已有 ODRC 或其他基线的预测，整理为长表 `sample_id,method,pred_group` 后统一评价：

```powershell
conda run -n visa_ad python evaluate_predictions.py `
  --manifest outputs/manifests/combined_manifest.csv `
  --predictions outputs/all_method_predictions.csv `
  --reference-method ODRC `
  --out-dir outputs/comparison
```

## 数据和标签来源

- MVTec LOCO AD 官方页：https://www.mvtec.com/research-teaching/datasets/mvtec-loco-ad
- LogicAL CVPRW 2024：https://openaccess.thecvf.com/content/CVPR2024W/VAND/html/Zhao_LogicAL_Towards_Logical_Anomaly_Synthesis_for_Unsupervised_Anomaly_Localization_CVPRW_2024_paper.html
- VisA 官方仓库：https://github.com/amazon-science/spot-diff
- VisA 官方 mask 类别：https://raw.githubusercontent.com/amazon-science/spot-diff/main/utils/id2class.py
- VID-AD 官方仓库：https://github.com/nkthiroto/VID-AD
- LogiCo 官方实现：https://github.com/cnulab/LogiCo

## 重要限制

当前项目审计文件已明确：只有当 V、O 和 R 都从目标图像按同一流程重新提取时，才能称为端到端分布迁移或外部数据集泛化。仅替换某一分支、继续使用 LOCO 缓存特征，只能称为 branch stress test。
