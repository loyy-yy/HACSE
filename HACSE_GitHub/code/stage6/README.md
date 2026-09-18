# Stage-6 experiment entry points

Run scripts from the repository root.

| Purpose | Script |
|---|---|
| Full V/O/D branch and fusion comparison | `run_stage6_full_branch_and_fusion_comparison_20260914.py` |
| Industrial evidence baselines | `run_stage6_industrial_representations_same_folds_20260914.py` |
| Frozen representation baselines | `run_stage6_foundation_representations_same_folds_20260914.py` |
| Fine-tuned RGB classifiers | `run_stage6_supervised_rgb_finetuned_20260914.py` |
| O-branch incremental analysis | `analyze_stage6_o_incremental_effect_20260917.py` |
| VisA feasibility/evaluation | `run_visa_vd_crossdataset_20260913.py` |
| Manuscript-facing tables | `build_stage6_expanded_comparison_tables_20260914.py` |

The remaining scripts in `scripts/` are dependency modules imported by these entry points.

## Required local artifacts

The public code package intentionally excludes datasets, model weights, derived features, and generated outputs. Before running the principal Stage-6 experiments, prepare the following paths relative to the repository root:

```text
code/full_loco_salad_csad_fusion_analysis.xlsx
code/stage2_three_class/00_inputs/aligned_input_scores.csv
code/stage6/outputs/stage6_o_v2/stage6_o_v2_features.csv
code/stage6/outputs/stage6_o_v2/stage6_o_v2_raw_components.jsonl
code/stage6/outputs/stage6_nested5fold_ablation/outer_fold_assignments.csv
code/stage6/outputs/stage6_nested5fold_ablation/nested5fold_selected_settings.csv
code/stage6/outputs/stage6_dino_global_v1_ceiling_dev/dino_global_v1_features.npz
```

Scripts create their output directories automatically. Dataset locations are configured through `MVTEC_LOCO_ROOT`, `VISA_ROOT`, and `VISA_RAW_TAR`.
