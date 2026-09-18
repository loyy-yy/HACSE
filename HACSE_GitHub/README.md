# HACSE

Official research code and experiment artifacts for **Heterogeneous Appearance--Composition--Semantic Evidence Fusion for Fine-Grained Industrial Anomaly Diagnosis**.

HACSE maps three heterogeneous evidence branches into a shared Normal/Logical/Structural probability space:

- **V**: appearance evidence derived from SALAD and CSAD responses;
- **O**: layout-aware composition evidence;
- **D**: frozen DINOv2-S/14 semantic evidence;
- **HACSE**: leakage-controlled probability-level fusion trained from inner out-of-fold predictions.

## Main results

All MVTec LOCO AD numbers below use the same category-and-class-stratified five-fold supervised diagnosis protocol.

| Method | Macro-F1 (%) | Structural-F1 (%) |
|---|---:|---:|
| DINOv2-S/14 frozen | 65.80 | 66.31 |
| SALAD + CSAD | 83.82 | 75.91 |
| V | 87.90 | 82.21 |
| V + O | 89.65 | 85.39 |
| V + D | 91.19 | 88.35 |
| **HACSE (V + O + D)** | **91.45** | **88.64** |

The repository contains the implementation and protocol documentation only; manuscript files, references, datasets, model weights, derived feature files, and generated experiment outputs are excluded.

## Repository layout

```text
HACSE_GitHub/
├── code/
│   ├── stage6/
│   │   ├── README.md
│   │   └── scripts/          # Core experiment and analysis scripts
│   └── stage7_generalization/ # Cross-dataset evaluation utilities
├── .gitignore
├── environment.example
└── requirements.txt
```

Raw datasets, derived features, generated outputs, pretrained model weights, image caches, and third-party repositories are intentionally excluded.

## Installation

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install a PyTorch build compatible with your CUDA version if GPU experiments are required.

## Dataset configuration

Download MVTec LOCO AD and VisA from their official sources. Do not commit the extracted datasets to this repository. Set paths through environment variables:

```powershell
$env:MVTEC_LOCO_ROOT = 'D:\datasets\mvtec_loco_anomaly_detection'
$env:VISA_ROOT = 'D:\datasets\ViSA_pytorch\1cls'
$env:VISA_RAW_TAR = 'D:\datasets\VisA_20220922.tar'
```

The expected dataset layouts are described below and in `code/stage7_generalization/README.md`.

MVTec LOCO AD should contain `breakfast_box`, `juice_bottle`, `pushpins`, `screw_bag`, and `splicing_connectors`. VisA should use the 12-category `1cls` layout.

## Reproducing the principal tables

Prepare the required feature files and fixed folds described in `code/stage6/README.md`, then run:

```bash
python code/stage6/scripts/run_stage6_full_branch_and_fusion_comparison_20260914.py
python code/stage6/scripts/build_stage6_expanded_comparison_tables_20260914.py
```

Additional entry points:

```bash
# SALAD-only, CSAD-only, and SALAD+CSAD
python code/stage6/scripts/run_stage6_industrial_representations_same_folds_20260914.py

# Frozen ResNet18, CLIP, and DINOv2 probes
python code/stage6/scripts/run_stage6_foundation_representations_same_folds_20260914.py

# Fine-tuned ResNet50, ConvNeXt-Tiny, and ViT-B/16
python code/stage6/scripts/run_stage6_supervised_rgb_finetuned_20260914.py

# Incremental contribution of O over V+D
python code/stage6/scripts/analyze_stage6_o_incremental_effect_20260917.py

# VisA feasibility audit or evaluation
python code/stage6/scripts/run_visa_vd_crossdataset_20260913.py --mode audit --visa-root /path/to/visa
```

The VisA directory currently records the feasibility audit. It must not be reported as a completed V+D result until the same 93-D SALAD/CSAD appearance evidence and 768-D frozen DINOv2 features have been generated for VisA.

## Reproducibility notes

- Outer folds are fixed by `outer_fold_assignments.csv`.
- Fusion heads use inner out-of-fold probabilities; outer-test samples are never used to train a fusion head.
- The random seed for the primary protocol is `20260827`.
- Development-only scripts are retained only when they are imported by a principal experiment entry point.

## Before making the repository public

Choose and add an appropriate software license before making the repository public. The current package intentionally does not assign a license on the author's behalf.
