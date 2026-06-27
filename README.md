# ADGAMIL

## Aggregate Dynamic Graph Representation with Attention-aware Multiple Instance Learning for Whole Slide Image Analysis

ADGAMIL is a weakly supervised framework for whole slide image (WSI) analysis. It is developed based on [CLAM](https://github.com/mahmoodlab/CLAM) and introduces aggregate dynamic graph representation into attention-aware multiple instance learning for slide-level classification.

This repository contains the complete workflow for WSI patching, feature extraction, model training, evaluation, and heatmap visualization.

## Supported Tasks

| Task | Classification | Feature dimension |
|---|---|---:|
| `TCGA_BRCA_UNI` | IDC vs. ILC | 1024 |
| `TCGA_BRCA_CHIEF` | IDC vs. ILC | 768 |
| `TCGA_NSCLC_UNI` | LUAD vs. LUSC | 1024 |
| `TCGA_NSCLC_CHIEF` | LUAD vs. LUSC | 768 |
| `PANDA_UNI` | Grade 0–5 | 1024 |
| `PANDA_CHIEF` | Grade 0–5 | 768 |

AUC is used for model selection on TCGA-BRCA and TCGA-NSCLC. Quadratic weighted kappa (QWK) is used for PANDA.

## Repository Structure

```text
ADGAMIL-main/
├── main.py
├── eval.py
├── create_patches.py
├── create_patches_fp.py
├── extract_features.py
├── extract_features_fp.py
├── create_heatmaps.py
├── create_splits_seq.py
├── create_splits_tcga_patient.py
├── create_splits_nsclc_uni_chief.py
├── environment.yml
├── dataset_csv/
├── dataset_modules/
├── heatmaps/
├── models/
│   └── ADGAMIL/
├── presets/
├── splits/
├── utils/
├── vis_utils/
└── wsi_core/
```

## Installation

```bash
conda env create -f environment.yml
conda activate <environment_name>
```

A CUDA-capable GPU is recommended.

## WSI Patching

```bash
python create_patches_fp.py \
    --source DATA_DIRECTORY \
    --save_dir PATCH_DIRECTORY \
    --patch_size 256 \
    --seg --patch --stitch
```

## Feature Extraction

```bash
CUDA_VISIBLE_DEVICES=0 python extract_features_fp.py \
    --data_h5_dir PATCH_DIRECTORY \
    --data_slide_dir DATA_DIRECTORY \
    --csv_path CSV_FILE \
    --feat_dir FEATURE_DIRECTORY \
    --batch_size 512 \
    --slide_ext .svs
```

Extracted slide features should be stored as `.pt` files under:

```text
FEATURE_DIRECTORY/pt_files/
```

## Dataset Format

Dataset CSV files are stored in `dataset_csv/` and should contain at least:

```text
case_id,slide_id,label
```

The `slide_id` must match the corresponding `.pt` filename.

Cross-validation split files are stored in `splits/`.

## Training

Before training, update the local feature directory paths in `main.py`.

Example:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
    --task TCGA_BRCA_UNI \
    --data_root_dir . \
    --split_dir TCGA_BRCA_UNI_100 \
    --results_dir ./results \
    --exp_code TCGA_BRCA_UNI \
    --model_type clam_sb \
    --embed_dim 1024 \
    --num_neighbors 5 \
    --k 5 \
    --weighted_sample \
    --early_stopping \
    --subtyping \
    --log_data
```

Available task names:

```text
TCGA_BRCA_UNI
TCGA_BRCA_CHIEF
TCGA_NSCLC_UNI
TCGA_NSCLC_CHIEF
PANDA_UNI
PANDA_CHIEF
```

Display all options with:

```bash
python main.py -h
```

## Evaluation

```bash
python eval.py -h
```

The training pipeline reports AUC, accuracy, macro recall, macro F1 score, specificity, and QWK.

## Heatmap Visualization

Configuration files are located in `heatmaps/configs/`.

```bash
CUDA_VISIBLE_DEVICES=0 python create_heatmaps.py --config CONFIG_FILE.yaml
```

## Main Features

- CLAM-based weakly supervised multiple instance learning
- Aggregate dynamic graph representation
- Attention-aware instance aggregation
- UNI and CHIEF feature support
- Patient-level and site-level split checking
- Class-weighted training and weighted sampling
- Mixed-precision training and gradient clipping
- Early stopping and cross-validation
- TensorBoard logging
- WSI heatmap visualization

## Acknowledgements

This project is developed based on CLAM:

> Lu, M. Y., Williamson, D. F. K., Chen, T. Y., et al. Data-efficient and weakly supervised computational pathology on whole-slide images. *Nature Biomedical Engineering*, 5, 555–570, 2021.

```bibtex
@article{lu2021data,
  title={Data-efficient and weakly supervised computational pathology on whole-slide images},
  author={Lu, Ming Y and Williamson, Drew FK and Chen, Tiffany Y and Chen, Richard J and Barbieri, Matteo and Mahmood, Faisal},
  journal={Nature Biomedical Engineering},
  volume={5},
  number={6},
  pages={555--570},
  year={2021}
}
```

Please also cite the feature encoder used in the experiment.

## License

This repository is intended for academic research. Parts of the project are developed from CLAM; please follow the license and usage requirements of the original CLAM repository.
