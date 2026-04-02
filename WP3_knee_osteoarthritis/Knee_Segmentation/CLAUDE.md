# Knee Segmentation Project

Knee joint space segmentation pipeline for Osteoarthritis (OA) research. Uses U-Net/DRN deep learning models to segment joint space in pre-cropped knee X-ray images, with tools for mask expansion and image masking.

## Environment

```bash
# Python environment
C:\Users\imran\miniconda3\envs\knee-segmentation\python.exe

# Activate environment
conda activate knee-segmentation
```

## Data Structure

### Source Data Location
```
C:\Users\imran\AI_hub_keskisuomi\WP3_knee_osteoarthritis\data\
├── train\data\
│   ├── 0\    # Healthy (grade 0) - 2,286 images
│   └── 2\    # OA (grade 2) - 1,516 images
├── val\data\
│   ├── 0\    # 328 images
│   └── 2\    # 212 images
└── test\data\
    ├── 0\    # 639 images
    └── 2\    # 447 images
```

**Total: 5,428 pre-cropped knee X-ray images (PNG format)**

### Dataset Split Summary
| Dataset | Grade 0 | Grade 2 | Total |
|---------|---------|---------|-------|
| Train   | 2,286   | 1,516   | 3,802 |
| Val     | 328     | 212     | 540   |
| Test    | 639     | 447     | 1,086 |
| **Total** | **3,253** | **2,175** | **5,428** |

## Core Scripts

### 1. Segmentation

#### `segment_simple.py`
Simple segmentation for pre-cropped images. Outputs masks only.

```bash
python segment_simple.py -a unet -m MODEL.pth -i <input_dir> -o <output_dir>
```

**Arguments:**
- `-a, --architecture`: `unet` or `drn` (default: unet)
- `-m, --model`: Path to model weights (required)
- `-i, --input-dir`: Input directory or single image file (required)
- `-o, --output-dir`: Output directory (default: out)
- `-nc, --n-classes`: Number of classes (default: 1)
- `-nch, --n-channels`: Number of input channels (default: 1)
- `-s, --scale`: Downscaling factor (default: 1.0)
- `-t, --threshold`: Mask threshold (default: 0.5)

#### `predict_precropped.py`
Segmentation with OA variable calculation (JSW, eminence measurements).

```bash
python predict_precropped.py -a unet -m MODEL.pth -i <input_dir> -sv
```

**Additional arguments:**
- `-sv, --save`: Save segmentation masks
- `-ps, --pixel-spacing`: Pixel spacing in mm/pixel (default: 0.143 0.143)

**Output:** `oa_variables.csv` with measurements

### 2. Mask Expansion

#### `expand_masks.py`
Morphological dilation to expand segmentation masks.

```bash
python expand_masks.py -i <mask_dir> -o <output_dir> -k <kernel_size> -n <iterations>
```

**Arguments:**
- `-i, --input`: Input directory with mask files (required)
- `-o, --output`: Output directory for expanded masks (required)
- `-k, --kernel-size`: Dilation kernel size (default: 5)
- `-n, --iterations`: Number of dilation iterations (default: 2)

**Expansion Presets:**
| Name | Kernel (k) | Iterations (n) | Description |
|------|------------|----------------|-------------|
| Small (S) | 3 | 1 | Minimal expansion |
| Medium (M) | 5 | 2 | Moderate expansion |
| Large (L) | 7 | 3 | Significant expansion |
| Extra Large (XL) | 15 | 5 | Very large expansion |
| **Extra Large (current)** | **9** | **4** | **Currently used preset** |

#### `expand_mask_horizontal.py`
Horizontal-only expansion (no vertical expansion).

```bash
python expand_mask_horizontal.py -i <mask_dir> -o <output_dir> -k <kernel_width> -n <iterations>
```

### 3. Mask Application

#### `apply_mask_blackout.py`
Apply full blackout to masked regions (joint space + adjacent subchondral bone when using expanded masks).

```bash
python apply_mask_blackout.py -i <image_dir> -m <mask_dir> -o <output_dir>
```

**Arguments:**
- `-i, --input`: Directory with original images (required)
- `-m, --masks`: Directory with mask files (required)
- `-o, --output`: Output directory (required)

#### `apply_mask_split.py`
Create left/right split masked versions.

```bash
python apply_mask_split.py -i <image_dir> -m <mask_dir> -l <left_output> -r <right_output>
```

**Arguments:**
- `-i, --input`: Directory with original images (required)
- `-m, --masks`: Directory with mask files (required)
- `-l, --left-output`: Output directory for left-masked images (required)
- `-r, --right-output`: Output directory for right-masked images (required)

### 4. Visualization

#### `generate_comparison_figures.py`
Generate 2x3 comparison figures showing pipeline stages (regular expansion).

```bash
python generate_comparison_figures.py --samples 3
```

**Output panels:**
1. Original image
2. Segmentation Mask (original)
3. Expanded Mask (Extra Large)
4. Full Blackout
5. Left Masked
6. Right Masked

**Output:** `comparison_figures/` directory

#### `generate_comparison_figures_horizontal.py`
Generate 2x3 comparison figures showing horizontal expansion pipeline stages.

```bash
python generate_comparison_figures_horizontal.py --samples 3
```

**Output panels:**
1. Original image
2. Segmentation Mask (original)
3. Horiz Expanded Mask (horizontal-only expansion)
4. Full Blackout
5. Left Masked
6. Right Masked

**Output:** `comparison_figures_horizontal/` directory

#### `generate_portfolio_figure.py`
Generate single comparison figure for a specific image.

### 5. Classification Pipeline

#### `prepare_classification_data.py`
Create classification-ready directory structures from flat ablated image folders by copying files.

```bash
python prepare_classification_data.py --base-dir . --output-dir classification_datasets
```

**Arguments:**
- `--base-dir`: Directory containing ablated folders (default: current dir)
- `--output-dir`: Output directory for classification-ready datasets (default: classification_datasets)

**Creates:** `classification_datasets/{blackout,medial_masked,lateral_masked}/{train,val,test}/{0,2}/` with copied PNGs.

**Note:** Uses `shutil.copy2()` for cross-platform compatibility (not symlinks).

#### `classify_train.py`
Train a ResNet-18 binary classifier (KL0 vs KL2) with ImageNet pretraining adapted for single-channel grayscale input.

```bash
python classify_train.py --data-dir knee_osteoarthritis_dataset --name baseline --epochs 25
python classify_train.py --data-dir classification_datasets/blackout --name blackout --epochs 25
```

**Arguments:**
- `--data-dir`: Root dataset directory containing train/val/test subdirs (required)
- `--name`: Experiment name for output directory (required)
- `--epochs`: Number of training epochs (default: 25)
- `--batch-size`: Batch size (default: 32)
- `--lr`: Learning rate (default: 1e-4)
- `--patience`: Early stopping patience (default: 7)
- `--gpu`: GPU index (default: 0)
- `--no-pretrained`: Train from scratch without ImageNet weights

**Key design choices:**
- ResNet-18 with ImageNet pretraining, conv1 adapted by averaging 3-channel weights to 1 channel
- Weighted CrossEntropyLoss for class imbalance (3,253 KL0 vs 2,175 KL2)
- AdamW optimizer with ReduceLROnPlateau scheduler
- Data augmentation: horizontal flip, rotation (±5°), translation (5%)

**Output:** `classification_results/{name}/best_model.pth`, `training_log.csv`

#### `classify_evaluate.py`
Evaluate a trained classifier on any test set. Produces accuracy, precision, recall, F1, confusion matrix, and confidence analysis.

```bash
# Self-evaluation
python classify_evaluate.py --model classification_results/baseline/best_model.pth \
    --test-dir knee_osteoarthritis_dataset/test --name baseline_self

# Cross-evaluation
python classify_evaluate.py --model classification_results/baseline/best_model.pth \
    --test-dir classification_datasets/blackout/test --name baseline_on_blackout
```

**Arguments:**
- `--model`: Path to best_model.pth checkpoint (required)
- `--test-dir`: Test directory with 0/ and 2/ subdirs (required)
- `--name`: Evaluation name for output files (required)
- `--save-per-image`: Save per-image predictions CSV with softmax probabilities

**Output:** `classification_results/evaluations/{name}.json` and optional `{name}_per_image.csv`

#### `run_all_experiments.py`
Master script to run all 4 classification experiment phases.

```bash
python run_all_experiments.py --epochs 25 --batch-size 32
python run_all_experiments.py --phase 2  # evaluation only
python run_all_experiments.py --phase 4  # summary only
```

**Phases:**
1. Train 4 models (baseline, blackout, medial_masked, lateral_masked)
2. Evaluate each model on its own test set
3. Cross-evaluate baseline model on all 4 test sets (confidence analysis)
4. Generate comparison summary with RQ-specific analysis

#### `sort_lateral_medial.py`
Sort left/right masked images into lateral/medial subfolders based on L/R filename suffix.

```bash
python sort_lateral_medial.py
```

**Anatomy mapping:**
- Left knee (L suffix): left side = medial, right side = lateral
- Right knee (R suffix): left side = lateral, right side = medial

### 6. Grad-CAM Visualization (RQ4)

#### `generate_gradcam.py`
Generate Grad-CAM heatmaps for trained ResNet-18 classifiers. Hooks into layer4 (last residual block).

```bash
# Single condition
python generate_gradcam.py --model classification_results/baseline/best_model.pth \
    --test-dir knee_osteoarthritis_dataset/test --name baseline --num-samples 10

# All 7 conditions (4 self + 3 cross-eval)
python generate_gradcam.py --all-conditions --num-samples 10
```

**Output:** `gradcam_results/{name}/` with `*_heatmap.png`, `*_gradcam.png` (overlay), `*_sidebyside.png` (3-panel), and `summary.json`.

### 7. Statistical Robustness

#### `run_robustness.py`
Multi-seed experiment runner for statistical significance analysis.

```bash
# Full robustness (5 seeds × 4 conditions = 20 training runs)
python run_robustness.py --seeds 5 --epochs 25

# Quick test (3 seeds, baseline only)
python run_robustness.py --seeds 3 --epochs 25 --conditions baseline

# Summary only (after training complete)
python run_robustness.py --summary-only
```

**Output:** `classification_results/robustness/robustness_summary.json` and `.csv` with mean ± std across seeds.

### 8. Utilities

#### `check_image_type.py`
Analyze image to determine if it's pre-cropped or full X-ray.

#### `knee_localizer.py`
Localize knee region in full X-ray images.

## Experiment Design

### 4 Experimental Conditions
| Condition | Dataset | What's Masked | Research Question |
|-----------|---------|---------------|-------------------|
| Baseline | knee_osteoarthritis_dataset | Nothing (original) | Control |
| Blackout | classification_datasets/blackout | Joint region removed (joint space + adjacent subchondral bone via expanded mask) | Main RQ |
| Medial Masked | classification_datasets/medial_masked | Medial compartment removed | Main RQ |
| Lateral Masked | classification_datasets/lateral_masked | Lateral compartment removed | Main RQ |

### Evaluation Matrix
- **Phase 2 (Self-eval):** Each model tested on its own test set → accuracy comparison
- **Phase 3 (Cross-eval):** Baseline model tested on all 4 test sets → confidence shift analysis
- **Phase 4 (Summary):** Comparison summary with accuracy drops and confidence metrics across conditions

### Hardware
- **GPU (authoritative results):** NVIDIA RTX 4080 Laptop GPU (12GB VRAM), ~22-28s per epoch, ~7 min per model
- **CPU (initial runs, superseded):** Intel Core i9-13980HX (32 threads), ~50 min per model
- **GPU speedup:** ~7× across all experiments
- **GPU environment:** `.venv_gpu` virtual environment with CUDA-enabled PyTorch

### Classification Results Directory
```
classification_results/
├── baseline/best_model.pth
├── blackout/best_model.pth
├── medial_masked/best_model.pth
├── lateral_masked/best_model.pth
├── evaluations/
│   ├── baseline_self.json
│   ├── blackout_self.json
│   ├── medial_masked_self.json
│   ├── lateral_masked_self.json
│   ├── baseline_on_baseline.json
│   ├── baseline_on_blackout.json
│   ├── baseline_on_medial_masked.json
│   └── baseline_on_lateral_masked.json
├── experiment_summary.json
├── robustness/
│   ├── robustness_summary.json
│   └── robustness_summary.csv
├── seed_42/{baseline,blackout,medial_masked,lateral_masked}/best_model.pth
├── seed_179/{baseline,blackout,medial_masked,lateral_masked}/best_model.pth
├── seed_316/{baseline,blackout,medial_masked,lateral_masked}/best_model.pth
├── seed_453/{baseline,blackout,medial_masked,lateral_masked}/best_model.pth
└── seed_590/{baseline,blackout,medial_masked,lateral_masked}/best_model.pth

gradcam_results/
├── baseline/           # 10 samples: heatmap, overlay, side-by-side
├── blackout/
├── medial_masked/
├── lateral_masked/
├── baseline_on_blackout/
├── baseline_on_medial/
└── baseline_on_lateral/
```

## Complete Pipeline Workflow

### Step 1: Run Segmentation
```bash
# For all 6 dataset folders
python segment_simple.py -a unet -m MODEL.pth \
    -i "C:/Users/imran/AI_hub_keskisuomi/WP3_knee_osteoarthritis/data/train/data/0" \
    -o results_train_0
```

### Step 2: Expand Masks (Extra Large: k=9, n=4)
```bash
python expand_masks.py -i results_train_0 -o results_train_0_extralarge -k 9 -n 4
```

### Step 3: Apply Masks (3 Types)

#### Full Blackout
```bash
python apply_mask_blackout.py \
    -i "C:/Users/imran/AI_hub_keskisuomi/WP3_knee_osteoarthritis/data/train/data/0" \
    -m results_train_0_extralarge \
    -o blackout_train_0_extralarge
```

#### Left/Right Split
```bash
python apply_mask_split.py \
    -i "C:/Users/imran/AI_hub_keskisuomi/WP3_knee_osteoarthritis/data/train/data/0" \
    -m results_train_0_extralarge \
    -l left_masked_train_0_extralarge \
    -r right_masked_train_0_extralarge
```

### Step 4: Generate Visualizations
```bash
python generate_comparison_figures.py --samples 2
```

## Output Directory Structure

### Mask Results
```
results_{dataset}_{grade}/                    # Original segmentation masks
results_{dataset}_{grade}_extralarge/         # Expanded masks (k=9, n=4)
results_{dataset}_{grade}_xl/                 # XL expanded masks (k=15, n=5)
results_{dataset}_{grade}_extralarge_horiz/   # Horizontal-only expansion
```

### Masked Images (Regular Expansion)
```
blackout_{dataset}_{grade}_extralarge/      # Full joint region blackout (space + bone margins)
left_masked_{dataset}_{grade}_extralarge/   # Left side masked
right_masked_{dataset}_{grade}_extralarge/  # Right side masked
```

### Masked Images (Horizontal Expansion)
```
blackout_{dataset}_{grade}_extralarge_horiz/      # Full joint region blackout (horizontal, space + bone margins)
left_masked_{dataset}_{grade}_extralarge_horiz/   # Left side masked (horizontal)
right_masked_{dataset}_{grade}_extralarge_horiz/  # Right side masked (horizontal)
```

### Visualizations
```
comparison_figures/                     # 2x3 comparison grids (regular expansion)
comparison_figures_horizontal/          # 2x3 comparison grids (horizontal expansion)
pipeline_visualization.png              # Single sample visualization
```

## Model Architecture

### U-Net
- Input channels: 1 (grayscale)
- Output classes: 1 (binary mask)
- Bilinear upsampling: True

### DRN (Dilated Residual Network)
- Model: drn_d_105
- Input channels: 1
- Output classes: 1

## Current Output Summary

### Regular Expansion (Extra Large k=9, n=4)

| Output Type | train_0 | train_2 | val_0 | val_2 | test_0 | test_2 | Total |
|-------------|---------|---------|-------|-------|--------|--------|-------|
| Expanded Masks | 2,286 | 1,516 | 328 | 212 | 639 | 447 | 5,428 |
| Full Blackout | 2,286 | 1,516 | 328 | 212 | 639 | 447 | 5,428 |
| Left Masked | 2,286 | 1,516 | 328 | 212 | 639 | 447 | 5,428 |
| Right Masked | 2,286 | 1,516 | 328 | 212 | 639 | 447 | 5,428 |

**Subtotal: 21,712** (5,428 x 4 types)

### Horizontal Expansion (Extra Large Horiz) — PRIMARY (used for thesis)

| Output Type | train_0 | train_2 | val_0 | val_2 | test_0 | test_2 | Total |
|-------------|---------|---------|-------|-------|--------|--------|-------|
| Horiz Expanded Masks | 2,286 | 1,516 | 328 | 212 | 639 | 447 | 5,428 |
| Full Blackout (Horiz) | 2,286 | 1,516 | 328 | 212 | 639 | 447 | 5,428 |
| Left Masked (Horiz) | 2,286 | 1,516 | 328 | 212 | 639 | 447 | 5,428 |
| Right Masked (Horiz) | 2,286 | 1,516 | 328 | 212 | 639 | 447 | 5,428 |

**Subtotal: 21,712 generated** (5,428 x 4 types)

**Thesis classification experiments used: 16,248 images** (horizontal expanded — blackout + left-masked + right-masked, actual file count across all train/val/test splits)

**Total output images: 43,424** (5,428 x 8 types, both variants)

## Dependencies

- PyTorch
- torchvision
- OpenCV (cv2)
- NumPy
- Pillow (PIL)
- pandas

## Notes

- All images are grayscale PNG format
- Mask files follow naming convention: `{image_id}_mask.png`
- Grade 0 = Healthy, Grade 2 = Osteoarthritis
- Pixel spacing default: 0.143 mm/pixel
