# Knee Segmentation & Image Processing Pipeline

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)
![Computer Vision](https://img.shields.io/badge/Computer_Vision-Deep_Learning-success)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

## Overview

This project provides a complete pipeline for knee X-ray image segmentation and processing using a trained U-Net model. The pipeline is designed for **pre-cropped knee images** (224×224 pixels) and includes four main processing steps:

1. **Mask Generation** - Segment joint space regions from knee X-rays
2. **Mask Expansion** - Enlarge masks to cover more bone area (optional)
3. **Full Blackout** - Replace entire segmented regions with black pixels  
4. **Split Blackout** - Create left-half and right-half masked versions

### Original Project Information
- Forks from https://github.com/fyu/drn and https://github.com/milesial/Pytorch-UNet
- A tool for segmenting the joint space of the knee automatically
- UNET and DRN architectures trained with histogram-equalized 5570 radiographs from OAI-dataset
- Tested on MOST dataset: UNet achieves IOU-score of 0.93, DRN achieves 0.90
- Modified to work with pre-cropped knee images for data augmentation and analysis

## Table of Contents
- [Setup](#setup)
- [Quick Start](#quick-start)
- [Pipeline Components](#pipeline-components)
  - [1. Mask Generation](#1-mask-generation-segment_simplepy)
  - [2. Mask Expansion (Regular & Horizontal)](#2-mask-expansion-expand_maskspy)
  - [3. Full Blackout](#3-full-blackout-apply_mask_blackoutpy)
  - [4. Split Blackout](#4-split-blackout-apply_mask_splitpy)
  - [5. Visualizations](#5-visualizations-generate_comparison_figurespy)
- [Classification & Experiments Pipeline](#classification--experiments-pipeline)
  - [1. Data Preparation](#1-data-preparation-prepare_classification_datapy)
  - [2. Classification Training & Evaluation](#2-classification-training--evaluation-run_all_experimentspy)
  - [3. Multi-seed Robustness Analysis](#3-multi-seed-robustness-analysis-run_robustnesspy)
  - [4. Grad-CAM Visualization](#4-grad-cam-visualization-generate_gradcampy)
- [Batch Processing](#batch-processing)
- [Directory Structure](#directory-structure)
- [Legacy Scripts](#legacy-scripts)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

---

## Setup

### Prerequisites
- Python 3.10 to 3.12 (Python 3.12 is recommended for GPU support)
- uv (fast Python package installer) or Conda
- Windows, Linux, or macOS
- (Optional) NVIDIA GPU (e.g., RTX 40-series) for hardware acceleration

### Environment Setup (Recommended: uv with GPU Support)

To fully utilize an NVIDIA GPU (like an RTX 4080), we recommend using `uv` to install PyTorch with CUDA 12.4.

1. **Create and activate a GPU-compatible virtual environment:**
```powershell
uv venv .venv_gpu --python 3.12
.venv_gpu\Scripts\activate
```

2. **Install GPU-accelerated PyTorch:**
```powershell
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

3. **Install remaining dependencies:**
```powershell
uv pip install pandas scipy scikit-learn matplotlib tqdm opencv-python pydicom scikit-image pillow
```

4. **Verify GPU is working:**
```powershell
python -c "import torch; print(f'GPU Found: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'No GPU')"
```

### Alternative Setup (Conda / CPU only)

1. **Create conda environment:**
```bash
conda create -n knee-segmentation python=3.10
conda activate knee-segmentation
```

2. **Install dependencies:**
```bash
pip install torch torchvision opencv-python scikit-image pydicom pandas numpy scipy matplotlib pillow
```

**Windows users**: If you encounter OpenCV DLL errors:
```bash
pip install opencv-python==4.10.0.84
```

3. **Verify model file**: Ensure `MODEL_unet.pth` exists in the directory

---

## Quick Start

### Single Folder Processing

Process one folder through the complete pipeline:

```powershell
# 1. Generate masks from original images
python segment_simple.py -i "C:\path\to\images" -m "MODEL_unet.pth" -o "masks_out"

# 2. (Optional) Expand masks to cover more bone area
python expand_masks.py -i "masks_out" -o "masks_expanded" -k 5 -n 2

# 3. Create full blackout version (entire gap darkened)
python apply_mask_blackout.py -i "C:\path\to\images" -m "masks_out" -o "blackedout"

# 4. Create left/right split versions
python apply_mask_split.py -i "C:\path\to\images" -m "masks_out" -l "left_masked" -r "right_masked"
```

**Note:** You can use expanded masks in steps 3 and 4 by replacing `"masks_out"` with `"masks_expanded"`

---

## Pipeline Components

### 1. Mask Generation (`segment_simple.py`)

Generates binary segmentation masks identifying joint space regions in knee X-rays using the trained U-Net model.

#### Usage
```bash
python segment_simple.py -i <input_dir> -m <model_path> -o <output_dir>
```

#### Parameters
| Parameter | Description | Default |
|-----------|-------------|---------|
| `-i, --input` | Directory containing original knee X-ray images (PNG) | Required |
| `-m, --model` | Path to trained U-Net model | `MODEL_unet.pth` |
| `-o, --output` | Directory to save generated masks | Required |

#### Output
- Binary mask images saved as `{original_filename}_mask.png`
- White pixels (255) = segmented joint space regions
- Black pixels (0) = background
- Typical segmentation: 0.4% - 10% of image area

#### Example
```bash
python segment_simple.py -i "data/test/data/0" -m "MODEL_unet.pth" -o "out_test_0"
```

**Expected Output:**
```
Found 639 images in data/test/data/0
Model loaded from MODEL_unet.pth
Processing: 100%|████████████| 639/639 [02:15<00:00, 4.73it/s]
Predictions saved to out_test_0/
```

---

### 2. Mask Expansion (`expand_masks.py`)

Expands segmentation masks to cover more area using morphological dilation. This is useful for:
- Including bone edges around joint space
- Creating larger masked regions for data augmentation
- Testing different mask sizes

#### Usage
```bash
python expand_masks.py -i <mask_dir> -o <output_dir> [-k <kernel_size>] [-n <iterations>]
```

#### Parameters
| Parameter | Description | Default |
|-----------|-------------|---------|
| `-i, --input` | Directory containing mask images (from step 1) | Required |
| `-o, --output` | Directory to save expanded masks | Required |
| `-k, --kernel-size` | Size of dilation kernel (larger = more expansion) | 5 |
| `-n, --iterations` | Number of dilation iterations (more = larger expansion) | 2 |

#### Output
- Expanded binary mask images saved as `{original_name}_mask.png`
- Same filenames as input masks for easy pipeline integration
- Expansion levels can be customized:
  - **Small (S)**: `-k 3 -n 1` (minimal expansion)
  - **Medium (M)**: `-k 5 -n 2` (default, moderate expansion)
  - **Large (L)**: `-k 7 -n 3` (significant expansion)
  - **Extra Large (XL)**: `-k 10 -n 5` (maximum expansion)

#### Example
```bash
# Default expansion (Medium)
python expand_masks.py -i "out_test_0" -o "out_test_0_expanded"

# Large expansion
python expand_masks.py -i "out_test_0" -o "out_test_0_expanded_L" -k 7 -n 3

# Extra large expansion
python expand_masks.py -i "out_test_0" -o "out_test_0_expanded_XL" -k 10 -n 5

# Horizontal-only expansion (preserves vertical constraints)
python expand_mask_horizontal.py -i "out_test_0" -o "out_test_0_expanded_horiz" -k 9 -n 4
```

**Expected Output:**
```
Original masks: 639
Output directory: out_test_0_expanded
Expansion settings: kernel_size=5, iterations=2

1/639: 9003175L_mask.png ✓
2/639: 9003175R_mask.png ✓
...
639/639: 9998089L_mask.png ✓

==================================================
✓ Success: 639/639
✗ Failed: 0/639
✓ Expanded masks saved to: out_test_0_expanded/
==================================================
```

---

### 3. Full Blackout (`apply_mask_blackout.py`)

Creates versions of original images where the **entire** segmented region is replaced with black pixels (value 0).

#### Usage
```bash
python apply_mask_blackout.py -i <input_dir> -m <mask_dir> -o <output_dir>
```

#### Parameters
| Parameter | Description | Required |
|-----------|-------------|----------|
| `-i, --input` | Directory containing original images | Yes |
| `-m, --masks` | Directory containing mask images (from step 1) | Yes |
| `-o, --output` | Directory to save blackedout images | Yes |

#### Output
- Original images with masked regions set to black (pixel value 0)
- Same dimensions as original images
- Useful for removing joint space information completely
- Can use regular or expanded masks

#### Example
```bash
# Using regular masks
python apply_mask_blackout.py -i "data/test/data/0" -m "out_test_0" -o "blackedout_test_0"

# Using expanded masks
python apply_mask_blackout.py -i "data/test/data/0" -m "out_test_0_expanded_L" -o "blackedout_test_0_expanded_L"
```

**Expected Output:**
```
Original images: 639
Mask directory: out_test_0
Output directory: blackedout_test_0

1/639: 9003175L.png ✓
2/639: 9003175R.png ✓
...
639/639: 9998089L.png ✓

==================================================
✓ Success: 639/639
✗ Failed: 0/639
✓ Output saved to: blackedout_test_0/
==================================================
```

---

### 4. Split Blackout (`apply_mask_split.py`)

Splits each mask **vertically down the middle** and creates two versions:
- **Left-masked**: Only left half of the mask applied (left side darkened)
- **Right-masked**: Only right half of the mask applied (right side darkened)

#### Usage
```bash
python apply_mask_split.py -i <input_dir> -m <mask_dir> -l <left_output_dir> -r <right_output_dir>
```

#### Parameters
| Parameter | Description | Required |
|-----------|-------------|----------|
| `-i, --input` | Directory containing original images | Yes |
| `-m, --masks` | Directory containing mask images (from step 1) | Yes |
| `-l, --output-left` | Directory to save left-masked images | Yes |
| `-r, --output-right` | Directory to save right-masked images | Yes |

#### Output
- **Left-masked folder**: Original images with left half of joint space darkened
- **Right-masked folder**: Original images with right half of joint space darkened
- Useful for analyzing left vs right knee differences or creating augmented datasets

#### Example
```bash
python apply_mask_split.py -i "data/test/data/0" -m "out_test_0" -l "left_masked_test_0" -r "right_masked_test_0"
```

**Expected Output:**
```
Original images: 639
Mask directory: out_test_0
Output left-masked: left_masked_test_0
Output right-masked: right_masked_test_0

1/639: 9003175L.png ✓
2/639: 9003175R.png ✓
...
639/639: 9998089L.png ✓

==================================================
✓ Success: 639/639
✗ Failed: 0/639
✓ Left-masked saved to: left_masked_test_0/
✓ Right-masked saved to: right_masked_test_0/
==================================================
```

---

### 5. Visualizations (`generate_comparison_figures.py`)

Generates 2x3 comparison grids of the pipeline stages (Original, Mask, Expanded, Blackout, Left Mask, Right Mask) for quick sanity checking and reporting. Includes horizontal-variants.

#### Usage
```bash
# Generate regular 2x3 comparison figures
python generate_comparison_figures.py --samples 3

# Generate horizontal-expansion comparison figures
python generate_comparison_figures_horizontal.py --samples 3
```
*Outputs are saved to `comparison_figures/` and `comparison_figures_horizontal/`.*

---

## Classification & Experiments Pipeline

A complete suite of tools is provided to train classifiers, benchmark robustness, and generate Explainable AI (XAI) visualizations (Grad-CAM).

### 1. Data Preparation (`prepare_classification_data.py`)

Before training, the flat ablated image folders must be organized into classification-ready datasets (train/val/test).

#### Usage
```powershell
# Step 1: Sort left/right masks into anatomical Medial/Lateral directories
python sort_lateral_medial.py

# Step 2: Structure datasets for PyTorch ImageFolder
python prepare_classification_data.py --base-dir . --output-dir classification_datasets
```
*This creates the `classification_datasets/` structure used by the ML models.*

### 2. Classification Training & Evaluation (`run_all_experiments.py`)

A master script that trains and evaluates classifiers over the 4 experimental condition datasets (baseline, blackout, lateral, medial). It performs cross-evaluations to test confidence shifts.

#### Usage
To run the full suite with GPU acceleration across 25 epochs:
```powershell
python run_all_experiments.py --epochs 25 --batch-size 32
```
To run only particular phases:
```powershell
python run_all_experiments.py --phase 2  # Evaluate existing models
python run_all_experiments.py --phase 3  # Cross-evaluation
```

### 3. Multi-seed Robustness Analysis (`run_robustness.py`)

Runs the classification experiments using multiple random seeds to provide statistically significant results (mean ± std). Essential for reducing variance and proving robustness across models.

#### Usage
```powershell
# Run a 5-seed robustness test across all 4 conditions
python run_robustness.py --seeds 5 --epochs 25

# Quick test with 1 seed on baseline data
python run_robustness.py --seeds 1 --conditions baseline --epochs 1
```
Results will save to `classification_results/robustness/robustness_summary.csv` for thesis tables.

### 4. Grad-CAM Visualization (`generate_gradcam.py`)

Produces Class Activation Maps (Grad-CAM) to see what image regions the ResNet-18 model is focusing on to classify KL0 vs KL2.

#### Usage
```powershell
# Generate comparison heatmaps for all 4 model conditions
python generate_gradcam.py --all-conditions --num-samples 10
```
Heatmap overlays are saved in the `gradcam_results/` directory, showing predictions side-by-side with original and heatmap outputs.

---

## Batch Processing

Pre-made batch files are included for easy processing. Just double-click to run or execute from command prompt.

### Available Batch Files

| Batch File | Purpose |
|------------|---------|
| `process_all.bat` | Complete pipeline for all 6 folders (test/train/val x 0/2) |
| `expand_all_masks.bat` | Create S/M/L/XL mask expansions for all results |
| `process_single_folder.bat` | Process a custom folder with arguments |
| `segment_only.bat` | Quick mask generation only |
| `apply_expanded_blackout.bat` | Create blackout images using expanded masks |
| `expand_horizontal_all.bat` | Batch script for horizontal mask expansions |
| `apply_horizontal_blackout_all.bat`| Batch script for horizontal blackouts |

### Complete Pipeline for All Folders

Process all datasets (test/train/val with subsets 0 and 2):

```batch
process_all.bat
```

This will:
1. Generate segmentation masks for all 6 folders
2. Create blackout images (joint space blacked out)
3. Create left/right split versions

**Output directories created:**
- `results_test_0`, `results_test_2`, etc. (masks)
- `blackedout_test_0`, `blackedout_test_2`, etc. (blackout images)
- `left_masked_test_0`, `right_masked_test_0`, etc. (split images)

### Batch Processing with Mask Expansion

After running `process_all.bat`, expand masks to different sizes:

```batch
expand_all_masks.bat
```

Creates 4 expansion levels for each mask directory:
- `*_expanded_S` - Small (kernel=3, iterations=1)
- `*_expanded_M` - Medium (kernel=5, iterations=2)
- `*_expanded_L` - Large (kernel=7, iterations=3)
- `*_expanded_XL` - Extra Large (kernel=10, iterations=5)

Then apply blackout with expanded masks:

```batch
apply_expanded_blackout.bat
```

### Process Single Folder

For custom folders or testing:

```batch
process_single_folder.bat "C:\path\to\images" "my_prefix"
```

Creates: `results_my_prefix`, `blackedout_my_prefix`, `left_masked_my_prefix`, `right_masked_my_prefix`

### Quick Segmentation Only

Generate masks without blackout/split processing:

```batch
segment_only.bat "C:\path\to\images" "output_masks"
```

---

## Directory Structure

After running the complete pipeline:

```
Knee_Segmentation/
├── MODEL_unet.pth                 # Trained U-Net model
│
├── segment_simple.py              # Script 1: Mask generation
├── expand_masks.py                # Script 2: Mask expansion (optional)
├── apply_mask_blackout.py         # Script 3: Full blackout
├── apply_mask_split.py            # Script 4: Split blackout
│
├── process_all.bat                # Batch: Process all folders
├── expand_all_masks.bat           # Batch: Expand all masks
├── process_single_folder.bat      # Batch: Process single folder
├── segment_only.bat               # Batch: Segmentation only
├── apply_expanded_blackout.bat    # Batch: Blackout with expanded masks
│
├── train.py                       # Legacy: Train models
├── predict.py                     # Legacy: Predict on DICOM files
├── predict_png.py                 # Legacy: Predict on full PNG X-rays
├── evaluate.py                    # Legacy: Model evaluation
│
├── README_UPDATED.md              # This comprehensive guide
├── USAGE_GUIDE.md                 # Detailed usage examples
├── RUNNING_INSTRUCTIONS.md        # Step-by-step guide
│
├── unet/                          # U-Net model architecture
│   ├── __init__.py
│   ├── unet_model.py
│   └── unet_parts.py
│
├── drn/                           # DRN model architecture
│   └── drn.py
│
├── utils/                         # Utilities
│   ├── __init__.py
│   ├── data_loading.py
│   ├── data_transforms.py
│   ├── metrics.py
│   └── utils.py
│
├── .venv_gpu/                     # GPU Virtual Environment (uv)
├── classification_datasets/       # Formatted datasets post-preparation
├── classification_results/        # Trained models, logs, & robustness CSVs
├── gradcam_results/               # Visual heatmaps & prediction overlays
├── comparison_figures/            # 2x3 visualization grids
│
├── results_test_0/                # Generated masks (Step 1)
│   ├── 9003175L_mask.png
│   ├── 9003175R_mask.png
│   └── ... (639 files)
│
├── results_test_0_expanded/       # Expanded masks (Step 2, optional)
│   ├── 9003175L_mask.png
│   ├── 9003175R_mask.png
│   └── ... (639 files)
│
├── blackedout_test_0/             # Full blackout (Step 3)
│   ├── 9003175L.png
│   ├── 9003175R.png
│   └── ... (639 files)
│
├── left_masked_test_0/            # Left-half masked (Step 4a)
│   ├── 9003175L.png
│   ├── 9003175R.png
│   └── ... (639 files)
│
└── right_masked_test_0/           # Right-half masked (Step 4b)
    ├── 9003175L.png
    ├── 9003175R.png
    └── ... (639 files)
```

---

## Image Specifications

| Property | Specification |
|----------|---------------|
| Input Format | PNG grayscale images |
| Expected Size | 224×224 pixels (pre-cropped knee regions) |
| Output Masks | 224×224 binary (0 or 255) |
| Blackout Value | 0 (pure black) |
| Processing | Preserves original dimensions |

---

## Legacy Scripts

The following scripts are from the original project and designed for full X-ray DICOMs:

### `train.py`
- Trains U-Net or DRN models
- Requires training_data folder with "data" and "target" subfolders
- Automatically creates validation split

### `predict.py`
- Original prediction script for DICOM files
- Supports full X-ray images with knee localization
- Not recommended for pre-cropped PNG images

**Usage:**
```bash
python predict.py -a unet -m MODEL_unet.pth -i <dicom_dir> -sv
```

**Arguments:**
- `-a, --architecture`: Model architecture (unet or drn)
- `-m, --model`: Model file path
- `-i, --input-dir`: Directory with DICOM images
- `-sv, --save`: Save processed images
- `-s, --scale`: Downscaling factor (default: 1)
- `-t, --mask-threshold`: Probability threshold (default: 0.5)

### `predict_png.py`
- Modified for PNG files
- Expects full X-ray images (not pre-cropped)
- Includes knee localization step

### `knee_localizer.py`
- Utilities for finding knee regions in full X-rays
- Not needed for pre-cropped 224×224 images

**Note:** For pre-cropped knee images, use `segment_simple.py` instead of the legacy scripts.

---

## Model Information

### U-Net Architecture
- **Input Channels**: 1 (grayscale)
- **Output Classes**: 1 (binary segmentation)
- **Target**: Joint space regions in knee X-rays
- **Training Data**: 5570 histogram-equalized radiographs from OAI dataset
- **Performance**: IOU score of 0.93 on MOST dataset

### Model File
- **Filename**: `MODEL_unet.pth`
- **Size**: ~62 MB
- **Framework**: PyTorch
- **Required for**: All three processing scripts

---

## Troubleshooting

### Common Issues

#### 1. "Mask not found" error
**Problem:** `apply_mask_blackout.py` or `apply_mask_split.py` can't find mask files

**Solution:**
- Ensure you've run `segment_simple.py` first to generate masks
- Check that mask directory path is correct (use absolute paths)
- Verify mask files exist and are named `{original}_mask.png`

```bash
# Check if masks exist
ls out_test_0/*.png | measure
```

---

#### 2. OpenCV DLL errors (Windows)
**Problem:** `ImportError: DLL load failed while importing cv2`

**Solution:**
```bash
# Remove conda opencv
conda remove opencv

# Install specific pip version
pip install opencv-python==4.10.0.84
```

---

#### 3. Out of memory errors
**Problem:** Processing crashes with memory error

**Solution:**
- Process folders individually instead of batch
- Close other applications to free RAM
- Reduce batch size if modifying scripts

---

#### 4. Conda not recognized in PowerShell
**Problem:** `conda: The term 'conda' is not recognized`

**Solution:** Use direct Python path:
```powershell
& "C:\Users\<username>\miniconda3\envs\knee-segmentation\python.exe" segment_simple.py -i "..." -m "..." -o "..."
```

---

#### 5. Model file not found
**Problem:** `FileNotFoundError: MODEL_unet.pth`

**Solution:**
- Ensure `MODEL_unet.pth` is in the current directory
- Or specify full path: `-m "C:\full\path\to\MODEL_unet.pth"`

---

### Performance Notes

| Metric | Value |
|--------|-------|
| Processing time | ~0.1-0.3 seconds per image |
| 639 images | 1-3 minutes total |
| GPU acceleration | Available with CUDA-enabled PyTorch |
| Memory usage | ~2-4 GB RAM |

---

## Additional Documentation

- **USAGE_GUIDE.md** - Detailed usage examples and common scenarios
- **RUNNING_INSTRUCTIONS.md** - Step-by-step execution guide with screenshots
- See these files for more comprehensive information

---

## Citation

If you use this code in your research, please cite the framework built upon for semantic processing:

```bibtex
@misc{Knee_Segmentation_2025,
  title={Knee Segmentation & Image Processing Pipeline for X-Ray Diagnostics},
  author={AI Hub Keski-Suomi},
  year={2025},
  url={https://github.com/AI-hub-keskisuomi/AI_hub_keskisuomi}
}
```

And acknowledge the **AI Hub Keski-Suomi** project.

---

## License

This project is licensed under the [MIT License](LICENSE) - see the LICENSE file for details.

---

## Contact

For questions or issues:
- **GitHub Repository**: [AI_hub_keskisuomi](https://github.com/AI-hub-keskisuomi/AI_hub_keskisuomi)
- **Project Path**: `WP3_knee_osteoarthritis/Knee_Segmentation`
- **Project**: AI Hub Keski-Suomi - WP3 Knee Osteoarthritis
- **Submit an Issue**: [GitHub Issues](https://github.com/AI-hub-keskisuomi/AI_hub_keskisuomi/issues)

---

## Version History

- **v3.0** (2026-03-31) - Massive overhaul including deep learning classification architecture (`run_all_experiments.py`), multi-seed statistical robustness testing (`run_robustness.py`), and visualization with XAI Grad-CAM output (`generate_gradcam.py`). Upgraded to Python 3.12 via `uv` with raw NVIDIA RTX GPU support natively integrated. Added dataset preparation pipelines and horizontal mask capabilities.
- **v2.1** (2025-10-23) - Added mask expansion functionality for data augmentation
- **v2.0** (2025-XX-XX) - Added processing pipeline (blackout and split scripts)
- **v1.0** (Original) - Initial segmentation model and training code
