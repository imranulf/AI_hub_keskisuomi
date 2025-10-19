# Knee Segmentation & Image Processing Pipeline


## Overview

This project provides a complete pipeline for knee X-ray image segmentation and processing using a trained U-Net model. The pipeline includes three main processing steps:

1. **Mask Generation** - Segment joint space regions from knee X-rays
2. **Full Blackout** - Replace entire segmented regions with black pixels  
3. **Split Blackout** - Create left-half and right-half masked versions

### Original Project Information
- Forks from https://github.com/fyu/drn and https://github.com/milesial/Pytorch-UNet
- A tool for segmenting the joint space of the knee automatically
- UNET and DRN architectures trained with histogram-equalized 5570 radiographs from OAI-dataset
- Tested on MOST dataset: UNet achieves IOU-score of 0.93, DRN achieves 0.90

## Table of Contents
- [Setup](#setup)
- [Quick Start](#quick-start)
- [Pipeline Components](#pipeline-components)
- [Batch Processing](#batch-processing)
- [Directory Structure](#directory-structure)
- [Legacy Scripts](#legacy-scripts)
- [Troubleshooting](#troubleshooting)

---

## Setup

### Prerequisites
- Python 3.10+
- Conda or Miniconda (recommended)
- Windows, Linux, or macOS

### Environment Setup

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

# 2. Create full blackout version (entire gap darkened)
python apply_mask_blackout.py -i "C:\path\to\images" -m "masks_out" -o "blackedout"

# 3. Create left/right split versions
python apply_mask_split.py -i "C:\path\to\images" -m "masks_out" -l "left_masked" -r "right_masked"
```

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

### 2. Full Blackout (`apply_mask_blackout.py`)

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

#### Example
```bash
python apply_mask_blackout.py -i "data/test/data/0" -m "out_test_0" -o "blackedout_test_0"
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

### 3. Split Blackout (`apply_mask_split.py`)

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

## Batch Processing

### Complete Pipeline for All Folders

For datasets organized as `test/data/0`, `test/data/2`, `train/data/0`, `train/data/2`, `val/data/0`, `val/data/2`:

**Save this as `process_all.ps1`:**

```powershell
# Complete pipeline batch processing script
$folders = @(
    @{input="C:\Users\imran\AI_hub_keskisuomi\WP3_knee_osteoarthritis\data\test\data\0"; 
      mask="out_test_0"; blackout="blackedout_test_0"; 
      left="left_masked_test_0"; right="right_masked_test_0"},
    @{input="C:\Users\imran\AI_hub_keskisuomi\WP3_knee_osteoarthritis\data\test\data\2"; 
      mask="out_test_2"; blackout="blackedout_test_2"; 
      left="left_masked_test_2"; right="right_masked_test_2"},
    @{input="C:\Users\imran\AI_hub_keskisuomi\WP3_knee_osteoarthritis\data\train\data\0"; 
      mask="out_train_0"; blackout="blackedout_train_0"; 
      left="left_masked_train_0"; right="right_masked_train_0"},
    @{input="C:\Users\imran\AI_hub_keskisuomi\WP3_knee_osteoarthritis\data\train\data\2"; 
      mask="out_train_2"; blackout="blackedout_train_2"; 
      left="left_masked_train_2"; right="right_masked_train_2"},
    @{input="C:\Users\imran\AI_hub_keskisuomi\WP3_knee_osteoarthritis\data\val\data\0"; 
      mask="out_val_0"; blackout="blackedout_val_0"; 
      left="left_masked_val_0"; right="right_masked_val_0"},
    @{input="C:\Users\imran\AI_hub_keskisuomi\WP3_knee_osteoarthritis\data\val\data\2"; 
      mask="out_val_2"; blackout="blackedout_val_2"; 
      left="left_masked_val_2"; right="right_masked_val_2"}
)

foreach ($f in $folders) {
    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host "Processing: $($f.input)" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    
    # Step 1: Generate masks
    Write-Host "`n[1/3] Generating masks..." -ForegroundColor Yellow
    python segment_simple.py -i $f.input -m "MODEL_unet.pth" -o $f.mask
    
    # Step 2: Create full blackout
    Write-Host "`n[2/3] Creating full blackout..." -ForegroundColor Yellow
    python apply_mask_blackout.py -i $f.input -m $f.mask -o $f.blackout
    
    # Step 3: Create split versions
    Write-Host "`n[3/3] Creating left/right split versions..." -ForegroundColor Yellow
    python apply_mask_split.py -i $f.input -m $f.mask -l $f.left -r $f.right
    
    Write-Host "`n✓ Completed: $($f.input)" -ForegroundColor Green
}

Write-Host "`n========================================" -ForegroundColor Green
Write-Host "ALL FOLDERS PROCESSED SUCCESSFULLY" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
```

**Run with:**
```powershell
.\process_all.ps1
```

---

## Directory Structure

After running the complete pipeline:

```
Knee_Segmentation/
├── MODEL_unet.pth                 # Trained U-Net model
├── segment_simple.py              # Script 1: Mask generation
├── apply_mask_blackout.py         # Script 2: Full blackout
├── apply_mask_split.py            # Script 3: Split blackout
├── README.md                      # This file
├── USAGE_GUIDE.md                 # Detailed usage examples
├── RUNNING_INSTRUCTIONS.md        # Step-by-step guide
│
├── out_test_0/                    # Generated masks (Step 1)
│   ├── 9003175L_mask.png
│   ├── 9003175R_mask.png
│   └── ... (639 files)
│
├── blackedout_test_0/             # Full blackout (Step 2)
│   ├── 9003175L.png
│   ├── 9003175R.png
│   └── ... (639 files)
│
├── left_masked_test_0/            # Left-half masked (Step 3a)
│   ├── 9003175L.png
│   ├── 9003175R.png
│   └── ... (639 files)
│
└── right_masked_test_0/           # Right-half masked (Step 3b)
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

If you use this code in your research, please cite:

```
[Add citation information here]
```

And acknowledge the AI Hub Keski-Suomi project.

---

## License

[Add license information here]

---

## Contact

For questions or issues:
- GitHub Issues: [Repository URL]
- Email: [Contact email]
- Project: AI Hub Keski-Suomi - WP3 Knee Osteoarthritis

---

## Version History

- **v2.0** (2025) - Added processing pipeline (blackout and split scripts)
- **v1.0** (Original) - Initial segmentation model and training code
