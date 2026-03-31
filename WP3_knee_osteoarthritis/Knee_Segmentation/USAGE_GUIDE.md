# Knee Segmentation - Complete Usage Guide

## 🎯 Summary

Your images are **224×224 pre-cropped knee regions**, not full X-rays. This project provides a complete pipeline for:
- Segmenting joint space regions
- Expanding masks to cover more bone area (optional)
- Creating blackout versions for data augmentation
- Splitting masks left/right for analysis

## ✅ Available Scripts

| Script | Purpose | Input | Output |
|--------|---------|-------|--------|
| `segment_simple.py` | Generate segmentation masks | Original images | Binary masks |
| `expand_masks.py` | Enlarge masks | Masks | Expanded masks |
| `apply_mask_blackout.py` | Black out masked regions | Images + Masks | Blackout images |
| `apply_mask_split.py` | Create left/right versions | Images + Masks | L/R masked images |

## 📁 Output Locations

After running segmentation and processing, your files are organized as:

```
C:\Users\imran\AI_hub_keskisuomi\WP3_knee_osteoarthritis\Knee_Segmentation\
│
├── results_test_0/              ← Segmentation masks (Step 1)
│   ├── 9003175L_mask.png        ← Binary masks (white = joint space)
│   ├── 9003175R_mask.png
│   └── ... (one per image)
│
├── results_test_0_expanded/     ← Expanded masks (Step 2, optional)
│   ├── 9003175L_mask.png        ← Enlarged masks
│   └── ...
│
├── blackedout_test_0/           ← Full blackout (Step 3)
│   ├── 9003175L.png             ← Original with masked area black
│   └── ...
│
├── left_masked_test_0/          ← Left-half masked (Step 4a)
│   ├── 9003175L.png             ← Left side darkened
│   └── ...
│
└── right_masked_test_0/         ← Right-half masked (Step 4b)
    ├── 9003175L.png             ← Right side darkened
    └── ...
```

## 🚀 How to Use

### 1. Basic Segmentation

```powershell
cd C:\Users\imran\AI_hub_keskisuomi\WP3_knee_osteoarthritis\Knee_Segmentation

# Process a directory of images
python segment_simple.py -m MODEL_unet.pth -i "C:\path\to\images" -o output_folder

# Process a single image
python segment_simple.py -m MODEL_unet.pth -i "C:\path\to\image.png" -o output_folder
```

**Options:**
- `-a unet` : Architecture (unet or drn, default: unet)
- `-m MODEL_unet.pth` : Your trained model (required)
- `-i <path>` : Input directory or single file (required)
- `-o <folder>` : Output directory for masks (default: "out")
- `-t 0.5` : Threshold (0-1, default: 0.5)
- `-s 1.0` : Scale factor (default: 1.0)

### 2. Expand Masks (Optional)

```powershell
# Default expansion (Medium)
python expand_masks.py -i "results_test_0" -o "results_test_0_expanded"

# Small expansion
python expand_masks.py -i "results_test_0" -o "results_test_0_expanded_S" -k 3 -n 1

# Large expansion
python expand_masks.py -i "results_test_0" -o "results_test_0_expanded_L" -k 7 -n 3

# Extra large expansion
python expand_masks.py -i "results_test_0" -o "results_test_0_expanded_XL" -k 10 -n 5
```

**Options:**
- `-i <dir>` : Input mask directory (required)
- `-o <dir>` : Output directory (required)
- `-k <size>` : Kernel size (default: 5, larger = more expansion)
- `-n <num>` : Iterations (default: 2, more = larger expansion)

### 3. Apply Full Blackout

```powershell
# Using regular masks
python apply_mask_blackout.py -i "C:\path\to\images" -m "results_test_0" -o "blackedout_test_0"

# Using expanded masks
python apply_mask_blackout.py -i "C:\path\to\images" -m "results_test_0_expanded_L" -o "blackedout_test_0_expanded_L"
```

**Options:**
- `-i <dir>` : Original images directory (required)
- `-m <dir>` : Mask directory (required)
- `-o <dir>` : Output directory (required)
- `-s <suffix>` : Mask filename suffix (default: "_mask")

### 4. Create Left/Right Split Versions

```powershell
# Using regular masks
python apply_mask_split.py -i "C:\path\to\images" -m "results_test_0" -l "left_masked_test_0" -r "right_masked_test_0"

# Using expanded masks
python apply_mask_split.py -i "C:\path\to\images" -m "results_test_0_expanded" -l "left_masked_test_0_exp" -r "right_masked_test_0_exp"
```

**Options:**
- `-i <dir>` : Original images directory (required)
- `-m <dir>` : Mask directory (required)
- `-l <dir>` : Output directory for left-masked images (required)
- `-r <dir>` : Output directory for right-masked images (required)

### Examples for Your Data

```powershell
# Test data folder 0 - Complete pipeline
python segment_simple.py -m MODEL_unet.pth -i "C:\Users\imran\AI_hub_keskisuomi\WP3_knee_osteoarthritis\data\test\data\0" -o results_test_0
python apply_mask_blackout.py -i "C:\Users\imran\AI_hub_keskisuomi\WP3_knee_osteoarthritis\data\test\data\0" -m results_test_0 -o blackedout_test_0
python apply_mask_split.py -i "C:\Users\imran\AI_hub_keskisuomi\WP3_knee_osteoarthritis\data\test\data\0" -m results_test_0 -l left_masked_test_0 -r right_masked_test_0

# Test data folder 2 - Complete pipeline
python segment_simple.py -m MODEL_unet.pth -i "C:\Users\imran\AI_hub_keskisuomi\WP3_knee_osteoarthritis\data\test\data\2" -o results_test_2
python apply_mask_blackout.py -i "C:\Users\imran\AI_hub_keskisuomi\WP3_knee_osteoarthritis\data\test\data\2" -m results_test_2 -o blackedout_test_2
python apply_mask_split.py -i "C:\Users\imran\AI_hub_keskisuomi\WP3_knee_osteoarthritis\data\test\data\2" -m results_test_2 -l left_masked_test_2 -r right_masked_test_2

# Train data folder 0 - Complete pipeline
python segment_simple.py -m MODEL_unet.pth -i "C:\Users\imran\AI_hub_keskisuomi\WP3_knee_osteoarthritis\data\train\data\0" -o results_train_0
python apply_mask_blackout.py -i "C:\Users\imran\AI_hub_keskisuomi\WP3_knee_osteoarthritis\data\train\data\0" -m results_train_0 -o blackedout_train_0
python apply_mask_split.py -i "C:\Users\imran\AI_hub_keskisuomi\WP3_knee_osteoarthritis\data\train\data\0" -m results_train_0 -l left_masked_train_0 -r right_masked_train_0

# Validation data folder 0 - Complete pipeline
python segment_simple.py -m MODEL_unet.pth -i "C:\Users\imran\AI_hub_keskisuomi\WP3_knee_osteoarthritis\data\val\data\0" -o results_val_0
python apply_mask_blackout.py -i "C:\Users\imran\AI_hub_keskisuomi\WP3_knee_osteoarthritis\data\val\data\0" -m results_val_0 -o blackedout_val_0
python apply_mask_split.py -i "C:\Users\imran\AI_hub_keskisuomi\WP3_knee_osteoarthritis\data\val\data\0" -m results_val_0 -l left_masked_val_0 -r right_masked_val_0
```

## 📊 Understanding the Output

### Segmentation Masks
Each output mask:
- **White pixels (255)**: Predicted joint space region
- **Black pixels (0)**: Background
- **Filename**: `{original_name}_mask.png`
- **Size**: Same as input image (224×224)

The script prints the percentage of image segmented for each file, e.g.:
```
1/639: 9003175L.png ✓ 4.8% segmented
```

### Expanded Masks
- Larger white regions covering more bone area
- Same format as original masks
- Can be used interchangeably in pipeline

### Blackout Images
- Original image with masked region set to black (0)
- Useful for removing joint space information
- Can test model robustness

### Split Images
- **Left-masked**: Left half of joint space darkened
- **Right-masked**: Right half of joint space darkened
- Useful for analyzing bilateral differences

## ⚠️ Why Other Scripts Failed

1. **predict.py** - Expected DICOM files with metadata
2. **predict_png.py** - Expected large full X-rays (1000+ pixels) to split left/right
3. Your images are:
   - Already cropped to single knees
   - Small (224×224)
   - PNG format without DICOM metadata
   
## 🔧 Troubleshooting

### "ModuleNotFoundError"
Make sure you're using the correct environment:
```powershell
& C:\Users\imran\miniconda3\envs\knee-segmentation\python.exe segment_simple.py ...
```

### Empty/Poor Segmentation
- Try adjusting threshold: `-t 0.3` or `-t 0.7`
- Check that MODEL_unet.pth is the correct trained model
- Verify images are grayscale knee X-rays

### Slow Processing
- CPU mode: ~1-2 seconds per image
- For 639 images: ~15-20 minutes total
- GPU would be much faster (install CUDA-compatible PyTorch)

## 📈 Next Steps

1. **Review Output**: Open some masks in an image viewer to check quality
2. **Batch Process**: Use `process_all.bat` to run on all data folders (just double-click!)
3. **Analysis**: Use masks for your osteoarthritis analysis
4. **OA Variables**: If you need joint space width, eminentia measurements, etc., those would require:
   - Proper pixel spacing calibration
   - Higher resolution images
   - Modified variable calculation code for pre-cropped images

## ⚠️ Script Comparison

| Script | Use Case | Your Images | Notes |
|--------|----------|-------------|-------|
| `predict.py` | Full DICOM X-rays | ❌ No | For full-size DICOM files |
| `predict_png.py` | Large full PNG X-rays | ❌ No | Expects 1000+ pixel images |
| `predict_precropped.py` | Pre-cropped with OA vars | ⚠️ Hangs | Calculates OA variables |
| `segment_simple.py` | Pre-cropped, masks only | ✅ Works! | **Use this** |
| `expand_masks.py` | Mask expansion | ✅ Works! | Optional step |
| `apply_mask_blackout.py` | Full blackout | ✅ Works! | For augmentation |
| `apply_mask_split.py` | Left/right split | ✅ Works! | For analysis |

## 💡 Tips

- **Use batch files**: Double-click `process_all.bat` for easy processing
- **Organize output**: Use descriptive output folder names
- **Keep logs**: Redirect output to file: `... > log.txt 2>&1`
- **Sample first**: Test on small subset before processing all images
- **Check CUDA**: For GPU acceleration, verify: `python -c "import torch; print(torch.cuda.is_available())"`

## 📦 Batch Files Available

| File | Description |
|------|-------------|
| `process_all.bat` | Process all 6 folders through complete pipeline |
| `expand_all_masks.bat` | Create S/M/L/XL expansion for all masks |
| `process_single_folder.bat` | Process a custom folder |
| `segment_only.bat` | Quick mask generation only |
| `apply_expanded_blackout.bat` | Blackout with expanded masks |

## 📞 Support

If you encounter issues:
1. Check error messages carefully
2. Verify input paths exist
3. Ensure environment has all packages
4. Try on a single image first to isolate problems
