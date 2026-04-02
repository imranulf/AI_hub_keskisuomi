# Knee Segmentation - Running Instructions

## Quick Reference

**For pre-cropped knee images (224×224):**
1. Use `segment_simple.py` to generate masks
2. Optionally use `expand_masks.py` to enlarge masks
3. Use `apply_mask_blackout.py` for full blackout
4. Use `apply_mask_split.py` for left/right versions

**Current workspace:** `.\Knee_Segmentation`

## Environment Setup

### 1. Activate Environment
You created an environment called `knee-segmentation`. To use it:

**Option A: From Anaconda Prompt**
```bash
conda activate knee-segmentation
```

**Option B: From PowerShell (if conda init was run)**
```powershell
conda activate knee-segmentation
```

**Option C: Run directly with full Python path (no activation needed)**
```powershell
& /path/to/-segmentation\python.exe <script.py>
```

### 2. Environment Contents
- Python 3.10
- PyTorch with CUDA support
- OpenCV (cv2)
- scikit-image
- pydicom
- pandas, numpy, scipy, matplotlib
- All other dependencies from environment_cuda.yml

## Running Predictions

### For Pre-Cropped PNG Images (Recommended)

Use `segment_simple.py` for PNG or JPG pre-cropped knee images:

```powershell
cd .\Knee_Segmentation

python segment_simple.py -m MODEL_unet.pth -i "C:\path\to\images" -o output_folder
```

#### Parameters:
- `-a unet` : Architecture (unet or drn, default: unet)
- `-m MODEL_unet.pth` : Path to trained model (required)
- `-i "C:\path\to\images"` : Input directory with PNG/JPG files OR single image file (required)
- `-o output_folder` : Output directory for masks (default: "out")
- `-nc 1` : Number of classes (1 for binary segmentation, default: 1)
- `-nch 1` : Number of input channels (1 for grayscale, default: 1)
- `-s 1.0` : Downscaling factor (1.0 = no downscaling, default: 1.0)
- `-t 0.5` : Mask threshold (0.0-1.0, default: 0.5)

#### Example for Your Test Data:
```powershell
python segment_simple.py -m MODEL_unet.pth -i "./data\test\data\0" -o results_test_0
```

### For Mask Expansion (Optional)

```powershell
# Medium expansion (default)
python expand_masks.py -i results_test_0 -o results_test_0_expanded

# Large expansion
python expand_masks.py -i results_test_0 -o results_test_0_expanded_L -k 7 -n 3
```

### For Applying Masks

```powershell
# Full blackout
python apply_mask_blackout.py -i "./data\test\data\0" -m results_test_0 -o blackedout_test_0

# Left/right split
python apply_mask_split.py -i "./data\test\data\0" -m results_test_0 -l left_masked_test_0 -r right_masked_test_0
```

### For DICOM Files (Original Setup)

Use `predict.py` for DICOM knee x-ray images:

```powershell
python predict.py -a unet -m MODEL_unet.pth -i "C:\path\to\dicom\files" -sv -nc 1
```

Note: DICOM files contain pixel spacing metadata, so no `-ps` parameter is needed.

## Output Files

### 1. Segmentation Masks
- Location: `Knee_Segmentation/<output_folder>/`
- Format: `{filename}_mask.png`
- Content: Binary segmentation masks (white = joint space, black = background)

### 2. Expanded Masks (if expansion step used)
- Location: `Knee_Segmentation/<output_folder>_expanded/`
- Format: `{filename}_mask.png`
- Content: Enlarged segmentation masks

### 3. Blackout Images (if blackout applied)
- Location: `Knee_Segmentation/<blackout_folder>/`
- Format: `{filename}.png`
- Content: Original images with masked regions set to black

### 4. Split Images (if split applied)
- Location: `Knee_Segmentation/<left_folder>/` and `<right_folder>/`
- Format: `{filename}.png`
- Content: Images with left or right half of mask applied

## Important Notes

### Image Requirements
- **Pre-cropped knee images**: Works with any size, optimized for 224×224
- **Format**: PNG or JPG grayscale images
- **Orientation**: Should show single knee region
- **No DICOM metadata required**: Works with standard image files

### Mask Expansion Levels
Choose expansion level based on your needs:
- **No expansion**: Original joint space only
- **Small (-k 3 -n 1)**: Minimal expansion, ~2-3 pixel border
- **Medium (-k 5 -n 2)**: Default, ~5-7 pixel border
- **Large (-k 7 -n 3)**: Significant expansion, ~10-12 pixel border
- **XL (-k 10 -n 5)**: Maximum expansion, ~20+ pixel border

### Error Handling
- The scripts process all images and skip ones that fail
- Common errors:
  - "Could not read image": Invalid image format or corrupted file
  - "Mask not found": Run segmentation step first
  - "Shape mismatch": Masks will be resized to match images automatically
- Failed images are logged but don't stop the batch process

### Performance
- **CPU mode**: ~0.1-0.3 seconds per image for segmentation
- **GPU mode** (if CUDA available): Much faster
- **639 images**: 1-3 minutes for segmentation on CPU
- **Check CUDA**: `python -c "import torch; print(torch.cuda.is_available())"`

## Troubleshooting

### "ModuleNotFoundError"
```powershell
pip install <missing_package>
```

### "DLL load failed"
Reinstall the problematic package:
```powershell
pip uninstall <package>
pip install <package>
```

### Conda not recognized in PowerShell
Initialize conda for PowerShell (one-time):
```powershell
# From Anaconda Prompt:
conda init powershell
# Then restart PowerShell
```

Or always use the full Python path (Option C above).

### Poor Segmentation Results
- Check that images are properly oriented knee x-rays
- Adjust `-t` threshold (try 0.3-0.7)
- Verify MODEL_unet.pth is the correct trained model
- Consider retraining the model on your specific dataset

### Masks Don't Align with Images
- Masks are automatically resized to match image dimensions
- Check that mask files have correct naming (`{original}_mask.png`)
- Ensure mask directory path is correct

### Want Larger/Smaller Masked Regions
- Use `expand_masks.py` to enlarge masks
- Adjust `-k` (kernel size) and `-n` (iterations) parameters
- Test different levels: S, M, L, XL

## Next Steps

1. **Validate Results**: Open some output masks to verify segmentation quality
2. **Test Expansion**: Try different expansion levels to find optimal size
3. **Process Other Folders**: Run on all your data folders (test, train, val)
4. **Batch Processing**: Use the pre-made batch files for automation
5. **Analysis**: Use the generated images for your osteoarthritis research

## Batch Files (Ready to Use)

Pre-made batch files are included in the project. Just double-click to run!

### Available Batch Files

| File | Description |
|------|-------------|
| `process_all.bat` | Process all 6 folders through complete pipeline |
| `expand_all_masks.bat` | Create S/M/L/XL expansion for all masks |
| `process_single_folder.bat` | Process a custom folder |
| `segment_only.bat` | Quick mask generation only |
| `apply_expanded_blackout.bat` | Blackout with expanded masks |

### Complete Pipeline - All Folders

Double-click `process_all.bat` or run from command prompt:

```batch
process_all.bat
```

This processes all 6 data folders (test_0, test_2, train_0, train_2, val_0, val_2) through:
1. Segmentation (generates masks)
2. Blackout (removes joint space)
3. Left/right split (bilateral analysis)

### Expand All Masks

After running `process_all.bat`, expand masks with:

```batch
expand_all_masks.bat
```

Creates 4 expansion sizes (S, M, L, XL) for all mask directories.

### Process Single Folder

For custom folders:

```batch
process_single_folder.bat "C:\path\to\images" "output_prefix"
```

### Segmentation Only

Quick mask generation:

```batch
segment_only.bat "C:\path\to\images" "output_dir"
```

### Apply Expanded Blackout

Create blackout images using expanded masks:

```batch
apply_expanded_blackout.bat
```
