# Knee Segmentation - Complete Guide

## 🎯 Summary

Your images are **224×224 pre-cropped knee regions**, not full X-rays. This required a different approach than the original code expected.

## ✅ Solution

Created `segment_simple.py` - a streamlined script that:
- ✓ Works with pre-cropped knee images (any size)
- ✓ Performs U-Net segmentation directly
- ✓ Saves binary segmentation masks
- ✓ Fast and reliable (no complex preprocessing)

## 📁 Output Locations

After running segmentation, your files are located at:

```
C:\Users\imran\AI_hub_keskisuomi\WP3_knee_osteoarthritis\Knee_Segmentation\
├── out_test/              ← Segmentation masks directory
│   ├── 9003175L_mask.png  ← Binary masks (white = joint space)
│   ├── 9003175R_mask.png
│   └── ... (one per image)
```

## 🚀 How to Use

### Basic Segmentation (Recommended)

```powershell
cd C:\Users\imran\AI_hub_keskisuomi\WP3_knee_osteoarthritis\Knee_Segmentation

# Process a directory of images
python segment_simple.py -a unet -m MODEL_unet.pth -i "C:\path\to\images" -o output_folder

# Process a single image
python segment_simple.py -a unet -m MODEL_unet.pth -i "C:\path\to\image.png" -o output_folder
```

### Options

- `-a unet` : Architecture (unet or drn)
- `-m MODEL_unet.pth` : Your trained model
- `-i <path>` : Input directory or single file
- `-o <folder>` : Output directory for masks (default: "out")
- `-t 0.5` : Threshold (0-1, default: 0.5)
- `-s 1.0` : Scale factor (default: 1.0)

### Examples for Your Data

```powershell
# Test data folder 0
python segment_simple.py -a unet -m MODEL_unet.pth -i "C:\Users\imran\AI_hub_keskisuomi\WP3_knee_osteoarthritis\data\test\data\0" -o results_test_0

# Test data folder 2
python segment_simple.py -a unet -m MODEL_unet.pth -i "C:\Users\imran\AI_hub_keskisuomi\WP3_knee_osteoarthritis\data\test\data\2" -o results_test_2

# Train data folder 0
python segment_simple.py -a unet -m MODEL_unet.pth -i "C:\Users\imran\AI_hub_keskisuomi\WP3_knee_osteoarthritis\data\train\data\0" -o results_train_0

# Validation data folder 0
python segment_simple.py -a unet -m MODEL_unet.pth -i "C:\Users\imran\AI_hub_keskisuomi\WP3_knee_osteoarthritis\data\val\data\0" -o results_val_0
```

## 📊 Understanding the Output

Each output mask:
- **White pixels (255)**: Predicted joint space region
- **Black pixels (0)**: Background
- **Filename**: `{original_name}_mask.png`
- **Size**: Same as input image (224×224)

The script prints the percentage of image segmented for each file, e.g.:
```
1/639: 9003175L.png ✓ 4.8% segmented
```

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
2. **Batch Process**: Run on all your data folders
3. **Analysis**: Use masks for your osteoarthritis analysis
4. **OA Variables**: If you need joint space width, eminentia measurements, etc., those would require:
   - Proper pixel spacing calibration
   - Higher resolution images
   - Modified variable calculation code for pre-cropped images

## 🆚 Script Comparison

| Script | Use Case | Your Images |
|--------|----------|-------------|
| `predict.py` | Full DICOM X-rays | ❌ No |
| `predict_png.py` | Large full PNG X-rays | ❌ No |
| `predict_precropped.py` | Pre-cropped with OA vars | ⚠️ Hangs |
| `segment_simple.py` | Pre-cropped, masks only | ✅ Works! |

## 💡 Tips

- **Organize output**: Use descriptive output folder names
- **Keep logs**: Redirect output to file: `... > log.txt 2>&1`
- **Sample first**: Test on small subset before processing all images
- **Check CUDA**: For GPU acceleration, verify: `python -c "import torch; print(torch.cuda.is_available())"`

## 📞 Support

If you encounter issues:
1. Check error messages carefully
2. Verify input paths exist
3. Ensure environment has all packages
4. Try on a single image first to isolate problems
