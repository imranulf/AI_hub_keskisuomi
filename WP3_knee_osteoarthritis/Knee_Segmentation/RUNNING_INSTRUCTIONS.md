# Knee Segmentation - Running Instructions

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
& C:\Users\imran\miniconda3\envs\knee-segmentation\python.exe <script.py>
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

### For PNG/JPG Images (Your Current Setup)

Use `predict_png.py` for PNG or JPG knee x-ray images:

```powershell
cd C:\Users\imran\AI_hub_keskisuomi\WP3_knee_osteoarthritis\Knee_Segmentation

python predict_png.py -a unet -m MODEL_unet.pth -i "C:\path\to\images" -sv -nc 1 -ps 0.143 0.143
```

#### Parameters:
- `-a unet` : Architecture (unet or drn)
- `-m MODEL_unet.pth` : Path to trained model
- `-i "C:\path\to\images"` : Input directory with PNG/JPG files OR single image file
- `-sv` or `--save` : Save segmentation masks to `out/` folder
- `-nc 1` : Number of classes (1 for binary segmentation)
- `-s 1.0` : Downscaling factor (1.0 = no downscaling)
- `-t 0.5` : Mask threshold (0.0-1.0)
- `-ps 0.143 0.143` : Pixel spacing in mm/pixel [row col]. **IMPORTANT**: Adjust this based on your actual image resolution!

#### Example for Your Test Data:
```powershell
python predict_png.py -a unet -m MODEL_unet.pth -i "C:\Users\imran\AI_hub_keskisuomi\WP3_knee_osteoarthritis\data\test\data\0" -sv -nc 1 -ps 0.143 0.143
```

### For DICOM Files (Original Setup)

Use `predict.py` for DICOM knee x-ray images:

```powershell
python predict.py -a unet -m MODEL_unet.pth -i "C:\path\to\dicom\files" -sv -nc 1
```

Note: DICOM files contain pixel spacing metadata, so no `-ps` parameter is needed.

## Output Files

### 1. Segmentation Masks (if `-sv` flag used)
- Location: `Knee_Segmentation/out/`
- Format: `{filename}_l.png` and `{filename}_r.png`
- Content: Binary segmentation masks (white = joint space, black = background)

### 2. OA Variables CSV
- Location: `Knee_Segmentation/oa_variables.csv`
- Content: Calculated osteoarthritis variables for each knee:
  - `name`: Image filename
  - `side`: "l" (left) or "r" (right)
  - `l_em_height`, `m_em_height`: Lateral and medial eminentia heights (mm)
  - `l_em_angle`, `m_em_angle`: Lateral and medial eminentia angles (degrees)
  - `l_min_jsw`, `m_min_jsw`: Lateral and medial minimum joint space width (mm)
  - `l_avg_jsw`, `m_avg_jsw`: Lateral and medial average joint space width (mm)
  - `tibia_model`: Linear regression coefficients for tibial plane

## Important Notes

### Pixel Spacing
The default pixel spacing is set to `0.143 mm/pixel` (approximately 0.177 DPI or 7 pixels/mm). This is critical for accurate physical measurements. **You must adjust this value** based on your actual image resolution:

- If your images have different resolution, calculate: `pixel_spacing = physical_size_mm / image_size_pixels`
- Common values:
  - High-res medical: 0.1-0.15 mm/pixel
  - Standard medical: 0.15-0.2 mm/pixel
  - Lower res: 0.2-0.3 mm/pixel

### Error Handling
- The script processes all images and skips ones that fail
- Common errors:
  - "float division by zero": Poor segmentation or unusual image content
  - "Could not read image": Invalid image format or corrupted file
- Failed images are logged but don't stop the batch process

### Performance
- CPU mode: ~2-5 seconds per image
- GPU mode (if CUDA available): ~0.5-1 second per image
- Check CUDA availability: `python -c "import torch; print(torch.cuda.is_available())"`

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
- Check pixel spacing value matches your images
- Consider retraining the model on your specific dataset

## Next Steps

1. **Validate Results**: Open some output masks in `out/` folder and verify segmentation quality
2. **Check CSV**: Open `oa_variables.csv` to see calculated measurements
3. **Adjust Pixel Spacing**: If you know the true physical dimensions, update the `-ps` parameter
4. **Process Other Folders**: Run on `data/test/data/2`, `data/train/data/0`, etc.
5. **Batch Processing**: Create a script to process all folders automatically

## Example Batch Script

Create `run_all.ps1`:
```powershell
$env_python = "C:\Users\imran\miniconda3\envs\knee-segmentation\python.exe"
$folders = @(
    "C:\Users\imran\AI_hub_keskisuomi\WP3_knee_osteoarthritis\data\test\data\0",
    "C:\Users\imran\AI_hub_keskisuomi\WP3_knee_osteoarthritis\data\test\data\2",
    "C:\Users\imran\AI_hub_keskisuomi\WP3_knee_osteoarthritis\data\train\data\0",
    "C:\Users\imran\AI_hub_keskisuomi\WP3_knee_osteoarthritis\data\train\data\2",
    "C:\Users\imran\AI_hub_keskisuomi\WP3_knee_osteoarthritis\data\val\data\0",
    "C:\Users\imran\AI_hub_keskisuomi\WP3_knee_osteoarthritis\data\val\data\2"
)

foreach ($folder in $folders) {
    $name = Split-Path $folder -Leaf
    Write-Host "Processing folder: $folder"
    & $env_python predict_png.py -a unet -m MODEL_unet.pth -i $folder -sv -nc 1 -ps 0.143 0.143
    Move-Item oa_variables.csv "oa_variables_$name.csv"
}
```

Run it:
```powershell
.\run_all.ps1
```
