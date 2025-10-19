# Running Instructions - Knee Segmentation Pipeline

Complete step-by-step guide for processing knee X-ray images through the segmentation and masking pipeline.

## Table of Contents
1. [Prerequisites Check](#1-prerequisites-check)
2. [Environment Activation](#2-environment-activation)
3. [Pipeline Execution](#3-pipeline-execution)
4. [Verification Steps](#4-verification-steps)
5. [Common Workflows](#5-common-workflows)
6. [Troubleshooting During Execution](#6-troubleshooting-during-execution)

---

## 1. Prerequisites Check

### Before Starting

Ensure you have:
- ✅ Python 3.10+ installed (via Conda/Miniconda)
- ✅ Environment created: `knee-segmentation`
- ✅ All dependencies installed
- ✅ Model file: `MODEL_unet.pth` in the Knee_Segmentation directory
- ✅ Input images organized in directories

### Verify Installation

Open PowerShell and run:

```powershell
# Check conda installation
conda --version

# Check Python environment
conda env list

# Expected output should show 'knee-segmentation' environment
```

### Verify Model File

```powershell
# Navigate to Knee_Segmentation directory
cd C:\Users\imran\AI_hub_keskisuomi\WP3_knee_osteoarthritis\Knee_Segmentation

# Check if model exists
Test-Path MODEL_unet.pth
# Should return: True
```

### Verify Input Images

```powershell
# Check if images exist (example for test/data/0)
Get-ChildItem "C:\Users\imran\AI_hub_keskisuomi\WP3_knee_osteoarthritis\data\test\data\0\*.png" | Measure-Object

# Should show count of PNG files
```

---

## 2. Environment Activation

### Option A: Standard Conda Activation

```powershell
# Activate environment
conda activate knee-segmentation

# Verify activation (prompt should show environment name)
# Example: (knee-segmentation) PS C:\...>

# Verify Python
python --version
# Should show: Python 3.10.x
```

### Option B: Direct Python Path (If conda not recognized)

```powershell
# Use direct path to environment Python
$PYTHON = "C:\Users\imran\miniconda3\envs\knee-segmentation\python.exe"

# Verify it works
& $PYTHON --version

# Use $PYTHON instead of 'python' in all commands
# Example: & $PYTHON segment_simple.py -i "..." -m "..." -o "..."
```

---

## 3. Pipeline Execution

### Workflow Overview

```
Input Images → [Step 1] → Masks → [Step 2] → Full Blackout
                      ↓
                  [Step 3] → Left/Right Split
```

---

### STEP 1: Generate Segmentation Masks

**Purpose:** Create binary masks identifying joint space regions

**Command Template:**
```powershell
python segment_simple.py -i <INPUT_DIR> -m MODEL_unet.pth -o <OUTPUT_DIR>
```

**Example - Process test/data/0:**
```powershell
cd C:\Users\imran\AI_hub_keskisuomi\WP3_knee_osteoarthritis\Knee_Segmentation

python segment_simple.py `
    -i "C:\Users\imran\AI_hub_keskisuomi\WP3_knee_osteoarthritis\data\test\data\0" `
    -m "MODEL_unet.pth" `
    -o "out_test_0"
```

**Expected Output:**
```
Found 639 images in C:\Users\imran\...\data\test\data\0
Model loaded from MODEL_unet.pth
Processing: 100%|████████████████████| 639/639 [02:15<00:00, 4.73it/s]
Predictions saved to out_test_0/
```

**What Happens:**
- Reads all PNG images from input directory
- Loads U-Net model
- Generates binary masks for each image
- Saves masks as `{filename}_mask.png`

**Time Required:** ~1-3 minutes for 639 images

---

### STEP 2: Create Full Blackout Images

**Purpose:** Replace entire segmented region with black pixels

**Command Template:**
```powershell
python apply_mask_blackout.py -i <INPUT_DIR> -m <MASK_DIR> -o <OUTPUT_DIR>
```

**Example - Process test/data/0:**
```powershell
python apply_mask_blackout.py `
    -i "C:\Users\imran\AI_hub_keskisuomi\WP3_knee_osteoarthritis\data\test\data\0" `
    -m "out_test_0" `
    -o "blackedout_test_0"
```

**Expected Output:**
```
Original images: 639
Mask directory: out_test_0
Output directory: blackedout_test_0

1/639: 9003175L.png ✓
2/639: 9003175R.png ✓
3/639: 9003316L.png ✓
...
639/639: 9998089L.png ✓

==================================================
✓ Success: 639/639
✗ Failed: 0/639
✓ Output saved to: blackedout_test_0/
==================================================
```

**What Happens:**
- Reads original images and corresponding masks
- Sets all white pixels in mask to black in original
- Saves modified images with same filenames

**Time Required:** ~30-60 seconds for 639 images

---

### STEP 3: Create Left/Right Split Versions

**Purpose:** Create separate images with left-half and right-half masks applied

**Command Template:**
```powershell
python apply_mask_split.py -i <INPUT_DIR> -m <MASK_DIR> -l <LEFT_OUTPUT_DIR> -r <RIGHT_OUTPUT_DIR>
```

**Example - Process test/data/0:**
```powershell
python apply_mask_split.py `
    -i "C:\Users\imran\AI_hub_keskisuomi\WP3_knee_osteoarthritis\data\test\data\0" `
    -m "out_test_0" `
    -l "left_masked_test_0" `
    -r "right_masked_test_0"
```

**Expected Output:**
```
Original images: 639
Mask directory: out_test_0
Output left-masked: left_masked_test_0
Output right-masked: right_masked_test_0

1/639: 9003175L.png ✓
2/639: 9003175R.png ✓
3/639: 9003316L.png ✓
...
639/639: 9998089L.png ✓

==================================================
✓ Success: 639/639
✗ Failed: 0/639
✓ Left-masked saved to: left_masked_test_0/
✓ Right-masked saved to: right_masked_test_0/
==================================================
```

**What Happens:**
- Splits each mask vertically at center
- Creates left-masked version (left half darkened)
- Creates right-masked version (right half darkened)
- Saves to two separate directories

**Time Required:** ~30-60 seconds for 639 images

---

## 4. Verification Steps

### After Each Step, Verify Output

#### Verify Step 1 (Masks)
```powershell
# Count generated masks
Get-ChildItem "out_test_0\*.png" | Measure-Object

# View a sample mask (opens in default image viewer)
ii "out_test_0\9003175L_mask.png"

# Expected: White regions showing joint space, black background
```

#### Verify Step 2 (Full Blackout)
```powershell
# Count blackedout images
Get-ChildItem "blackedout_test_0\*.png" | Measure-Object

# Compare original vs blackedout
ii "C:\Users\imran\AI_hub_keskisuomi\WP3_knee_osteoarthritis\data\test\data\0\9003175L.png"
ii "blackedout_test_0\9003175L.png"

# Expected: Joint space region should be black in blackedout version
```

#### Verify Step 3 (Split Versions)
```powershell
# Count left-masked images
Get-ChildItem "left_masked_test_0\*.png" | Measure-Object

# Count right-masked images
Get-ChildItem "right_masked_test_0\*.png" | Measure-Object

# View samples
ii "left_masked_test_0\9003175L.png"
ii "right_masked_test_0\9003175L.png"

# Expected: 
# - Left-masked: Left half of joint space darkened
# - Right-masked: Right half of joint space darkened
```

---

## 5. Common Workflows

### Workflow A: Process Single Folder (Manual)

**Use Case:** Process one folder at a time with verification between steps

```powershell
# Set variables for easier command execution
$INPUT_DIR = "C:\Users\imran\AI_hub_keskisuomi\WP3_knee_osteoarthritis\data\test\data\0"
$MASK_DIR = "out_test_0"
$BLACKOUT_DIR = "blackedout_test_0"
$LEFT_DIR = "left_masked_test_0"
$RIGHT_DIR = "right_masked_test_0"

# Navigate to script directory
cd C:\Users\imran\AI_hub_keskisuomi\WP3_knee_osteoarthritis\Knee_Segmentation

# Step 1: Generate masks
python segment_simple.py -i $INPUT_DIR -m "MODEL_unet.pth" -o $MASK_DIR

# Verify masks were created
Get-ChildItem "$MASK_DIR\*.png" | Measure-Object

# Step 2: Create full blackout
python apply_mask_blackout.py -i $INPUT_DIR -m $MASK_DIR -o $BLACKOUT_DIR

# Verify blackout images
Get-ChildItem "$BLACKOUT_DIR\*.png" | Measure-Object

# Step 3: Create split versions
python apply_mask_split.py -i $INPUT_DIR -m $MASK_DIR -l $LEFT_DIR -r $RIGHT_DIR

# Verify split images
Get-ChildItem "$LEFT_DIR\*.png" | Measure-Object
Get-ChildItem "$RIGHT_DIR\*.png" | Measure-Object

Write-Host "✓ All steps completed successfully!" -ForegroundColor Green
```

---

### Workflow B: Process All Standard Folders (Batch)

**Use Case:** Process all 6 standard folders automatically

```powershell
# Save this as process_all_folders.ps1

# Navigate to script directory
cd C:\Users\imran\AI_hub_keskisuomi\WP3_knee_osteoarthritis\Knee_Segmentation

# Define all folders to process
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

# Process each folder
foreach ($f in $folders) {
    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host "Processing: $($f.input)" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    
    # Check if input folder exists
    if (!(Test-Path $f.input)) {
        Write-Host "✗ Input folder not found: $($f.input)" -ForegroundColor Red
        continue
    }
    
    # Step 1: Generate masks
    Write-Host "`n[1/3] Generating masks..." -ForegroundColor Yellow
    python segment_simple.py -i $f.input -m "MODEL_unet.pth" -o $f.mask
    if ($LASTEXITCODE -ne 0) {
        Write-Host "✗ Mask generation failed!" -ForegroundColor Red
        continue
    }
    
    # Step 2: Create full blackout
    Write-Host "`n[2/3] Creating full blackout..." -ForegroundColor Yellow
    python apply_mask_blackout.py -i $f.input -m $f.mask -o $f.blackout
    if ($LASTEXITCODE -ne 0) {
        Write-Host "✗ Full blackout failed!" -ForegroundColor Red
        continue
    }
    
    # Step 3: Create split versions
    Write-Host "`n[3/3] Creating left/right split versions..." -ForegroundColor Yellow
    python apply_mask_split.py -i $f.input -m $f.mask -l $f.left -r $f.right
    if ($LASTEXITCODE -ne 0) {
        Write-Host "✗ Split creation failed!" -ForegroundColor Red
        continue
    }
    
    Write-Host "`n✓ Completed: $($f.input)" -ForegroundColor Green
}

Write-Host "`n========================================" -ForegroundColor Green
Write-Host "ALL FOLDERS PROCESSED" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
```

**To Run:**
```powershell
.\process_all_folders.ps1
```

**Expected Duration:** ~15-30 minutes for all 6 folders

---

### Workflow C: Process Only Specific Steps

**Use Case:** You only need masks, or only need blackout versions

#### Only Generate Masks (No Blackout/Split)
```powershell
$folders = @("test\data\0", "test\data\2", "train\data\0", "train\data\2", "val\data\0", "val\data\2")

foreach ($f in $folders) {
    $name = $f -replace '[\\/]', '_'
    python segment_simple.py `
        -i "C:\Users\imran\AI_hub_keskisuomi\WP3_knee_osteoarthritis\data\$f" `
        -m "MODEL_unet.pth" `
        -o "out_$name"
}
```

#### Only Create Blackout (Assuming Masks Exist)
```powershell
$folders = @(
    @{input="data\test\data\0"; mask="out_test_0"; out="blackedout_test_0"}
    # Add more as needed
)

foreach ($f in $folders) {
    python apply_mask_blackout.py -i $f.input -m $f.mask -o $f.out
}
```

---

## 6. Troubleshooting During Execution

### Issue: Script Hangs or Freezes

**Symptoms:** Script appears stuck with no progress

**Solutions:**
1. Check if model is loading (first run is slower)
2. Press `Ctrl+C` to cancel, then restart
3. Check available RAM (close other programs)
4. Process fewer images or split into batches

---

### Issue: "Mask not found" Errors

**Symptoms:** `apply_mask_blackout.py` or `apply_mask_split.py` reports missing masks

**Solutions:**
```powershell
# Check if masks were actually created
Get-ChildItem "out_test_0\*.png"

# If empty, re-run Step 1
python segment_simple.py -i "..." -m "MODEL_unet.pth" -o "out_test_0"

# Verify mask naming convention
Get-ChildItem "out_test_0\*_mask.png" | Select-Object -First 5
```

---

### Issue: Partial Processing (Some Images Fail)

**Symptoms:** Success count is less than total count

**Example:** `✓ Success: 635/639`

**Investigation:**
```powershell
# Check which images failed (look for error messages in output)

# Manually check specific image
python -c "import cv2; img = cv2.imread('problematic_image.png'); print(img.shape if img is not None else 'Failed')"

# Skip problematic images or fix them manually
```

---

### Issue: Output Folders Not Created

**Symptoms:** Script completes but no output directory

**Solutions:**
```powershell
# Check current directory
Get-Location

# Ensure you're in the correct directory
cd C:\Users\imran\AI_hub_keskisuomi\WP3_knee_osteoarthritis\Knee_Segmentation

# Check if directories were created
Get-ChildItem -Directory | Where-Object {$_.Name -like "out_*" -or $_.Name -like "blackedout_*"}

# Manually create if needed
New-Item -ItemType Directory -Path "out_test_0"
```

---

### Issue: Permission Errors

**Symptoms:** `PermissionError: [WinError 5] Access is denied`

**Solutions:**
```powershell
# Run PowerShell as Administrator
# Or change output location to user directory

# Use different output path
python segment_simple.py `
    -i "..." `
    -m "MODEL_unet.pth" `
    -o "C:\Users\imran\Documents\knee_output"
```

---

### Issue: Memory Errors

**Symptoms:** `MemoryError` or system becomes unresponsive

**Solutions:**
1. Close other applications
2. Process folders one at a time
3. Restart computer to clear memory
4. Modify scripts to process in smaller batches

---

## Summary Checklist

Before starting:
- [ ] Environment activated
- [ ] In correct directory (Knee_Segmentation)
- [ ] Model file exists
- [ ] Input images verified

For each folder:
- [ ] Step 1: Masks generated successfully
- [ ] Step 2: Blackout images created
- [ ] Step 3: Split images created
- [ ] All outputs verified

After completion:
- [ ] All expected directories exist
- [ ] File counts match expectations
- [ ] Sample images visually verified
- [ ] Results backed up (optional)

---

## Quick Reference Commands

```powershell
# Navigate to working directory
cd C:\Users\imran\AI_hub_keskisuomi\WP3_knee_osteoarthritis\Knee_Segmentation

# Activate environment
conda activate knee-segmentation

# Step 1: Generate masks
python segment_simple.py -i "<input>" -m "MODEL_unet.pth" -o "<mask_out>"

# Step 2: Full blackout
python apply_mask_blackout.py -i "<input>" -m "<mask_out>" -o "<blackout_out>"

# Step 3: Split versions
python apply_mask_split.py -i "<input>" -m "<mask_out>" -l "<left_out>" -r "<right_out>"

# Verify outputs
Get-ChildItem "<output_dir>\*.png" | Measure-Object
```

---

## Support

If you encounter issues not covered here:
1. Check README.md for general troubleshooting
2. Check USAGE_GUIDE.md for detailed examples
3. Review error messages carefully
4. Contact project maintainers

---

**Document Version:** 2.0  
**Last Updated:** October 2025  
**For:** AI Hub Keski-Suomi - WP3 Knee Osteoarthritis
