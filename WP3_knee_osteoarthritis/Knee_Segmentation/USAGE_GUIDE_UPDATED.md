# Usage Guide - Knee Segmentation Pipeline

Comprehensive usage examples and practical scenarios for the knee segmentation and image processing pipeline.

## Table of Contents
1. [Basic Usage Examples](#1-basic-usage-examples)
2. [Advanced Scenarios](#2-advanced-scenarios)
3. [Use Case Workflows](#3-use-case-workflows)
4. [Output Interpretation](#4-output-interpretation)
5. [Tips and Best Practices](#5-tips-and-best-practices)
6. [FAQ](#6-faq)

---

## 1. Basic Usage Examples

### Example 1.1: Process a Single Test Folder

**Scenario:** You have knee X-ray images in `test/data/0` and want to generate all outputs.

```powershell
# Navigate to working directory
cd C:\Users\imran\AI_hub_keskisuomi\WP3_knee_osteoarthritis\Knee_Segmentation

# Activate environment
conda activate knee-segmentation

# Step 1: Generate masks
python segment_simple.py `
    -i "C:\Users\imran\AI_hub_keskisuomi\WP3_knee_osteoarthritis\data\test\data\0" `
    -m "MODEL_unet.pth" `
    -o "out_test_0"

# Step 2: Create full blackout
python apply_mask_blackout.py `
    -i "C:\Users\imran\AI_hub_keskisuomi\WP3_knee_osteoarthritis\data\test\data\0" `
    -m "out_test_0" `
    -o "blackedout_test_0"

# Step 3: Create left/right split
python apply_mask_split.py `
    -i "C:\Users\imran\AI_hub_keskisuomi\WP3_knee_osteoarthritis\data\test\data\0" `
    -m "out_test_0" `
    -l "left_masked_test_0" `
    -r "right_masked_test_0"
```

**Result:** Four output directories created with 639 processed images each.

---

### Example 1.2: Process Only Training Data

**Scenario:** You only need to process training folders.

```powershell
# Define training folders
$train_folders = @(
    @{input="C:\Users\imran\AI_hub_keskisuomi\WP3_knee_osteoarthritis\data\train\data\0";
      mask="out_train_0"; blackout="blackedout_train_0"; 
      left="left_masked_train_0"; right="right_masked_train_0"},
    @{input="C:\Users\imran\AI_hub_keskisuomi\WP3_knee_osteoarthritis\data\train\data\2";
      mask="out_train_2"; blackout="blackedout_train_2"; 
      left="left_masked_train_2"; right="right_masked_train_2"}
)

# Process each training folder
foreach ($f in $train_folders) {
    Write-Host "Processing: $($f.input)" -ForegroundColor Cyan
    python segment_simple.py -i $f.input -m "MODEL_unet.pth" -o $f.mask
    python apply_mask_blackout.py -i $f.input -m $f.mask -o $f.blackout
    python apply_mask_split.py -i $f.input -m $f.mask -l $f.left -r $f.right
    Write-Host "✓ Completed" -ForegroundColor Green
}
```

---

### Example 1.3: Generate Masks Only (No Blackout)

**Scenario:** You only need segmentation masks for analysis, not modified images.

```powershell
# Process all folders - masks only
$all_folders = @(
    "C:\Users\imran\AI_hub_keskisuomi\WP3_knee_osteoarthritis\data\test\data\0",
    "C:\Users\imran\AI_hub_keskisuomi\WP3_knee_osteoarthritis\data\test\data\2",
    "C:\Users\imran\AI_hub_keskisuomi\WP3_knee_osteoarthritis\data\train\data\0",
    "C:\Users\imran\AI_hub_keskisuomi\WP3_knee_osteoarthritis\data\train\data\2",
    "C:\Users\imran\AI_hub_keskisuomi\WP3_knee_osteoarthritis\data\val\data\0",
    "C:\Users\imran\AI_hub_keskisuomi\WP3_knee_osteoarthritis\data\val\data\2"
)

$i = 0
foreach ($folder in $all_folders) {
    $output_name = "masks_" + $i
    python segment_simple.py -i $folder -m "MODEL_unet.pth" -o $output_name
    $i++
}
```

---

### Example 1.4: Using Relative Paths

**Scenario:** Your data is in the same directory structure as the scripts.

```powershell
# Assuming this structure:
# Knee_Segmentation/
#   ├── segment_simple.py
#   ├── MODEL_unet.pth
#   └── my_images/
#       ├── image1.png
#       └── image2.png

# Use relative paths
python segment_simple.py -i "my_images" -m "MODEL_unet.pth" -o "my_masks"
python apply_mask_blackout.py -i "my_images" -m "my_masks" -o "my_blackout"
python apply_mask_split.py -i "my_images" -m "my_masks" -l "my_left" -r "my_right"
```

---

## 2. Advanced Scenarios

### Example 2.1: Process Images from Multiple Sources

**Scenario:** You have images from different studies in different locations.

```powershell
# Define custom sources
$custom_sources = @(
    @{name="Study_A"; path="D:\Research\KneeData\StudyA\images"},
    @{name="Study_B"; path="D:\Research\KneeData\StudyB\images"},
    @{name="Study_C"; path="E:\External\KneeImages"}
)

# Process each source
foreach ($source in $custom_sources) {
    $mask_dir = "masks_$($source.name)"
    $blackout_dir = "blackout_$($source.name)"
    $left_dir = "left_$($source.name)"
    $right_dir = "right_$($source.name)"
    
    Write-Host "`nProcessing: $($source.name)" -ForegroundColor Cyan
    
    python segment_simple.py -i $source.path -m "MODEL_unet.pth" -o $mask_dir
    python apply_mask_blackout.py -i $source.path -m $mask_dir -o $blackout_dir
    python apply_mask_split.py -i $source.path -m $mask_dir -l $left_dir -r $right_dir
    
    # Create summary
    $count = (Get-ChildItem "$mask_dir\*.png" | Measure-Object).Count
    Write-Host "✓ Processed $count images from $($source.name)" -ForegroundColor Green
}
```

---

### Example 2.2: Reprocess Only Failed Images

**Scenario:** Previous run failed for some images, reprocess only those.

```powershell
# Get list of original images
$original_images = Get-ChildItem "C:\...\data\test\data\0\*.png" | Select-Object -ExpandProperty Name

# Get list of successfully processed masks
$processed_masks = Get-ChildItem "out_test_0\*_mask.png" | ForEach-Object {
    $_.Name -replace '_mask\.png$', '.png'
}

# Find missing images
$missing = $original_images | Where-Object { $processed_masks -notcontains $_ }

if ($missing.Count -gt 0) {
    Write-Host "Found $($missing.Count) images that need reprocessing" -ForegroundColor Yellow
    
    # Create temporary directory with missing images
    $temp_dir = "temp_reprocess"
    New-Item -ItemType Directory -Path $temp_dir -Force
    
    foreach ($img in $missing) {
        Copy-Item "C:\...\data\test\data\0\$img" -Destination $temp_dir
    }
    
    # Reprocess
    python segment_simple.py -i $temp_dir -m "MODEL_unet.pth" -o "out_test_0_retry"
    
    # Move successful masks to main directory
    Move-Item "out_test_0_retry\*.png" -Destination "out_test_0" -Force
    
    # Cleanup
    Remove-Item $temp_dir -Recurse -Force
    Remove-Item "out_test_0_retry" -Recurse -Force
    
    Write-Host "✓ Reprocessing complete" -ForegroundColor Green
} else {
    Write-Host "✓ All images already processed" -ForegroundColor Green
}
```

---

### Example 2.3: Process with Quality Check

**Scenario:** Verify mask quality before creating blackout versions.

```powershell
# Step 1: Generate masks
python segment_simple.py -i "data\test\data\0" -m "MODEL_unet.pth" -o "out_test_0"

# Step 2: Quality check - analyze segmentation percentages
Write-Host "`nAnalyzing mask quality..." -ForegroundColor Cyan

$analysis = Get-ChildItem "out_test_0\*.png" | ForEach-Object {
    # Use Python to calculate white pixel percentage
    $result = python -c @"
import cv2
import numpy as np
mask = cv2.imread('$($_.FullName)', cv2.IMREAD_GRAYSCALE)
if mask is not None:
    white_pixels = np.sum(mask > 0)
    total_pixels = mask.size
    percentage = (white_pixels / total_pixels) * 100
    print(f'{percentage:.2f}')
else:
    print('0.00')
"@
    [PSCustomObject]@{
        File = $_.Name
        SegmentationPercent = [double]$result
    }
}

# Show statistics
$analysis | Measure-Object -Property SegmentationPercent -Average -Minimum -Maximum | 
    Format-List Count, Average, Minimum, Maximum

# Flag suspicious masks (too high or too low segmentation)
$suspicious = $analysis | Where-Object { $_.SegmentationPercent -lt 0.1 -or $_.SegmentationPercent -gt 15 }

if ($suspicious.Count -gt 0) {
    Write-Host "`n⚠ Found $($suspicious.Count) suspicious masks:" -ForegroundColor Yellow
    $suspicious | Format-Table -AutoSize
} else {
    Write-Host "`n✓ All masks passed quality check" -ForegroundColor Green
}

# Proceed with blackout if quality is acceptable
Read-Host "Press Enter to continue with blackout processing, or Ctrl+C to cancel"
python apply_mask_blackout.py -i "data\test\data\0" -m "out_test_0" -o "blackedout_test_0"
```

---

### Example 2.4: Create Different Output Formats

**Scenario:** You need outputs in different directories for different analyses.

```powershell
# Process once
python segment_simple.py -i "data\test\data\0" -m "MODEL_unet.pth" -o "masks"

# Create multiple blackout versions with different naming conventions
$output_configs = @(
    @{name="analysis_full_blackout"; use_split=$false},
    @{name="ml_training_left"; use_split=$true; side="left"},
    @{name="ml_training_right"; use_split=$true; side="right"}
)

foreach ($config in $output_configs) {
    if ($config.use_split) {
        if ($config.side -eq "left") {
            python apply_mask_split.py -i "data\test\data\0" -m "masks" `
                -l $config.name -r "temp_ignore"
            Remove-Item "temp_ignore" -Recurse -Force
        } else {
            python apply_mask_split.py -i "data\test\data\0" -m "masks" `
                -l "temp_ignore" -r $config.name
            Remove-Item "temp_ignore" -Recurse -Force
        }
    } else {
        python apply_mask_blackout.py -i "data\test\data\0" -m "masks" -o $config.name
    }
    Write-Host "✓ Created: $($config.name)" -ForegroundColor Green
}
```

---

## 3. Use Case Workflows

### Use Case 3.1: Machine Learning Dataset Preparation

**Goal:** Create augmented dataset with original, blackout, left-masked, and right-masked versions.

```powershell
# Define output structure
$base_output = "ML_Dataset"
New-Item -ItemType Directory -Path $base_output -Force

# Create subdirectories
$splits = @("train", "val", "test")
$versions = @("original", "full_blackout", "left_masked", "right_masked")

foreach ($split in $splits) {
    foreach ($version in $versions) {
        New-Item -ItemType Directory -Path "$base_output\$split\$version" -Force
    }
}

# Process each split
$input_dirs = @{
    "train" = "C:\...\data\train\data\0"
    "val" = "C:\...\data\val\data\0"
    "test" = "C:\...\data\test\data\0"
}

foreach ($split in $splits) {
    Write-Host "`nProcessing $split set..." -ForegroundColor Cyan
    
    $input_dir = $input_dirs[$split]
    $mask_dir = "temp_masks_$split"
    
    # Generate masks
    python segment_simple.py -i $input_dir -m "MODEL_unet.pth" -o $mask_dir
    
    # Copy original images
    Copy-Item "$input_dir\*.png" -Destination "$base_output\$split\original\"
    
    # Create full blackout
    python apply_mask_blackout.py -i $input_dir -m $mask_dir `
        -o "$base_output\$split\full_blackout"
    
    # Create split versions
    python apply_mask_split.py -i $input_dir -m $mask_dir `
        -l "$base_output\$split\left_masked" `
        -r "$base_output\$split\right_masked"
    
    # Cleanup temporary masks
    Remove-Item $mask_dir -Recurse -Force
    
    Write-Host "✓ $split set complete" -ForegroundColor Green
}

Write-Host "`n✓ ML Dataset created in: $base_output" -ForegroundColor Green
Get-ChildItem $base_output -Recurse -Directory | 
    ForEach-Object {
        $count = (Get-ChildItem $_.FullName -File | Measure-Object).Count
        Write-Host "  $($_.FullName): $count files"
    }
```

---

### Use Case 3.2: Comparative Analysis Study

**Goal:** Create side-by-side comparisons of original vs processed images.

```powershell
# Process images
python segment_simple.py -i "study_images" -m "MODEL_unet.pth" -o "study_masks"
python apply_mask_blackout.py -i "study_images" -m "study_masks" -o "study_blackout"

# Create comparison directory
New-Item -ItemType Directory -Path "comparisons" -Force

# Generate side-by-side comparisons using Python
python -c @"
import cv2
import numpy as np
from pathlib import Path

# Get all images
images = list(Path('study_images').glob('*.png'))

for img_path in images:
    # Read images
    original = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    mask_path = Path('study_masks') / (img_path.stem + '_mask.png')
    blackout_path = Path('study_blackout') / img_path.name
    
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    blackout = cv2.imread(str(blackout_path), cv2.IMREAD_GRAYSCALE)
    
    # Create side-by-side comparison
    # Convert mask to color overlay
    mask_overlay = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    mask_overlay[mask > 0] = [0, 0, 255]  # Red overlay on segmented region
    
    # Stack horizontally: Original | Mask Overlay | Blackout
    original_bgr = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    blackout_bgr = cv2.cvtColor(blackout, cv2.COLOR_GRAY2BGR)
    comparison = np.hstack([original_bgr, mask_overlay, blackout_bgr])
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(comparison, 'Original', (10, 30), font, 0.7, (255, 255, 255), 2)
    cv2.putText(comparison, 'Segmentation', (original.shape[1] + 10, 30), font, 0.7, (255, 255, 255), 2)
    cv2.putText(comparison, 'Blackout', (original.shape[1]*2 + 10, 30), font, 0.7, (255, 255, 255), 2)
    
    # Save
    output_path = Path('comparisons') / img_path.name
    cv2.imwrite(str(output_path), comparison)
    print(f'Created: {output_path.name}')

print('✓ All comparisons created')
"@
```

---

### Use Case 3.3: Batch Analysis with Reporting

**Goal:** Process all folders and generate summary report.

```powershell
# Create report file
$report_file = "processing_report_$(Get-Date -Format 'yyyyMMdd_HHmmss').txt"

"Knee Segmentation Processing Report" | Out-File $report_file
"=" * 50 | Out-File $report_file -Append
"Started: $(Get-Date)" | Out-File $report_file -Append
"" | Out-File $report_file -Append

# Define folders
$folders = @(
    "test\data\0", "test\data\2",
    "train\data\0", "train\data\2",
    "val\data\0", "val\data\2"
)

$total_processed = 0
$total_time = Measure-Command {
    foreach ($folder in $folders) {
        $folder_name = $folder -replace '[\\/]', '_'
        $input_path = "C:\Users\imran\AI_hub_keskisuomi\WP3_knee_osteoarthritis\data\$folder"
        
        "`nProcessing: $folder" | Out-File $report_file -Append
        "-" * 50 | Out-File $report_file -Append
        
        # Count input images
        $input_count = (Get-ChildItem "$input_path\*.png" | Measure-Object).Count
        "Input images: $input_count" | Out-File $report_file -Append
        
        # Process
        $step_time = Measure-Command {
            python segment_simple.py -i $input_path -m "MODEL_unet.pth" -o "out_$folder_name" 2>&1 | Out-Null
            python apply_mask_blackout.py -i $input_path -m "out_$folder_name" -o "blackout_$folder_name" 2>&1 | Out-Null
            python apply_mask_split.py -i $input_path -m "out_$folder_name" -l "left_$folder_name" -r "right_$folder_name" 2>&1 | Out-Null
        }
        
        # Verify outputs
        $mask_count = (Get-ChildItem "out_$folder_name\*.png" | Measure-Object).Count
        $blackout_count = (Get-ChildItem "blackout_$folder_name\*.png" | Measure-Object).Count
        $left_count = (Get-ChildItem "left_$folder_name\*.png" | Measure-Object).Count
        $right_count = (Get-ChildItem "right_$folder_name\*.png" | Measure-Object).Count
        
        "Masks generated: $mask_count" | Out-File $report_file -Append
        "Blackout created: $blackout_count" | Out-File $report_file -Append
        "Left-masked created: $left_count" | Out-File $report_file -Append
        "Right-masked created: $right_count" | Out-File $report_file -Append
        "Processing time: $($step_time.TotalMinutes.ToString('F2')) minutes" | Out-File $report_file -Append
        
        $success = ($mask_count -eq $input_count -and $blackout_count -eq $input_count -and 
                    $left_count -eq $input_count -and $right_count -eq $input_count)
        
        if ($success) {
            "Status: ✓ SUCCESS" | Out-File $report_file -Append
            $total_processed += $input_count
        } else {
            "Status: ✗ INCOMPLETE" | Out-File $report_file -Append
        }
    }
}

# Summary
"" | Out-File $report_file -Append
"=" * 50 | Out-File $report_file -Append
"SUMMARY" | Out-File $report_file -Append
"=" * 50 | Out-File $report_file -Append
"Total images processed: $total_processed" | Out-File $report_file -Append
"Total processing time: $($total_time.TotalMinutes.ToString('F2')) minutes" | Out-File $report_file -Append
"Completed: $(Get-Date)" | Out-File $report_file -Append

# Display report
Get-Content $report_file
Write-Host "`n✓ Report saved to: $report_file" -ForegroundColor Green
```

---

## 4. Output Interpretation

### Understanding Mask Files

**Mask Format:**
- Filename: `{original_name}_mask.png`
- Values: 0 (black) = background, 255 (white) = segmented region
- Size: Same as original (224×224)

**Interpreting Segmentation:**
```powershell
# Check mask statistics
python -c @"
import cv2
import numpy as np
from pathlib import Path

masks = list(Path('out_test_0').glob('*_mask.png'))
print(f'Total masks: {len(masks)}\n')

percentages = []
for mask_path in masks:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    white_pixels = np.sum(mask > 0)
    percentage = (white_pixels / mask.size) * 100
    percentages.append(percentage)

percentages = np.array(percentages)
print(f'Segmentation Statistics:')
print(f'  Mean: {percentages.mean():.2f}%')
print(f'  Median: {np.median(percentages):.2f}%')
print(f'  Min: {percentages.min():.2f}%')
print(f'  Max: {percentages.max():.2f}%')
print(f'  Std Dev: {percentages.std():.2f}%')
"@
```

**Expected Results:**
- Normal joint space: 0.4% - 10% of image
- Unusually low (<0.1%): Possible detection failure or very narrow joint
- Unusually high (>15%): Possible over-segmentation or artifact

---

### Understanding Blackout Images

**Full Blackout:**
- Original image with segmented regions set to pixel value 0
- Joint space information completely removed
- Useful for testing models without joint space features

**Visual Check:**
```powershell
# Open original and blackout side-by-side
ii "data\test\data\0\9003175L.png"
ii "blackedout_test_0\9003175L.png"

# Look for: Joint space should be completely black in blackout version
```

---

### Understanding Split Masks

**Left-Masked Version:**
- Only LEFT half of mask applied
- If image shows a left knee (labeled "L"), the medial or lateral side is darkened depending on orientation
- Right half remains original

**Right-Masked Version:**
- Only RIGHT half of mask applied
- Complements left-masked version
- Left half remains original

**Use Cases:**
- Data augmentation for machine learning
- Analyzing left vs right differences
- Testing model robustness to partial occlusions

---

## 5. Tips and Best Practices

### Tip 5.1: Directory Organization

**Recommended Structure:**
```
Project/
├── Knee_Segmentation/
│   ├── MODEL_unet.pth
│   ├── segment_simple.py
│   ├── apply_mask_blackout.py
│   ├── apply_mask_split.py
│   │
│   ├── outputs/
│   │   ├── masks/
│   │   ├── blackout/
│   │   ├── left_masked/
│   │   └── right_masked/
│   │
│   └── logs/
└── data/
    ├── test/
    ├── train/
    └── val/
```

### Tip 5.2: Naming Conventions

**Consistent naming makes batch processing easier:**

```powershell
# Use descriptive prefixes
$input_dir = "data_test_0"
$mask_dir = "masks_test_0"
$blackout_dir = "blackout_test_0"
$left_dir = "left_test_0"
$right_dir = "right_test_0"

# Or use dates for versioning
$date = Get-Date -Format "yyyyMMdd"
$mask_dir = "masks_test_0_$date"
```

### Tip 5.3: Performance Optimization

**Speed up processing:**

1. **Use SSD for storage** (faster I/O)
2. **Close unnecessary applications** (free RAM)
3. **Process in parallel** (if you have multiple GPUs):
   ```powershell
   # Start multiple processes for different folders
   Start-Job { python segment_simple.py -i "folder1" -m "MODEL_unet.pth" -o "out1" }
   Start-Job { python segment_simple.py -i "folder2" -m "MODEL_unet.pth" -o "out2" }
   ```

4. **GPU acceleration** (install CUDA PyTorch):
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

### Tip 5.4: Backup Strategy

**Always backup outputs:**

```powershell
# Create timestamped backup
$backup_dir = "backups\$(Get-Date -Format 'yyyyMMdd_HHmmss')"
New-Item -ItemType Directory -Path $backup_dir -Force

# Copy all output directories
$outputs = @("out_*", "blackout_*", "left_masked_*", "right_masked_*")
foreach ($pattern in $outputs) {
    Get-ChildItem -Directory -Filter $pattern | 
        ForEach-Object {
            Copy-Item $_.FullName -Destination "$backup_dir\$($_.Name)" -Recurse
        }
}

Write-Host "✓ Backup created: $backup_dir" -ForegroundColor Green
```

### Tip 5.5: Logging and Monitoring

**Create detailed logs:**

```powershell
# Start logging
$log_file = "processing_log_$(Get-Date -Format 'yyyyMMdd_HHmmss').txt"
Start-Transcript -Path $log_file

# Run your commands
python segment_simple.py -i "..." -m "MODEL_unet.pth" -o "..."

# Stop logging
Stop-Transcript

# Log file now contains all output and errors
```

---

## 6. FAQ

### Q6.1: How long does processing take?

**A:** Depends on number of images and hardware:
- 600-700 images: 1-3 minutes for masks, 30-60 seconds each for blackout/split
- Total for one folder (all steps): ~3-5 minutes
- All 6 folders: ~20-30 minutes

### Q6.2: Can I process non-PNG images?

**A:** Currently scripts are designed for PNG. To process other formats:

```powershell
# Convert JPEG to PNG first
Get-ChildItem "*.jpg" | ForEach-Object {
    python -c "import cv2; img=cv2.imread('$($_.Name)'); cv2.imwrite('$($_.BaseName).png', img)"
}

# Then process PNGs normally
python segment_simple.py -i "." -m "MODEL_unet.pth" -o "masks"
```

### Q6.3: What if images are not 224×224?

**A:** The model expects 224×224. If different size:

```powershell
# Resize all images first
python -c @"
import cv2
from pathlib import Path

for img_path in Path('input_dir').glob('*.png'):
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    resized = cv2.resize(img, (224, 224))
    cv2.imwrite(f'resized_dir/{img_path.name}', resized)
"@

# Then process resized images
python segment_simple.py -i "resized_dir" -m "MODEL_unet.pth" -o "masks"
```

### Q6.4: Can I adjust the segmentation threshold?

**A:** The current `segment_simple.py` uses default threshold (0.5). To modify:

1. Open `segment_simple.py`
2. Find the line: `pred = (torch.sigmoid(pred) > 0.5).float()`
3. Change `0.5` to your desired value (e.g., `0.3` for more sensitive, `0.7` for stricter)

Or add as command-line argument (requires script modification).

### Q6.5: How do I verify all outputs are correct?

**A:** Use this verification script:

```powershell
$folders = @("test_0", "test_2", "train_0", "train_2", "val_0", "val_2")

foreach ($folder in $folders) {
    $input = "data\test\data\$folder"
    $mask = "out_$folder"
    $blackout = "blackout_$folder"
    $left = "left_$folder"
    $right = "right_$folder"
    
    if (Test-Path $input) {
        $input_count = (Get-ChildItem "$input\*.png" | Measure-Object).Count
        $mask_count = (Get-ChildItem "$mask\*.png" | Measure-Object).Count
        $blackout_count = (Get-ChildItem "$blackout\*.png" | Measure-Object).Count
        $left_count = (Get-ChildItem "$left\*.png" | Measure-Object).Count
        $right_count = (Get-ChildItem "$right\*.png" | Measure-Object).Count
        
        $status = if ($input_count -eq $mask_count -and $mask_count -eq $blackout_count -and
                      $blackout_count -eq $left_count -and $left_count -eq $right_count) {
            "✓ PASS"
        } else {
            "✗ FAIL"
        }
        
        Write-Host "$folder : Input=$input_count, Mask=$mask_count, Blackout=$blackout_count, Left=$left_count, Right=$right_count - $status"
    }
}
```

### Q6.6: What if I get memory errors?

**A:** Process in smaller batches:

```powershell
# Split input directory into smaller chunks
$all_images = Get-ChildItem "large_input_dir\*.png"
$batch_size = 100
$batch_num = 0

for ($i = 0; $i -lt $all_images.Count; $i += $batch_size) {
    $batch = $all_images[$i..[Math]::Min($i + $batch_size - 1, $all_images.Count - 1)]
    
    # Create temporary batch directory
    $batch_dir = "batch_$batch_num"
    New-Item -ItemType Directory -Path $batch_dir -Force
    $batch | Copy-Item -Destination $batch_dir
    
    # Process batch
    python segment_simple.py -i $batch_dir -m "MODEL_unet.pth" -o "masks_$batch_num"
    
    # Cleanup
    Remove-Item $batch_dir -Recurse -Force
    $batch_num++
}

# Merge all mask directories
New-Item -ItemType Directory -Path "all_masks" -Force
Get-ChildItem "masks_*\*.png" | Move-Item -Destination "all_masks"
```

---

## Support and Resources

- **README.md**: Overview and quick reference
- **RUNNING_INSTRUCTIONS.md**: Step-by-step execution guide
- **This file (USAGE_GUIDE.md)**: Detailed examples and scenarios

For additional help or to report issues, contact the project maintainers.

---

**Document Version:** 2.0  
**Last Updated:** October 2025  
**For:** AI Hub Keski-Suomi - WP3 Knee Osteoarthritis
