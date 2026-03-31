@echo off
REM ============================================================================
REM Knee Segmentation - Process Single Folder
REM Usage: process_single_folder.bat <input_dir> <output_prefix>
REM Example: process_single_folder.bat "C:\path\to\images" "my_output"
REM ============================================================================

setlocal enabledelayedexpansion

set "MODEL=MODEL_unet.pth"

REM Check arguments
if "%~1"=="" (
    echo.
    echo USAGE: process_single_folder.bat ^<input_dir^> ^<output_prefix^>
    echo.
    echo Arguments:
    echo   input_dir     : Directory containing knee X-ray images (PNG/JPG)
    echo   output_prefix : Prefix for output directories (optional, default: output)
    echo.
    echo Example:
    echo   process_single_folder.bat "C:\data\knee_images" "test_run"
    echo.
    echo This will create:
    echo   - results_test_run       : Segmentation masks
    echo   - blackedout_test_run    : Images with joint space blacked out
    echo   - left_masked_test_run   : Left half masked
    echo   - right_masked_test_run  : Right half masked
    echo.
    pause
    exit /b 1
)

set "INPUT_DIR=%~1"
if "%~2"=="" (
    set "PREFIX=output"
) else (
    set "PREFIX=%~2"
)

REM Check if model exists
if not exist "%MODEL%" (
    echo ERROR: Model file not found: %MODEL%
    echo Please ensure MODEL_unet.pth is in the current directory.
    pause
    exit /b 1
)

REM Check if input directory exists
if not exist "%INPUT_DIR%" (
    echo ERROR: Input directory not found: %INPUT_DIR%
    pause
    exit /b 1
)

set "MASK_DIR=results_%PREFIX%"
set "BLACKOUT_DIR=blackedout_%PREFIX%"
set "LEFT_DIR=left_masked_%PREFIX%"
set "RIGHT_DIR=right_masked_%PREFIX%"

echo ============================================================================
echo                    KNEE SEGMENTATION - SINGLE FOLDER
echo ============================================================================
echo.
echo Input:  %INPUT_DIR%
echo Prefix: %PREFIX%
echo Model:  %MODEL%
echo.
echo Output directories:
echo   Masks:    %MASK_DIR%
echo   Blackout: %BLACKOUT_DIR%
echo   Left:     %LEFT_DIR%
echo   Right:    %RIGHT_DIR%
echo.

REM Step 1: Generate segmentation masks
echo ========================================
echo [1/3] Generating segmentation masks...
echo ========================================
python segment_simple.py -m "%MODEL%" -i "%INPUT_DIR%" -o "%MASK_DIR%"
if errorlevel 1 (
    echo ERROR: Segmentation failed
    pause
    exit /b 1
)

REM Step 2: Apply full blackout
echo.
echo ========================================
echo [2/3] Creating blackout images...
echo ========================================
python apply_mask_blackout.py -i "%INPUT_DIR%" -m "%MASK_DIR%" -o "%BLACKOUT_DIR%"
if errorlevel 1 (
    echo ERROR: Blackout failed
    pause
    exit /b 1
)

REM Step 3: Create left/right split versions
echo.
echo ========================================
echo [3/3] Creating left/right split versions...
echo ========================================
python apply_mask_split.py -i "%INPUT_DIR%" -m "%MASK_DIR%" -l "%LEFT_DIR%" -r "%RIGHT_DIR%"
if errorlevel 1 (
    echo ERROR: Split failed
    pause
    exit /b 1
)

echo.
echo ============================================================================
echo                         PROCESSING COMPLETE
echo ============================================================================
echo.
echo Output created:
echo   %MASK_DIR%     - Segmentation masks (*_mask.png)
echo   %BLACKOUT_DIR% - Blackout images
echo   %LEFT_DIR%     - Left-half masked images
echo   %RIGHT_DIR%    - Right-half masked images
echo.

pause
