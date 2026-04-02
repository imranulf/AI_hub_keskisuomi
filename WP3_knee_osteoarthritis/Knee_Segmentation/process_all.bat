@echo off
REM ============================================================================
REM Knee Segmentation - Complete Pipeline Batch Processing
REM Processes all folders: test/train/val with subsets 0 and 2
REM ============================================================================

setlocal enabledelayedexpansion

REM Configuration
set "BASE_DATA=./data"
set "MODEL=MODEL_unet.pth"

REM Check if model exists
if not exist "%MODEL%" (
    echo ERROR: Model file not found: %MODEL%
    echo Please ensure MODEL_unet.pth is in the current directory.
    pause
    exit /b 1
)

echo ============================================================================
echo                    KNEE SEGMENTATION BATCH PROCESSING
echo ============================================================================
echo.
echo Data path: %BASE_DATA%
echo Model: %MODEL%
echo.

REM Process each folder
call :process_folder "test" "0"
call :process_folder "test" "2"
call :process_folder "train" "0"
call :process_folder "train" "2"
call :process_folder "val" "0"
call :process_folder "val" "2"

echo.
echo ============================================================================
echo                         ALL PROCESSING COMPLETE
echo ============================================================================
echo.
echo Output directories created:
echo   - results_*        : Segmentation masks
echo   - blackedout_*     : Images with joint space blacked out
echo   - left_masked_*    : Images with left half of joint space masked
echo   - right_masked_*   : Images with right half of joint space masked
echo.

pause
goto :eof

REM ============================================================================
REM Function to process a single folder
REM Usage: call :process_folder "split" "subset"
REM ============================================================================
:process_folder
set "SPLIT=%~1"
set "SUBSET=%~2"
set "FOLDER_NAME=%SPLIT%_%SUBSET%"

set "INPUT_DIR=%BASE_DATA%\%SPLIT%\data\%SUBSET%"
set "MASK_DIR=results_%FOLDER_NAME%"
set "BLACKOUT_DIR=blackedout_%FOLDER_NAME%"
set "LEFT_DIR=left_masked_%FOLDER_NAME%"
set "RIGHT_DIR=right_masked_%FOLDER_NAME%"

echo.
echo ========================================
echo Processing: %FOLDER_NAME%
echo Input: %INPUT_DIR%
echo ========================================

REM Check if input directory exists
if not exist "%INPUT_DIR%" (
    echo WARNING: Input directory not found: %INPUT_DIR%
    echo Skipping %FOLDER_NAME%...
    goto :eof
)

REM Step 1: Generate segmentation masks
echo.
echo [1/3] Generating segmentation masks...
python segment_simple.py -m "%MODEL%" -i "%INPUT_DIR%" -o "%MASK_DIR%"
if errorlevel 1 (
    echo ERROR: Segmentation failed for %FOLDER_NAME%
    goto :eof
)

REM Step 2: Apply full blackout
echo.
echo [2/3] Creating blackout images...
python apply_mask_blackout.py -i "%INPUT_DIR%" -m "%MASK_DIR%" -o "%BLACKOUT_DIR%"
if errorlevel 1 (
    echo ERROR: Blackout failed for %FOLDER_NAME%
    goto :eof
)

REM Step 3: Create left/right split versions
echo.
echo [3/3] Creating left/right split versions...
python apply_mask_split.py -i "%INPUT_DIR%" -m "%MASK_DIR%" -l "%LEFT_DIR%" -r "%RIGHT_DIR%"
if errorlevel 1 (
    echo ERROR: Split failed for %FOLDER_NAME%
    goto :eof
)

echo.
echo SUCCESS: Completed %FOLDER_NAME%
goto :eof
