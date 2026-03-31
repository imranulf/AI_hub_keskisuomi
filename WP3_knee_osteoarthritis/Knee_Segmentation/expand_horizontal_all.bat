@echo off
REM ============================================================================
REM Expand Masks Horizontally - All Directories
REM Extends masks from left to right edge for each row with joint space
REM ============================================================================

setlocal enabledelayedexpansion

REM Activate conda environment
call C:\Users\imran\miniconda3\Scripts\activate.bat torch_env

echo ============================================================================
echo             HORIZONTAL MASK EXPANSION - ALL DIRECTORIES
echo ============================================================================
echo.
echo This will expand masks horizontally to cover the full image width
echo wherever joint space is detected (creates horizontal band).
echo.

REM Define mask folders to process
set "MASK_FOLDERS=results_test_0 results_test_2 results_train_0 results_train_2 results_val_0 results_val_2"

for %%F in (%MASK_FOLDERS%) do (
    if exist "%%F" (
        echo Processing: %%F
        python expand_mask_horizontal.py -i "%%F" -o "%%F_horizontal"
        echo.
    ) else (
        echo WARNING: Mask directory not found: %%F
    )
)

echo ============================================================================
echo                         EXPANSION COMPLETE
echo ============================================================================
echo.
echo Created directories: *_horizontal
echo.

pause
