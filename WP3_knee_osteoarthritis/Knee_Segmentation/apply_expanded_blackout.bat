@echo off
REM ============================================================================
REM Knee Segmentation - Apply Blackout with Expanded Masks
REM Creates blackout images using different expansion levels
REM ============================================================================

setlocal enabledelayedexpansion

echo ============================================================================
echo              APPLY BLACKOUT WITH EXPANDED MASKS
echo ============================================================================
echo.

REM Configuration
set "BASE_DATA=./data"

REM Define folders and their input paths
set "FOLDERS=test_0 test_2 train_0 train_2 val_0 val_2"

for %%F in (%FOLDERS%) do (
    REM Parse folder name to get split and subset
    for /f "tokens=1,2 delims=_" %%A in ("%%F") do (
        set "SPLIT=%%A"
        set "SUBSET=%%B"
    )

    set "INPUT_DIR=!BASE_DATA!\!SPLIT!\data\!SUBSET!"

    REM Check if input directory exists
    if not exist "!INPUT_DIR!" (
        echo WARNING: Input not found: !INPUT_DIR!
        echo Skipping %%F...
    ) else (
        echo ========================================
        echo Processing: %%F
        echo ========================================

        REM Apply blackout with each expansion level
        for %%E in (S M L XL) do (
            set "MASK_DIR=results_%%F_expanded_%%E"
            set "OUTPUT_DIR=blackedout_%%F_expanded_%%E"

            if exist "!MASK_DIR!" (
                echo [%%E] Creating !OUTPUT_DIR!...
                python apply_mask_blackout.py -i "!INPUT_DIR!" -m "!MASK_DIR!" -o "!OUTPUT_DIR!"
            ) else (
                echo [%%E] Mask directory not found: !MASK_DIR! - skipping
            )
        )
        echo.
    )
)

echo ============================================================================
echo                         PROCESSING COMPLETE
echo ============================================================================
echo.
echo Created blackout directories with expanded masks:
echo   blackedout_*_expanded_S  : Using small expansion
echo   blackedout_*_expanded_M  : Using medium expansion
echo   blackedout_*_expanded_L  : Using large expansion
echo   blackedout_*_expanded_XL : Using extra large expansion
echo.

pause
