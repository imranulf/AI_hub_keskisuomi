@echo off
REM ============================================================================
REM Knee Segmentation - Segmentation Only (Quick Mask Generation)
REM Usage: segment_only.bat <input_dir> <output_dir>
REM ============================================================================

setlocal

set "MODEL=MODEL_unet.pth"

REM Check arguments
if "%~1"=="" (
    echo.
    echo USAGE: segment_only.bat ^<input_dir^> ^<output_dir^>
    echo.
    echo Arguments:
    echo   input_dir  : Directory containing knee X-ray images (PNG/JPG)
    echo   output_dir : Directory to save segmentation masks
    echo.
    echo Example:
    echo   segment_only.bat "C:\data\knee_images" "masks_output"
    echo.
    pause
    exit /b 1
)

set "INPUT_DIR=%~1"
if "%~2"=="" (
    set "OUTPUT_DIR=masks_out"
) else (
    set "OUTPUT_DIR=%~2"
)

REM Check if model exists
if not exist "%MODEL%" (
    echo ERROR: Model file not found: %MODEL%
    pause
    exit /b 1
)

REM Check if input directory exists
if not exist "%INPUT_DIR%" (
    echo ERROR: Input directory not found: %INPUT_DIR%
    pause
    exit /b 1
)

echo ============================================================================
echo                    KNEE SEGMENTATION - MASK GENERATION
echo ============================================================================
echo.
echo Input:  %INPUT_DIR%
echo Output: %OUTPUT_DIR%
echo Model:  %MODEL%
echo.

python segment_simple.py -m "%MODEL%" -i "%INPUT_DIR%" -o "%OUTPUT_DIR%"

if errorlevel 1 (
    echo.
    echo ERROR: Segmentation failed
    pause
    exit /b 1
)

echo.
echo ============================================================================
echo                         SEGMENTATION COMPLETE
echo ============================================================================
echo Masks saved to: %OUTPUT_DIR%
echo.

pause
