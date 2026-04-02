@echo off
REM ============================================================================
REM Apply Horizontal Blackout - All Directories
REM Creates three types: left blackout, right blackout, full blackout
REM ============================================================================

setlocal enabledelayedexpansion

REM Configuration
set "BASE_DATA=./data"

echo ============================================================================
echo          HORIZONTAL BLACKOUT - THREE TYPES (LEFT, RIGHT, FULL)
echo ============================================================================
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
echo   - *_horizontal_left   : Left side of joint band blacked out
echo   - *_horizontal_right  : Right side of joint band blacked out
echo   - *_horizontal_full   : Entire horizontal band blacked out
echo.

pause
goto :eof

REM ============================================================================
REM Function to process a single folder
REM ============================================================================
:process_folder
set "SPLIT=%~1"
set "SUBSET=%~2"
set "FOLDER_NAME=%SPLIT%_%SUBSET%"

set "INPUT_DIR=%BASE_DATA%\%SPLIT%\data\%SUBSET%"
set "MASK_DIR=results_%FOLDER_NAME%_horizontal"
set "OUTPUT_LEFT=blackedout_%FOLDER_NAME%_horizontal_left"
set "OUTPUT_RIGHT=blackedout_%FOLDER_NAME%_horizontal_right"
set "OUTPUT_FULL=blackedout_%FOLDER_NAME%_horizontal_full"

echo.
echo ========================================
echo Processing: %FOLDER_NAME%
echo ========================================

REM Check if input directory exists
if not exist "%INPUT_DIR%" (
    echo WARNING: Input not found: %INPUT_DIR%
    goto :eof
)

REM Check if horizontal mask directory exists
if not exist "%MASK_DIR%" (
    echo WARNING: Horizontal masks not found: %MASK_DIR%
    echo Run expand_horizontal_all.bat first!
    goto :eof
)

REM Apply horizontal blackout (creates left, right, and full)
python apply_horizontal_blackout.py -i "%INPUT_DIR%" -m "%MASK_DIR%" -l "%OUTPUT_LEFT%" -r "%OUTPUT_RIGHT%" -f "%OUTPUT_FULL%"

echo SUCCESS: Completed %FOLDER_NAME%
goto :eof
