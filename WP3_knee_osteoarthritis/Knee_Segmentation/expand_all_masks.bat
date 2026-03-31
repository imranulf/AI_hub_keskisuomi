@echo off
REM ============================================================================
REM Knee Segmentation - Expand All Masks
REM Creates multiple expansion levels (S, M, L, XL) for all mask directories
REM ============================================================================

setlocal enabledelayedexpansion

echo ============================================================================
echo                    MASK EXPANSION - ALL DIRECTORIES
echo ============================================================================
echo.
echo Expansion Levels:
echo   S  (Small)       : kernel=3, iterations=1
echo   M  (Medium)      : kernel=5, iterations=2
echo   L  (Large)       : kernel=7, iterations=3
echo   XL (Extra Large) : kernel=10, iterations=5
echo.

REM Define mask folders to process
set "MASK_FOLDERS=results_test_0 results_test_2 results_train_0 results_train_2 results_val_0 results_val_2"

for %%F in (%MASK_FOLDERS%) do (
    if exist "%%F" (
        echo ========================================
        echo Expanding: %%F
        echo ========================================

        REM Small expansion
        echo [S] Creating %%F_expanded_S...
        python expand_masks.py -i "%%F" -o "%%F_expanded_S" -k 3 -n 1

        REM Medium expansion
        echo [M] Creating %%F_expanded_M...
        python expand_masks.py -i "%%F" -o "%%F_expanded_M" -k 5 -n 2

        REM Large expansion
        echo [L] Creating %%F_expanded_L...
        python expand_masks.py -i "%%F" -o "%%F_expanded_L" -k 7 -n 3

        REM Extra Large expansion
        echo [XL] Creating %%F_expanded_XL...
        python expand_masks.py -i "%%F" -o "%%F_expanded_XL" -k 10 -n 5

        echo SUCCESS: Expanded %%F
        echo.
    ) else (
        echo WARNING: Mask directory not found: %%F
        echo          Run process_all.bat first to generate masks.
        echo.
    )
)

echo ============================================================================
echo                         EXPANSION COMPLETE
echo ============================================================================
echo.
echo Created expansion directories:
echo   *_expanded_S  : Small expansion
echo   *_expanded_M  : Medium expansion
echo   *_expanded_L  : Large expansion
echo   *_expanded_XL : Extra large expansion
echo.

pause
