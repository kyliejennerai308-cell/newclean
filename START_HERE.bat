@echo off
REM ============================================================================
REM VINYL PLAYMAT RESTORATION - START HERE
REM ============================================================================
REM USAGE:
REM   1. Put your scanned images in the "scans" folder
REM   2. Double-click this START_HERE.bat file
REM   3. Wait for processing to complete
REM   4. Find cleaned images in "scans/output" folder
REM ============================================================================

REM Change to the directory where this batch file lives
cd /d "%~dp0"

title Vinyl Playmat Restoration - New Colour Regime
color 0A
echo.
echo ============================================================================
echo     VINYL PLAYMAT RESTORATION - NEW COLOUR REGIME
echo ============================================================================
echo.
echo This script will:
echo   - Check Python installation
echo   - Install required packages (opencv-python, numpy)
echo   - Process all images in the "scans" folder
echo   - Use GPU acceleration automatically when available
echo   - Output cleaned images to "scans/output"
echo.
echo ============================================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not found! Please install Python 3.8 or higher.
    echo.
    pause
    exit /b 1
)

REM Install required packages
echo Installing required packages...
python -m pip install opencv-python numpy --upgrade
if errorlevel 1 (
    echo.
    echo [ERROR] Failed to install required packages
    echo.
    pause
    exit /b 1
)

echo [OK] Packages installed successfully
echo.

REM Run the restoration script (no arguments — paths are built in)
echo ============================================================================
echo Processing images...
echo ============================================================================
echo.

python restore_playmat_hsv.py

if errorlevel 1 (
    echo.
    echo [ERROR] Processing failed - check error messages above
    echo.
    pause
    exit /b 1
)

echo.
echo ============================================================================
echo     PROCESSING COMPLETE!
echo ============================================================================
echo.
echo Your cleaned images are in: scans\output\
echo.
pause
