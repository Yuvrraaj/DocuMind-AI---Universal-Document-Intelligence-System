@echo off
REM ──────────────────────────────────────────────────────────────
REM  Insurance Document Analyzer Launcher
REM  Usage: Double-click this file to start the application.
REM ──────────────────────────────────────────────────────────────

:: 1. Move to the directory containing this batch file
cd /d "%~dp0"

:: 2. (Optional) Activate virtual environment if present
if exist venv\Scripts\activate.bat (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo No virtual environment found; using system Python.
)

:: 3. Run the Python GUI script
echo Starting Insurance Document Analyzer...
python insurance_query_system.py

:: 4. Wait for user to close window
echo.
pause
