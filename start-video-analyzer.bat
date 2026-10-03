@echo off
chcp 65001 >nul
cd /d "%~dp0"

if "%~1"=="" (
    echo Usage: start-video-analyzer.bat ^<video_file^>
    pause
    exit /b 1
)

if not exist venv-gpu\Scripts\activate.bat (
    echo Error: venv-gpu not found. See README for setup.
    pause
    exit /b 1
)
call venv-gpu\Scripts\activate.bat

echo Starting video analysis: %~1
python video-analyzer.py "%~1"
pause
