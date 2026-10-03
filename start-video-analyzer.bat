@echo off
chcp 65001 >nul
cd /d "%~dp0"

set /p ENV_LINE=<.env

for /f "tokens=2 delims==" %%a in ("%ENV_LINE%") do set VIDEO_FILE=%%a

if "%VIDEO_FILE%"=="" (
    echo Error: .env file not found or empty
    pause
    exit /b 1
)

echo Starting video analysis: %VIDEO_FILE%
python video-analyzer.py %VIDEO_FILE%
pause
