#!/bin/bash

cd "$(dirname "$0")"

if [ -z "$1" ]; then
    echo "Usage: $0 <video_file> [extra analyzer args, e.g. --no-display]"
    exit 1
fi

# On Windows (Git Bash) use Scripts/activate; on Linux/macOS use bin/activate
if [ -f venv-gpu/Scripts/activate ]; then
    source venv-gpu/Scripts/activate
elif [ -f venv-gpu/bin/activate ]; then
    source venv-gpu/bin/activate
else
    echo "Error: virtual environment venv-gpu not found. See README for setup."
    exit 1
fi

echo "Starting video analysis: $1"
python video-analyzer.py "$@"
