#!/bin/bash

cd "$(dirname "$0")"

if [ ! -f .env ]; then
    echo "Error: .env file not found"
    exit 1
fi

# On Windows (Git Bash) use Scripts/activate; on Linux/macOS use bin/activate
if [ -f venv-gpu/Scripts/activate ]; then
    source venv-gpu/Scripts/activate
elif [ -f venv-gpu/bin/activate ]; then
    source venv-gpu/bin/activate
else
    echo "Error: virtual environment venv-gpu not found"
    exit 1
fi

set -a
source .env
set +a

if [ -z "$RTSP_URL" ]; then
    echo "Error: RTSP_URL not set in .env"
    exit 1
fi

# Strip everything up to and including the last "@" so credentials don't appear in the console
echo "Starting stream analysis: ${RTSP_URL##*@}"
python stream-analyzer.py "$RTSP_URL"
