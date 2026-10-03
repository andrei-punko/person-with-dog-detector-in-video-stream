#!/bin/bash

cd "$(dirname "$0")"

if [ ! -f .env ]; then
    echo "Error: .env file not found"
    exit 1
fi

# On Windows use Scripts/activate; on Linux/macOS use bin/activate
source venv-gpu/Scripts/activate

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
