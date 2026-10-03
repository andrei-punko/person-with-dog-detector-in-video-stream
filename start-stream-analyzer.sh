#!/bin/bash

cd "$(dirname "$0")"

if [ ! -f .env ]; then
    echo "Error: .env file not found"
    exit 1
fi

source venv-gpu/Scripts/activate

set -a
source .env
set +a

if [ -z "$RTSP_URL" ]; then
    echo "Error: RTSP_URL not set in .env"
    exit 1
fi

echo "Starting stream analysis: $RTSP_URL"
python stream-analyzer.py "$RTSP_URL"
