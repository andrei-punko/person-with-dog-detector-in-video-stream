# Detector of person + dog pair in video stream or file

[![CI Build and Test](https://github.com/andrei-punko/person-with-dog-detector-in-video-stream/actions/workflows/python-ci-config.yml/badge.svg)](https://github.com/andrei-punko/person-with-dog-detector-in-video-stream/actions/workflows/python-ci-config.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Detects people and dogs in an RTSP stream or a video file using YOLO + TensorRT.  
When a person and a dog are found closer than a configurable pixel threshold, a screenshot is saved to `screenshots/`.

## Requirements

- Windows or Linux
- NVIDIA GPU with CUDA (tested with CUDA 12.1 on RTX 3060)
- Python 3.12
- TensorRT (installed as part of `requirements.txt`)

## Setup

### 1. Clone the repository

```bash
git clone <repo-url>
cd person-dog-detector
```

### 2. Create Python virtual environment

```bash
py -3.12 -m venv venv-gpu
```

Activate it:

- **Windows:** `source venv-gpu/Scripts/activate`
- **Linux / macOS:** `source venv-gpu/bin/activate`

### 3. Install dependencies

```bash
pip install -r requirements.txt
pip install "torch>=2.5.1" "torchvision>=0.20.1" --index-url https://download.pytorch.org/whl/cu121
```

### 4. Choose the YOLO model (n/s/m/l)

Place `yolo26l.pt` in the `models/` folder:

```
models/
└── yolo26l.pt
```

On CPU we could use `.pt` model, but on GPU to speed up it's better to use `.engine` model

### 5. Export (from .pt) to TensorRT engine to speed up calculations

Run once to produce `models/yolo26l.engine` (requires a connected GPU):

```bash
python utils/convert-model-to-engine.py
```

### 6. Configure credentials of RTSP stream from IP-camera

Copy `.env-sample` to `.env` and fill in your RTSP URL:

```bash
cp .env-sample .env
```

Populate user & password inside RTSP URL

## Usage

### Analyze an RTSP stream

```bash
# Using the launcher (reads RTSP_URL from .env):
bash start-stream-analyzer.sh

# Or directly:
python stream-analyzer.py rtsp://user:password@192.168.1.80:554/stream1
```

The script reconnects automatically if the stream drops. Press **Q** in the video window to stop.

Use `--no-display` to run without a video window (e.g. on a headless server):

```bash
python stream-analyzer.py --no-display rtsp://user:password@192.168.1.80:554/stream1
```

### Analyze a video file

```bash
# Using the launcher (pass the file as an argument):
./start-video-analyzer.sh videos/01.mp4

# Or directly:
python video-analyzer.py videos/01.mp4

# Without a video window:
python video-analyzer.py --no-display videos/01.mp4

# Analyse every 3rd frame (about 3x faster than analysing every frame):
python video-analyzer.py --no-display --stride 3 videos/01.mp4
```

Analysis stops after `video.max_duration_sec` seconds (default: 3 minutes) or when the file ends.
`video.frame_stride` (or `--stride N`) sets how many frames are skipped: `1` analyses everything,
higher values are proportionally faster. Screenshot timestamps stay in video time.

### Analyze every video in a folder (batch)

```bash
# All videos in ./videos, one process per file, headless (no preview window):
./batch-video-analyzer.sh

# Only *.mp4, including sub-folders, keep the preview window, retry a failed file twice:
./batch-video-analyzer.sh --pattern '*.mp4' --recursive --display --retry 2

# Resume: files already analyzed by a previous batch run are skipped
./batch-video-analyzer.sh --skip-done        # --reset-done clears that record

# Maximum speed: every 5th frame (overrides video.frame_stride from config.yaml)
./batch-video-analyzer.sh --stride 5

# See what would run, without running anything:
./batch-video-analyzer.sh --list
./batch-video-analyzer.sh --dry-run
```

Each file is analyzed by its own `start-video-analyzer.sh` process, so a broken file is reported and
the batch continues (`--stop-on-error` to stop instead). At the end a summary line prints
`ok/failed/skipped` and how many screenshots were newly saved; the exit code is `1` if any file failed.

| Option | Meaning |
|--------|---------|
| `--videos-dir DIR` | Folder to scan (default: `videos`) |
| `--pattern GLOB` | File name pattern, e.g. `*.mp4` (default: mp4/mov/mkv/avi/m4v/mpg/mpeg/ts/webm) |
| `--recursive` | Also scan sub-folders |
| `--display` | Show the preview window for each file (default is `--no-display`) |
| `--config FILE` | Passed to every run, same as in the single-file scripts |
| `--stride N` | Analyse every Nth frame, overriding `video.frame_stride` |
| `--skip-done` / `--reset-done` | Skip files already processed (`logs/batch-done/` markers) / clear them |
| `--retry N` | Retry a failing file `N` times |
| `--stop-on-error` | Abort the batch on the first failure |
| `--list` / `--dry-run` | Print the file list / the commands, run nothing |

Everything after `--` is passed to `video-analyzer.py` as-is. To use a different option set per run,
write it into a `--config` override file instead.

Your could use video files from tests fixture [folder](tests/fixtures) to check how it works (they used by [video analyzer tests](tests/test_video_analyzer.py))

## How it works

1. Frames are passed through YOLO tracking (TensorRT engine, FP16). Only every
   `video.frame_stride`-th frame is analysed, so skipped frames are never run through the model.
2. Detections are filtered by per-class confidence thresholds.
3. A geometric filter drops low-confidence dog detections whose center falls inside a person box (common false positive: person wearing a
   hood).
4. The distance between each person center and each dog center, divided by the person's box height, is compared against `pairs.max_distance_ratio`. This keeps the rule the same for near and far pairs.
5. Pairs are tracked by tracker ID: a screenshot is saved when a pair starts and then every `pairs.snapshot_interval_sec` while it persists.
   A pair ends after `pairs.lost_timeout_sec` without being seen.

## Settings

All settings live in [`config.yaml`](config.yaml), with a comment for every value. To use different
values without editing it, pass an override file; only the keys it contains are replaced:

```bash
python stream-analyzer.py --config my.yaml rtsp://user:password@192.168.1.80:554/stream1
```

Main settings:

| Key                                      | Default        | Description                                                                |
|------------------------------------------|----------------|----------------------------------------------------------------------------|
| `pairs.max_distance_ratio`               | `1.0`          | Max centre distance in person heights (scale-invariant) to count as a pair |
| `detection.conf_person` / `conf_dog`     | `0.2` / `0.005` | Per-class confidence floor (dog is low on purpose, to see distant dogs)   |
| `video.max_duration_sec`                 | `1800`         | Video analysis time limit (seconds)                                       |
| `video.frame_stride`                     | `1`            | Analyse every Nth video frame; higher is faster, `1` is most accurate      |
| `model.imgsz`                            | `1280`         | Model input size, must match the exported engine                           |
| `model.device`                           | `cuda:0`       | Inference deivce. CPU (`cpu`) or GPU (`cuda:0`) could be chosen            |
| `model.tracker`                          | `bytetrack.yaml` | Tracker config; `botsort.yaml` only for a moving camera                  |
| `stream.timeout_ms`                      | `5000`         | RTSP connect / read timeout (ms)                                           |
| `stream.reconnect_min_delay_sec` / `max` | `1` / `30`     | Reconnect back-off range (seconds)                                         |

### Batch processing speed

Most of the time goes into the model, so the number of inferences per second of video decides the
batch duration. Two settings do most of the work:

| Change | Effect on a 1440p 15 fps recording (RTX 3060, `yolo26l` @ 1280) |
|--------|-----------------------------------------------------------------|
| `model.tracker: bytetrack.yaml` instead of BoT-SORT | 23.5 → 40.6 analysed fps (~1.7x) |
| `video.frame_stride: 3` on top of that | 40.6 → 117.8 video fps (~5x overall) |

The tracker gain comes from skipping BoT-SORT's per-frame camera-motion estimation, which is
pointless for a fixed camera. The stride gain comes from running the model on every third frame.
Running several files in parallel does **not** help: the GPU is already saturated (2 processes
gave only ~12% more total throughput while competing for the same GPU).

## Project structure

```
person-dog-detector/
├── models/
│   ├── yolo26l.pt              # Source weights (download separately)
│   └── yolo26l.engine          # TensorRT engine (generated by convert-model-to-engine.py)
├── screenshots/                # Saved event frames (created automatically)
├── logs/                       # Log files (created automatically)
├── pdd/                        # Shared code
│   ├── config.py               # config.yaml loading and overrides
│   ├── logs.py                 # Logging setup (rotating files)
│   ├── sources.py              # URL redaction, threaded RTSP capture
│   ├── screenshots.py          # ScreenshotSaver
│   ├── detection.py            # Detections, pairs, PairTracker, drawing
│   └── pipeline.py             # Startup, per-frame processing, display
├── tests/                      # pytest tests for pdd/
├── utils/
│   ├── convert-model-to-engine.py  # One-time TensorRT export
│   └── check-cuda.py           # Verify CUDA availability
├── config.yaml                 # All settings
├── stream-analyzer.py          # RTSP stream analyser
├── video-analyzer.py           # Video file analyser
├── start-stream-analyzer.sh    # Launcher (reads .env), Windows Git Bash / Linux / macOS
├── start-video-analyzer.sh     # Video file launcher
├── batch-video-analyzer.sh     # Runs the video launcher over a whole folder of videos
├── requirements.txt
├── requirements-dev.txt
├── .env-sample
└── README.md
```

## Screenshot naming

| Source     | Example                                            |
|------------|----------------------------------------------------|
| Stream     | `192_168_1_80_554_stream1_20261003_154356_123.jpg` |
| Video file | `01_7.7s.jpg`                                      |

## Logs

Each script writes to its own log file and to the console:

| Script               | Log file                   |
|----------------------|----------------------------|
| `stream-analyzer.py` | `logs/stream-analyzer.log` |
| `video-analyzer.py`  | `logs/video-analyzer.log`  |

The directory is set by `logging.dir` in `config.yaml` and is created automatically.

Both log files are git-ignored. Logs are rotated: each file is capped at 30 MB and the 3 most recent
backups are kept (`*.log.1`, `*.log.2`, `*.log.3`).

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The tests cover the `pdd` package (config, URL redaction, screenshot saving, detection filtering, pair tracking,
threaded stream capture) and need neither a GPU nor a camera.

## Appendixes

### Verify is CUDA installed properly

```bash
python utils/check-cuda.py
```

Prints the PyTorch version and whether CUDA is available.
