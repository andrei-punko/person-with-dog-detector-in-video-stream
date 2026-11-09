#!/bin/bash
# Run start-video-analyzer.sh once for every video file in a folder (default: videos/).
# Each file is analysed in its own Python process, so a broken file never stops the batch.
#
# Output of every run still goes to logs/video-analyzer.log, screenshots to screenshots/.

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR" || exit 1

VIDEOS_DIR=videos
PATTERN=""
RECURSIVE=0
HEADLESS=1
CONFIG=""
STRIDE=""
SKIP_DONE=0
RESET_DONE=0
STOP_ON_ERROR=0
RETRY=0
DRY_RUN=0
LIST_ONLY=0
EXTRA=()
DONE_DIR=logs/batch-done
VIDEO_EXTS="mp4 mov mkv avi m4v mpg mpeg ts webm"

usage() {
    cat <<EOF
Usage: $(basename "$0") [options] [-- extra args passed to video-analyzer.py]

Batch-analyze every video file in a folder with start-video-analyzer.sh.

Options:
  --videos-dir DIR   Folder to scan (default: videos)
  --pattern GLOB     File name pattern, e.g. '*.mp4' (default: any known video extension)
  --recursive        Scan sub-folders too
  --display          Show the preview window for every file (default: --no-display)
  --config FILE      Passed through to every run as --config FILE
  --stride N         Analyse every Nth frame (default: video.frame_stride in config.yaml)
  --skip-done        Skip files already processed by a previous batch (markers in $DONE_DIR/)
  --reset-done       Delete those markers first, i.e. analyze everything again
  --retry N          Retry a failed file N times (default: 0)
  --stop-on-error    Stop the batch after the first failure
  --list             Only list the files that would be processed
  --dry-run          Print the commands instead of running them
  -h, --help         This help

Examples:
  $(basename "$0")
  $(basename "$0") --display
  $(basename "$0") --skip-done --config my.yaml
  $(basename "$0") --videos-dir tests/fixtures --recursive --pattern '*.mp4'
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --videos-dir) VIDEOS_DIR="${2:-}"; shift 2 ;;
        --pattern) PATTERN="${2:-}"; shift 2 ;;
        --recursive) RECURSIVE=1; shift ;;
        --display) HEADLESS=0; shift ;;
        --config) CONFIG="${2:-}"; shift 2 ;;
        --stride) STRIDE="${2:-}"; shift 2 ;;
        --skip-done) SKIP_DONE=1; shift ;;
        --reset-done) RESET_DONE=1; shift ;;
        --retry) RETRY="${2:-0}"; shift 2 ;;
        --stop-on-error) STOP_ON_ERROR=1; shift ;;
        --list) LIST_ONLY=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h | --help) usage; exit 0 ;;
        --) shift; EXTRA=("$@"); break ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
    esac
done

if [ ! -d "$VIDEOS_DIR" ]; then
    echo "Error: videos folder not found: $VIDEOS_DIR" >&2
    echo "Create it and put video files there, or pass --videos-dir DIR." >&2
    exit 1
fi
if [ ! -f start-video-analyzer.sh ]; then
    echo "Error: start-video-analyzer.sh not found in $ROOT_DIR" >&2
    exit 1
fi

# Activate the venv once, so the config probe below and the child scripts both use it.
if [ -f venv-gpu/Scripts/activate ]; then
    source venv-gpu/Scripts/activate
elif [ -f venv-gpu/bin/activate ]; then
    source venv-gpu/bin/activate
else
    echo "Warning: virtual environment venv-gpu not found, using the system Python." >&2
fi

if [ "$RESET_DONE" -eq 1 ]; then
    rm -rf "$DONE_DIR"
    echo "Cleared processed-file markers in $DONE_DIR/"
fi

# --- File list ---

find_args=("$VIDEOS_DIR")
[ "$RECURSIVE" -eq 0 ] && find_args+=(-maxdepth 1)
find_args+=(-type f ! -name '.*')

if [ -n "$PATTERN" ]; then
    find_args+=(-iname "$PATTERN")
else
    find_args+=('(')
    first=1
    for ext in $VIDEO_EXTS; do
        [ "$first" -eq 1 ] || find_args+=(-o)
        find_args+=(-iname "*.$ext")
        first=0
    done
    find_args+=(')')
fi

files=()
while IFS= read -r -d '' f; do files+=("$f"); done < <(find "${find_args[@]}" -print0 2>/dev/null | sort -z)

if [ "${#files[@]}" -eq 0 ]; then
    echo "No video files found in $VIDEOS_DIR/"
    exit 0
fi

# Screenshots folder as the analyzers will see it (config.yaml plus --config overrides).
SCREENSHOTS_DIR=$(python -c "from pdd.config import load_config; print(load_config('${CONFIG}' or None)['screenshots']['dir'])" 2>/dev/null)
[ -z "$SCREENSHOTS_DIR" ] && SCREENSHOTS_DIR=screenshots

echo "Found ${#files[@]} video file(s) in $VIDEOS_DIR/"
echo "Screenshots: $SCREENSHOTS_DIR/    Log: logs/video-analyzer.log"
echo

count_screenshots() {
    # Screenshots of this source: file names start with pdd.sources.source_label of the file.
    local n
    n=$(find "$SCREENSHOTS_DIR" -maxdepth 1 -type f -iname "${1}_*.jpg" 2>/dev/null | wc -l)
    printf '%s' "${n//[![:digit:]]/}"
}

total=0 ok=0 failed=0 skipped=0 new_shots=0
batch_start=$(date +%s)

for f in "${files[@]}"; do
    total=$((total + 1))
    base=$(basename "$f")
    stem=${base%.*}
    rel=${f#"$VIDEOS_DIR"/}

    # Label used in screenshot names: non-alphanumeric characters become underscores.
    label=$(printf '%s' "$stem" | sed -E 's/[^A-Za-z0-9]+/_/g; s/^_+//; s/_+$//')
    [ -z "$label" ] && label=source
    marker="$DONE_DIR/$(printf '%s' "$rel" | sed -E 's#[^A-Za-z0-9._-]+#_#g')"

    if [ ! -s "$f" ]; then
        echo "[$total/${#files[@]}] SKIP, empty file: $rel"
        skipped=$((skipped + 1))
        continue
    fi
    if [ "$SKIP_DONE" -eq 1 ] && [ -f "$marker" ]; then
        echo "[$total/${#files[@]}] SKIP, processed before: $rel"
        skipped=$((skipped + 1))
        continue
    fi
    if [ "$LIST_ONLY" -eq 1 ]; then
        echo "[$total/${#files[@]}] $rel"
        continue
    fi

    cmd=(bash "$ROOT_DIR/start-video-analyzer.sh" "$f")
    [ "$HEADLESS" -eq 1 ] && cmd+=(--no-display)
    [ -n "$CONFIG" ] && cmd+=(--config "$CONFIG")
    [ -n "$STRIDE" ] && cmd+=(--stride "$STRIDE")
    cmd+=(${EXTRA[@]+"${EXTRA[@]}"})

    echo "[$total/${#files[@]}] $rel"
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "  ${cmd[*]}"
        continue
    fi

    before=$(count_screenshots "$label")
    start=$(date +%s)
    attempt=0
    while : ; do
        "${cmd[@]}"
        code=$?
        attempt=$((attempt + 1))
        [ $code -eq 0 ] || [ $attempt -gt "$RETRY" ] && break
        echo "  Exit code $code, retry $attempt/$RETRY..."
    done
    elapsed=$(( $(date +%s) - start ))
    after=$(count_screenshots "$label")
    delta=$((after - before))
    [ $delta -lt 0 ] && delta=0

    if [ $code -eq 0 ]; then
        mkdir -p "$DONE_DIR"
        printf 'done at %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" > "$marker"
        ok=$((ok + 1))
        echo "  OK in ${elapsed}s, screenshots: $after (new: $delta)"
    else
        failed=$((failed + 1))
        echo "  FAILED (exit $code) in ${elapsed}s - see logs/video-analyzer.log"
        [ "$STOP_ON_ERROR" -eq 1 ] && { echo "Stopping the batch (--stop-on-error)."; break; }
    fi
    new_shots=$((new_shots + delta))
    echo
done

echo "Batch finished in $(( $(date +%s) - batch_start ))s: ok=$ok failed=$failed skipped=$skipped of ${#files[@]} file(s), new screenshots=$new_shots"
[ "$failed" -gt 0 ] && exit 1
exit 0
