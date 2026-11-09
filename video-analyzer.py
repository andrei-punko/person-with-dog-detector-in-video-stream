import sys

import cv2

from pdd.detection import PairTracker
from pdd.pipeline import handle_pair_events, load_model, process_frame, setup, show_frame, track_kwargs
from pdd.screenshots import ScreenshotSaver
from pdd.sources import redact_url, source_label

args, cfg, logger = setup("Detect persons and dogs in a video file.", "Path to the video file", "video-analyzer.log",
                          add_stride=True)

max_duration = cfg["video"]["max_duration_sec"]
stride = args.stride if args.stride is not None else cfg["video"]["frame_stride"]
if stride < 1:
    logger.error(f"Frame stride must be 1 or more, got {stride}")
    sys.exit(2)
logger.info(f"Analyzing file: {redact_url(args.source)}")

cap = cv2.VideoCapture(args.source)
if not cap.isOpened():
    logger.error(f"Could not open video file: {args.source}")
    sys.exit(1)

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()
if fps <= 0:
    logger.error(f"Could not read FPS from video file: {args.source}")
    sys.exit(1)

max_frames = min(int(fps * max_duration), total_frames)
logger.info(f"FPS: {fps:.2f}, total frames: {total_frames}, analysing up to {max_frames} frames (~{max_duration}s), "
            f"every {stride} frame(s) = ~{fps / stride:.1f} analysed fps")

screenshots = ScreenshotSaver.from_config(cfg, source_label(args.source))
pair_tracker = PairTracker.from_config(cfg)
logger.info(f"Screenshots will be saved to: {cfg['screenshots']['dir']}/")

# Load and warm up the TensorRT model before processing the file
model = load_model(cfg, logger)

# stream=True yields frames one by one without loading the whole video into memory.
# vid_stride makes the loader skip frames in between, which is the main speed knob.
results = model.track(source=args.source, stream=True, vid_stride=stride, **track_kwargs(cfg))
logger.info("Analysis started. Press 'Q' in the video window to stop.")


# --- Main loop ---

for frame_idx, result in enumerate(results):
    # The loader yields the last frame of each group of `stride` frames, so the real video time is
    # about frame_idx * stride / fps; the sub-frame offset is irrelevant for timers and file names.
    if frame_idx * stride >= max_frames:
        logger.info(f"Limit reached: {max_duration}s, stopping.")
        break

    frame = result.orig_img
    time_sec = frame_idx * stride / fps
    process_frame(frame, result.boxes, time_sec, f"{time_sec:.1f}s", cfg, pair_tracker, screenshots, logger,
                  log_suffix=f", time={time_sec:.1f}s")

    if not args.no_display and show_frame("Video Analysis", frame, cfg):
        logger.info("Analysis interrupted by user.")
        break

handle_pair_events(pair_tracker.finish(), None, 0, "", screenshots, logger)

cv2.destroyAllWindows()
logger.info("Analysis finished.")
