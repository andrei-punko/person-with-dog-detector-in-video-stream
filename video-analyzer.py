import sys

import cv2

from pdd.detection import PairTracker
from pdd.pipeline import handle_pair_events, load_model, process_frame, setup, show_frame, track_kwargs
from pdd.screenshots import ScreenshotSaver
from pdd.sources import redact_url, source_label

args, cfg, logger = setup("Detect persons and dogs in a video file.", "Path to the video file", "video-analyzer.log")

max_duration = cfg["video"]["max_duration_sec"]
logger.info(f"Analyzing file: {redact_url(args.source)}")

cap = cv2.VideoCapture(args.source)
if not cap.isOpened():
    logger.error(f"Could not open video file: {args.source}")
    sys.exit(1)

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()

max_frames = min(int(fps * max_duration), total_frames)
logger.info(f"FPS: {fps}, total frames: {total_frames}, analysing: {max_frames} frames (~{max_duration}s)")

screenshots = ScreenshotSaver.from_config(cfg, source_label(args.source))
pair_tracker = PairTracker.from_config(cfg)
logger.info(f"Screenshots will be saved to: {cfg['screenshots']['dir']}/")

# Load and warm up the TensorRT model before processing the file
model = load_model(cfg, logger)

# stream=True yields frames one by one without loading the whole video into memory
results = model.track(source=args.source, stream=True, **track_kwargs(cfg))
logger.info("Analysis started. Press 'Q' in the video window to stop.")


# --- Main loop ---

for frame_idx, result in enumerate(results):
    if frame_idx >= max_frames:
        logger.info(f"Limit reached: {max_duration}s, stopping.")
        break

    frame = result.orig_img.copy()
    time_sec = frame_idx / fps
    process_frame(frame, result.boxes, time_sec, f"{time_sec:.1f}s", cfg, pair_tracker, screenshots, logger,
                  log_suffix=f", time={time_sec:.1f}s")

    if not args.no_display and show_frame("Video Analysis", frame, cfg):
        logger.info("Analysis interrupted by user.")
        break

handle_pair_events(pair_tracker.finish(), None, 0, "", screenshots, logger)

cv2.destroyAllWindows()
logger.info("Analysis finished.")
