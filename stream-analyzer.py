import time

import cv2

from pdd.detection import PairTracker
from pdd.pipeline import (
    format_timestamp, handle_pair_events, load_model, process_frame, quit_pressed, setup, show_frame,
    track_kwargs,
)
from pdd.screenshots import ScreenshotSaver
from pdd.sources import ThreadedVideoCapture, redact_url, source_label

args, cfg, logger = setup("Detect persons and dogs in an RTSP stream.", "RTSP or other stream URL", "stream-analyzer.log")

screenshots = ScreenshotSaver.from_config(cfg, source_label(args.source))
pair_tracker = PairTracker.from_config(cfg)
logger.info(f"Stream source: {redact_url(args.source)}")

# Load and warm up the TensorRT model before opening the stream
model = load_model(cfg, logger)
tracking = track_kwargs(cfg)

cap = ThreadedVideoCapture.from_config(args.source, logger, cfg).start()
logger.info("Analysis started. Press 'Q' in the video window to stop.")


# --- Main loop ---

try:
    while True:
        frame = cap.read(timeout=1.0)
        if frame is None:
            # No frames (stream down): still let lost pairs expire
            handle_pair_events(pair_tracker.update({}, time.time()), None, time.time(), "", screenshots, logger)
            if not args.no_display and quit_pressed():
                logger.info("Analysis stopped by user.")
                break
            continue

        now = time.time()
        result = model.track(source=frame, **tracking)[0]
        # To log per-frame inference timing: from pdd.logs import log_time; log_time(logger, result)
        process_frame(frame, result.boxes, now, format_timestamp(now), cfg, pair_tracker, screenshots, logger)

        if not args.no_display and show_frame("Stream", frame, cfg):
            logger.info("Analysis stopped by user.")
            break

except KeyboardInterrupt:
    logger.info("Analysis interrupted by Ctrl+C.")
finally:
    handle_pair_events(pair_tracker.finish(), None, time.time(), "", screenshots, logger)
    cap.stop()
    if not args.no_display:
        cv2.destroyAllWindows()
