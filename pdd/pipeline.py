"""Code shared by stream-analyzer.py and video-analyzer.py: startup, per-frame processing, display."""
import argparse
import os
import time

import cv2
import numpy as np

from pdd.config import load_config
from pdd.detection import (
    CLASS_DOG, CLASS_PERSON, class_thresholds, collect_detections, draw_detections, find_pairs,
)
from pdd.logs import setup_logging

QUIT_KEY = ord('q')


def setup(description, source_help, log_file):
    """Parse command-line arguments, load the config and configure logging.

    Returns (args, cfg, logger). args.source is the stream URL or video file path.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("source", help=source_help)
    parser.add_argument("--no-display", action="store_true", help="Disable the video window (for headless servers)")
    parser.add_argument("--config", help="YAML file overriding values from config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    log = cfg["logging"]
    logger = setup_logging(os.path.join(log["dir"], log_file), log["max_bytes"], log["backup_count"])
    return args, cfg, logger


def load_model(cfg, logger):
    """Load the TensorRT model and run one dummy inference so the first real frame is not slow."""
    from ultralytics import YOLO  # imported here: it is slow and not needed by tests

    model = YOLO(cfg["model"]["path"])
    logger.info("Warming up TensorRT model...")
    model.track(source=np.zeros((640, 640, 3), dtype=np.uint8), device=cfg["model"]["device"], verbose=False)
    return model


def track_kwargs(cfg):
    """Arguments for model.track() common to both scripts."""
    return {
        "show": False,
        "classes": [CLASS_PERSON, CLASS_DOG],
        # Ask the model for everything above the lowest threshold; per-class filtering happens later
        "conf": min(class_thresholds(cfg).values()),
        "imgsz": cfg["model"]["imgsz"],
        "device": cfg["model"]["device"],
        "verbose": False,
        # Keep tracker state between frames so IDs stay stable
        "persist": True,
    }


def format_timestamp(t):
    """Format a wall-clock time as YYYYMMDD_HHMMSS_mmm for screenshot names."""
    return f"{time.strftime('%Y%m%d_%H%M%S', time.localtime(t))}_{int(t * 1000) % 1000:03d}"


def handle_pair_events(events, frame, now, suffix, screenshots, logger, log_suffix=""):
    """Log pair events and save a screenshot for each start/ongoing event.

    suffix is the screenshot name suffix; log_suffix is appended to every log line.
    frame may be None when only "end" events are possible (no new frame available).
    """
    for event, (pkey, dkey), distance, duration in events:
        if event == "end":
            logger.info(f"Pair ended: {pkey} <-> {dkey}, duration={duration:.1f}s{log_suffix}")
            continue
        label = "Person with dog" if event == "start" else f"Person with dog (still together, {duration:.0f}s)"
        logger.info(f"{label}: {pkey} <-> {dkey}, distance={distance:.0f}px{log_suffix}")
        screenshots.save(frame, now, suffix, logger)


def process_frame(frame, boxes, now, suffix, cfg, tracker, screenshots, logger, log_suffix=""):
    """Detect persons and dogs, draw them on the frame, track pairs and report pair events."""
    persons, dogs = collect_detections(boxes, class_thresholds(cfg), cfg["detection"]["dog_inside_person_max_conf"])
    draw_detections(frame, persons, dogs)
    pairs = find_pairs(persons, dogs, cfg["pairs"]["distance_threshold"])
    handle_pair_events(tracker.update(pairs, now), frame, now, suffix, screenshots, logger, log_suffix)


def quit_pressed(delay_ms=1):
    """Poll the preview window for the quit key. Returns True if it was pressed."""
    return cv2.waitKey(delay_ms) & 0xFF == QUIT_KEY


def show_frame(title, frame, cfg):
    """Show the frame in a window, downscaled to fit the configured size. Returns True if quit was pressed."""
    d = cfg["display"]
    h, w = frame.shape[:2]
    scale = min(d["max_width"] / w, d["max_height"] / h, 1.0)
    shown = cv2.resize(frame, (int(w * scale), int(h * scale))) if scale < 1.0 else frame
    cv2.imshow(title, shown)
    return quit_pressed()
