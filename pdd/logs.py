"""Logging setup and logging helpers."""
import logging
import os
import sys
from logging.handlers import RotatingFileHandler


def setup_logging(log_file, max_bytes, backup_count):
    """Configure logging to write to a rotating file and the console with a shared format."""
    os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count, encoding='utf-8'),
            logging.StreamHandler(sys.stdout),
        ]
    )
    return logging.getLogger()


def log_time(logger, res):
    """Log inference timing and effective FPS for a YOLO result (debugging aid)."""
    preprocess_speed = res.speed.get('preprocess', 0.0)
    inference_speed = res.speed.get('inference', 0.0)
    postprocess_speed = res.speed.get('postprocess', 0.0)
    total_speed = preprocess_speed + inference_speed + postprocess_speed
    fps_hardware = 1000 / total_speed if total_speed > 0 else 0.0
    logger.info(f"SPEED: Inference: {inference_speed:.1f}ms | Total: {total_speed:.1f}ms ({fps_hardware:.1f} FPS)")
