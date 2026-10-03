import logging
import os
import re
import sys
from urllib.parse import urlsplit

import cv2


def setup_logging(log_file):
    """Configure logging to write to both a file and the console with a shared format."""
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout),
        ]
    )
    return logging.getLogger()


def redact_url(source):
    """Return the source string with login, password and query parameters removed, safe for logging."""
    parts = urlsplit(source)
    has_credentials = parts.username is not None or parts.password is not None
    if not has_credentials and not parts.query:
        return source
    host = parts.hostname or ""
    if parts.port:
        host += f":{parts.port}"
    netloc = f"***@{host}" if has_credentials else parts.netloc
    return parts._replace(netloc=netloc, query="***" if parts.query else "").geturl()


def source_label(source):
    """Return a short filesystem-safe label for a source: video basename or stream host/port/path."""
    if "://" in source:
        parts = urlsplit(source)
        raw = f"{parts.hostname or 'stream'}_{parts.port or ''}_{parts.path}"
    else:
        raw = os.path.splitext(os.path.basename(source))[0]
    return re.sub(r"[^A-Za-z0-9]+", "_", raw).strip("_") or "source"


class ScreenshotSaver:
    """Save frames to <directory>/<label>_<suffix>.jpg, at most once per min_interval seconds."""

    def __init__(self, directory, label, min_interval=0.33):
        self.directory = directory
        self.label = label
        self.min_interval = min_interval
        self.last_time = None
        os.makedirs(directory, exist_ok=True)

    def save(self, frame, timestamp, suffix, logger):
        """Save frame if min_interval has passed since the last save. Returns the path or None."""
        if self.last_time is not None and timestamp - self.last_time < self.min_interval:
            return None
        path = os.path.join(self.directory, f"{self.label}_{suffix}.jpg")
        cv2.imwrite(path, frame)
        self.last_time = timestamp
        logger.info(f"  Screenshot saved: {path}")
        return path
