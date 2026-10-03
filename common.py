import logging
import os
import re
import sys
import threading
import time
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


class ThreadedVideoCapture:
    """Read a stream in a background thread and always expose only the newest frame.

    The thread owns the connection: it opens the stream, reconnects with exponential
    back-off on failure, and drops old frames so slow inference never builds up lag.
    """

    def __init__(self, url, logger, timeout_ms=5000, min_delay=1, max_delay=30):
        self.url = url
        self.logger = logger
        self.timeout_ms = timeout_ms
        self.min_delay = min_delay
        self.max_delay = max_delay
        self._url_safe = redact_url(url)
        self._frame = None
        self._cond = threading.Condition()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()
        return self

    def _open(self):
        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG, [
            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self.timeout_ms,
            cv2.CAP_PROP_READ_TIMEOUT_MSEC, self.timeout_ms,
        ])
        if not cap.isOpened():
            cap.release()
            return None
        return cap

    def _run(self):
        delay = self.min_delay
        connected_before = False
        while not self._stop.is_set():
            cap = self._open()
            if cap is None:
                self.logger.warning(f"Could not connect to {self._url_safe}, retrying in {delay}s")
                self._stop.wait(delay)
                delay = min(delay * 2, self.max_delay)
                continue

            self.logger.info("Stream reconnected." if connected_before else "Stream connected.")
            connected_before = True
            delay = self.min_delay
            while not self._stop.is_set():
                ret, frame = cap.read()
                if not ret:
                    self.logger.warning("Stream lost or frame read error, reconnecting...")
                    break
                with self._cond:
                    self._frame = frame
                    self._cond.notify_all()
            cap.release()
            self._stop.wait(self.min_delay)

    def read(self, timeout=1.0):
        """Return the newest unconsumed frame, or None if none arrives within timeout seconds."""
        with self._cond:
            if self._frame is None:
                self._cond.wait(timeout)
            frame, self._frame = self._frame, None
            return frame

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=self.timeout_ms / 1000 + 1)
