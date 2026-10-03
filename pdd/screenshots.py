"""Saving event screenshots."""
import os

import cv2


class ScreenshotSaver:
    """Save frames to <directory>/<label>_<suffix>.jpg, at most once per min_interval seconds."""

    def __init__(self, directory, label, min_interval, jpeg_quality, max_width):
        self.directory = directory
        self.label = label
        self.min_interval = min_interval
        self.jpeg_quality = jpeg_quality
        self.max_width = max_width
        self.last_time = None
        os.makedirs(directory, exist_ok=True)

    @classmethod
    def from_config(cls, cfg, label):
        """Create a saver from the "screenshots" section of the config."""
        s = cfg["screenshots"]
        return cls(s["dir"], label, s["min_interval_sec"], s["jpeg_quality"], s["max_width"])

    def save(self, frame, timestamp, suffix, logger):
        """Save frame if min_interval has passed since the last save. Returns the path or None."""
        if self.last_time is not None and timestamp - self.last_time < self.min_interval:
            return None
        path = os.path.join(self.directory, f"{self.label}_{suffix}.jpg")
        h, w = frame.shape[:2]
        if self.max_width and w > self.max_width:
            frame = cv2.resize(frame, (self.max_width, round(h * self.max_width / w)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        self.last_time = timestamp
        logger.info(f"  Screenshot saved: {path}")
        return path
