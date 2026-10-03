import logging
import math
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


# A dog detection below this confidence whose centre lies inside a person box is
# treated as a false positive (the common "hood as dog" mistake).
DOG_INSIDE_PERSON_MAX_CONF = 0.15


def draw_bounding_box(frame, x1, y1, x2, y2, cls, conf):
    """Draw a bounding box and class label on the frame (COCO class 0 = person, otherwise dog)."""
    label = "person" if cls == 0 else "dog"
    color = (0, 255, 0) if cls == 0 else (0, 0, 255)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def collect_detections(boxes, conf_thresholds, default_conf=0.25):
    """Split YOLO boxes into persons and dogs, applying per-class thresholds and the false-positive filter.

    Returns two lists of {"id": int | None, "coords": (x1, y1, x2, y2), "center": (cx, cy), "conf": float}.
    "id" is the tracker ID and is None when the tracker has not assigned one yet.
    """
    persons = []
    dogs = []
    if boxes is None or len(boxes) == 0:
        return persons, dogs

    for box in boxes:
        cls = int(box.cls)
        conf = float(box.conf)
        if conf < conf_thresholds.get(cls, default_conf):
            continue
        x1, y1, x2, y2 = box.xyxy[0].int().tolist()
        track_id = int(box.id) if getattr(box, "id", None) is not None else None
        det = {
            "id": track_id,
            "coords": (x1, y1, x2, y2),
            "center": ((x1 + x2) / 2, (y1 + y2) / 2),
            "conf": conf,
        }
        if cls == 0:
            persons.append(det)
        elif cls == 16:
            dogs.append(det)

    def inside_person(dog):
        cx, cy = dog["center"]
        return any(
            px1 <= cx <= px2 and py1 <= cy <= py2
            for px1, py1, px2, py2 in (p["coords"] for p in persons)
        )

    dogs = [d for d in dogs if not (d["conf"] < DOG_INSIDE_PERSON_MAX_CONF and inside_person(d))]
    return persons, dogs


def find_pairs(persons, dogs, distance_threshold):
    """Return {(person_key, dog_key): distance} for every person/dog pair closer than the threshold.

    Keys are tracker IDs ("p3", "d7"); a detection without an ID falls back to its list index ("p#0").
    """
    pairs = {}
    for i, p in enumerate(persons):
        pkey = f"p{p['id']}" if p["id"] is not None else f"p#{i}"
        for j, d in enumerate(dogs):
            dkey = f"d{d['id']}" if d["id"] is not None else f"d#{j}"
            distance = math.dist(p["center"], d["center"])
            if distance < distance_threshold:
                pairs[(pkey, dkey)] = distance
    return pairs


class PairTracker:
    """Turn per-frame person/dog pairs into events: a pair starts, is still going, or ends.

    A pair is considered ended when it has not been seen for lost_timeout seconds, so a few
    missed detections do not split one encounter into several. While a pair persists, an
    "ongoing" event is emitted every snapshot_interval seconds (e.g. to save another screenshot).
    All times are in seconds on whatever clock the caller uses (wall clock or video time).
    """

    def __init__(self, lost_timeout=2.0, snapshot_interval=5.0):
        self.lost_timeout = lost_timeout
        self.snapshot_interval = snapshot_interval
        self.active = {}

    def update(self, pairs, now):
        """Return a list of (event, key, distance, duration) tuples; event is "start", "ongoing" or "end"."""
        events = []
        for key, distance in pairs.items():
            state = self.active.get(key)
            if state is None:
                self.active[key] = {"start": now, "last_seen": now, "last_snap": now}
                events.append(("start", key, distance, 0.0))
                continue
            state["last_seen"] = now
            if now - state["last_snap"] >= self.snapshot_interval:
                state["last_snap"] = now
                events.append(("ongoing", key, distance, now - state["start"]))

        for key in [k for k, s in self.active.items() if k not in pairs and now - s["last_seen"] >= self.lost_timeout]:
            state = self.active.pop(key)
            events.append(("end", key, None, state["last_seen"] - state["start"]))
        return events

    def finish(self):
        """End all active pairs (call when the input is over)."""
        events = [("end", key, None, s["last_seen"] - s["start"]) for key, s in self.active.items()]
        self.active.clear()
        return events
