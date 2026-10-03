import logging
import os
import time
from logging.handlers import RotatingFileHandler

import cv2
import numpy as np
import pytest

import common
from common import (
    PairTracker, ScreenshotSaver, ThreadedVideoCapture, collect_detections,
    find_pairs, redact_url, source_label,
)

LOGGER = logging.getLogger("tests")
THRESHOLDS = {0: 0.2, 16: 0.02}


# --- redact_url / source_label ---

def test_redact_url_removes_credentials():
    assert redact_url("rtsp://user:secret@192.168.1.80:554/stream1") == "rtsp://***@192.168.1.80:554/stream1"


def test_redact_url_removes_query():
    assert "abc" not in redact_url("rtsp://host/stream?token=abc")


@pytest.mark.parametrize("source", ["videos/01.mp4", "rtsp://192.168.1.80:554/s"])
def test_redact_url_leaves_safe_sources_unchanged(source):
    assert redact_url(source) == source


def test_source_label_for_stream_has_no_credentials():
    label = source_label("rtsp://user:secret@192.168.1.80:554/stream1")
    assert label == "192_168_1_80_554_stream1"
    assert "secret" not in label and "user" not in label


def test_source_label_for_video_is_basename_without_extension():
    assert source_label("videos/My Clip.mp4") == "My_Clip"


def test_source_label_is_filesystem_safe():
    assert source_label("rtsp://host:1/a/b?c=d").replace("_", "").isalnum()


# --- ScreenshotSaver ---

def blank_frame(w=640, h=360):
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_screenshot_respects_min_interval(tmp_path):
    saver = ScreenshotSaver(str(tmp_path), "x", min_interval=1.0)
    assert saver.save(blank_frame(), 0.0, "a", LOGGER) is not None
    assert saver.save(blank_frame(), 0.5, "b", LOGGER) is None
    assert saver.save(blank_frame(), 1.0, "c", LOGGER) is not None
    assert sorted(os.listdir(tmp_path)) == ["x_a.jpg", "x_c.jpg"]


def test_screenshot_downscales_wide_frames(tmp_path):
    saver = ScreenshotSaver(str(tmp_path), "x", max_width=320)
    path = saver.save(blank_frame(640, 360), 0.0, "a", LOGGER)
    assert cv2.imread(path).shape[:2] == (180, 320)


def test_screenshot_keeps_narrow_frames(tmp_path):
    saver = ScreenshotSaver(str(tmp_path), "x", max_width=1920)
    path = saver.save(blank_frame(640, 360), 0.0, "a", LOGGER)
    assert cv2.imread(path).shape[:2] == (360, 640)


def test_screenshot_lower_quality_gives_smaller_file(tmp_path):
    noisy = np.random.default_rng(0).integers(0, 255, (360, 640, 3), dtype=np.uint8)
    hi = ScreenshotSaver(str(tmp_path), "hi", jpeg_quality=95).save(noisy, 0.0, "a", LOGGER)
    lo = ScreenshotSaver(str(tmp_path), "lo", jpeg_quality=40).save(noisy, 0.0, "a", LOGGER)
    assert os.path.getsize(lo) < os.path.getsize(hi)


# --- collect_detections ---

class _Coords:
    def __init__(self, values):
        self.values = values

    def int(self):
        return self

    def tolist(self):
        return list(self.values)


class FakeBox:
    """Minimal stand-in for an ultralytics box."""

    def __init__(self, cls, conf, xyxy, track_id=None):
        self.cls = cls
        self.conf = conf
        self.id = track_id
        self.xyxy = [_Coords(xyxy)]


def test_collect_splits_persons_and_dogs_with_ids():
    boxes = [
        FakeBox(0, 0.9, (0, 0, 100, 200), track_id=3),
        FakeBox(16, 0.5, (300, 100, 350, 150), track_id=7),
    ]
    persons, dogs = collect_detections(boxes, THRESHOLDS)
    assert [p["id"] for p in persons] == [3]
    assert [d["id"] for d in dogs] == [7]
    assert dogs[0]["center"] == (325.0, 125.0)


def test_collect_missing_track_id_is_none():
    persons, _ = collect_detections([FakeBox(0, 0.9, (0, 0, 10, 10))], THRESHOLDS)
    assert persons[0]["id"] is None


def test_collect_applies_per_class_thresholds():
    boxes = [
        FakeBox(0, 0.1, (0, 0, 10, 10)),          # person below 0.2
        FakeBox(16, 0.01, (100, 100, 120, 120)),  # dog below 0.02
        FakeBox(16, 0.03, (200, 200, 220, 220)),  # dog above 0.02
    ]
    persons, dogs = collect_detections(boxes, THRESHOLDS)
    assert persons == []
    assert len(dogs) == 1


@pytest.mark.parametrize("dog_conf, dog_box, expected_dogs", [
    (0.05, (40, 40, 60, 60), 0),        # low confidence, inside person -> "hood" false positive
    (0.5, (40, 40, 60, 60), 1),         # confident, inside person -> kept
    (0.05, (300, 100, 350, 150), 1),    # low confidence, outside person -> kept
])
def test_collect_hood_filter(dog_conf, dog_box, expected_dogs):
    boxes = [FakeBox(0, 0.9, (0, 0, 100, 200)), FakeBox(16, dog_conf, dog_box)]
    _, dogs = collect_detections(boxes, THRESHOLDS)
    assert len(dogs) == expected_dogs


@pytest.mark.parametrize("boxes", [None, []])
def test_collect_empty_boxes(boxes):
    assert collect_detections(boxes, THRESHOLDS) == ([], [])


# --- find_pairs ---

def det(track_id, cx, cy):
    return {"id": track_id, "center": (cx, cy)}


def test_find_pairs_uses_track_ids():
    pairs = find_pairs([det(3, 0, 0)], [det(7, 30, 40)], 100)
    assert list(pairs) == [("p3", "d7")]
    assert pairs[("p3", "d7")] == pytest.approx(50.0)


def test_find_pairs_distance_equal_to_threshold_is_not_a_pair():
    assert find_pairs([det(1, 0, 0)], [det(2, 100, 0)], 100) == {}


def test_find_pairs_far_apart():
    assert find_pairs([det(1, 0, 0)], [det(2, 500, 0)], 100) == {}


def test_find_pairs_falls_back_to_index_without_ids():
    pairs = find_pairs([det(None, 0, 0)], [det(None, 10, 0)], 100)
    assert list(pairs) == [("p#0", "d#0")]


def test_find_pairs_multiple():
    persons = [det(1, 0, 0), det(2, 1000, 0)]
    dogs = [det(5, 10, 0), det(6, 1010, 0)]
    assert set(find_pairs(persons, dogs, 100)) == {("p1", "d5"), ("p2", "d6")}


# --- PairTracker ---

KEY = ("p1", "d2")


def run(tracker, timeline):
    """timeline: list of (time, pair_present). Returns [(time, event)]."""
    out = []
    for now, present in timeline:
        pairs = {KEY: 50.0} if present else {}
        out += [(now, e[0]) for e in tracker.update(pairs, now)]
    return out


def test_tracker_starts_once_while_together():
    t = PairTracker(lost_timeout=2.0, snapshot_interval=5.0)
    assert run(t, [(0, True), (1, True), (2, True)]) == [(0, "start")]


def test_tracker_ongoing_every_snapshot_interval():
    t = PairTracker(lost_timeout=2.0, snapshot_interval=5.0)
    result = run(t, [(0, True), (4, True), (5, True), (9, True), (10, True)])
    assert result == [(0, "start"), (5, "ongoing"), (10, "ongoing")]


def test_tracker_ends_after_lost_timeout():
    t = PairTracker(lost_timeout=2.0, snapshot_interval=5.0)
    result = run(t, [(0, True), (1, True), (2, False), (2.9, False), (3.0, False)])
    assert result == [(0, "start"), (3.0, "end")]


def test_tracker_short_gap_does_not_split_pair():
    t = PairTracker(lost_timeout=2.0, snapshot_interval=5.0)
    assert run(t, [(0, True), (1, False), (1.5, False), (2, True)]) == [(0, "start")]


def test_tracker_end_reports_visible_duration():
    t = PairTracker(lost_timeout=2.0, snapshot_interval=5.0)
    t.update({KEY: 10.0}, 0)
    t.update({KEY: 10.0}, 4)
    assert t.update({}, 6.5) == [("end", KEY, None, 4)]


def test_tracker_new_start_after_end():
    t = PairTracker(lost_timeout=1.0, snapshot_interval=5.0)
    assert run(t, [(0, True), (1, False), (2, True)]) == [(0, "start"), (1, "end"), (2, "start")]


def test_tracker_finish_ends_active_pairs():
    t = PairTracker()
    t.update({KEY: 10.0}, 0)
    t.update({KEY: 10.0}, 3)
    assert t.finish() == [("end", KEY, None, 3)]
    assert t.finish() == []


# --- setup_logging ---

def test_setup_logging_uses_rotating_file_handler(tmp_path, monkeypatch):
    root = logging.getLogger()
    monkeypatch.setattr(root, "handlers", [])
    monkeypatch.setattr(root, "level", root.level)

    logger = common.setup_logging(str(tmp_path / "t.log"), max_bytes=1000, backup_count=2)

    file_handlers = [h for h in logger.handlers if isinstance(h, RotatingFileHandler)]
    assert len(file_handlers) == 1
    assert file_handlers[0].maxBytes == 1000
    assert file_handlers[0].backupCount == 2
    for h in logger.handlers:
        h.close()


# --- ThreadedVideoCapture ---

class FakeVideoCapture:
    """Scripted cv2.VideoCapture replacement.

    Each opened instance plays the next script from SCRIPTS: a list of frames where None means a
    read failure, or None for a connection that fails to open. With no scripts left, open fails.
    """

    SCRIPTS = []
    opened = 0

    def __init__(self, *args, **kwargs):
        type(self).opened += 1
        self.script = type(self).SCRIPTS.pop(0) if type(self).SCRIPTS else None

    def isOpened(self):
        return self.script is not None

    def read(self):
        if not self.script:
            time.sleep(0.01)
            return False, None
        item = self.script.pop(0)
        return (False, None) if item is None else (True, item)

    def release(self):
        pass


def frame_of(value):
    return np.full((4, 4, 3), value, dtype=np.uint8)


@pytest.fixture
def make_capture(monkeypatch):
    FakeVideoCapture.SCRIPTS = []
    FakeVideoCapture.opened = 0
    monkeypatch.setattr(common.cv2, "VideoCapture", FakeVideoCapture)
    created = []

    def factory(scripts=()):
        FakeVideoCapture.SCRIPTS = list(scripts)
        cap = ThreadedVideoCapture(
            "rtsp://user:secret@host/s", LOGGER, timeout_ms=100, min_delay=0.01, max_delay=0.02
        ).start()
        created.append(cap)
        return cap

    yield factory
    for cap in created:
        cap.stop()


def test_capture_returns_latest_frame_and_drops_old_ones(make_capture):
    cap = make_capture([[frame_of(1), frame_of(2), frame_of(3), None]])
    time.sleep(0.3)  # let the thread read everything while nobody consumes
    frame = cap.read(timeout=1.0)
    assert frame is not None
    assert int(frame[0, 0, 0]) == 3


def test_capture_frame_is_consumed_once(make_capture):
    cap = make_capture([[frame_of(1), None]])
    assert cap.read(timeout=1.0) is not None
    assert cap.read(timeout=0.1) is None


def test_capture_read_times_out_without_frames(make_capture):
    cap = make_capture()
    started = time.time()
    assert cap.read(timeout=0.2) is None
    assert time.time() - started < 1.0


def test_capture_reconnects_after_stream_loss(make_capture):
    cap = make_capture([[frame_of(1), None], None, [frame_of(9), None]])
    seen = []
    deadline = time.time() + 3
    while time.time() < deadline and 9 not in seen:
        frame = cap.read(timeout=0.2)
        if frame is not None:
            seen.append(int(frame[0, 0, 0]))
    assert 1 in seen and 9 in seen
    assert FakeVideoCapture.opened >= 3


def test_capture_stop_ends_thread(make_capture):
    cap = make_capture()
    cap.stop()
    assert not cap._thread.is_alive()


def test_capture_logs_do_not_contain_credentials(make_capture, caplog):
    with caplog.at_level(logging.INFO, logger="tests"):
        cap = make_capture()
        time.sleep(0.15)
        cap.stop()
    assert "Could not connect" in caplog.text
    assert "secret" not in caplog.text
