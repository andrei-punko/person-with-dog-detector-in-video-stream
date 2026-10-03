import logging

import numpy as np
import pytest

from pdd.config import load_config
from pdd.detection import PairTracker
from pdd.pipeline import format_timestamp, handle_pair_events, process_frame, show_frame, track_kwargs
from tests.test_detection import FakeBox

LOGGER = logging.getLogger("tests")


class FakeScreenshots:
    def __init__(self):
        self.saved = []

    def save(self, frame, timestamp, suffix, logger):
        self.saved.append((timestamp, suffix))


def test_format_timestamp_shape():
    ts = format_timestamp(1_700_000_000.123)
    date, clock, ms = ts.split("_")
    assert len(date) == 8 and len(clock) == 6 and ms == "123"


def test_track_kwargs_uses_lowest_threshold_and_config():
    cfg = load_config()
    kw = track_kwargs(cfg)
    assert kw["conf"] == min(cfg["detection"]["conf_person"], cfg["detection"]["conf_dog"])
    assert kw["imgsz"] == cfg["model"]["imgsz"]
    assert kw["classes"] == [0, 16]
    assert kw["persist"] is True


def test_handle_pair_events_logs_and_saves(caplog):
    shots = FakeScreenshots()
    events = [("start", ("p1", "d2"), 42.0, 0.0), ("end", ("p3", "d4"), None, 5.0)]
    with caplog.at_level(logging.INFO, logger="tests"):
        handle_pair_events(events, object(), 1.5, "sfx", shots, LOGGER, log_suffix=", time=1.5s")
    assert shots.saved == [(1.5, "sfx")]
    assert "Person with dog: p1 <-> d2, distance=42px, time=1.5s" in caplog.text
    assert "Pair ended: p3 <-> d4, duration=5.0s, time=1.5s" in caplog.text


def test_handle_pair_events_end_needs_no_frame():
    shots = FakeScreenshots()
    handle_pair_events([("end", ("p1", "d2"), None, 1.0)], None, 0, "", shots, LOGGER)
    assert shots.saved == []


def test_process_frame_detects_pair_and_saves_screenshot():
    cfg = load_config()
    frame = np.zeros((300, 600, 3), dtype=np.uint8)
    boxes = [
        FakeBox(0, 0.9, (0, 0, 100, 200), track_id=1),
        FakeBox(16, 0.5, (110, 100, 170, 160), track_id=2),
    ]
    shots = FakeScreenshots()
    tracker = PairTracker.from_config(cfg)
    process_frame(frame, boxes, 10.0, "s", cfg, tracker, shots, LOGGER)
    assert shots.saved == [(10.0, "s")]
    assert frame.any()  # boxes were drawn


def test_process_frame_far_apart_does_nothing():
    cfg = load_config()
    frame = np.zeros((300, 900, 3), dtype=np.uint8)
    boxes = [
        FakeBox(0, 0.9, (0, 0, 100, 200), track_id=1),
        FakeBox(16, 0.5, (700, 100, 760, 160), track_id=2),
    ]
    shots = FakeScreenshots()
    process_frame(frame, boxes, 0.0, "s", cfg, PairTracker.from_config(cfg), shots, LOGGER)
    assert shots.saved == []


@pytest.mark.parametrize("w, h, expected", [
    (3840, 2160, (1920, 1080)),
    (1280, 720, (1280, 720)),
])
def test_show_frame_scales_down_only(monkeypatch, w, h, expected):
    import pdd.pipeline as pipeline
    shown = {}
    monkeypatch.setattr(pipeline.cv2, "imshow", lambda title, f: shown.update(shape=f.shape[:2]))
    monkeypatch.setattr(pipeline.cv2, "waitKey", lambda delay: ord("q"))
    cfg = load_config()
    assert show_frame("t", np.zeros((h, w, 3), dtype=np.uint8), cfg) is True
    assert shown["shape"] == (expected[1], expected[0])
