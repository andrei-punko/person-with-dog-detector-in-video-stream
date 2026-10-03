import logging
import time

import numpy as np
import pytest

from pdd import sources
from pdd.sources import ThreadedVideoCapture, redact_url, source_label

LOGGER = logging.getLogger("tests")


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
    monkeypatch.setattr(sources.cv2, "VideoCapture", FakeVideoCapture)
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


def test_capture_from_config():
    cfg = {"stream": {"timeout_ms": 1, "reconnect_min_delay_sec": 2, "reconnect_max_delay_sec": 3}}
    cap = ThreadedVideoCapture.from_config("rtsp://h/s", LOGGER, cfg)
    assert (cap.timeout_ms, cap.min_delay, cap.max_delay) == (1, 2, 3)
