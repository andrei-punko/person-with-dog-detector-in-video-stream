import logging
import os

import cv2
import numpy as np

from pdd.screenshots import ScreenshotSaver

LOGGER = logging.getLogger("tests")


def saver(tmp_path, label="x", min_interval=0.0, jpeg_quality=65, max_width=1920):
    return ScreenshotSaver(str(tmp_path), label, min_interval, jpeg_quality, max_width)


def blank_frame(w=640, h=360):
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_respects_min_interval(tmp_path):
    s = saver(tmp_path, min_interval=1.0)
    assert s.save(blank_frame(), 0.0, "a", LOGGER) is not None
    assert s.save(blank_frame(), 0.5, "b", LOGGER) is None
    assert s.save(blank_frame(), 1.0, "c", LOGGER) is not None
    assert sorted(os.listdir(tmp_path)) == ["x_a.jpg", "x_c.jpg"]


def test_downscales_wide_frames(tmp_path):
    path = saver(tmp_path, max_width=320).save(blank_frame(640, 360), 0.0, "a", LOGGER)
    assert cv2.imread(path).shape[:2] == (180, 320)


def test_keeps_narrow_frames(tmp_path):
    path = saver(tmp_path).save(blank_frame(640, 360), 0.0, "a", LOGGER)
    assert cv2.imread(path).shape[:2] == (360, 640)


def test_zero_max_width_disables_scaling(tmp_path):
    path = saver(tmp_path, max_width=0).save(blank_frame(640, 360), 0.0, "a", LOGGER)
    assert cv2.imread(path).shape[:2] == (360, 640)


def test_lower_quality_gives_smaller_file(tmp_path):
    noisy = np.random.default_rng(0).integers(0, 255, (360, 640, 3), dtype=np.uint8)
    hi = saver(tmp_path, "hi", jpeg_quality=95).save(noisy, 0.0, "a", LOGGER)
    lo = saver(tmp_path, "lo", jpeg_quality=40).save(noisy, 0.0, "a", LOGGER)
    assert os.path.getsize(lo) < os.path.getsize(hi)


def test_from_config(tmp_path):
    cfg = {"screenshots": {"dir": str(tmp_path / "s"), "min_interval_sec": 1, "jpeg_quality": 50, "max_width": 100}}
    s = ScreenshotSaver.from_config(cfg, "lbl")
    assert (s.label, s.min_interval, s.jpeg_quality, s.max_width) == ("lbl", 1, 50, 100)
    assert os.path.isdir(tmp_path / "s")
