import logging
from logging.handlers import RotatingFileHandler
from types import SimpleNamespace

from pdd.logs import log_time, setup_logging


def test_setup_logging_uses_rotating_file_handler(tmp_path, monkeypatch):
    root = logging.getLogger()
    monkeypatch.setattr(root, "handlers", [])
    monkeypatch.setattr(root, "level", root.level)

    logger = setup_logging(str(tmp_path / "t.log"), max_bytes=1000, backup_count=2)

    file_handlers = [h for h in logger.handlers if isinstance(h, RotatingFileHandler)]
    assert len(file_handlers) == 1
    assert file_handlers[0].maxBytes == 1000
    assert file_handlers[0].backupCount == 2
    for h in logger.handlers:
        h.close()


def test_log_time_reports_fps(caplog):
    result = SimpleNamespace(speed={"preprocess": 1.0, "inference": 8.0, "postprocess": 1.0})
    with caplog.at_level(logging.INFO, logger="tests"):
        log_time(logging.getLogger("tests"), result)
    assert "100.0 FPS" in caplog.text
