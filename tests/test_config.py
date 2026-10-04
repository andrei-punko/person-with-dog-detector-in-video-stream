from pdd.config import load_config


def test_default_config_has_all_sections():
    cfg = load_config()
    for section in ("model", "detection", "pairs", "screenshots", "logging", "stream", "video", "display"):
        assert section in cfg


def test_default_dog_threshold_stays_low():
    assert load_config()["detection"]["conf_dog"] <= 0.02


def test_override_replaces_only_given_keys(tmp_path):
    override = tmp_path / "my.yaml"
    override.write_text("pairs:\n  distance_threshold: 250\n", encoding="utf-8")
    cfg = load_config(str(override))
    assert cfg["pairs"]["distance_threshold"] == 250
    assert cfg["pairs"]["lost_timeout_sec"] == load_config()["pairs"]["lost_timeout_sec"]


def test_empty_override_file_keeps_defaults(tmp_path):
    override = tmp_path / "empty.yaml"
    override.write_text("", encoding="utf-8")
    assert load_config(str(override)) == load_config()
