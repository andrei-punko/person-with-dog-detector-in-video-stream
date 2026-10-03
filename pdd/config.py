"""Loading of the YAML settings shared by both analyzer scripts."""
import os

import yaml

# config.yaml lives in the project root, one level above this package
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")


def _merge(base, override):
    """Recursively merge override into base and return base."""
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value
    return base


def _read(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(path=None):
    """Load the default config.yaml, optionally overridden by the keys present in the file at path."""
    config = _read(DEFAULT_CONFIG_PATH)
    if path:
        _merge(config, _read(path))
    return config
