"""
Configuration loader for QICNN.
"""

import yaml
from pathlib import Path


def load_config(path: str = "configs/config.yaml") -> dict:
    """Load YAML configuration and return as a nested dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def merge_configs(base: dict, override: dict) -> dict:
    """Deep-merge *override* into *base* (modifies base in-place)."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            merge_configs(base[key], value)
        else:
            base[key] = value
    return base
