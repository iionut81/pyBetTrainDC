"""Central configuration loader.

Reads config.yaml once and exposes a flat module-level dict ``CFG``.
All scripts import from here instead of hardcoding thresholds.

Usage:
    from config import CFG
    min_prob = CFG["dc"]["min_probability"]
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"

_cache: Dict[str, Any] | None = None


def load_config(path: Path | str | None = None) -> Dict[str, Any]:
    """Load and cache the YAML config. Thread-safe enough for single-process use."""
    global _cache
    if _cache is not None and path is None:
        return _cache
    p = Path(path) if path else _CONFIG_PATH
    with open(p, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if path is None:
        _cache = cfg
    return cfg


CFG = load_config()