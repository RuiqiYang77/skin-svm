"""Small filesystem and JSON helpers used across experiment scripts.

These functions keep artifact directory creation and JSON writing consistent
across training, prediction, and evaluation entry points.
"""

import json
from pathlib import Path


def ensure_dir(path):
    """Create a directory if needed and return it as a Path object."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data, path):
    """Save JSON output with stable indentation for experiment artifacts."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
