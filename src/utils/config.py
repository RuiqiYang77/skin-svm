"""Read and write YAML configuration files for experiments.

Training saves the resolved configuration next to model outputs so each run can
be reproduced from its artifact directory.
"""

from pathlib import Path

import yaml


def load_config(config_path):
    """Load a YAML config file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_config(config, output_path):
    """Save the resolved config used by an experiment."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)
