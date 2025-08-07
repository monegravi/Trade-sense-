import os
from typing import Any, Dict
import yaml


def load_config(path: str = None) -> Dict[str, Any]:
    cfg_path = path or os.getenv("CONFIG_PATH", "config/config.yaml")
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)