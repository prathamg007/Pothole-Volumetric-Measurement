from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

SERVER_ROOT = Path(__file__).resolve().parent.parent.parent


@lru_cache(maxsize=1)
def load_config(path: str | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else SERVER_ROOT / "config" / "config.yaml"
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def resolve_path(relative: str | Path) -> Path:
    p = Path(relative)
    return p if p.is_absolute() else SERVER_ROOT / p
