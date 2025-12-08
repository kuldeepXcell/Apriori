"""Centralized project paths for loaders and runtimes."""

from pathlib import Path

# /home/.../Apriori/src/app/config/paths.py -> parents[3] == project root
BASE_DIR = Path(__file__).resolve().parents[3]
CONFIG_DIR = BASE_DIR / "config"
PIPELINES_DIR = CONFIG_DIR / "pipelines"
SCHEMAS_DIR = CONFIG_DIR / "schemas"
DATA_DIR = BASE_DIR / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
STATIC_DIR = BASE_DIR / "static"


def ensure_directories() -> None:
    """Create expected directories if they do not exist."""
    for path in (
        CONFIG_DIR,
        PIPELINES_DIR,
        SCHEMAS_DIR,
        DATA_DIR,
        PROCESSED_DATA_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


