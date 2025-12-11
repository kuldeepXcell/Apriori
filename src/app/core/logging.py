"""Shared logging setup with colored output and module tagging."""

from enum import StrEnum
import logging
from typing import Iterable

from rich.logging import RichHandler


class ModuleName(StrEnum):
    PIPELINE = "pipeline"
    RETRIEVAL = "retrieval"
    RERANK = "rerank"
    INGESTION = "ingestion"
    PREPROCESS = "preprocess"
    UI = "ui"
    ADAPTER = "adapter"
    STEPS = "steps"


class ModuleNameFilter(logging.Filter):
    """Ensure records have module_name; allow override via extra."""

    def __init__(self, default_module: str = "-") -> None:
        super().__init__()
        self.default_module = default_module

    def filter(self, record: logging.LogRecord) -> bool:  # type: ignore[override]
        if not hasattr(record, "module_name"):
            record.module_name = self.default_module
        return True


def setup_logging(level: str = "INFO") -> None:
    """Configure root logging once at process start."""
    handler = RichHandler(rich_tracebacks=True, markup=False)
    handler.addFilter(ModuleNameFilter())
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s | %(levelname)s | %(module_name)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[handler],
    )
    # Tame noisy third-party debug logs to keep app logs readable.
    noisy_loggers: Iterable[str] = (
        "httpx",
        "httpcore",
        "openai",
        "qdrant_client",
        "watchdog",
        "PIL",
        "PIL.PngImagePlugin",
        "urllib3",
        "streamlit.runtime",
    )
    for noisy in noisy_loggers:
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a logger with the configured formatting."""
    return logging.getLogger(name)

