from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, Optional

import yaml

from app.configs.pipeline_config import PipelineConfig


class PipelineRegistry:
    """Loads pipeline configurations from YAML files and exposes them by name."""

    def __init__(self, config_dir: Optional[Path] = None):
        self.config_dir = config_dir or Path(__file__).resolve().parents[2] / "pipelines"
        self._pipelines: Dict[str, PipelineConfig] = {}
        self._load_all()

    def _load_all(self) -> None:
        if not self.config_dir.exists():
            msg = f"Pipeline config directory not found: {self.config_dir}"
            raise FileNotFoundError(msg)

        for file_path in sorted(self.config_dir.glob("*.yaml")):
            with file_path.open("r", encoding="utf-8") as fp:
                raw = yaml.safe_load(fp) or {}
            config = PipelineConfig.model_validate(raw)
            self._pipelines[config.name] = config

        if not self._pipelines:
            raise RuntimeError(f"No pipeline configs found in {self.config_dir}")

    def get_pipeline(self, name: Optional[str] = None) -> PipelineConfig:
        """Return a specific pipeline, defaulting to DEFAULT_PIPELINE env value."""
        target = name or os.getenv("DEFAULT_PIPELINE")
        if target is None:
            raise ValueError(
                "Pipeline name not specified and DEFAULT_PIPELINE is not set."
            )
        try:
            return self._pipelines[target]
        except KeyError as exc:
            raise KeyError(f"Pipeline '{target}' is not registered.") from exc

    def list_pipelines(self) -> Iterable[PipelineConfig]:
        return self._pipelines.values()


pipeline_registry = PipelineRegistry()
