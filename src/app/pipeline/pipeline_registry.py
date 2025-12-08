"""Registry that loads pipeline configs and exposes Pipeline objects."""

from pathlib import Path
from typing import List

from app.config import paths
from app.config.loader import load_pipeline_configs
from app.pipeline.pipeline import Pipeline


class PipelineRegistry:
    def __init__(self, pipelines_dir: Path):
        self.pipelines_dir = pipelines_dir
        self._pipelines = self._load_pipelines()

    def _load_pipelines(self) -> List[Pipeline]:
        configs = load_pipeline_configs(self.pipelines_dir)
        return [Pipeline(config) for config in configs]

    def get_all_pipelines(self) -> List[Pipeline]:
        return self._pipelines


pipeline_registry = PipelineRegistry(paths.PIPELINES_DIR)


