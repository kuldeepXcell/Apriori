from typing import List, Dict
from app.configs.default_pipelines import get_pipeline_configs
from app.pipelines.base_pipeline import RetrievalPipeline

class PipelineRegistry:
    """
    Registry for managing multiple retrieval pipelines.
    """
    
    def __init__(self):
        self._pipelines: Dict[str, RetrievalPipeline] = {}
        self._load_default_pipelines()
    
    def _load_default_pipelines(self):
        """Load all default pipeline configurations."""
        configs = get_pipeline_configs()
        for config in configs:
            self._pipelines[config.name] = RetrievalPipeline(config)
    
    def get_all_pipelines(self) -> List[RetrievalPipeline]:
        """Returns all registered pipelines."""
        return list(self._pipelines.values())
    
    def get_pipeline_by_name(self, name: str) -> RetrievalPipeline:
        """Get a specific pipeline by name."""
        if name not in self._pipelines:
            raise KeyError(f"Pipeline '{name}' not found")
        return self._pipelines[name]
    
    def get_pipeline_names(self) -> List[str]:
        """Get names of all registered pipelines."""
        return list(self._pipelines.keys())

# Global registry instance
pipeline_registry = PipelineRegistry()
