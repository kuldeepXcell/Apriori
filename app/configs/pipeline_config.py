from pydantic import BaseModel, Field
from typing import List, Callable, Optional

class PipelineConfig(BaseModel):
    """
    Configuration for a retrieval pipeline.
    Defines all the models and processing steps for a single experiment.
    """
    name: str = Field(..., description="Unique pipeline identifier")
    description: str = Field(default="", description="Human-readable description")
    
    # Embedding Configuration
    embedding_model: str = Field(..., description="OpenAI embedding model name")
    embedding_dim: int = Field(..., description="Embedding vector dimension")
    
    # LLM Configuration
    llm_model: str = Field(default="gpt-4o", description="LLM model for re-ranking")
    llm_temperature: float = Field(default=0.0, description="Temperature for LLM")
    
    # Qdrant Configuration
    collection_name: str = Field(..., description="Qdrant collection name for this pipeline")
    
    # Search Parameters
    search_limit: int = Field(default=15, description="Number of candidates to retrieve")
    rerank_limit: int = Field(default=5, description="Number of results after re-ranking")
    
    # Processing (function names as strings, resolved at runtime)
    preprocessing_steps: List[str] = Field(default_factory=list, description="Preprocessing function names")
    postprocessing_steps: List[str] = Field(default_factory=list, description="Postprocessing function names")
    
    class Config:
        from_attributes = True
