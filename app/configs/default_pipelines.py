from app.configs.pipeline_config import PipelineConfig

# Define default pipelines for experimentation
DEFAULT_PIPELINES = [
    PipelineConfig(
        name="baseline",
        description="Baseline: text-embedding-3-small + GPT-4o",
        embedding_model="text-embedding-3-small",
        embedding_dim=1536,
        llm_model="gpt-4o",
        collection_name="financial_indicators_baseline",
        search_limit=15,
        rerank_limit=5
    ),
    PipelineConfig(
        name="fast",
        description="Fast: text-embedding-ada-002 + GPT-3.5-turbo",
        embedding_model="text-embedding-ada-002",
        embedding_dim=1536,
        llm_model="gpt-3.5-turbo",
        collection_name="financial_indicators_fast",
        search_limit=15,
        rerank_limit=5
    ),
    # Add more pipeline configs here as needed
]

def get_pipeline_configs():
    """Returns all configured pipelines."""
    return DEFAULT_PIPELINES
