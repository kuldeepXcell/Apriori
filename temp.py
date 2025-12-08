import sys
from pathlib import Path

import yaml

# Ensure src/ is on PYTHONPATH so `app` imports resolve when run standalone.
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / "src"))

from app.adapters import (  # noqa: E402
    openai_embedding_adapter,
    qdrant_adapter,
    sparse_embedding_adapter,
)
from app.config import paths  # noqa: E402

raw = yaml.safe_load((paths.PIPELINES_DIR / "baseline_hybrid.yaml").read_text())
vector_store_cfg = raw["vector_store"]
collection = vector_store_cfg["collection_name"]

indicator_id = "test-1"
definition = "Gross Domestic Product measures the monetary value of final goods and services."
question = "What is GDP?"
application = "Used to compare economic output and growth between countries."
keywords = ["gdp", "growth", "economy", "output"]

dense_vectors = {
    "definition": openai_embedding_adapter.embed_one(definition),
    "question": openai_embedding_adapter.embed_one(question),
    "application": openai_embedding_adapter.embed_one(application),
}

sparse = next(sparse_embedding_adapter.encode_keywords(keywords))
sparse_vec = qdrant_adapter.make_sparse_vector(sparse.indices, sparse.values)

qdrant_adapter.ensure_named_collection(collection, vector_store_cfg, force_recreate=False)

point = qdrant_adapter.build_point(
    point_id=indicator_id,
    dense_vectors=dense_vectors,
    sparse_vector=sparse_vec,
    payload={
        "name": "GDP",
        "definition": definition,
        "keywords": keywords,
        "source": "manual-smoke-test",
    },
)
qdrant_adapter.upsert_points(collection, [point])
print("Upserted point:", indicator_id)