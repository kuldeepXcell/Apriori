import sys
import os
from pathlib import Path

# Add project root to python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from app.services.ingestion import ingestion_service
from app.configs.default_pipelines import get_pipeline_configs

if __name__ == "__main__":
    print("Loading pipeline configurations...")
    configs = get_pipeline_configs()
    print(f"Found {len(configs)} pipeline configurations:")
    for config in configs:
        print(f"  - {config.name}: {config.description}")
    
    print("\nStarting ingestion for all pipelines...")
    ingestion_service.run_ingestion(configs)
