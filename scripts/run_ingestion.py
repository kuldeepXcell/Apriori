import sys
import os
from pathlib import Path

# Add project root to python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from app.services.ingestion import ingestion_service

if __name__ == "__main__":
    ingestion_service.run_ingestion()
