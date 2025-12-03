import argparse
import sys
from pathlib import Path

# Add project root to python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from app.services.ingestion import ingestion_service
from app.configs.default_pipelines import get_pipeline_configs

def parse_indicator_range(range_arg: str) -> tuple[int, int]:
    """
    Parses a 1-indexed inclusive range string (e.g., "5-10") into zero-based indices.
    Returns:
        Tuple of (start_idx, end_idx) suitable for slicing (end exclusive).
    """
    try:
        start_str, end_str = [part.strip() for part in range_arg.split("-", maxsplit=1)]
        start = int(start_str)
        end = int(end_str)
    except ValueError as exc:
        raise ValueError("Range must be in the form START-END, e.g., 5-10") from exc

    if start < 1 or end < start:
        raise ValueError("Range must be positive and START must be <= END.")

    # Convert to zero-based indexing with end exclusive
    return start - 1, end

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest financial indicators into Qdrant.")
    parser.add_argument(
        "--range",
        dest="index_range",
        type=str,
        help="1-indexed inclusive range of indicators to ingest (e.g., 5-10).",
    )
    args = parser.parse_args()

    indicator_range = None
    if args.index_range:
        indicator_range = parse_indicator_range(args.index_range)
        print(f"Configured indicator slice: {args.index_range} (1-indexed inclusive).")

    print("Loading pipeline configurations...")
    configs = get_pipeline_configs()
    print(f"Found {len(configs)} pipeline configurations:")
    for config in configs:
        print(f"  - {config.name}: {config.description}")
    
    print("\nStarting ingestion for all pipelines...")
    ingestion_service.run_ingestion(configs, index_range=indicator_range)
