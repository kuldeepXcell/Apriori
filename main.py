from pathlib import Path
import sys

# Ensure the src/ directory is importable when running `streamlit run main.py`.
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from app.ui.app import run_app  # noqa: E402


if __name__ == "__main__":
    run_app()
