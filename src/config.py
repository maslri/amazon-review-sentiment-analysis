from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Local directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Kaggle
KAGGLE_INPUT = Path("/kaggle/input")

IS_KAGGLE = os.path.exists("/kaggle/input")
