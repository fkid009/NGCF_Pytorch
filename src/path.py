from pathlib import Path

BASE_DIR = Path(__file__).parents[1]

SRC_PATH = BASE_DIR / "src"

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

