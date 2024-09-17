from pathlib import Path

__all__ = [
    "BASE_DIR",
    "SIAPY_DIR",
    "TEST_DATA_DIR",
]

BASE_DIR = Path(__file__).parent.parent.parent.absolute()
SIAPY_DIR = Path(BASE_DIR, "siapy")
TEST_DATA_DIR = Path(BASE_DIR, "tests", "data")
