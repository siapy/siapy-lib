"""Configuration constants and paths for SiaPy library.

This module defines important directory paths and configuration constants
used throughout the SiaPy library, including base directories and test data locations.
"""

from pathlib import Path

__all__ = [
    "BASE_DIR",
    "SIAPY_DIR",
    "TEST_DATA_DIR",
]

BASE_DIR = Path(__file__).parent.parent.parent.absolute()
SIAPY_DIR = Path(BASE_DIR, "siapy")
TEST_DATA_DIR = Path(BASE_DIR, "tests", "data")
