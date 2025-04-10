import hashlib
import logging
import os
import tarfile
from pathlib import Path

import requests  # type: ignore[import]

#################################################################
#                    TEST DATA CONFIGURATION                    #
#################################################################

DATA_VERSION = "testdata-v1"

#################################################################

base_url = "https://github.com/siapy/siapy-lib/releases/download"
data_dir = Path(__file__).parent / "data"
archive_name = f"{DATA_VERSION}.tar.gz"
archive_path = data_dir / archive_name
checksum_url = f"{base_url}/{DATA_VERSION}/{archive_name}.sha256"
archive_url = f"{base_url}/{DATA_VERSION}/{archive_name}"

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
os.makedirs(data_dir, exist_ok=True)


def calculate_checksum(file_path: Path) -> str:
    """Calculate SHA-256 checksum of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def download_file(url: str, save_path: Path) -> None:
    """Download a file from URL to the specified path."""
    logger.info(f"Downloading {url} to {save_path}")
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(save_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)


def extract_archive(archive_path: Path, extract_dir: Path) -> None:
    """Extract the downloaded archive to the specified directory."""
    logger.info(f"Extracting {archive_path} to {extract_dir}")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=extract_dir)


def verify_testdata_integrity() -> bool:
    """Ensure test data is available, downloading and extracting if necessary."""
    if data_dir.exists() and any(data_dir.iterdir()):
        try:
            response = requests.get(checksum_url)
            response.raise_for_status()
            expected_checksum = response.text.strip().split()[0]

            if archive_path.exists():
                current_checksum = calculate_checksum(archive_path)
                if current_checksum == expected_checksum:
                    logger.info("Test data is up-to-date")
                    extract_archive(archive_path, data_dir.parent)
                    logger.info("Test data successfully extracted")
                    return True
                else:
                    logger.info("Test data checksum mismatch, re-downloading...")
            else:
                logger.info("Archive not found, downloading...")
        except requests.RequestException:
            logger.warning("Could not verify remote checksum, using existing data")
            return False

    try:
        download_file(archive_url, archive_path)
        logger.info("Test data archive downloaded successfully")
        extract_archive(archive_path, data_dir.parent)
        logger.info("Test data successfully extracted")
        return True
    except requests.RequestException as e:
        logger.error(f"Failed to download test data: {e}")
        raise RuntimeError(f"Could not download test data: {e}")


if __name__ == "__main__":
    verify_testdata_integrity()
