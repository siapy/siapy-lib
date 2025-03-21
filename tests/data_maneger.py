import hashlib
import logging
import os
import tarfile
from pathlib import Path
from typing import Optional

import requests


class TestDataManager:
    """Manages test data downloads and verification."""

    def __init__(
        self,
        base_url: str = "https://github.com/username/siapy-lib/releases/download",
        version: str = "testdata-v1",
        data_dir: Optional[str] = None,
    ):
        self.version = version
        self.base_url = base_url
        if data_dir is None:
            # Default to the tests/data directory relative to this file
            self.data_dir = Path(__file__).parent / "data"
        else:
            self.data_dir = Path(data_dir)

        self.archive_name = f"{version}.tar.gz"
        self.archive_path = self.data_dir.parent / self.archive_name
        self.checksum_url = f"{self.base_url}/{version}/{self.archive_name}.sha256"
        self.archive_url = f"{self.base_url}/{version}/{self.archive_name}"

        # Create data directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)

    def _calculate_checksum(self, file_path):
        """Calculate SHA-256 checksum of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def _download_file(self, url, save_path):
        """Download a file from URL to the specified path."""
        logging.info(f"Downloading {url} to {save_path}")
        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

    def _extract_archive(self):
        """Extract the downloaded archive to the data directory."""
        logging.info(f"Extracting {self.archive_path} to {self.data_dir.parent}")
        with tarfile.open(self.archive_path, "r:gz") as tar:
            tar.extractall(path=self.data_dir.parent)

    def ensure_data_available(self):
        """
        Ensure test data is available, downloading and extracting if necessary.

        Returns:
        --------
        bool
            True if data was downloaded or updated, False if it was already up-to-date
        """
        # Check if the data directory exists and has content
        if self.data_dir.exists() and any(self.data_dir.iterdir()):
            # Try to get the expected checksum
            try:
                response = requests.get(self.checksum_url)
                response.raise_for_status()
                expected_checksum = response.text.strip().split()[0]

                # If we have the archive, verify its checksum
                if self.archive_path.exists():
                    current_checksum = self._calculate_checksum(self.archive_path)
                    if current_checksum == expected_checksum:
                        logging.info("Test data is up-to-date")
                        return False
                    logging.info("Test data checksum mismatch, re-downloading")
                else:
                    logging.info("Archive not found but data directory exists")
                    return False
            except requests.RequestException:
                logging.warning("Could not verify remote checksum, using existing data")
                return False

        # Download the archive
        try:
            self._download_file(self.archive_url, self.archive_path)
            self._extract_archive()
            logging.info("Test data successfully downloaded and extracted")
            return True
        except requests.RequestException as e:
            logging.error(f"Failed to download test data: {e}")
            raise RuntimeError(f"Could not download test data: {e}")


if __name__ == "__main__":
    data_manager = TestDataManager(
        base_url="https://github.com/username/siapy-lib/releases/download",
        version="testdata-v1",
    )
    data_manager.ensure_data_available()
    print(data_manager.data_dir)
