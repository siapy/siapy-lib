import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest
import spectral as sp

from siapy.entities.images import SpectralImage
from siapy.utils.images import save_image
from tests.fixtures import spectral_images  # noqa: F401


@pytest.mark.manual
def test_save_image_manual(spectral_images):
    image = spectral_images.vnir_np
    with TemporaryDirectory() as save_path:
        save_path = Path(save_path, "test_image.hdr")
        save_image(image, save_path)
        assert save_path.exists()


def test_save_image():
    with TemporaryDirectory() as tmpdir:
        image = np.random.default_rng().random((100, 100))
        save_path = Path(tmpdir, "test_image.hdr")
        save_image(image, save_path)
        assert save_path.exists()


def test_overwrite_argument():
    with TemporaryDirectory() as tmpdir:
        image = np.random.default_rng().random((100, 100))
        save_path = Path(tmpdir, "test_image.hdr")
        save_image(image, save_path)
        with pytest.raises(sp.io.envi.EnviException):
            save_image(image, save_path, overwrite=False)


def test_metadata_argument():
    with TemporaryDirectory() as tmpdir:
        image = np.random.default_rng().random((100, 100))
        metadata = {"description": "test"}
        save_path = Path(tmpdir, "test_image.hdr")
        save_image(image, save_path, metadata=metadata)
        image_disc = SpectralImage.envi_open(hdr_path=save_path)
        assert image_disc.metadata["description"] == "test"


def test_dtype_argument():
    with TemporaryDirectory() as tmpdir:
        image = np.random.default_rng().random((100, 100))
        save_path = Path(tmpdir, "test_image.hdr")
        save_image(image, save_path, dtype=np.uint16)
        image_disc = SpectralImage.envi_open(hdr_path=save_path)
        assert image_disc.file.dtype == "<u2"


def test_path_argument():
    with TemporaryDirectory() as tmpdir:
        image = np.random.default_rng().random((100, 100))
        save_path_str = os.path.join(tmpdir, "test_image.hdr")
        save_image(image, save_path_str)
        assert Path(save_path_str).exists()
