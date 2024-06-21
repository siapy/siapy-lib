from types import SimpleNamespace

import pytest

from siapy.core.configs import TEST_DATA_DIR
from siapy.entities import SpectralImage

image_vnir_hdr_path = TEST_DATA_DIR / "vnir.hdr"
image_vnir_img_path = TEST_DATA_DIR / "vnir.hyspex"
image_swir_hdr_path = TEST_DATA_DIR / "swir.hdr"
image_swir_img_path = TEST_DATA_DIR / "swir.hyspex"


def test_envi_open():
    spectral_image_vnir = SpectralImage.envi_open(
        hdr_path=image_vnir_hdr_path,
        img_path=image_vnir_img_path,
    )
    assert isinstance(spectral_image_vnir, SpectralImage)

    spectral_image_swir = SpectralImage.envi_open(
        hdr_path=image_swir_hdr_path,
        img_path=image_swir_img_path,
    )
    assert isinstance(spectral_image_swir, SpectralImage)


@pytest.fixture
def spectral_images():
    spectral_image_vnir = SpectralImage.envi_open(
        hdr_path=image_vnir_hdr_path,
        img_path=image_vnir_img_path,
    )
    spectral_image_swir = SpectralImage.envi_open(
        hdr_path=image_swir_hdr_path,
        img_path=image_swir_img_path,
    )
    return SimpleNamespace(vnir=spectral_image_vnir, swir=spectral_image_swir)


def test_fixture_spectral_images(spectral_images):
    assert isinstance(spectral_images.vnir, SpectralImage)
    assert isinstance(spectral_images.swir, SpectralImage)
