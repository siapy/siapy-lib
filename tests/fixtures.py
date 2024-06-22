from types import SimpleNamespace

import pytest

from siapy.entities import SpectralImage
from tests.configs import (
    image_swir_hdr_path,
    image_swir_img_path,
    image_vnir_hdr_path,
    image_vnir_img_path,
)


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
