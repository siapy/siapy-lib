from types import SimpleNamespace

import numpy as np
import pytest

from siapy.entities import Pixels, SpectralImage
from tests.configs import (
    image_swir_hdr_path,
    image_swir_img_path,
    image_vnir_hdr_path,
    image_vnir_img_path,
)


class SpectralImages(SimpleNamespace):
    vnir: SpectralImage
    swir: SpectralImage
    vnir_np: np.ndarray
    swir_np: np.ndarray


@pytest.fixture
def spectral_images() -> SpectralImages:
    spectral_image_vnir = SpectralImage.envi_open(
        header_path=image_vnir_hdr_path,
        image_path=image_vnir_img_path,
    )
    spectral_image_swir = SpectralImage.envi_open(
        header_path=image_swir_hdr_path,
        image_path=image_swir_img_path,
    )
    spectral_image_vnir_np = spectral_image_vnir.to_numpy()
    spectral_image_swir_np = spectral_image_swir.to_numpy()
    return SpectralImages(
        vnir=spectral_image_vnir,
        swir=spectral_image_swir,
        vnir_np=spectral_image_vnir_np,
        swir_np=spectral_image_swir_np,
    )


class CorrespondingPixels(SimpleNamespace):
    vnir: Pixels
    swir: Pixels


@pytest.fixture
def corresponding_pixels() -> CorrespondingPixels:
    pixels_vnir = np.array(
        [
            [1007, 620],
            [417, 1052],
            [439, 1582],
            [1100, 1866],
            [832, 1090],
            [1133, 1079],
            [854, 1407],
            [1138, 1413],
        ]
    )
    pixels_swir = np.array(
        [
            [252, 110],
            [99, 219],
            [107, 354],
            [268, 422],
            [207, 230],
            [279, 225],
            [210, 309],
            [283, 309],
        ]
    )
    return CorrespondingPixels(
        vnir=Pixels.from_iterable(pixels_vnir),
        swir=Pixels.from_iterable(pixels_swir),
    )
