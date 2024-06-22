import numpy as np
import pytest

from siapy.transformations.corregistrator import Corregistrator
from siapy.utils.plotting import pixels_select_click  # noqa: F401
from tests.fixtures import corresponding_pixels, spectral_images  # noqa: F401


@pytest.mark.manual
def test_pixels_select_click_manual(spectral_images, corresponding_pixels):
    # image_vnir = spectral_images.vnir
    # image_swir = spectral_images.swir
    # pixels_vnir = pixels_select_click(image_vnir)
    # pixels_swir = pixels_select_click(image_swir)
    pixels_vnir = corresponding_pixels.vnir
    pixels_swir = corresponding_pixels.swir

    corregistrator = Corregistrator()
    matx, _ = corregistrator.align(pixels_swir, pixels_vnir, plot_progress=False)
    pixels_transformed = corregistrator.transform(pixels_vnir, matx)
    assert (
        np.sqrt(np.sum((pixels_swir.to_numpy() - pixels_transformed.to_numpy()) ** 2))
        < 10
    )
