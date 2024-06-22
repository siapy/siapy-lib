import numpy as np
import pytest
from PIL import Image as PILImage

from siapy.utils.plotting import pixels_select_click, validate_and_convert_image
from tests.fixtures import spectral_images  # noqa: F401


@pytest.mark.manual
def test_pixels_select_click(spectral_images):
    image_vnir = spectral_images.vnir
    image_swir = spectral_images.swir

    pixels_vnir = pixels_select_click(image_vnir.to_display())
    pixels_swir = pixels_select_click(image_swir.to_display())


def test_validate_and_convert_image_with_spectral_image(spectral_images):
    image_vnir = spectral_images.vnir
    assert isinstance(validate_and_convert_image(image_vnir), np.ndarray)
    assert isinstance(validate_and_convert_image(image_vnir.to_display()), np.ndarray)


def test_validate_and_convert_image_with_image():
    mock_image = PILImage.new("RGB", (100, 100))
    result = validate_and_convert_image(mock_image)
    assert isinstance(result, np.ndarray) and result.shape == (100, 100, 3)


def test_validate_and_convert_image_with_numpy_array():
    mock_numpy_array = np.random.rand(100, 100, 3)
    assert np.array_equal(
        validate_and_convert_image(mock_numpy_array), mock_numpy_array
    )


def test_validate_and_convert_image_with_invalid_input():
    mock_numpy_array = np.random.rand(100, 100, 4)
    with pytest.raises(ValueError):
        validate_and_convert_image(mock_numpy_array)
