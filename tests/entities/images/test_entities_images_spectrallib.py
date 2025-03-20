import numpy as np
import pytest

from siapy.core.exceptions import InvalidFilepathError, InvalidInputError
from siapy.entities.images.spectral_lib import SpectralLibImage, _parse_description


def test_open_valid(configs):
    spectral_image_vnir = SpectralLibImage.open(
        header_path=configs.image_vnir_hdr_path,
        image_path=configs.image_vnir_img_path,
    )
    assert isinstance(spectral_image_vnir, SpectralLibImage)


def test_open_invalid():
    with pytest.raises(InvalidFilepathError):
        SpectralLibImage.open(header_path="invalid_header_path")


def test_rows(spectral_images):
    assert isinstance(spectral_images.vnir.image.rows, int)
    assert isinstance(spectral_images.swir.image.rows, int)


def test_cols(spectral_images):
    assert isinstance(spectral_images.vnir.image.cols, int)
    assert isinstance(spectral_images.swir.image.cols, int)


def test_description(spectral_images):
    vnir_desc = spectral_images.vnir.image.description
    swir_desc = spectral_images.swir.image.description
    assert isinstance(vnir_desc, dict)
    assert isinstance(swir_desc, dict)
    required_keys = ["ID"]
    assert all(key in vnir_desc.keys() for key in required_keys)
    assert all(key in swir_desc.keys() for key in required_keys)


def test_remove_nan(spectral_images):
    image = np.array([[[1, 2, np.nan], [4, 2, 6]], [[np.nan, 8, 9], [10, 11, 12]]])
    result = spectral_images.vnir.image._remove_nan(image.copy())

    assert np.array_equal(result[0, 0], np.array([0, 0, 0]))
    assert np.array_equal(result[0, 1], np.array([4, 2, 6]))
    assert np.array_equal(result[1, 0], np.array([0, 0, 0]))
    assert np.array_equal(result[1, 1], np.array([10, 11, 12]))

    # Call the _remove_nan method with a non-default nan_value
    result = spectral_images.vnir.image._remove_nan(image.copy(), nan_value=99)

    # Check that all nan values have been replaced with 99
    assert (result == 99).sum() == 6


def test_parse_description_simple():
    description = "Frameperiod = 20060\nIntegration time = 20000"
    expected = {"Frameperiod": 20060, "Integration time": 20000}
    assert _parse_description(description) == expected


def test_parse_description_with_floats_and_ints():
    description = "Binning = 2\nPixelsize x = 0.000187"
    expected = {"Binning": 2, "Pixelsize x": 0.000187}
    assert _parse_description(description) == expected


def test_parse_description_with_commas():
    description = "Rotating stage position = 0.000000,15.700000,degrees"
    expected = {"Rotating stage position": [0.000000, 15.700000, "degrees"]}
    assert _parse_description(description) == expected


def test_parse_description_empty_value():
    description = "Comment ="
    expected = {"Comment": ""}
    assert _parse_description(description) == expected


def test_parse_description_invalid_format_raises_value_error():
    description = "This is not a valid format"
    with pytest.raises(InvalidInputError):
        _parse_description(description)
