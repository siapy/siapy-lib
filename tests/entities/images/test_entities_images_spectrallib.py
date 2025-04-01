from pathlib import Path

import numpy as np
import pytest
import spectral as sp
from PIL import Image

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


def test_file_property(spectral_images):
    assert isinstance(
        spectral_images.vnir.image.file,
        (sp.io.envi.BilFile, sp.io.envi.BipFile, sp.io.envi.BsqFile),
    )


def test_filepath_property(spectral_images, configs):
    assert isinstance(spectral_images.vnir.image.filepath, Path)
    assert spectral_images.vnir.image.filepath.name == configs.image_vnir_img_path.name


def test_metadata_property(spectral_images):
    metadata = spectral_images.vnir.image.metadata
    assert isinstance(metadata, dict)
    required_keys = ["default bands", "wavelength", "description"]
    assert all(key in metadata for key in required_keys)


def test_shape_property(spectral_images):
    shape = spectral_images.vnir.image.shape
    assert isinstance(shape, tuple)
    assert len(shape) == 3
    assert all(isinstance(x, int) for x in shape)


def test_rows_property(spectral_images):
    assert isinstance(spectral_images.vnir.image.rows, int)
    assert isinstance(spectral_images.swir.image.rows, int)


def test_cols_property(spectral_images):
    assert isinstance(spectral_images.vnir.image.cols, int)
    assert isinstance(spectral_images.swir.image.cols, int)


def test_bands_property(spectral_images):
    bands = spectral_images.vnir.image.bands
    assert isinstance(bands, int)
    assert bands > 0


def test_default_bands_property(spectral_images):
    default_bands = spectral_images.vnir.image.default_bands
    assert isinstance(default_bands, list)
    assert all(isinstance(x, int) for x in default_bands)
    assert np.array_equal(default_bands, [55, 41, 12])


def test_wavelengths_property(spectral_images):
    wavelengths = spectral_images.vnir.image.wavelengths
    assert isinstance(wavelengths, list)
    assert all(isinstance(x, float) for x in wavelengths)
    assert len(wavelengths) == 160


def test_description_property(spectral_images):
    vnir_desc = spectral_images.vnir.image.description
    swir_desc = spectral_images.swir.image.description
    assert isinstance(vnir_desc, dict)
    assert isinstance(swir_desc, dict)
    required_keys = ["ID"]
    assert all(key in vnir_desc.keys() for key in required_keys)
    assert all(key in swir_desc.keys() for key in required_keys)


def test_camera_id_property(spectral_images, configs):
    camera_id = spectral_images.vnir.image.camera_id
    assert isinstance(camera_id, str)
    assert camera_id == configs.image_vnir_name


def test_to_display(spectral_images):
    image = spectral_images.vnir.image.to_display(equalize=True)
    assert isinstance(image, Image.Image)
    assert image.mode == "RGB"
    assert image.size == (spectral_images.vnir.image.cols, spectral_images.vnir.image.rows)

    # Test without equalization
    image_no_equalize = spectral_images.vnir.image.to_display(equalize=False)
    assert isinstance(image_no_equalize, Image.Image)


def test_to_numpy(spectral_images):
    array = spectral_images.vnir.image.to_numpy()
    assert isinstance(array, np.ndarray)
    assert array.shape == spectral_images.vnir.image.shape

    # Test with nan_value
    array_no_nans = spectral_images.vnir.image.to_numpy(nan_value=0.0)
    assert not np.any(np.isnan(array_no_nans))


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


def test_to_xarray(spectral_images):
    spectral_image_vnir = spectral_images.vnir
    xarray_vnir = spectral_image_vnir.to_xarray()

    assert xarray_vnir is not None
    assert hasattr(xarray_vnir, "dims")
    assert hasattr(xarray_vnir, "coords")
    assert hasattr(xarray_vnir, "attrs")

    assert xarray_vnir.dims == ("y", "x", "band")
    assert xarray_vnir.shape == spectral_image_vnir.shape

    assert len(xarray_vnir.coords["y"]) == spectral_image_vnir.shape[0]
    assert len(xarray_vnir.coords["x"]) == spectral_image_vnir.shape[1]
    assert len(xarray_vnir.coords["band"]) == spectral_image_vnir.shape[2]

    assert xarray_vnir.coords["band"].shape[0] == len(spectral_image_vnir.wavelengths)
    assert all(xarray_vnir.coords["band"].values == spectral_image_vnir.wavelengths)

    assert xarray_vnir.attrs == spectral_image_vnir.metadata

    assert np.array_equal(xarray_vnir.values, spectral_image_vnir.to_numpy(), equal_nan=True)


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
