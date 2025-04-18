import numpy as np
import pytest
from PIL import Image

from siapy.core.exceptions import InvalidInputError
from siapy.core.types import XarrayType
from siapy.entities.images.mock import MockImage


def test_initialization():
    # Valid 3D array
    valid_array = np.random.rand(100, 100, 3).astype(np.float32)
    mock_img = MockImage(array=valid_array)
    assert isinstance(mock_img, MockImage)

    # Invalid 2D array
    invalid_array = np.random.rand(100, 100).astype(np.float32)
    with pytest.raises(InvalidInputError):
        MockImage(array=invalid_array)


def test_open():
    test_array = np.random.rand(100, 100, 3).astype(np.float32)
    mock_img = MockImage.open(array=test_array)
    assert isinstance(mock_img, MockImage)


def test_properties():
    test_array = np.random.rand(50, 60, 5).astype(np.float32)
    mock_img = MockImage(array=test_array)

    assert mock_img.filepath.is_absolute() is False
    assert isinstance(mock_img.metadata, dict)
    assert len(mock_img.metadata) == 0
    assert mock_img.shape == (50, 60, 5)
    assert mock_img.bands == 5
    assert mock_img.default_bands == [0, 1, 2]
    assert mock_img.wavelengths == [0, 1, 2, 3, 4]
    assert mock_img.camera_id == ""


def test_default_bands_fewer_than_three():
    test_array = np.random.rand(50, 60, 2).astype(np.float32)
    mock_img = MockImage(array=test_array)
    assert mock_img.default_bands == [0, 1]


def test_to_display():
    # Test RGB case
    rgb_array = np.random.rand(50, 60, 3).astype(np.float32)
    mock_rgb = MockImage(array=rgb_array)

    img_eq = mock_rgb.to_display(equalize=True)
    assert isinstance(img_eq, Image.Image)
    assert img_eq.size == (60, 50)
    assert img_eq.mode == "RGB"

    img_no_eq = mock_rgb.to_display(equalize=False)
    assert isinstance(img_no_eq, Image.Image)

    # Test single band case
    single_band_array = np.random.rand(50, 60, 1).astype(np.float32)
    mock_single = MockImage(array=single_band_array)

    img_single = mock_single.to_display()
    assert isinstance(img_single, Image.Image)
    assert img_single.size == (60, 50)


def test_to_display_with_nans():
    array_with_nans = np.random.rand(50, 60, 3).astype(np.float32)
    array_with_nans[10:20, 10:20, :] = np.nan

    mock_img = MockImage(array=array_with_nans)
    img = mock_img.to_display()

    assert isinstance(img, Image.Image)
    # Image should be created successfully despite NaNs


def test_to_numpy():
    test_array = np.random.rand(50, 60, 3).astype(np.float32)
    mock_img = MockImage(array=test_array)

    # Without NaN handling
    result = mock_img.to_numpy()
    assert isinstance(result, np.ndarray)
    assert result.shape == (50, 60, 3)
    assert result.dtype == np.float32
    assert np.array_equal(result, test_array)

    # With NaN handling
    result_no_nans = mock_img.to_numpy(nan_value=0.0)
    assert isinstance(result_no_nans, np.ndarray)
    assert not np.any(np.isnan(result_no_nans))


def test_to_xarray():
    test_array = np.random.rand(50, 60, 3).astype(np.float32)
    mock_img = MockImage(array=test_array)

    result = mock_img.to_xarray()
    assert isinstance(result, XarrayType)
    assert "y" in result.dims
    assert "x" in result.dims
    assert "band" in result.dims
    assert result.attrs["camera_id"] == ""
    assert result.shape == (50, 60, 3)

    np.testing.assert_array_equal(result.coords["band"].values, mock_img.wavelengths)
    np.testing.assert_array_equal(result.coords["x"].values, np.arange(mock_img.shape[1]))
    np.testing.assert_array_equal(result.coords["y"].values, np.arange(mock_img.shape[0]))
