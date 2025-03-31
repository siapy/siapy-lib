from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from siapy.core.exceptions import InvalidFilepathError
from siapy.core.types import XarrayType
from siapy.entities.images.rasterio_lib import RasterioLibImage


def test_open_valid(configs):
    raster = RasterioLibImage.open(configs.image_micasense_merged)
    assert isinstance(raster, RasterioLibImage)


def test_open_invalid():
    with pytest.raises(InvalidFilepathError):
        RasterioLibImage.open("nonexistent.tif")


def test_file(configs):
    raster = RasterioLibImage.open(configs.image_micasense_merged)
    file = raster.file
    assert isinstance(file, XarrayType)
    assert hasattr(file, "dims")
    assert hasattr(file, "values")
    assert hasattr(file, "attrs")
    assert "band" in file.dims
    assert "x" in file.dims
    assert "y" in file.dims


def test_properties(configs):
    raster = RasterioLibImage.open(configs.image_micasense_merged)

    assert isinstance(raster.filepath, Path)
    assert isinstance(raster.metadata, dict)
    assert isinstance(raster.shape, tuple)
    assert len(raster.shape) == 3
    assert isinstance(raster.rows, int)
    assert isinstance(raster.cols, int)
    assert isinstance(raster.bands, int)
    assert isinstance(raster.default_bands, list)
    assert all(isinstance(x, int) for x in raster.default_bands)
    assert isinstance(raster.wavelengths, np.ndarray)
    assert isinstance(raster.camera_id, str)


def test_shape_consistency(configs):
    raster = RasterioLibImage.open(configs.image_micasense_merged)
    shape = raster.shape
    assert shape[0] == raster.rows
    assert shape[1] == raster.cols
    assert shape[2] == raster.bands


def test_to_display(configs):
    raster = RasterioLibImage.open(configs.image_micasense_merged)
    img_eq = raster.to_display(equalize=True)
    assert isinstance(img_eq, Image.Image)
    img_no_eq = raster.to_display(equalize=False)
    assert isinstance(img_no_eq, Image.Image)


def test_numpy(configs):
    raster = RasterioLibImage.open(configs.image_micasense_merged)
    arr = raster.to_numpy()
    assert isinstance(arr, np.ndarray)
    assert arr.shape == raster.shape


def test_to_numpy_with_nan_handling(configs):
    raster = RasterioLibImage.open(configs.image_micasense_merged)

    arr_no_nans = raster.to_numpy(nan_value=0.0)
    assert isinstance(arr_no_nans, np.ndarray)
    assert arr_no_nans.shape == raster.shape
    assert not np.any(np.isnan(arr_no_nans))


def test_to_xarray(configs):
    raster = RasterioLibImage.open(configs.image_micasense_merged)
    assert isinstance(raster.to_xarray(), XarrayType)
