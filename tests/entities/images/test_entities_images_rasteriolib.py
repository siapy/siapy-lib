import numpy as np
import pytest

from siapy.core.exceptions import InvalidFilepathError, InvalidInputError
from siapy.entities.images.rasterio_lib import RasterioLibImage


def test_open_valid(configs):
    raster = RasterioLibImage.open(
    )
    assert isinstance(raster, RasterioLibImage)
