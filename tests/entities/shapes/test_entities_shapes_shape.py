import geopandas as gpd
import pytest

from siapy.core.exceptions import InvalidFilepathError
from siapy.entities import Shape


@pytest.fixture
def point_shapefile(configs):
    return Shape.open_shapefile(configs.shapefile_point)


@pytest.fixture
def buffer_shapefile(configs):
    return Shape.open_shapefile(configs.shapefile_buffer)


def test_open_shapefile(point_shapefile, buffer_shapefile):
    assert isinstance(point_shapefile.df, gpd.GeoDataFrame)
    assert isinstance(buffer_shapefile.df, gpd.GeoDataFrame)


def test_invalid_shapefile_path():
    with pytest.raises(InvalidFilepathError):
        Shape.open_shapefile("nonexistent.shp")


def test_shape_type(point_shapefile, buffer_shapefile):
    assert point_shapefile.shape_type == "point"
    assert buffer_shapefile.shape_type == "polygon"
