import geopandas as gpd
import numpy as np
import pytest

from siapy.core.exceptions import InvalidFilepathError
from siapy.entities import Shapefile


@pytest.fixture
def point_shapefile(configs):
    return Shapefile.from_path(configs.shapefile_point)


@pytest.fixture
def buffer_shapefile(configs):
    return Shapefile.from_path(configs.shapefile_buffer)


def test_shapefile_loading(point_shapefile, buffer_shapefile):
    assert isinstance(point_shapefile, Shapefile)
    assert isinstance(point_shapefile.df, gpd.GeoDataFrame)
    assert isinstance(buffer_shapefile, Shapefile)
    assert isinstance(buffer_shapefile.df, gpd.GeoDataFrame)


def test_invalid_shapefile_path():
    with pytest.raises(InvalidFilepathError):
        Shapefile.from_path("nonexistent.shp")


def test_geometry_type(point_shapefile, buffer_shapefile):
    assert point_shapefile.geometry_type == "Point"
    assert buffer_shapefile.geometry_type == "Polygon"


def test_geometry_types(point_shapefile):
    types = point_shapefile.geometry_types
    assert isinstance(types, list)
    assert all(isinstance(t, str) for t in types)


def test_consistent_geometry(point_shapefile):
    assert point_shapefile.has_consistent_geometry_type()


def test_length(point_shapefile):
    assert len(point_shapefile) > 0
    assert len(point_shapefile) == len(point_shapefile.df)


def test_to_numpy(point_shapefile):
    numpy_array = point_shapefile.to_numpy()
    assert isinstance(numpy_array, np.ndarray)
    assert len(numpy_array) == len(point_shapefile)


def test_coordinates_attributes():
    assert Shapefile.coords.X == "x"
    assert Shapefile.coords.Y == "y"
