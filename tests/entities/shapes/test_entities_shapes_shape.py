import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

from siapy.core.exceptions import ConfigurationError, InvalidFilepathError, InvalidTypeError
from siapy.entities import Shape
from siapy.entities.pixels import PixelCoordinate, Pixels


@pytest.fixture(scope="module")
def point_shapefile(configs):
    return Shape.open_shapefile(configs.shapefile_point)


@pytest.fixture(scope="module")
def buffer_shapefile(configs):
    return Shape.open_shapefile(configs.shapefile_buffer)


def test_open_shapefile(point_shapefile, buffer_shapefile):
    assert isinstance(point_shapefile.df, gpd.GeoDataFrame)
    assert isinstance(buffer_shapefile.df, gpd.GeoDataFrame)


def test_invalid_shapefile_path():
    with pytest.raises(InvalidFilepathError):
        Shape.open_shapefile("nonexistent.shp")


def test_shape_initialization_errors():
    with pytest.raises(ConfigurationError):
        Shape()

    with pytest.raises(ConfigurationError):
        point = Point(0, 0)
        Shape(geometry=point, geo_dataframe=gpd.GeoDataFrame())


def test_shape_array_interface():
    shape = Shape.from_rectangle(0, 0, 2, 2, label="test_shape")

    # Test implicit conversion to numpy array
    array = np.asarray(shape)
    assert isinstance(array, np.ndarray)
    assert np.array_equal(array, shape.to_numpy())


def test_basic_shape_creation():
    point = Shape.from_point(1.0, 2.0, label="test_point")
    assert point.is_point
    assert not point.is_multi
    assert point.label == "test_point"

    line_points = [
        [[0, 0], (1, 1)],
        [PixelCoordinate(2, 2), (3, 3)],
    ]
    for points in line_points:
        pixels = Pixels.from_iterable(points)
        line = Shape.from_line(pixels, label="test_line")
        assert line.is_line
        assert not line.is_multi

    # Test polygon creation with automatic closure
    polygon_points = [(0, 0), (2, 0), (2, 2), (0, 2)]
    polygon = Shape.from_polygon(Pixels.from_iterable(polygon_points))
    assert polygon.is_polygon
    assert not polygon.is_multi
    # Verify automatic closure
    exterior_coords = list(polygon.exterior[0].coords)
    assert exterior_coords[0] == exterior_coords[-1]


def test_multi_geometry_creation():
    # MultiPoint
    points = Pixels.from_iterable([(0, 0), (1, 1), (2, 2)])
    multi_point = Shape.from_multipoint(points)
    assert multi_point.is_point and multi_point.is_multi

    # MultiLine
    lines = [Pixels.from_iterable([(0, 0), (1, 1)]), Pixels.from_iterable([(2, 2), (3, 3)])]
    multi_line = Shape.from_multiline(lines)
    assert multi_line.is_line and multi_line.is_multi

    # MultiPolygon
    polygons = [
        Pixels.from_iterable([(0, 0), (1, 0), (1, 1), (0, 1)]),
        Pixels.from_iterable([(2, 2), (3, 2), (3, 3), (2, 3)]),
    ]
    multi_polygon = Shape.from_multipolygon(polygons)
    assert multi_polygon.is_polygon and multi_polygon.is_multi


def test_special_shape_creation():
    # Rectangle
    rect = Shape.from_rectangle(0, 0, 2, 2)
    assert rect.is_polygon
    assert rect.bounds.iloc[0].tolist() == [0, 0, 2, 2]

    # Circle
    circle = Shape.from_circle(PixelCoordinate(0, 0), radius=1.0)
    assert circle.is_polygon
    bounds = circle.bounds.iloc[0]
    assert -1 <= bounds.minx <= -0.9  # Allow for floating point imprecision
    assert -1 <= bounds.miny <= -0.9
    assert 0.9 <= bounds.maxx <= 1
    assert 0.9 <= bounds.maxy <= 1

    # Polygon with holes
    exterior = Pixels.from_iterable([(0, 0), (4, 0), (4, 4), (0, 4)])
    hole = Pixels.from_iterable([(1, 1), (3, 1), (3, 3), (1, 3)])
    shape = Shape.from_polygon(exterior, holes=[hole])
    assert shape.is_polygon
    assert not shape.is_multi
    # Verify the hole was created
    assert len(list(shape.geometry[0].interiors)) == 1


def test_invalid_polygon_creation():
    with pytest.raises(ConfigurationError):
        # Try to create polygon with less than 3 points
        invalid_pixels = Pixels.from_iterable([(0, 0), (1, 1)])
        Shape.from_polygon(invalid_pixels)


def test_shape_type_queries():
    point = Shape.from_point(0, 0)
    assert point.is_point
    assert not point.is_line
    assert not point.is_polygon
    assert not point.is_multi

    line = Shape.from_line(Pixels.from_iterable([(0, 0), (1, 1)]))
    assert not line.is_point
    assert line.is_line
    assert not line.is_polygon
    assert not line.is_multi

    multipoint = Shape.from_multipoint(Pixels.from_iterable([(0, 0), (1, 1)]))
    assert multipoint.is_point
    assert multipoint.is_multi


def test_invalid_multiline():
    with pytest.raises(ConfigurationError):
        Shape.from_multiline([])


def test_invalid_multipolygon():
    with pytest.raises(ConfigurationError):
        Shape.from_multipolygon([])


def test_geometry_pass_properties():
    points = Pixels.from_iterable([(0, 0), (2, 0), (2, 2), (0, 2)])
    shape = Shape.from_polygon(points)

    # Test all geometric properties
    properties = {
        "boundary": gpd.GeoSeries,
        "bounds": pd.DataFrame,
        "centroid": gpd.GeoSeries,
        "convex_hull": gpd.GeoSeries,
        "envelope": gpd.GeoSeries,
        "exterior": gpd.GeoSeries,
    }

    for prop_name, expected_type in properties.items():
        prop_value = getattr(shape, prop_name)
        assert isinstance(prop_value, expected_type), f"{prop_name} type mismatch"


def test_shape_label_property():
    shape = Shape.from_point(1.0, 2.0, label="initial")

    # Test the getter
    assert shape.label == "initial"

    # Test setting a valid label
    shape.label = "new_label"
    assert shape.label == "new_label"

    # Error
    with pytest.raises(InvalidTypeError):
        shape.label = 123  # Not a string


def test_shape_geometry_property():
    point = Shape.from_point(1.0, 2.0)
    assert isinstance(point.geometry, gpd.GeoSeries)

    # Test geometry setter
    new_geometry = gpd.GeoSeries([Point(0, 0)])
    point.geometry = new_geometry
    assert point.geometry.equals(new_geometry)


def test_shape_geometry_property_errors():
    with pytest.raises(InvalidTypeError):
        shape = Shape.from_point(1.0, 2.0)
        shape.geometry = "invalid"


def test_shape_geometry_empty_geodataframe():
    empty_gdf = gpd.GeoDataFrame()
    shape = Shape.from_geodataframe(empty_gdf)

    with pytest.raises(ConfigurationError):
        _ = shape.geometry


def test_file_operations(tmp_path):
    original = Shape.from_point(1.0, 2.0, label="test")

    # Test multiple file formats
    formats = {"shp": "ESRI Shapefile", "geojson": "GeoJSON", "gpkg": "GPKG"}

    for ext, driver in formats.items():
        filepath = tmp_path / f"test.{ext}"
        original.to_file(filepath, driver=driver)
        assert filepath.exists()

        loaded = Shape.open_shapefile(filepath)
        assert loaded.is_point
        assert loaded.geometry.equals(original.geometry)


def test_to_numpy():
    points = Pixels.from_iterable([(0, 0), (2, 0), (2, 2), (0, 2)])
    shape = Shape.from_polygon(points)

    array = shape.to_numpy()
    assert isinstance(array, np.ndarray)


def test_shape_buffer():
    shape = Shape.from_point(0, 0)
    buffered = shape.buffer(1.0)

    # The result should be a polygon
    assert buffered.is_polygon
    assert not buffered.is_point

    # The buffer should have increased the bounds
    assert buffered.bounds.iloc[0].tolist() != shape.bounds.iloc[0].tolist()

    # Check buffer with different distances
    small_buffer = shape.buffer(0.5)
    large_buffer = shape.buffer(2.0)

    # Larger buffer should have greater area
    assert large_buffer.geometry.area[0] > small_buffer.geometry.area[0]

    # Original shape should remain unchanged
    assert shape.is_point


def test_shape_intersection():
    # Create two overlapping shapes
    shape1 = Shape.from_rectangle(0, 0, 2, 2)
    shape2 = Shape.from_rectangle(1, 1, 3, 3)

    # Calculate intersection
    intersection = shape1.intersection(shape2)

    # Check that intersection is correct
    assert intersection.is_polygon
    bounds = intersection.bounds.iloc[0]
    assert bounds.minx == 1.0
    assert bounds.miny == 1.0
    assert bounds.maxx == 2.0
    assert bounds.maxy == 2.0

    # Test with non-overlapping shapes
    shape3 = Shape.from_rectangle(5, 5, 6, 6)
    empty_intersection = shape1.intersection(shape3)
    assert empty_intersection.geometry.is_empty[0]

    # Original shapes should remain unchanged
    assert shape1.bounds.iloc[0].tolist() == [0, 0, 2, 2]
    assert shape2.bounds.iloc[0].tolist() == [1, 1, 3, 3]


def test_shape_union():
    # Create two shapes
    shape1 = Shape.from_rectangle(0, 0, 2, 2)
    shape2 = Shape.from_rectangle(1, 1, 3, 3)

    # Calculate union
    union = shape1.union(shape2)

    # Check that union is correct
    assert union.is_polygon
    bounds = union.bounds.iloc[0]
    assert bounds.minx == 0.0
    assert bounds.miny == 0.0
    assert bounds.maxx == 3.0
    assert bounds.maxy == 3.0

    # Union area should be less than sum of individual areas due to overlap
    area_sum = shape1.geometry.area[0] + shape2.geometry.area[0]
    assert union.geometry.area[0] < area_sum

    # Original shapes should remain unchanged
    assert shape1.bounds.iloc[0].tolist() == [0, 0, 2, 2]
    assert shape2.bounds.iloc[0].tolist() == [1, 1, 3, 3]


def test_shape_copy():
    # Create a shape
    original = Shape.from_rectangle(0, 0, 2, 2, label="original")
    copied = original.copy()

    # Verify properties are the same
    assert copied.bounds.equals(original.bounds)
    assert copied.geometry.equals(original.geometry)
    assert copied.label == original.label
    assert copied.is_polygon == original.is_polygon

    # Verify it's a deep copy by modifying one and checking the other
    new_geometry = gpd.GeoSeries([Point(0, 0)])
    copied.geometry = new_geometry

    # Original should be unchanged
    assert not copied.geometry.equals(original.geometry)
    assert copied.is_point
    assert original.is_polygon


def test_shape_type_property():
    # Create different shape types and verify their shape_type property
    point = Shape.from_point(0, 0)
    assert point.shape_type == "point"

    line = Shape.from_line(Pixels.from_iterable([(0, 0), (1, 1)]))
    assert line.shape_type == "linestring"

    polygon = Shape.from_polygon(Pixels.from_iterable([(0, 0), (2, 0), (2, 2), (0, 2)]))
    assert polygon.shape_type == "polygon"

    multipoint = Shape.from_multipoint(Pixels.from_iterable([(0, 0), (1, 1)]))
    assert multipoint.shape_type == "multipoint"

    # Test that shape_type is correctly updated when geometry changes
    new_geometry = gpd.GeoSeries([Point(0, 0)])
    polygon.geometry = new_geometry
    assert polygon.shape_type == "point"


def test_df_property():
    shape = Shape.from_rectangle(0, 0, 2, 2, label="test")

    # Test the df property
    assert isinstance(shape.df, gpd.GeoDataFrame)
    assert not shape.df.empty
    assert "geometry" in shape.df.columns

    # Test modification through df property
    # Add a new column
    shape.df["attribute"] = "test_value"
    assert "attribute" in shape.df.columns
    assert shape.df["attribute"][0] == "test_value"

    # Verify that df provides direct access to the underlying GeoDataFrame
    original_id = id(shape.df)
    shape.df["new_attr"] = 100
    assert id(shape.df) == original_id  # Should be the same object
    assert "new_attr" in shape.df.columns
