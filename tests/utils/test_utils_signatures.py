import numpy as np
import pytest

from siapy.entities import Pixels, Shape, SpectralImage
from siapy.utils.signatures import get_signatures_within_convex_hull
from siapy.utils.plots import display_image_with_areas


def test_get_signatures_within_convex_hull(configs):
    raster = SpectralImage.rasterio_open(configs.image_micasense_merged)
    point_shape = Shape.open_shapefile(configs.shapefile_point)
    buffer_shape = Shape.open_shapefile(configs.shapefile_buffer)
    raster.geometric_shapes.shapes = [point_shape, buffer_shape]
    signatures_point = get_signatures_within_convex_hull(raster, point_shape)
    signatures_buffer = get_signatures_within_convex_hull(raster, buffer_shape)

    assert len(signatures_point) == 17
    assert all([len(sp) == 1 for sp in signatures_point])
    assert len(signatures_buffer) == 17
    assert all([len(sb) > 1 for sb in signatures_buffer])


@pytest.mark.manual
def test_convex_hull_visualization():
    points = Pixels.from_iterable([(3, 4), (24, 8), (15, 23)])
    shape = Shape.from_line(points)

    image_mock = SpectralImage.from_numpy(np.zeros((30, 30, 3)))
    signatures_within = get_signatures_within_convex_hull(image_mock, shape)[0]
    display_image_with_areas(image_mock, signatures_within.pixels, color="red")


def test_get_signatures_within_rectangular_shape():
    image_data = np.zeros((100, 100, 3))
    image_mock = SpectralImage.from_numpy(image_data)

    shape = Shape.from_rectangle(10, 21, 12, 23)
    signatures = get_signatures_within_convex_hull(image_mock, shape)[0]

    pixels_list = signatures.pixels.to_list()
    expected_points = [[u, v] for u in range(10, 13) for v in range(21, 24)]
    assert len(pixels_list) == len(expected_points)

    # Convert to sets for comparison (order may differ)
    pixels_set = {tuple(p) for p in signatures.pixels.as_type(int).to_list()}
    expected_set = {tuple(p) for p in expected_points}
    assert pixels_set == expected_set


def test_get_signatures_within_point_shape():
    image_data = np.zeros((30, 30, 3))
    image_mock = SpectralImage.from_numpy(image_data)

    points = [[10, 15], [12, 23]]
    shape = Shape.from_multipoint(Pixels.from_iterable(points))

    signatures = get_signatures_within_convex_hull(image_mock, shape)[0]

    pixels_list = signatures.pixels.as_type(int).to_list()
    assert len(pixels_list) == 2

    # Convert to sets for comparison (order may differ)
    pixels_set = {tuple(p) for p in pixels_list}
    expected_set = {tuple(p) for p in points}
    assert pixels_set == expected_set


def test_get_signatures_within_triangle_shape():
    image_data = np.zeros((10, 10, 3))
    image_mock = SpectralImage.from_numpy(image_data)

    points = Pixels.from_iterable([(2, 1), (3, 3), (2, 3)])
    shape = Shape.from_polygon(points)

    signatures = get_signatures_within_convex_hull(image_mock, shape)[0]

    # Expected points in the triangle
    expected_points = [[2, 1], [2, 2], [2, 3], [3, 3]]

    # Convert to sets for comparison (order may differ)
    pixels_set = {tuple(p) for p in signatures.pixels.as_type(int).to_list()}
    expected_set = {tuple(p) for p in expected_points}
    assert pixels_set == expected_set


def test_get_signatures_within_complex_shape():
    image_data = np.zeros((3, 3, 3))
    image_mock = SpectralImage.from_numpy(image_data)

    # Create an L-shaped polygon
    points = Pixels.from_iterable([(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)])
    shape = Shape.from_polygon(points)

    signatures = get_signatures_within_convex_hull(image_mock, shape)[0]

    # The convex hull should fill in the "L" shape
    # Expected points in the 3x3 grid
    expected_points = [
        [0, 0],
        [0, 1],
        [0, 2],
        [1, 0],
        [1, 1],
        [1, 2],
        [2, 0],
        [2, 1],
    ]

    # Convert to sets for comparison (order may differ)
    pixels_set = {tuple(p) for p in signatures.pixels.as_type(int).to_list()}
    expected_set = {tuple(p) for p in expected_points}
    assert pixels_set == expected_set
