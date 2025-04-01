import numpy as np
import pytest

from siapy.core.exceptions import InvalidInputError
from siapy.entities import Pixels
from siapy.entities.shapes import FreeDraw, Point, Rectangle, ShapeGeometry, create_shape_from_geometry
from siapy.utils.plots import display_image_with_areas


@pytest.mark.manual
def test_freedraw_convex_hull_manual():
    pixels_input = [(3, 4), (24, 8), (15, 23)]
    image_mock = np.zeros((30, 30, 3))
    pixels = Pixels.from_iterable(pixels_input)
    freedraw = FreeDraw(pixels=pixels)
    convex_hull_output = freedraw.convex_hull()
    display_image_with_areas(image_mock, convex_hull_output, color="red")


def test_create_shape_from_enum():
    pixels = Pixels.from_iterable([(10, 20), (30, 40)])
    rectangle = create_shape_from_geometry(shape=ShapeGeometry.RECTANGLE, pixels=pixels)
    assert isinstance(rectangle, Rectangle)
    point = create_shape_from_geometry(shape=ShapeGeometry.POINT, pixels=pixels)
    assert isinstance(point, Point)
    freedraw = create_shape_from_geometry(shape=ShapeGeometry.FREEDRAW, pixels=pixels)
    assert isinstance(freedraw, FreeDraw)


def test_create_shape_with_label():
    pixels = Pixels.from_iterable([(10, 20)])
    label = "Test Label"
    rectangle = create_shape_from_geometry(shape="rectangle", pixels=pixels, label=label)
    assert rectangle.label == label
    point = create_shape_from_geometry(shape=ShapeGeometry.POINT, pixels=pixels, label=label)
    assert point.label == label


def test_create_shape_with_invalid_string():
    pixels = Pixels.from_iterable([(10, 20)])
    with pytest.raises(InvalidInputError) as excinfo:
        create_shape_from_geometry(shape="invalid_shape", pixels=pixels)
    assert "Unsupported shape type" in str(excinfo.value)


def test_create_shape_with_invalid_type():
    """Test that invalid shape types raise the correct exception."""
    pixels = Pixels.from_iterable([(10, 20)])
    with pytest.raises(InvalidInputError) as excinfo:
        create_shape_from_geometry(shape=123, pixels=pixels)
    assert "Unsupported shape type" in str(excinfo.value)


def test_create_shape_with_empty_pixels():
    """Test creating shapes with empty pixels."""
    pixels = Pixels.from_iterable([])
    rectangle = create_shape_from_geometry(shape="rectangle", pixels=pixels)
    assert len(rectangle.pixels) == 0


def test_created_shape_pixels_preservation():
    """Test that created shapes preserve the original pixels."""
    pixel_coords = [(10, 20), (30, 40)]
    pixels = Pixels.from_iterable(pixel_coords)
    for shape_type in ["rectangle", "point", "freedraw"]:
        shape = create_shape_from_geometry(shape=shape_type, pixels=pixels)
        assert shape.pixels == pixels


def test_shape_convex_hull_method_exists():
    """Test that all shapes have a convex_hull method."""
    pixels = Pixels.from_iterable([(10, 20), (30, 40)])

    for shape_type in ["rectangle", "point", "freedraw"]:
        shape = create_shape_from_geometry(shape=shape_type, pixels=pixels)
        assert hasattr(shape, "convex_hull")
        assert callable(shape.convex_hull)

        # Verify method returns Pixels object
        result = shape.convex_hull()
        assert isinstance(result, Pixels)


def test_none_label_handling():
    """Test that None labels are handled correctly."""
    pixels = Pixels.from_iterable([(10, 20)])
    shape = create_shape_from_geometry(shape="rectangle", pixels=pixels, label=None)
    assert shape.label == ""


def test_shape_pixels_property():
    pixels_input = [(10, 15), (20, 25)]
    pixels = Pixels.from_iterable(pixels_input)
    shape = Rectangle(pixels=pixels)
    pixels_output = shape.pixels
    assert pixels_output == pixels


def test_shape_label_property():
    pixels = Pixels.from_iterable([(10, 15)])
    label_input = "Test Shape"
    shape = Rectangle(pixels=pixels, label=label_input)
    label_output = shape.label
    assert label_output == label_input


def test_rectangle_convex_hull():
    pixels_input = [(10, 21), (12, 23)]
    pixels = Pixels.from_iterable(pixels_input)
    rectangle = Rectangle(pixels=pixels)
    convex_hull_output = rectangle.convex_hull()

    expected_pixels = Pixels.from_iterable([(u, v) for u in range(10, 13) for v in range(21, 24)])
    assert convex_hull_output.df.equals(expected_pixels.df)


def test_point_convex_hull():
    pixels_input = [(10, 15), (12, 23)]
    pixels = Pixels.from_iterable(pixels_input)
    point = Point(pixels=pixels)
    convex_hull_output = point.convex_hull()

    expected_pixels = Pixels.from_iterable(pixels_input)
    assert convex_hull_output.df.equals(expected_pixels.df)


def test_freedraw_convex_hull():
    pixels_input = [(2, 1), (3, 3), (2, 3)]
    pixels_output = [(2, 2), (2, 3), (3, 3)]
    pixels = Pixels.from_iterable(pixels_input)
    freedraw = FreeDraw(pixels=pixels)
    convex_hull_output = freedraw.convex_hull()
    expected_pixels = Pixels.from_iterable(pixels_output)
    assert convex_hull_output.df.equals(expected_pixels.df)
    # image_mock = np.zeros((10, 10, 3))
    # display_selected_areas(image_mock, convex_hull_output, color="red")
