import numpy as np
import pytest

from siapy.entities import Pixels, Shape
from siapy.entities.shapes import FreeDraw, Point, Rectangle
from siapy.utils.plots import display_selected_areas


@pytest.mark.manual
def test_freedraw_convex_hull_manual():
    pixels_input = [(3, 4), (24, 8), (15, 23)]
    image_mock = np.zeros((30, 30, 3))
    pixels = Pixels.from_iterable(pixels_input)
    freedraw = FreeDraw(pixels=pixels)
    convex_hull_output = freedraw.convex_hull()
    display_selected_areas(image_mock, convex_hull_output, color="red")


def test_from_shape_type():
    pixels_input = [(10, 15), (60, 66)]
    pixels = Pixels.from_iterable(pixels_input)
    rectangle = Shape.from_shape_type(shape_type="rectangle", pixels=pixels)

    assert isinstance(rectangle, Rectangle)
    point = Shape.from_shape_type(shape_type="point", pixels=pixels)
    assert isinstance(point, Point)
    freedraw = Shape.from_shape_type(shape_type="freedraw", pixels=pixels)
    assert isinstance(freedraw, FreeDraw)

    with pytest.raises(ValueError):
        Shape.from_shape_type(shape_type="shape", pixels=pixels)


def test_shape_init():
    pixels_input = [(10, 15), (60, 66)]
    with pytest.raises(TypeError):
        Shape(pixels=pixels_input)


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

    expected_pixels = Pixels.from_iterable(
        [(u, v) for u in range(10, 13) for v in range(21, 24)]
    )
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
