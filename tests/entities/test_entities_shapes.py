import numpy as np
import pytest

from siapy.core.exceptions import InvalidInputError
from siapy.entities import Pixels, Shape
from siapy.entities.shapes import (
    _SHAPE_TYPE_FREEDRAW,
    _SHAPE_TYPE_POINT,
    _SHAPE_TYPE_RECTANGLE,
    FreeDraw,
    GeometricShapes,
    Point,
    Rectangle,
)
from siapy.utils.plots import display_image_with_areas


@pytest.mark.manual
def test_freedraw_convex_hull_manual():
    pixels_input = [(3, 4), (24, 8), (15, 23)]
    image_mock = np.zeros((30, 30, 3))
    pixels = Pixels.from_iterable(pixels_input)
    freedraw = FreeDraw(pixels=pixels)
    convex_hull_output = freedraw.convex_hull()
    display_image_with_areas(image_mock, convex_hull_output, color="red")


def test_from_shape_type():
    pixels_input = [(10, 15), (60, 66)]
    pixels = Pixels.from_iterable(pixels_input)
    rectangle = Shape.from_shape_type(shape_type="rectangle", pixels=pixels)

    assert isinstance(rectangle, Rectangle)
    point = Shape.from_shape_type(shape_type="point", pixels=pixels)
    assert isinstance(point, Point)
    freedraw = Shape.from_shape_type(shape_type="freedraw", pixels=pixels)
    assert isinstance(freedraw, FreeDraw)

    with pytest.raises(InvalidInputError):
        Shape.from_shape_type(shape_type="shape", pixels=pixels)


def test_shape_init():
    pixels_input = [(10, 15), (60, 66)]
    with pytest.raises(TypeError):
        Shape(pixels=pixels_input)


def test_shape_type_property():
    pixels = Pixels.from_iterable([(10, 15), (20, 25)])
    rectangle = Rectangle(pixels=pixels)
    assert rectangle.shape_type == _SHAPE_TYPE_RECTANGLE
    point = Point(pixels=pixels)
    assert point.shape_type == _SHAPE_TYPE_POINT
    freedraw = FreeDraw(pixels=pixels)
    assert freedraw.shape_type == _SHAPE_TYPE_FREEDRAW


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


# Geometric shapes


def test_geometric_shapes_getter(spectral_images):
    assert isinstance(spectral_images.vnir.geometric_shapes, GeometricShapes)
    assert len(spectral_images.vnir.geometric_shapes) == 0


def test_geometric_shapes_setter_valid(spectral_images, corresponding_pixels):
    pixels_vnir = corresponding_pixels.vnir
    rect = Shape.from_shape_type(shape_type="rectangle", pixels=pixels_vnir)
    shapes = [rect, rect]
    spectral_images.vnir.geometric_shapes.shapes = shapes
    assert spectral_images.vnir.geometric_shapes.shapes == shapes


def test_geometric_shapes_setter_invalid(spectral_images):
    # Assuming invalid_shape is not an instance of Shape
    invalid_shape = "not a shape"
    shapes = [invalid_shape]

    with pytest.raises(InvalidInputError):
        spectral_images.vnir.geometric_shapes.shapes = shapes


def test_geometric_shapes_iteration(spectral_images, corresponding_pixels):
    rect = Shape.from_shape_type(shape_type="rectangle", pixels=corresponding_pixels.vnir)
    spectral_images.vnir.geometric_shapes.shapes = [rect, rect]
    for shape in spectral_images.vnir.geometric_shapes:
        assert isinstance(shape, Shape)


def test_geometric_shapes_getitem(spectral_images, corresponding_pixels):
    rect = Shape.from_shape_type(shape_type="rectangle", pixels=corresponding_pixels.vnir)
    spectral_images.vnir.geometric_shapes.shapes = [rect]
    assert spectral_images.vnir.geometric_shapes[0] == rect


def test_geometric_shapes_setitem_valid(spectral_images, corresponding_pixels):
    rect = Shape.from_shape_type(shape_type="rectangle", pixels=corresponding_pixels.vnir)
    spectral_images.vnir.geometric_shapes.shapes = [rect]
    new_rect = Shape.from_shape_type(shape_type="rectangle", pixels=corresponding_pixels.vnir)
    spectral_images.vnir.geometric_shapes[0] = new_rect
    assert spectral_images.vnir.geometric_shapes[0] == new_rect


def test_geometric_shapes_setitem_invalid(spectral_images):
    invalid_shape = "not a shape"
    spectral_images.vnir.geometric_shapes.shapes = []
    with pytest.raises(InvalidInputError):
        spectral_images.vnir.geometric_shapes[0] = invalid_shape


def test_geometric_shapes_len(spectral_images, corresponding_pixels):
    rect = Shape.from_shape_type(shape_type="rectangle", pixels=corresponding_pixels.vnir)
    spectral_images.vnir.geometric_shapes.shapes = [rect, rect]
    assert len(spectral_images.vnir.geometric_shapes) == 2


def test_geometric_shapes_eq(spectral_images, corresponding_pixels):
    rect1 = Shape.from_shape_type(shape_type="rectangle", pixels=corresponding_pixels.vnir)
    rect2 = Shape.from_shape_type(shape_type="rectangle", pixels=corresponding_pixels.vnir)
    spectral_images.vnir.geometric_shapes.shapes = [rect1, rect2]
    other_geometric_shapes = GeometricShapes(spectral_images.vnir)
    other_geometric_shapes.shapes = [rect1, rect2]
    assert spectral_images.vnir.geometric_shapes == other_geometric_shapes


def test_geometric_shapes_append_valid(spectral_images, corresponding_pixels):
    rect = Shape.from_shape_type(shape_type="rectangle", pixels=corresponding_pixels.vnir)
    spectral_images.vnir.geometric_shapes.clear()
    spectral_images.vnir.geometric_shapes.append(rect)
    assert rect in spectral_images.vnir.geometric_shapes
    assert len(spectral_images.vnir.geometric_shapes) == 1


def test_geometric_shapes_append_invalid(spectral_images):
    invalid_shape = "not a shape"
    with pytest.raises(InvalidInputError):
        spectral_images.vnir.geometric_shapes.append(invalid_shape)


def test_geometric_shapes_extend_valid(spectral_images, corresponding_pixels):
    rect = Shape.from_shape_type(shape_type="rectangle", pixels=corresponding_pixels.vnir)
    spectral_images.vnir.geometric_shapes.clear()
    spectral_images.vnir.geometric_shapes.extend([rect, rect])
    assert len(spectral_images.vnir.geometric_shapes) == 2


def test_geometric_shapes_extend_invalid(spectral_images):
    invalid_shape = "not a shape"
    with pytest.raises(InvalidInputError):
        spectral_images.vnir.geometric_shapes.extend([invalid_shape])


def test_geometric_shapes_insert_valid(spectral_images, corresponding_pixels):
    rect = Shape.from_shape_type(shape_type="rectangle", pixels=corresponding_pixels.vnir)
    spectral_images.vnir.geometric_shapes.insert(0, rect)
    assert spectral_images.vnir.geometric_shapes[0] == rect


def test_geometric_shapes_insert_invalid(spectral_images):
    invalid_shape = "not a shape"
    with pytest.raises(InvalidInputError):
        spectral_images.vnir.geometric_shapes.insert(0, invalid_shape)


def test_geometric_shapes_remove_valid(spectral_images, corresponding_pixels):
    rect = Shape.from_shape_type(shape_type="rectangle", pixels=corresponding_pixels.vnir)
    spectral_images.vnir.geometric_shapes.clear()
    spectral_images.vnir.geometric_shapes.append(rect)
    spectral_images.vnir.geometric_shapes.remove(rect)
    assert rect not in spectral_images.vnir.geometric_shapes


def test_geometric_shapes_remove_invalid(spectral_images):
    invalid_shape = "not a shape"
    with pytest.raises(InvalidInputError):
        spectral_images.vnir.geometric_shapes.remove(invalid_shape)


def test_geometric_shapes_pop(spectral_images, corresponding_pixels):
    rect = Shape.from_shape_type(shape_type="rectangle", pixels=corresponding_pixels.vnir)
    spectral_images.vnir.geometric_shapes.clear()
    spectral_images.vnir.geometric_shapes.append(rect)
    popped_shape = spectral_images.vnir.geometric_shapes.pop()
    assert popped_shape == rect
    assert len(spectral_images.vnir.geometric_shapes) == 0


def test_geometric_shapes_clear(spectral_images, corresponding_pixels):
    rect = Shape.from_shape_type(shape_type="rectangle", pixels=corresponding_pixels.vnir)
    spectral_images.vnir.geometric_shapes.append(rect)
    spectral_images.vnir.geometric_shapes.clear()
    assert len(spectral_images.vnir.geometric_shapes) == 0


def test_geometric_shapes_index_valid(spectral_images, corresponding_pixels):
    rect = Shape.from_shape_type(shape_type="rectangle", pixels=corresponding_pixels.vnir)
    spectral_images.vnir.geometric_shapes.append(rect)
    index = spectral_images.vnir.geometric_shapes.index(rect)
    assert index == 0


def test_geometric_shapes_index_invalid(spectral_images, corresponding_pixels):
    rect = Shape.from_shape_type(shape_type="rectangle", pixels=corresponding_pixels.vnir)
    invalid_shape = "not a shape"
    spectral_images.vnir.geometric_shapes.append(rect)
    with pytest.raises(InvalidInputError):
        spectral_images.vnir.geometric_shapes.index(invalid_shape)


def test_geometric_shapes_count(spectral_images, corresponding_pixels):
    rect = Shape.from_shape_type(shape_type="rectangle", pixels=corresponding_pixels.vnir)
    spectral_images.vnir.geometric_shapes.clear()
    spectral_images.vnir.geometric_shapes.append(rect)
    spectral_images.vnir.geometric_shapes.append(rect)
    count = spectral_images.vnir.geometric_shapes.count(rect)
    assert count == 2


def test_geometric_shapes_reverse(spectral_images, corresponding_pixels):
    rect1 = Shape.from_shape_type(shape_type="rectangle", pixels=corresponding_pixels.vnir)
    rect2 = Shape.from_shape_type(shape_type="rectangle", pixels=corresponding_pixels.vnir)
    spectral_images.vnir.geometric_shapes.append(rect1)
    spectral_images.vnir.geometric_shapes.append(rect2)
    spectral_images.vnir.geometric_shapes.reverse()
    assert spectral_images.vnir.geometric_shapes[0] == rect2
    assert spectral_images.vnir.geometric_shapes[1] == rect1


def test_geometric_shapes_sort(spectral_images, corresponding_pixels):
    rect1 = Shape.from_shape_type(shape_type="rectangle", pixels=corresponding_pixels.vnir, label="B")
    rect2 = Shape.from_shape_type(shape_type="rectangle", pixels=corresponding_pixels.vnir, label="A")
    spectral_images.vnir.geometric_shapes.clear()
    spectral_images.vnir.geometric_shapes.append(rect1)
    spectral_images.vnir.geometric_shapes.append(rect2)
    spectral_images.vnir.geometric_shapes.sort(key=lambda shape: shape.label)
    assert spectral_images.vnir.geometric_shapes[0].label == "A"
    assert spectral_images.vnir.geometric_shapes[1].label == "B"


def test_geometric_shapes_get_by_name_found(spectral_images, corresponding_pixels):
    rect1 = Shape.from_shape_type(shape_type="rectangle", pixels=corresponding_pixels.vnir, label="Rect1")
    rect2 = Shape.from_shape_type(shape_type="rectangle", pixels=corresponding_pixels.vnir, label="Rect2")
    spectral_images.vnir.geometric_shapes.shapes = [rect1, rect2]
    found_shape = spectral_images.vnir.geometric_shapes.get_by_name("Rect1")
    assert found_shape == rect1
    assert found_shape.label == "Rect1"


def test_geometric_shapes_get_by_name_not_found(spectral_images, corresponding_pixels):
    # Create a shape
    rect = Shape.from_shape_type(shape_type="rectangle", pixels=corresponding_pixels.vnir, label="Rect")
    spectral_images.vnir.geometric_shapes.shapes = [rect]
    found_shape = spectral_images.vnir.geometric_shapes.get_by_name("NonExistent")
    assert found_shape is None
