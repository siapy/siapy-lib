import pytest

from siapy.core.exceptions import InvalidInputError
from siapy.entities.shapes import GeometricShapes, Shape


@pytest.fixture(scope="module")
def rectangle_shape():
    x_min = 10
    y_min = 15
    x_max = 60
    y_max = 66
    return Shape.from_rectangle(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)


def test_geometric_shapes_getter(spectral_images):
    assert isinstance(spectral_images.vnir.geometric_shapes, GeometricShapes)
    assert len(spectral_images.vnir.geometric_shapes) == 0


def test_geometric_shapes_setter_valid(rectangle_shape, spectral_images):
    rect = rectangle_shape
    shapes = [rect, rect]
    spectral_images.vnir.geometric_shapes.shapes = shapes
    assert spectral_images.vnir.geometric_shapes.shapes == shapes


def test_geometric_shapes_setter_invalid(spectral_images):
    # Assuming invalid_shape is not an instance of Shape
    invalid_shape = "not a shape"
    shapes = [invalid_shape]

    with pytest.raises(InvalidInputError):
        spectral_images.vnir.geometric_shapes.shapes = shapes


def test_geometric_shapes_iteration(rectangle_shape, spectral_images):
    rect = rectangle_shape
    spectral_images.vnir.geometric_shapes.shapes = [rect, rect]
    for shape in spectral_images.vnir.geometric_shapes:
        assert isinstance(shape, Shape)


def test_geometric_shapes_getitem(rectangle_shape, spectral_images):
    rect = rectangle_shape
    spectral_images.vnir.geometric_shapes.shapes = [rect]
    assert spectral_images.vnir.geometric_shapes[0] == rect


def test_geometric_shapes_setitem_valid(rectangle_shape, spectral_images):
    rect = rectangle_shape
    spectral_images.vnir.geometric_shapes.shapes = [rect]
    new_rect = rectangle_shape.copy()
    spectral_images.vnir.geometric_shapes[0] = new_rect
    assert spectral_images.vnir.geometric_shapes[0] == new_rect


def test_geometric_shapes_setitem_invalid(spectral_images):
    invalid_shape = "not a shape"
    spectral_images.vnir.geometric_shapes.shapes = []
    with pytest.raises(InvalidInputError):
        spectral_images.vnir.geometric_shapes[0] = invalid_shape


def test_geometric_shapes_len(rectangle_shape, spectral_images):
    rect = rectangle_shape
    spectral_images.vnir.geometric_shapes.shapes = [rect, rect]
    assert len(spectral_images.vnir.geometric_shapes) == 2


def test_geometric_shapes_eq(rectangle_shape, spectral_images):
    rect1 = rectangle_shape.copy()
    rect2 = rectangle_shape.copy()
    spectral_images.vnir.geometric_shapes.shapes = [rect1, rect2]
    other_geometric_shapes = GeometricShapes(spectral_images.vnir)
    other_geometric_shapes.shapes = [rect1, rect2]
    assert spectral_images.vnir.geometric_shapes == other_geometric_shapes


def test_geometric_shapes_eq_invalid(spectral_images):
    with pytest.raises(InvalidInputError):
        spectral_images.vnir.geometric_shapes == "not a geometric shapes"


def test_geometric_shapes_append_valid(rectangle_shape, spectral_images):
    rect = rectangle_shape
    spectral_images.vnir.geometric_shapes.clear()
    spectral_images.vnir.geometric_shapes.append(rect)
    assert rect in spectral_images.vnir.geometric_shapes
    assert len(spectral_images.vnir.geometric_shapes) == 1


def test_geometric_shapes_append_invalid(spectral_images):
    invalid_shape = "not a shape"
    with pytest.raises(InvalidInputError):
        spectral_images.vnir.geometric_shapes.append(invalid_shape)


def test_geometric_shapes_extend_valid(rectangle_shape, spectral_images):
    rect = rectangle_shape
    spectral_images.vnir.geometric_shapes.clear()
    spectral_images.vnir.geometric_shapes.extend([rect, rect])
    assert len(spectral_images.vnir.geometric_shapes) == 2


def test_geometric_shapes_extend_invalid(spectral_images):
    invalid_shape = "not a shape"
    with pytest.raises(InvalidInputError):
        spectral_images.vnir.geometric_shapes.extend([invalid_shape])


def test_geometric_shapes_insert_valid(rectangle_shape, spectral_images):
    rect = rectangle_shape
    spectral_images.vnir.geometric_shapes.insert(0, rect)
    assert spectral_images.vnir.geometric_shapes[0] == rect


def test_geometric_shapes_insert_invalid(spectral_images):
    invalid_shape = "not a shape"
    with pytest.raises(InvalidInputError):
        spectral_images.vnir.geometric_shapes.insert(0, invalid_shape)


def test_geometric_shapes_remove_valid(rectangle_shape, spectral_images):
    rect = rectangle_shape
    spectral_images.vnir.geometric_shapes.clear()
    spectral_images.vnir.geometric_shapes.append(rect)
    spectral_images.vnir.geometric_shapes.remove(rect)
    assert rect not in spectral_images.vnir.geometric_shapes


def test_geometric_shapes_remove_invalid(spectral_images):
    invalid_shape = "not a shape"
    with pytest.raises(InvalidInputError):
        spectral_images.vnir.geometric_shapes.remove(invalid_shape)


def test_geometric_shapes_pop(rectangle_shape, spectral_images):
    rect = rectangle_shape
    spectral_images.vnir.geometric_shapes.clear()
    spectral_images.vnir.geometric_shapes.append(rect)
    popped_shape = spectral_images.vnir.geometric_shapes.pop()
    assert popped_shape == rect
    assert len(spectral_images.vnir.geometric_shapes) == 0


def test_geometric_shapes_clear(rectangle_shape, spectral_images):
    rect = rectangle_shape
    spectral_images.vnir.geometric_shapes.append(rect)
    spectral_images.vnir.geometric_shapes.clear()
    assert len(spectral_images.vnir.geometric_shapes) == 0


def test_geometric_shapes_index_valid(rectangle_shape, spectral_images):
    rect = rectangle_shape
    spectral_images.vnir.geometric_shapes.append(rect)
    index = spectral_images.vnir.geometric_shapes.index(rect)
    assert index == 0


def test_geometric_shapes_index_invalid(rectangle_shape, spectral_images):
    rect = rectangle_shape
    invalid_shape = "not a shape"
    spectral_images.vnir.geometric_shapes.append(rect)
    with pytest.raises(InvalidInputError):
        spectral_images.vnir.geometric_shapes.index(invalid_shape)


def test_geometric_shapes_count(rectangle_shape, spectral_images):
    rect = rectangle_shape
    spectral_images.vnir.geometric_shapes.clear()
    spectral_images.vnir.geometric_shapes.append(rect)
    spectral_images.vnir.geometric_shapes.append(rect)
    count = spectral_images.vnir.geometric_shapes.count(rect)
    assert count == 2


def test_geometric_shapes_reverse(rectangle_shape, spectral_images):
    rect1 = rectangle_shape.copy()
    rect2 = rectangle_shape.copy()
    spectral_images.vnir.geometric_shapes.append(rect1)
    spectral_images.vnir.geometric_shapes.append(rect2)
    spectral_images.vnir.geometric_shapes.reverse()
    assert spectral_images.vnir.geometric_shapes[0] == rect2
    assert spectral_images.vnir.geometric_shapes[1] == rect1


def test_geometric_shapes_sort(rectangle_shape, spectral_images):
    rect1 = rectangle_shape.copy()
    rect1.label = "B"
    rect2 = rectangle_shape.copy()
    rect2.label = "A"
    spectral_images.vnir.geometric_shapes.clear()
    spectral_images.vnir.geometric_shapes.append(rect1)
    spectral_images.vnir.geometric_shapes.append(rect2)
    spectral_images.vnir.geometric_shapes.sort(key=lambda shape: shape.label)
    assert spectral_images.vnir.geometric_shapes[0].label == "A"
    assert spectral_images.vnir.geometric_shapes[1].label == "B"


def test_geometric_shapes_get_by_name_found(rectangle_shape, spectral_images):
    rect1 = rectangle_shape.copy()
    rect1.label = "Rect1"
    rect2 = rectangle_shape.copy()
    rect2.label = "Rect2"
    spectral_images.vnir.geometric_shapes.shapes = [rect1, rect2]
    found_shape = spectral_images.vnir.geometric_shapes.get_by_name("Rect1")
    assert found_shape == rect1
    assert found_shape.label == "Rect1"


def test_geometric_shapes_get_by_name_not_found(rectangle_shape, spectral_images):
    rect = rectangle_shape.copy()
    rect.label = "Rect"
    spectral_images.vnir.geometric_shapes.shapes = [rect]
    found_shape = spectral_images.vnir.geometric_shapes.get_by_name("NonExistent")
    assert found_shape is None


def test_geometric_shapes_shapes_property_returns_copy(rectangle_shape, spectral_images):
    rect = rectangle_shape
    spectral_images.vnir.geometric_shapes.shapes = [rect]
    shapes = spectral_images.vnir.geometric_shapes.shapes
    shapes.append(rect)  # Modifying the copy shouldn't affect the original
    assert len(spectral_images.vnir.geometric_shapes) == 1


def test_geometric_shapes_index_with_start_stop(rectangle_shape, spectral_images):
    rect1 = rectangle_shape.copy()
    rect2 = rectangle_shape.copy()
    spectral_images.vnir.geometric_shapes.shapes = [rect1, rect2, rect1]
    index = spectral_images.vnir.geometric_shapes.index(rect1, 1, 3)
    assert index == 1


def test_geometric_shapes_check_shape_type_invalid_single():
    with pytest.raises(InvalidInputError):
        from siapy.entities.shapes.geometric_shapes import _check_shape_type

        _check_shape_type("not a shape", is_list=False)


def test_geometric_shapes_sort_invalid_key(rectangle_shape, spectral_images):
    rect = rectangle_shape
    spectral_images.vnir.geometric_shapes.shapes = [rect]
    with pytest.raises(TypeError):
        spectral_images.vnir.geometric_shapes.sort(key=123)  # invalid key type
