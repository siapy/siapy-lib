import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable, Iterator, Optional

from siapy.core.exceptions import InvalidInputError

from .shape import Shape

if TYPE_CHECKING:
    from siapy.entities import SpectralImage


__all__ = [
    "GeometricShapes",
]


@dataclass
class GeometricShapes:
    def __init__(
        self,
        image: "SpectralImage[Any]",
        geometric_shapes: list["Shape"] | None = None,
    ):
        self._image = image
        self._geometric_shapes = geometric_shapes if geometric_shapes is not None else []
        _check_shape_type(self._geometric_shapes, is_list=True)

    def __repr__(self) -> str:
        return f"GeometricShapes(\n{self._geometric_shapes}\n)"

    def __iter__(self) -> Iterator["Shape"]:
        return iter(self.shapes)

    def __getitem__(self, index: int) -> "Shape":
        return self.shapes[index]

    def __setitem__(self, index: int, shape: "Shape") -> None:
        _check_shape_type(shape, is_list=False)
        self._geometric_shapes[index] = shape

    def __len__(self) -> int:
        return len(self._geometric_shapes)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, GeometricShapes):
            raise InvalidInputError(
                {
                    "other_type": type(other).__name__,
                },
                "Comparison is only supported between GeometricShapes instances.",
            )
        return self._geometric_shapes == other._geometric_shapes and self._image == other._image

    @property
    def shapes(self) -> list["Shape"]:
        return self._geometric_shapes.copy()

    @shapes.setter
    def shapes(self, shapes: list["Shape"]) -> None:
        _check_shape_type(shapes, is_list=True)
        self._geometric_shapes = shapes

    def append(self, shape: "Shape") -> None:
        _check_shape_type(shape, is_list=False)
        self._geometric_shapes.append(shape)

    def extend(self, shapes: Iterable["Shape"]) -> None:
        _check_shape_type(shapes, is_list=True)
        self._geometric_shapes.extend(shapes)

    def insert(self, index: int, shape: "Shape") -> None:
        _check_shape_type(shape, is_list=False)
        self._geometric_shapes.insert(index, shape)

    def remove(self, shape: "Shape") -> None:
        _check_shape_type(shape, is_list=False)
        self._geometric_shapes.remove(shape)

    def pop(self, index: int = -1) -> "Shape":
        return self._geometric_shapes.pop(index)

    def clear(self) -> None:
        self._geometric_shapes.clear()

    def index(self, shape: "Shape", start: int = 0, stop: int = sys.maxsize) -> int:
        _check_shape_type(shape, is_list=False)
        return self._geometric_shapes.index(shape, start, stop)

    def count(self, shape: "Shape") -> int:
        _check_shape_type(shape, is_list=False)
        return self._geometric_shapes.count(shape)

    def reverse(self) -> None:
        self._geometric_shapes.reverse()

    def sort(self, key: Any = None, reverse: bool = False) -> None:
        self._geometric_shapes.sort(key=key, reverse=reverse)

    def get_by_name(self, name: str) -> Optional["Shape"]:
        names = [shape.label for shape in self.shapes]
        if name in names:
            index = names.index(name)
            return self.shapes[index]
        return None


def _check_shape_type(shapes: "Shape" | Iterable["Shape"], is_list: bool = False) -> None:
    if is_list and isinstance(shapes, Shape):
        raise InvalidInputError(
            {
                "shapes_type": type(shapes).__name__,
            },
            "Expected an iterable of Shape instances, but got a single Shape instance.",
        )

    if not is_list and isinstance(shapes, Shape):
        return

    if not isinstance(shapes, Iterable):
        raise InvalidInputError(
            {
                "shapes_type": type(shapes).__name__,
            },
            "Shapes must be an instance of Shape or an iterable of Shape instances.",
        )

    if not all(isinstance(shape, Shape) for shape in shapes):
        raise InvalidInputError(
            {
                "invalid_items": [type(shape).__name__ for shape in shapes if not isinstance(shape, Shape)],
            },
            "All items must be instances of Shape subclass.",
        )
