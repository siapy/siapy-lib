import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable, Iterator, Literal

import numpy as np
from matplotlib.path import Path

from siapy.core.exceptions import InvalidInputError, MethodNotImplementedError

from .pixels import Pixels

if TYPE_CHECKING:
    from .images import SpectralImage

__all__ = [
    "Shape",
    "GeometricShapes",
    "Rectangle",
    "Point",
    "FreeDraw",
]

_SHAPE_TYPE_RECTANGLE = "rectangle"
_SHAPE_TYPE_POINT = "point"
_SHAPE_TYPE_FREEDRAW = "freedraw"
ShapeType = Literal[_SHAPE_TYPE_RECTANGLE, _SHAPE_TYPE_POINT, _SHAPE_TYPE_FREEDRAW]  # type: ignore


@dataclass
class Shape(ABC):
    def __init__(
        self,
        shape_type: ShapeType,
        pixels: Pixels,
        label: str | None = None,
    ):
        self._shape_type = shape_type
        self._pixels = pixels
        self._label = label

    @classmethod
    def from_shape_type(
        cls,
        shape_type: ShapeType,
        pixels: Pixels,
        label: str | None = None,
    ) -> "Shape":
        types_map: dict[ShapeType, type[Shape]] = {
            _SHAPE_TYPE_RECTANGLE: Rectangle,
            _SHAPE_TYPE_POINT: Point,
            _SHAPE_TYPE_FREEDRAW: FreeDraw,
        }
        if shape_type in types_map:
            return types_map[shape_type](
                shape_type=shape_type,
                pixels=pixels,
                label=label,
            )
        else:
            raise InvalidInputError(
                {
                    "shape_type": shape_type,
                },
                f"Unsupported shape type: {shape_type}",
            )

    @property
    def shape_type(self) -> str:
        return self._shape_type

    @property
    def pixels(self) -> Pixels:
        return self._pixels

    @property
    def label(self) -> str | None:
        return self._label

    @abstractmethod
    def convex_hull(self):
        raise MethodNotImplementedError(self.__class__.__name__, "convex_hull")


@dataclass
class GeometricShapes:
    def __init__(
        self,
        image: "SpectralImage",
        geometric_shapes: list["Shape"] | None = None,
    ):
        self._image = image
        self._geometric_shapes = geometric_shapes if geometric_shapes is not None else []

    def __iter__(self) -> Iterator[Shape]:
        return iter(self.shapes)

    def __getitem__(self, index: int) -> Shape:
        return self.shapes[index]

    def __setitem__(self, index: int, shape: Shape):
        self._check_shape_type(shape)
        self._geometric_shapes[index] = shape

    def __len__(self) -> int:
        return len(self._geometric_shapes)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, GeometricShapes):
            return NotImplemented
        return self._geometric_shapes == other._geometric_shapes and self._image == other._image

    @property
    def shapes(self) -> list["Shape"]:
        return self._geometric_shapes.copy()

    @shapes.setter
    def shapes(self, shapes: list["Shape"]):
        self._check_shape_type(shapes)
        self._geometric_shapes = shapes

    def append(self, shape: "Shape"):
        self._check_shape_type(shape)
        self._geometric_shapes.append(shape)

    def extend(self, shapes: Iterable["Shape"]):
        self._check_shape_type(shapes)
        self._geometric_shapes.extend(shapes)

    def insert(self, index: int, shape: "Shape"):
        self._check_shape_type(shape)
        self._geometric_shapes.insert(index, shape)

    def remove(self, shape: "Shape"):
        self._check_shape_type(shape)
        self._geometric_shapes.remove(shape)

    def pop(self, index: int = -1) -> "Shape":
        return self._geometric_shapes.pop(index)

    def clear(self):
        self._geometric_shapes.clear()

    def index(self, shape: "Shape", start: int = 0, stop: int = sys.maxsize) -> int:
        self._check_shape_type(shape)
        return self._geometric_shapes.index(shape, start, stop)

    def count(self, shape: "Shape") -> int:
        self._check_shape_type(shape)
        return self._geometric_shapes.count(shape)

    def reverse(self):
        self._geometric_shapes.reverse()

    def sort(self, key: Any = None, reverse: bool = False):
        self._geometric_shapes.sort(key=key, reverse=reverse)

    def get_by_name(self, name: str) -> Shape | None:
        names = [shape.label for shape in self.shapes]
        if name in names:
            index = names.index(name)
            return self.shapes[index]
        return None

    def _check_shape_type(self, shapes: Shape | Iterable[Shape]):
        if isinstance(shapes, Shape):
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


class Rectangle(Shape):
    def __init__(self, pixels: Pixels, label: str | None = None, **kwargs: Any):
        super().__init__(_SHAPE_TYPE_RECTANGLE, pixels, label)

    def convex_hull(self) -> Pixels:
        # Rectangle is defined by two opposite corners
        u1, u2 = self.pixels.u()
        v1, v2 = self.pixels.v()

        pixels_inside = []
        for u_coord in range(min(u1, u2), max(u1, u2) + 1):
            for v_coord in range(min(v1, v2), max(v1, v2) + 1):
                pixels_inside.append((u_coord, v_coord))
        return Pixels.from_iterable(pixels_inside)


class Point(Shape):
    def __init__(self, pixels: Pixels, label: str | None = None, **kwargs: Any):
        super().__init__(_SHAPE_TYPE_POINT, pixels, label)

    def convex_hull(self) -> Pixels:
        return self.pixels


class FreeDraw(Shape):
    def __init__(self, pixels: Pixels, label: str | None = None, **kwargs: Any):
        super().__init__(_SHAPE_TYPE_FREEDRAW, pixels, label)

    def convex_hull(self) -> Pixels:
        if len(self.pixels) < 3:
            return self.pixels

        points = self.pixels.to_numpy()
        points_path = Path(points)

        # Create a grid of points that covers the convex hull area
        u_min, v_min = points.min(axis=0)
        u_max, v_max = points.max(axis=0)
        u, v = np.meshgrid(np.arange(u_min, u_max + 1), np.arange(v_min, v_max + 1))
        grid_points = np.vstack((u.flatten(), v.flatten())).T

        # Filter points that are inside the convex hull
        inside_points = grid_points[points_path.contains_points(grid_points)]

        return Pixels.from_iterable(inside_points.tolist())
