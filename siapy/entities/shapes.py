from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from matplotlib.path import Path

from .pixels import Pixels

SHAPE_TYPE_RECTANGLE = "rectangle"
SHAPE_TYPE_POINT = "point"
SHAPE_TYPE_FREEDRAW = "freedraw"
ShapeType = Literal[SHAPE_TYPE_RECTANGLE, SHAPE_TYPE_POINT, SHAPE_TYPE_FREEDRAW]  # type: ignore


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
            SHAPE_TYPE_RECTANGLE: Rectangle,
            SHAPE_TYPE_POINT: Point,
            SHAPE_TYPE_FREEDRAW: FreeDraw,
        }
        if shape_type in types_map:
            return types_map[shape_type](
                shape_type=shape_type,
                pixels=pixels,
                label=label,
            )
        else:
            raise ValueError(f"Unsupported shape type: {shape_type}")

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
        raise NotImplementedError(
            "convex_hull() method is not implemented for the base Shape class."
        )


class Rectangle(Shape):
    def __init__(self, pixels: Pixels, label: str | None = None, **kwargs: Any):
        super().__init__(SHAPE_TYPE_RECTANGLE, pixels, label)

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
        super().__init__(SHAPE_TYPE_POINT, pixels, label)

    def convex_hull(self) -> Pixels:
        return self.pixels


class FreeDraw(Shape):
    def __init__(self, pixels: Pixels, label: str | None = None, **kwargs: Any):
        super().__init__(SHAPE_TYPE_FREEDRAW, pixels, label)

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
