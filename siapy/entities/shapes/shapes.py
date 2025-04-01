from enum import Enum

import numpy as np
from matplotlib.path import Path

from siapy.core.exceptions import InvalidInputError

from ..pixels import Pixels
from .interfaces import ShapeBase

__all__ = [
    "Rectangle",
    "Point",
    "FreeDraw",
    "ShapeGeometry",
    "create_shape_from_geometry",
]


class ShapeGeometry(Enum):
    RECTANGLE = "rectangle"
    POINT = "point"
    FREEDRAW = "freedraw"


def create_shape_from_geometry(
    shape: str | ShapeGeometry,
    pixels: Pixels,
    label: str | None = None,
) -> ShapeBase:
    """
    Create a shape instance based on the provided shape geometry.

    Args:
        shape: The shape type
        pixels: The pixels that define the shape
        label: Optional label for the shape

    Returns:
        An instance of the appropriate shape class

    Raises:
        InvalidInputError: If the shape type is not supported
    """
    if isinstance(shape, str):
        try:
            shape = ShapeGeometry(shape)
        except ValueError:
            pass

    shape_classes: dict[ShapeGeometry, type[ShapeBase]] = {
        ShapeGeometry.RECTANGLE: Rectangle,
        ShapeGeometry.POINT: Point,
        ShapeGeometry.FREEDRAW: FreeDraw,
    }

    if shape not in shape_classes:
        raise InvalidInputError(
            {
                "shape_name": shape,
            },
            "Unsupported shape type",
        )

    assert isinstance(shape, ShapeGeometry), "Shape must be of type ShapeGeometry"
    return shape_classes[shape](pixels=pixels, label=label)


class Rectangle(ShapeBase):
    def convex_hull(self) -> Pixels:
        # Rectangle is defined by two opposite corners
        u1, u2 = self.pixels.u()
        v1, v2 = self.pixels.v()

        pixels_inside = []
        for u_coord in range(min(u1, u2), max(u1, u2) + 1):
            for v_coord in range(min(v1, v2), max(v1, v2) + 1):
                pixels_inside.append((u_coord, v_coord))
        return Pixels.from_iterable(pixels_inside)


class Point(ShapeBase):
    def convex_hull(self) -> Pixels:
        return self.pixels


class FreeDraw(ShapeBase):
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
