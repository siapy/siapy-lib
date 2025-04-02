from enum import Enum
from pathlib import Path
from typing import Optional

import geopandas as gpd
from shapely.geometry import LineString, MultiLineString, MultiPoint, MultiPolygon, Point, Polygon
from shapely.geometry.base import BaseGeometry

from siapy.core.exceptions import ConfigurationError, InvalidTypeError
from siapy.entities.pixels import PixelCoordinate, Pixels

__all__ = [
    "Shape",
    "ShapeGeometryEnum",
]


class ShapeGeometryEnum(Enum):
    """
    Geometry Types:
    - Point: Single coordinate point (x,y)
    - LineString: Series of connected points forming a line
    - Polygon: Closed shape with interior area
    - MultiPoint: Collection of independent points
    - MultiLineString: Collection of independent lines
    - MultiPolygon: Collection of independent polygons
    """

    POINT = "point"
    LINE = "linestring"
    POLYGON = "polygon"
    MULTIPOINT = "multipoint"
    MULTILINE = "multilinestring"
    MULTIPOLYGON = "multipolygon"


class Shape:
    """
    Unified shape class that can be created from shapefiles or programmatically.

    This class uses GeoDataFrame as its primary internal representation.
    Direct initialization is possible but using class methods is recommended.
    """

    def __init__(
        self,
        label: str = "",
        geometry: Optional[BaseGeometry] = None,
        geo_dataframe: Optional[gpd.GeoDataFrame] = None,
    ):
        """Initialize Shape with either a geometry or geodataframe"""
        self._label = label

        if geo_dataframe is not None and geometry is not None:
            raise ConfigurationError("Cannot provide both geometry and geodataframe")

        if geo_dataframe is not None:
            self._geodataframe = geo_dataframe
        elif geometry is not None:
            self._geodataframe = gpd.GeoDataFrame(geometry=[geometry])
        else:
            raise ConfigurationError("Must provide either geometry or geodataframe")

    @classmethod
    def open_shapefile(cls, filepath: str | Path, label: str = "") -> "Shape":
        geo_df = gpd.read_file(filepath)
        return cls(geo_dataframe=geo_df, label=label)

    @classmethod
    def from_geometry(cls, geometry: BaseGeometry, label: str = "") -> "Shape":
        if not isinstance(geometry, BaseGeometry):
            raise InvalidTypeError(
                input_value=geometry,
                allowed_types=BaseGeometry,
                message="Geometry must be of type BaseGeometry",
            )
        return cls(geometry=geometry, label=label)

    @classmethod
    def from_point(cls, x: float, y: float, label: str = "") -> "Shape":
        return cls(geometry=Point(x, y), label=label)

    @classmethod
    def from_multipoint(cls, points: "Pixels", label: str = "") -> "Shape":
        if len(points) < 1:
            raise ConfigurationError("At least one point is required")
        coords = points.to_list()
        return cls(geometry=MultiPoint(coords), label=label)

    @classmethod
    def from_line(cls, pixels: Pixels, label: str = "") -> "Shape":
        if len(pixels) < 2:
            raise ConfigurationError("At least two points are required for a line")

        return cls(geometry=LineString(pixels.to_list()), label=label)

    @classmethod
    def from_multiline(cls, line_segments: list[Pixels], label: str = "") -> "Shape":
        if not line_segments:
            raise ConfigurationError("At least one line segment is required")

        lines = [LineString(segment.to_list()) for segment in line_segments]
        multi_line = MultiLineString(lines)
        return cls(geometry=multi_line, label=label)

    @classmethod
    def from_polygon(cls, exterior: Pixels, holes: Optional[list[Pixels]] = None, label: str = "") -> "Shape":
        if len(exterior) < 3:
            raise ConfigurationError("At least three points are required for a polygon")

        exterior_coords = exterior.to_list()
        # Close the polygon if not already closed
        if exterior_coords[0] != exterior_coords[-1]:
            exterior_coords.append(exterior_coords[0])

        if holes:
            # Close each hole if not already closed
            closed_holes = []
            for hole in holes:
                hole_coords = hole.to_list()
                if hole_coords[0] != hole_coords[-1]:
                    hole_coords.append(hole_coords[0])
                closed_holes.append(hole_coords)
            geometry = Polygon(exterior_coords, closed_holes)
        else:
            geometry = Polygon(exterior_coords)

        return cls(geometry=geometry, label=label)

    @classmethod
    def from_multipolygon(cls, polygons: list[Pixels], label: str = "") -> "Shape":
        if not polygons:
            raise ConfigurationError("At least one polygon is required")

        polygon_objects = []
        for pixels in polygons:
            coords = pixels.to_list()
            # Close the polygon if not already closed
            if coords[0] != coords[-1]:
                coords.append(coords[0])
            polygon_objects.append(Polygon(coords))

        multi_polygon = MultiPolygon(polygon_objects)
        return cls(geometry=multi_polygon, label=label)

    @classmethod
    def from_rectangle(cls, x_min: int, y_min: int, x_max: int, y_max: int, label: str = "") -> "Shape":
        coords = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
        return cls(geometry=Polygon(coords), label=label)

    @classmethod
    def from_circle(cls, center: PixelCoordinate, radius: float, label: str = "") -> "Shape":
        point = Point(center)
        circle = point.buffer(radius)
        return cls(geometry=circle, label=label)

    @property
    def label(self) -> str:
        return self._label

    @property
    def geometry(self) -> BaseGeometry:
        """Return the Shapely geometry."""
        if self._geodataframe.empty:
            raise ConfigurationError("No geometry data available")
        return self._geodataframe.geometry.iloc[0]

    @property
    def shape_type(self) -> str:
        """Return the type of the shape geometry."""
        return self.geometry.geom_type.lower()

    @property
    def properties(self) -> dict:
        """Return additional properties associated with the shape."""
        return self._geodataframe.iloc[0].to_dict()

    @property
    def is_multi(self) -> bool:
        return self.shape_type.startswith("multi")

    @property
    def is_point(self) -> bool:
        return self.shape_type in (ShapeGeometryEnum.POINT.value, ShapeGeometryEnum.MULTIPOINT.value)

    @property
    def is_line(self) -> bool:
        return self.shape_type in (ShapeGeometryEnum.LINE.value, ShapeGeometryEnum.MULTILINE.value)

    @property
    def is_polygon(self) -> bool:
        return self.shape_type in (ShapeGeometryEnum.POLYGON.value, ShapeGeometryEnum.MULTIPOLYGON.value)

    def to_pixels(self) -> Pixels:
        """Convert the shape to pixels representation."""
        geom = self.geometry
        if geom.geom_type == "Point":
            return Pixels.from_iterable([(int(geom.x), int(geom.y))])

        bounds = geom.bounds  # (minx, miny, maxx, maxy)
        if geom.geom_type in ("Polygon", "Rectangle"):
            return Pixels.from_iterable(
                [
                    (int(bounds[0]), int(bounds[1])),  # min corner
                    (int(bounds[2]), int(bounds[3])),  # max corner
                ]
            )

        coords = list(geom.coords) if hasattr(geom, "coords") else []
        return Pixels.from_iterable([(int(x), int(y)) for x, y in coords])

    def convex_hull(self) -> "Shape":
        """Return the convex hull of this shape."""
        return Shape(geometry=self.geometry.convex_hull)

    def buffer(self, distance: float) -> "Shape":
        """Create a buffer around this shape."""
        return Shape(geometry=self.geometry.buffer(distance))

    def intersection(self, other: "Shape") -> "Shape":
        """Find the intersection between this shape and another."""
        return Shape(geometry=self.geometry.intersection(other.geometry))

    def union(self, other: "Shape") -> "Shape":
        """Find the union of this shape with another."""
        return Shape(geometry=self.geometry.union(other.geometry))

    def to_file(self, filepath: str | Path, driver: str = "ESRI Shapefile") -> None:
        """Save the shape to a file."""
        self._geodataframe.to_file(filepath, driver=driver)
