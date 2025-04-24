from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from shapely.geometry import LineString, MultiLineString, MultiPoint, MultiPolygon, Point, Polygon
from shapely.geometry.base import BaseGeometry

from siapy.core.exceptions import ConfigurationError, InvalidFilepathError, InvalidInputError, InvalidTypeError
from siapy.entities.pixels import CoordinateInput, PixelCoordinate, Pixels, validate_pixel_input

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


@dataclass
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

    def __repr__(self) -> str:
        return f"Shape(label='{self.label}', geometry_type={self.shape_type})"

    def __len__(self) -> int:
        return len(self.df)

    def __array__(self, dtype: np.dtype | None = None) -> NDArray[np.floating[Any]]:
        """Convert this shape object to a numpy array when requested by NumPy."""
        array = self.to_numpy()
        if dtype is not None:
            return array.astype(dtype)
        return array

    @classmethod
    def open_shapefile(cls, filepath: str | Path, label: str = "") -> "Shape":
        filepath = Path(filepath)
        if not filepath.exists():
            raise InvalidFilepathError(filepath)
        try:
            geo_df = gpd.read_file(filepath)
        except Exception as e:
            raise InvalidInputError({"filepath": str(filepath)}, f"Failed to open shapefile: {e}") from e
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
    def from_geodataframe(cls, geo_dataframe: gpd.GeoDataFrame, label: str = "") -> "Shape":
        if not isinstance(geo_dataframe, gpd.GeoDataFrame):
            raise InvalidTypeError(
                input_value=geo_dataframe,
                allowed_types=gpd.GeoDataFrame,
                message="GeoDataFrame must be of type GeoDataFrame",
            )
        return cls(geo_dataframe=geo_dataframe, label=label)

    @classmethod
    def from_point(cls, x: float, y: float, label: str = "") -> "Shape":
        return cls(geometry=Point(x, y), label=label)

    @classmethod
    def from_multipoint(cls, points: Pixels | pd.DataFrame | Iterable[CoordinateInput], label: str = "") -> "Shape":
        points = validate_pixel_input(points)
        if len(points) < 1:
            raise ConfigurationError("At least one point is required")
        coords = points.to_list()
        return cls(geometry=MultiPoint(coords), label=label)

    @classmethod
    def from_line(cls, pixels: Pixels | pd.DataFrame | Iterable[CoordinateInput], label: str = "") -> "Shape":
        pixels = validate_pixel_input(pixels)
        if len(pixels) < 2:
            raise ConfigurationError("At least two points are required for a line")

        return cls(geometry=LineString(pixels.to_list()), label=label)

    @classmethod
    def from_multiline(
        cls, line_segments: list[Pixels | pd.DataFrame | Iterable[CoordinateInput]], label: str = ""
    ) -> "Shape":
        if not line_segments:
            raise ConfigurationError("At least one line segment is required")

        lines = []
        for segment in line_segments:
            validated_segment = validate_pixel_input(segment)
            lines.append(LineString(validated_segment.to_list()))

        multi_line = MultiLineString(lines)
        return cls(geometry=multi_line, label=label)

    @classmethod
    def from_polygon(
        cls,
        exterior: Pixels | pd.DataFrame | Iterable[CoordinateInput],
        holes: Optional[list[Pixels | pd.DataFrame | Iterable[CoordinateInput]]] = None,
        label: str = "",
    ) -> "Shape":
        exterior = validate_pixel_input(exterior)
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
                validated_hole = validate_pixel_input(hole)
                hole_coords = validated_hole.to_list()
                if hole_coords[0] != hole_coords[-1]:
                    hole_coords.append(hole_coords[0])
                closed_holes.append(hole_coords)
            geometry = Polygon(exterior_coords, closed_holes)
        else:
            geometry = Polygon(exterior_coords)

        return cls(geometry=geometry, label=label)

    @classmethod
    def from_multipolygon(
        cls, polygons: list[Pixels | pd.DataFrame | Iterable[CoordinateInput]], label: str = ""
    ) -> "Shape":
        if not polygons:
            raise ConfigurationError("At least one polygon is required")

        polygon_objects = []
        for pixels in polygons:
            validated_pixels = validate_pixel_input(pixels)
            coords = validated_pixels.to_list()
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
    def df(self) -> gpd.GeoDataFrame:
        return self._geodataframe

    @property
    def label(self) -> str:
        return self._label

    @label.setter
    def label(self, label: str) -> None:
        if not isinstance(label, str):
            raise InvalidTypeError(
                input_value=label,
                allowed_types=str,
                message="Label must be a string",
            )
        self._label = label

    @property
    def geometry(self) -> gpd.GeoSeries:
        if self.df.empty:
            raise ConfigurationError("No geometry data available in GeoDataFrame")
        return self.df.geometry

    @geometry.setter
    def geometry(self, geometry: gpd.GeoSeries) -> None:
        if not isinstance(geometry, gpd.GeoSeries):
            raise InvalidTypeError(
                input_value=geometry,
                allowed_types=gpd.GeoSeries,
                message="Geometry must be of type GeoSeries",
            )
        self.df.geometry = geometry

    @property
    def shape_type(self) -> str:
        return self.geometry.geom_type[0].lower()

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

    @property
    def boundary(self) -> gpd.GeoSeries:
        return self.geometry.boundary

    @property
    def bounds(self) -> pd.DataFrame:
        return self.geometry.bounds

    @property
    def centroid(self) -> gpd.GeoSeries:
        return self.geometry.centroid

    @property
    def convex_hull(self) -> gpd.GeoSeries:
        return self.geometry.convex_hull

    @property
    def envelope(self) -> gpd.GeoSeries:
        return self.geometry.envelope

    @property
    def exterior(self) -> gpd.GeoSeries:
        if not self.is_polygon:
            raise ConfigurationError("Exterior is only applicable to Polygon geometries")
        return self.geometry.exterior

    def copy(self) -> "Shape":
        """Create a deep copy of the Shape instance."""
        copied_df = self.df.copy(deep=True)
        return Shape(label=self.label, geo_dataframe=copied_df)

    def buffer(self, distance: float) -> "Shape":
        buffered_geometry = self.geometry.buffer(distance)
        result = self.copy()
        result.geometry = buffered_geometry
        return result

    def intersection(self, other: "Shape") -> "Shape":
        intersection_geometry = self.geometry.intersection(other.geometry)
        result = self.copy()
        result.geometry = intersection_geometry
        return result

    def union(self, other: "Shape") -> "Shape":
        union_geometry = self.geometry.union(other.geometry)
        result = self.copy()
        result.geometry = union_geometry
        return result

    def to_file(self, filepath: str | Path, driver: str = "ESRI Shapefile") -> None:
        self._geodataframe.to_file(filepath, driver=driver)

    def to_numpy(self) -> NDArray[np.floating[Any]]:
        return self.df.to_numpy()
