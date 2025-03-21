from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, ClassVar, NamedTuple, cast

import geopandas as gpd
import numpy as np
from shapely.geometry.base import BaseGeometry

from siapy.core.exceptions import InvalidFilepathError

__all__ = [
    "Shapefile",
]


class Coordinates(NamedTuple):
    X: Annotated[str, "x coordinate in global coordinate system"] = "x"
    Y: Annotated[str, "x coordinate in global coordinate system"] = "y"


@dataclass
class Shapefile:
    _data: gpd.GeoDataFrame
    coords: ClassVar[Coordinates] = Coordinates()

    def __len__(self) -> int:
        return len(self.df)

    @classmethod
    def from_path(cls, filepath: str | Path) -> "Shapefile":
        if not Path(filepath).exists():
            raise InvalidFilepathError(filepath)
        shapefile = gpd.read_file(filepath)
        return cls(shapefile)

    @property
    def df(self) -> gpd.GeoDataFrame:
        return self._data

    # @property
    # def filepath(self) -> Path:
    #     return self._filepath

    @property
    def geometry_type(self) -> str:
        """Return the geometry type of the first feature in the shapefile."""
        geom = cast(BaseGeometry, self.df.geometry.iloc[0])
        return geom.geom_type

    @property
    def geometry_types(self) -> list[str]:
        """Return a list of all geometry types in the shapefile."""
        return [geom.geom_type for geom in self.df.geometry]

    def has_consistent_geometry_type(self) -> bool:
        """Check if all features have the same geometry type."""
        types = set(self.geometry_types)
        return len(types) == 1

    def to_numpy(self) -> np.ndarray:
        return self.df.to_numpy()
