from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Iterable, NamedTuple, Sequence

import numpy as np
import pandas as pd

from siapy.core.exceptions import InvalidInputError

__all__ = [
    "Pixels",
    "PixelCoordinate",
]


@dataclass
class HomogeneousCoordinate:
    X: str = "x"  # x coordinate on the image
    Y: str = "y"  # y coordinate on the image
    H: str = "h"  # homogeneous coordinate


class PixelCoordinate(NamedTuple):
    x: int  # x coordinate on the image
    y: int  # y coordinate on the image


CoordinateInput = PixelCoordinate | tuple[int, int] | Sequence[int]


@dataclass
class Pixels:
    _data: pd.DataFrame
    coords: ClassVar[HomogeneousCoordinate] = HomogeneousCoordinate()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx) -> PixelCoordinate:
        row = self.df.iloc[idx]
        return PixelCoordinate(x=int(row[self.coords.X]), y=int(row[self.coords.Y]))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Pixels):
            return False
        return self.df.equals(other.df)

    def __post_init__(self) -> None:
        validate_pixel_input_dimensions(self._data)

    @classmethod
    def from_iterable(cls, iterable: Iterable[CoordinateInput]) -> "Pixels":
        df = pd.DataFrame(iterable, columns=[cls.coords.X, cls.coords.Y])
        validate_pixel_input_dimensions(df)
        return cls(df)

    @classmethod
    def load_from_parquet(cls, filepath: str | Path) -> "Pixels":
        df = pd.read_parquet(filepath)
        validate_pixel_input_dimensions(df)
        return cls(df)

    @property
    def df(self) -> pd.DataFrame:
        return self._data

    def df_homogenious(self) -> pd.DataFrame:
        df_homo = self.df.copy()
        df_homo[self.coords.H] = 1
        return df_homo

    def u(self) -> pd.Series:
        return self.df[self.coords.X]

    def v(self) -> pd.Series:
        return self.df[self.coords.Y]

    def to_numpy(self) -> np.ndarray:
        return self.df.to_numpy()

    def to_list(self) -> list[PixelCoordinate]:
        return self.df.values.tolist()

    def save_to_parquet(self, filepath: str | Path) -> None:
        self.df.to_parquet(filepath, index=True)


def validate_pixel_input_dimensions(df: pd.DataFrame):
    if df.shape[1] != 2:
        raise InvalidInputError(
            message="Invalid input dimensions: expected 2 columns (u, v), got",
            input_value=df.shape[1],
        )
    if sorted(df.columns) != sorted([HomogeneousCoordinate.X, HomogeneousCoordinate.Y]):
        raise InvalidInputError(
            message=f"Invalid column names: expected ['{HomogeneousCoordinate.X}', '{HomogeneousCoordinate.Y}'], got",
            input_value=sorted(df.columns),
        )
