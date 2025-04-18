from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Iterable, NamedTuple, Sequence, TypeAlias

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from siapy.core.exceptions import InvalidInputError, InvalidTypeError

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
    x: float  # x coordinate on the image
    y: float  # y coordinate on the image


CoordinateInput: TypeAlias = PixelCoordinate | tuple[float, float] | Sequence[float]


@dataclass
class Pixels:
    _data: pd.DataFrame
    coords: ClassVar[HomogeneousCoordinate] = HomogeneousCoordinate()

    def __len__(self) -> int:
        return len(self.df)

    def __repr__(self) -> str:
        return f"Pixels(\n{self.df}\n)"

    def __getitem__(self, indices: Any) -> "Pixels":
        df_slice = self.df.iloc[indices]
        if isinstance(df_slice, pd.Series):
            df_slice = df_slice.to_frame().T
        return Pixels(df_slice)

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

    def u(self) -> "pd.Series[float]":
        # TODO: change to u -> x
        return self.df[self.coords.X]

    def v(self) -> "pd.Series[float]":
        return self.df[self.coords.Y]

    def to_numpy(self) -> NDArray[np.floating[Any]]:
        return self.df.to_numpy()

    def to_list(self) -> list[PixelCoordinate]:
        return self.df.values.tolist()

    def save_to_parquet(self, filepath: str | Path) -> None:
        self.df.to_parquet(filepath, index=True)

    def as_type(self, dtype: type) -> "Pixels":
        converted_df = self.df.copy()
        converted_df[self.coords.X] = converted_df[self.coords.X].astype(dtype)
        converted_df[self.coords.Y] = converted_df[self.coords.Y].astype(dtype)
        return Pixels(converted_df)

    def get_coordinate(self, idx: int) -> PixelCoordinate:
        row = self.df.iloc[idx]
        return PixelCoordinate(x=row[self.coords.X], y=row[self.coords.Y])


def validate_pixel_input_dimensions(df: pd.DataFrame | pd.Series) -> None:
    if isinstance(df, pd.Series):
        raise InvalidTypeError(
            input_value=df,
            allowed_types=pd.DataFrame,
            message="Expected a DataFrame, but got a Series.",
        )

    if df.empty:
        raise InvalidInputError(
            message="Input DataFrame is empty.",
            input_value=df,
        )

    if df.shape[1] != 2:
        raise InvalidInputError(
            message="Invalid input dimensions: expected 2 columns (x, y), got",
            input_value=df.shape[1],
        )

    if sorted(df.columns) != sorted([HomogeneousCoordinate.X, HomogeneousCoordinate.Y]):
        raise InvalidInputError(
            message=f"Invalid column names: expected ['{HomogeneousCoordinate.X}', '{HomogeneousCoordinate.Y}'], got",
            input_value=sorted(df.columns),
        )
