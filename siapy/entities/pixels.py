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
    "CoordinateInput",
    "HomogeneousCoordinate",
    "validate_pixel_input",
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

    def __array__(self, dtype: np.dtype | None = None) -> NDArray[np.floating[Any]]:
        """Convert this pixels object to a numpy array when requested by NumPy."""
        array = self.to_numpy()
        if dtype is not None:
            return array.astype(dtype)
        return array

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

    def x(self) -> "pd.Series[float]":
        return self.df[self.coords.X]

    def y(self) -> "pd.Series[float]":
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


def validate_pixel_input(input_data: Pixels | pd.DataFrame | Iterable[CoordinateInput]) -> Pixels:
    """Validates and converts various input types to Pixels object."""
    try:
        if isinstance(input_data, Pixels):
            return input_data

        if isinstance(input_data, pd.DataFrame):
            validate_pixel_input_dimensions(input_data)
            return Pixels(input_data)

        if isinstance(input_data, np.ndarray):
            if input_data.ndim != 2 or input_data.shape[1] != 2:
                raise InvalidInputError(
                    input_value=input_data.shape,
                    message=f"NumPy array must be 2D with shape (n, 2), got shape {input_data.shape}",
                )
            return Pixels(pd.DataFrame(input_data, columns=[HomogeneousCoordinate.X, HomogeneousCoordinate.Y]))

        if isinstance(input_data, Iterable):
            return Pixels.from_iterable(input_data)  # type: ignore

        raise InvalidTypeError(
            input_value=input_data,
            allowed_types=(Pixels, pd.DataFrame, np.ndarray, Iterable),
            message=f"Unsupported input type: {type(input_data).__name__}",
        )

    except Exception as e:
        if isinstance(e, (InvalidTypeError, InvalidInputError)):
            raise

        raise InvalidInputError(
            input_value=input_data,
            message=f"Failed to convert input to Pixels: {str(e)}"
            f"\nExpected a Pixels instance or an iterable (e.g. list, np.array, tuple, pd.DataFrame)."
            f"\nThe input must contain 2D coordinates with x and y values.",
        )
