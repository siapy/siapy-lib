from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from siapy.core.exceptions import InvalidInputError

from .pixels import Pixels

__all__ = [
    "Signatures",
    "Signals",
]


@dataclass
class Signals:
    _data: pd.DataFrame

    def __repr__(self) -> str:
        return f"Signals(\n{self.df}\n)"

    @classmethod
    def load_from_parquet(cls, filepath: str | Path) -> "Signals":
        df = pd.read_parquet(filepath)
        return cls(df)

    @property
    def df(self) -> pd.DataFrame:
        return self._data

    def to_numpy(self) -> np.ndarray:
        return self.df.to_numpy()

    def mean(self) -> np.ndarray:
        return np.nanmean(self.to_numpy(), axis=0)

    def save_to_parquet(self, filepath: str | Path) -> None:
        self.df.to_parquet(filepath, index=True)


@dataclass
class Signatures:
    _pixels: Pixels
    _signals: Signals

    def __repr__(self) -> str:
        return f"Signatures(\n{self.pixels}\n{self.signals}\n)"

    @classmethod
    def from_array_and_pixels(cls, image: np.ndarray, pixels: Pixels) -> "Signatures":
        pixels = pixels.as_type(int)
        u = pixels.u()
        v = pixels.v()

        if image.ndim != 3:
            raise InvalidInputError(f"Expected a 3-dimensional array, but got {image.ndim}-dimensional array.")
        if np.max(u) >= image.shape[1] or np.max(v) >= image.shape[0]:
            raise InvalidInputError(
                f"Pixel coordinates exceed image dimensions: "
                f"image shape is {image.shape}, but max u={np.max(u)}, max v={np.max(v)}."
            )

        signals_list = image[v, u, :]
        signals = Signals(pd.DataFrame(signals_list))
        return cls(pixels, signals)

    @classmethod
    def from_dataframe(cls, dataframe: pd.DataFrame) -> "Signatures":
        if not all(coord in dataframe.columns for coord in [Pixels.coords.X, Pixels.coords.Y]):
            raise InvalidInputError(
                dataframe.columns.tolist(),
                f"DataFrame must include columns for both '{Pixels.coords.X}' and '{Pixels.coords.Y}' coordinates.",
            )
        pixels = Pixels(dataframe[[Pixels.coords.X, Pixels.coords.Y]])
        signals = Signals(dataframe.drop(columns=[Pixels.coords.X, Pixels.coords.Y]))
        return cls(pixels, signals)

    @classmethod
    def from_dataframe_multiindex(cls, df: pd.DataFrame) -> "Signatures":
        if not isinstance(df.columns, pd.MultiIndex):
            raise InvalidInputError(
                type(df.columns),
                "DataFrame must have MultiIndex columns",
            )

        pixel_data = df.xs("pixel", axis=1, level="category")
        signal_data = df.xs("signal", axis=1, level="category")

        assert isinstance(pixel_data, pd.DataFrame)
        assert isinstance(signal_data, pd.DataFrame)

        pixels = Pixels(pixel_data)
        signals = Signals(signal_data)

        return cls(pixels, signals)

    @classmethod
    def open_parquet(cls, filepath: str | Path) -> "Signatures":
        df = pd.read_parquet(filepath)
        return cls.from_dataframe(df)

    @property
    def pixels(self) -> Pixels:
        return self._pixels

    @property
    def signals(self) -> Signals:
        return self._signals

    def to_dataframe(self) -> pd.DataFrame:
        return pd.concat([self.pixels.df, self.signals.df], axis=1)

    def to_dataframe_multiindex(self) -> pd.DataFrame:
        pixel_columns = pd.MultiIndex.from_tuples(
            [("pixel", "x"), ("pixel", "y")],
            names=["category", "coordinate"],
        )
        signal_columns = pd.MultiIndex.from_tuples(
            [("signal", col) for col in self.signals.df.columns],
            names=["category", "channel"],
        )

        pixel_df = pd.DataFrame(self.pixels.df.values, columns=pixel_columns)
        signal_df = pd.DataFrame(self.signals.df.values, columns=signal_columns)
        return pd.concat([pixel_df, signal_df], axis=1)

    def to_numpy(self) -> np.ndarray:
        return self.to_dataframe().to_numpy()

    def save_to_parquet(self, filepath: str | Path) -> None:
        self.to_dataframe().to_parquet(filepath, index=True)
