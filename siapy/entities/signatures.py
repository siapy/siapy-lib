from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from siapy.core.exceptions import InvalidInputError

from .pixels import Pixels

__all__ = [
    "Signatures",
    "Signals",
]


@dataclass
class Signals:
    _data: pd.DataFrame

    def __len__(self) -> int:
        return len(self.df)

    def __repr__(self) -> str:
        return f"Signals(\n{self.df}\n)"

    def __getitem__(self, indices: Any) -> "Signals":
        df_slice = self.df.iloc[indices]
        if isinstance(df_slice, pd.Series):
            df_slice = df_slice.to_frame().T
        return Signals(df_slice)

    @classmethod
    def from_iterable(cls, iterable: Iterable) -> "Signals":
        df = pd.DataFrame(iterable)
        return cls(df)

    @classmethod
    def load_from_parquet(cls, filepath: str | Path) -> "Signals":
        df = pd.read_parquet(filepath)
        return cls(df)

    @property
    def df(self) -> pd.DataFrame:
        return self._data

    def to_numpy(self) -> NDArray[np.floating[Any]]:
        return self.df.to_numpy()

    def mean(self) -> NDArray[np.floating[Any]]:
        return np.nanmean(self.to_numpy(), axis=0)

    def save_to_parquet(self, filepath: str | Path) -> None:
        self.df.to_parquet(filepath, index=True)


@dataclass
class Signatures:
    _pixels: Pixels
    _signals: Signals

    def __repr__(self) -> str:
        return f"Signatures(\n{self.pixels}\n{self.signals}\n)"

    def __len__(self) -> int:
        return len(self.pixels.df)

    def __getitem__(self, indices: Any) -> "Signatures":
        pixels = self.pixels[indices]
        signals = self.signals[indices]
        return Signatures(pixels, signals)

    def __post_init__(self) -> None:
        validate_inputs(self.pixels, self.signals)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Signatures":
        pixels_df = pd.DataFrame(data["pixels"])
        signals_df = pd.DataFrame(data["signals"])
        pixels = Pixels(pixels_df)
        signals = Signals(signals_df)
        validate_inputs(pixels, signals)
        return cls(pixels, signals)

    @classmethod
    def from_array_and_pixels(cls, image: NDArray[np.floating[Any]], pixels: Pixels) -> "Signatures":
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
        validate_inputs(pixels, signals)
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
        validate_inputs(pixels, signals)
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
        validate_inputs(pixels, signals)
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

    def to_numpy(self) -> NDArray[np.floating[Any]]:
        return self.to_dataframe().to_numpy()

    def to_dict(self) -> dict[str, Any]:
        return {
            "pixels": self.pixels.df.to_dict(),
            "signals": self.signals.df.to_dict(),
        }

    def reset_index(self) -> "Signatures":
        return Signatures(
            Pixels(self.pixels.df.reset_index(drop=True)), Signals(self.signals.df.reset_index(drop=True))
        )

    def save_to_parquet(self, filepath: str | Path) -> None:
        self.to_dataframe().to_parquet(filepath, index=True)

    def copy(self) -> "Signatures":
        pixels_df = self.pixels.df.copy()
        signals_df = self.signals.df.copy()
        return Signatures(Pixels(pixels_df), Signals(signals_df))


def validate_inputs(pixels: Pixels, signals: Signals) -> None:
    if len(pixels) != len(signals):
        raise InvalidInputError(
            f"Pixels and signals must have the same number of rows: {len(pixels)} pixels, {len(signals)} signals."
        )
