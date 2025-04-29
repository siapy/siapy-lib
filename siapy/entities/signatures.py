from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from siapy.core import logger
from siapy.core.exceptions import InvalidInputError, InvalidTypeError

from .pixels import CoordinateInput, Pixels, validate_pixel_input

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

    def __array__(self, dtype: np.dtype | None = None) -> NDArray[np.floating[Any]]:
        """Convert this signals object to a numpy array when requested by NumPy."""
        array = self.to_numpy()
        if dtype is not None:
            return array.astype(dtype)
        return array

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

    def average_signal(self, axis: int | tuple[int, ...] | Sequence[int] | None = 0) -> NDArray[np.floating[Any]]:
        return np.nanmean(self.to_numpy(), axis=axis)

    def save_to_parquet(self, filepath: str | Path) -> None:
        self.df.to_parquet(filepath, index=True)


def validate_signal_input(input_data: Signals | pd.DataFrame | Iterable[Sequence[float]]) -> Signals:
    """Validates and converts various input types to Signals object."""
    try:
        if isinstance(input_data, Signals):
            return input_data

        if isinstance(input_data, pd.DataFrame):
            return Signals(input_data)

        if isinstance(input_data, np.ndarray):
            if input_data.ndim not in (1, 2):
                raise InvalidInputError(
                    input_value=input_data.shape,
                    message=f"NumPy array must be 1D or 2D, got shape {input_data.shape}",
                )
            if input_data.ndim == 1:
                input_data = input_data.reshape(1, -1)  # Reshape 1D array to 2D if needed
            return Signals(pd.DataFrame(input_data))

        if isinstance(input_data, Iterable):
            return Signals.from_iterable(input_data)

        raise InvalidTypeError(
            input_value=input_data,
            allowed_types=(Signals, pd.DataFrame, np.ndarray, Iterable),
            message=f"Unsupported input type: {type(input_data).__name__}",
        )

    except Exception as e:
        if isinstance(e, (InvalidTypeError, InvalidInputError)):
            raise

        raise InvalidInputError(
            input_value=input_data,
            message=f"Failed to convert input to Signals: {str(e)}"
            f"\nExpected a Signals instance or an iterable (e.g. list, np.array, pd.DataFrame)."
            f"\nThe input must contain spectral signal values.",
        )


@dataclass
class Signatures:
    pixels: Pixels
    signals: Signals

    def __repr__(self) -> str:
        return f"Signatures(\n{self.pixels}\n{self.signals}\n)"

    def __len__(self) -> int:
        return len(self.pixels.df)

    def __getitem__(self, indices: Any) -> "Signatures":
        pixels = self.pixels[indices]
        signals = self.signals[indices]
        return Signatures(pixels, signals)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Signatures):
            return False
        return self.pixels.df.equals(other.pixels.df) and self.signals.df.equals(other.signals.df)

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
    def from_array_and_pixels(
        cls, array: NDArray[np.floating[Any]], pixels: Pixels | pd.DataFrame | Iterable[CoordinateInput]
    ) -> "Signatures":
        pixels = validate_pixel_input(pixels)
        if pd.api.types.is_float_dtype(pixels.df.dtypes.x) or pd.api.types.is_float_dtype(pixels.df.dtypes.y):
            logger.warning("Pixel DataFrame contains float values. Converting to integers.")
            pixels = pixels.as_type(int)

        x = pixels.x()
        y = pixels.y()

        if array.ndim != 3:
            raise InvalidInputError(f"Expected a 3-dimensional array, but got {array.ndim}-dimensional array.")
        if np.max(x) >= array.shape[1] or np.max(y) >= array.shape[0]:
            raise InvalidInputError(
                f"Pixel coordinates exceed image dimensions: "
                f"image shape is {array.shape}, but max u={np.max(x)}, max v={np.max(y)}."
            )

        signals_list = array[y, x, :]
        signals = Signals(pd.DataFrame(signals_list))
        validate_inputs(pixels, signals)
        return cls(pixels, signals)

    @classmethod
    def from_signals_and_pixels(
        cls,
        signals: Signals | pd.DataFrame | Iterable[Any],
        pixels: Pixels | pd.DataFrame | Iterable[CoordinateInput],
    ) -> "Signatures":
        pixels = validate_pixel_input(pixels)
        signals = validate_signal_input(signals)
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

    def to_numpy(self) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
        return self.pixels.to_numpy(), self.signals.to_numpy()

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
