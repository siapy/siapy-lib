from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from siapy.utils.general import get_classmethods

from .pixels import Pixels


@dataclass
class Signals:
    _data: pd.DataFrame

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
class SignaturesFilter:
    def __init__(self, pixels: Pixels, signals: Signals):
        self.pixels = pixels
        self.signals = signals

    def build(self) -> "Signatures":
        return Signatures._create(self.pixels, self.signals)

    def rows(self, rows: list[int] | slice | list[bool]) -> "SignaturesFilter":
        filtered_pixels_df = self.pixels.df.iloc[rows]
        filtered_signals_df = self.signals.df.iloc[rows]
        filtered_pixels = Pixels(pd.DataFrame(filtered_pixels_df))
        filtered_signals = Signals(pd.DataFrame(filtered_signals_df))
        return SignaturesFilter(filtered_pixels, filtered_signals)

    def cols(self, cols: list[int] | slice | list[bool]) -> "SignaturesFilter":
        filtered_signals_df = self.signals.df.iloc[:, cols]
        filtered_signals = Signals(pd.DataFrame(filtered_signals_df))
        return SignaturesFilter(self.pixels, filtered_signals)


@dataclass
class Signatures:
    _pixels: Pixels
    _signals: Signals

    def __init__(self, *args: Any, **kwargs: Any):
        raise RuntimeError(
            f"Use any of the @classmethod to create a new instance: {get_classmethods(Signatures)}"
        )

    @classmethod
    def _create(cls, pixels: Pixels, signals: Signals) -> "Signatures":
        instance = object.__new__(cls)
        instance._pixels = pixels
        instance._signals = signals
        return instance

    @classmethod
    def from_array_and_pixels(cls, image: np.ndarray, pixels: Pixels) -> "Signatures":
        u = pixels.u()
        v = pixels.v()
        signals_list = list(image[v, u, :])
        signals = Signals(pd.DataFrame(signals_list))
        return cls._create(pixels, signals)

    @classmethod
    def from_dataframe(cls, dataframe: pd.DataFrame) -> "Signatures":
        if not all(
            coord in dataframe.columns for coord in [Pixels.coords.U, Pixels.coords.V]
        ):
            raise ValueError(
                f"DataFrame must include columns for both '{Pixels.coords.U}'"
                f" and '{Pixels.coords.V}' coordinates."
            )
        pixels = Pixels(dataframe[[Pixels.coords.U, Pixels.coords.V]])
        signals = Signals(dataframe.drop(columns=[Pixels.coords.U, Pixels.coords.V]))
        return cls._create(pixels, signals)

    @classmethod
    def load_from_parquet(cls, filepath: str | Path) -> "Signatures":
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

    def to_numpy(self) -> np.ndarray:
        return self.to_dataframe().to_numpy()

    def filter(self) -> SignaturesFilter:
        return SignaturesFilter(self.pixels, self.signals)

    def save_to_parquet(self, filepath: str | Path) -> None:
        self.to_dataframe().to_parquet(filepath, index=True)
