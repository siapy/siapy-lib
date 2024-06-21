from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .pixels import Pixels


@dataclass
class Signals:
    _data: pd.DataFrame

    @property
    def df(self) -> pd.DataFrame:
        return self._data

    def to_numpy(self) -> np.ndarray:
        return self.df.to_numpy()

    def mean(self) -> np.ndarray:
        return np.nanmean(self.to_numpy(), axis=0)


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
            "Use Signatures.from_array_and_pixels() to create a new instance."
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

    @property
    def pixels(self) -> Pixels:
        return self._pixels

    @property
    def signals(self) -> Signals:
        return self._signals

    def df(self) -> pd.DataFrame:
        return pd.concat([self.pixels.df, self.signals.df], axis=1)

    def to_numpy(self) -> np.ndarray:
        return self.df().to_numpy()

    def filter(self) -> SignaturesFilter:
        return SignaturesFilter(self.pixels, self.signals)
