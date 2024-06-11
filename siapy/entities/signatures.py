from dataclasses import dataclass
from typing import Annotated, ClassVar

import numpy as np
import pandas as pd

from .pixels import Pixels


@dataclass
class Signals:
    _data: pd.DataFrame

    @property
    def df(self) -> pd.DataFrame:
        return self._data


@dataclass
class Signatures:
    _data: pd.DataFrame

    # Constants:
    SIG: Annotated[ClassVar[str], "Spectral signature values"] = "signature"
    PIX: Annotated[ClassVar[str], "Pixels locations"] = "pixels"

    def __init__(self, data: pd.DataFrame):
        raise RuntimeError(
            "Use Signatures.from_array_and_pixels() to create a new instance."
        )

    @classmethod
    def from_array_and_pixels(cls, image: np.ndarray, pixels: Pixels):
        pixels_df = pixels.df
        u = pixels_df.get(Pixels.U)
        v = pixels_df.get(Pixels.V)
        signatures = list(image[v, u, :])

        signatures_df = pd.DataFrame(signatures)
        data = pd.concat(
            [pixels_df, signatures_df], axis=1, keys=[Signatures.PIX, Signatures.SIG]
        )
        return cls._create(data)

    @classmethod
    def _create(cls, data: pd.DataFrame):
        instance = object.__new__(cls)
        instance._data = data
        return instance

    @property
    def df(self) -> pd.DataFrame:
        return self._data

    def signals(self) -> Signals:
        return Signals(self.df[Signatures.SIG])

    def pixels(self) -> Pixels:
        return Pixels(self.df[Signatures.PIX])

    # def to_numpy(self) -> np.ndarray:
    #     return np.vstack(self.df_filtered(only).to_numpy().flatten())
