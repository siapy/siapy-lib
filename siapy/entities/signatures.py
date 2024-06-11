from dataclasses import dataclass
from enum import Enum
from typing import Annotated, ClassVar

import numpy as np
import pandas as pd

from .pixels import Pixels


class SigFilterEnum(Enum):
    SIGNATURES = "signatures"
    PIXELS = "pixels"


@dataclass
class Signatures:
    _data: pd.DataFrame

    # Constants:
    SIG: Annotated[ClassVar[str], "Spectral signature values"] = "signature"

    @classmethod
    def from_array_and_pixels(cls, image: np.ndarray, pixels: Pixels):
        pixels_df = pixels.df
        u = pixels_df.get(Pixels.U)
        v = pixels_df.get(Pixels.V)
        signatures = list(image[v, u, :])

        data_out = pd.DataFrame({Signatures.SIG: signatures})
        data_out = pd.concat([pixels_df, data_out], axis=1)
        return cls(data_out)

    @property
    def df(self) -> pd.DataFrame:
        return self._data

    def df_filtered(self, only: SigFilterEnum | None = None) -> pd.DataFrame:
        if only == SigFilterEnum.SIGNATURES:
            return self.df[[self.SIG]]
        elif only == SigFilterEnum.PIXELS:
            return self.df[[Pixels.U, Pixels.V]]
        elif only is None:
            return self.df
        else:
            raise ValueError(f"Invalid argument: {only}. Expected {SigFilterEnum}")

    # def to_numpy(self, only: SigFilterEnum | None = None) -> np.ndarray:
    #     return np.vstack(self.df_filtered(only).to_numpy())
