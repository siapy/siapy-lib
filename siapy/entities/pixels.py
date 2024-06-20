from dataclasses import dataclass
from typing import Annotated, ClassVar, Iterable, NamedTuple

import numpy as np
import pandas as pd


class Coordinates(NamedTuple):
    U: Annotated[str, "u - x coordinate on the image"] = "u"
    V: Annotated[str, "v - y coordinate on the image"] = "v"
    H: Annotated[str, "h - homogenious coordinate"] = "h"


@dataclass
class Pixels:
    _data: pd.DataFrame
    coords: ClassVar[Coordinates] = Coordinates()

    @classmethod
    def from_iterable(
        cls,
        iterable: Iterable[
            tuple[
                Annotated[int, "u coordinate on the image"],
                Annotated[int, "v coordinate on the image"],
            ]
        ],
    ):
        df = pd.DataFrame(iterable, columns=[Pixels.coords.U, Pixels.coords.V])
        return cls(df)

    @property
    def df(self) -> pd.DataFrame:
        return self._data

    def df_homogenious(self) -> pd.DataFrame:
        df_homo = self.df.copy()
        df_homo[Pixels.coords.H] = 1
        return df_homo

    def u(self) -> pd.Series:
        return self.df[Pixels.coords.U]

    def v(self) -> pd.Series:
        return self.df[Pixels.coords.V]

    def to_numpy(self) -> np.ndarray:
        return self.df.to_numpy()
