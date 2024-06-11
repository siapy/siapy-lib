from dataclasses import dataclass
from typing import Annotated, ClassVar, Iterable

import pandas as pd


@dataclass
class Pixels:
    _data: pd.DataFrame

    # Constants:
    U: Annotated[ClassVar[str], "u - x coordinate on the image"] = "u"
    V: Annotated[ClassVar[str], "v - y coordinate on the image"] = "v"
    H: Annotated[ClassVar[str], "h - homogenious coordinate"] = "h"

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
        df = pd.DataFrame(iterable, columns=[Pixels.U, Pixels.V])
        return cls(df)

    @property
    def df(self) -> pd.DataFrame:
        return self._data

    def df_homogenious(self) -> pd.DataFrame:
        df_homo = self.df.copy()
        df_homo[Pixels.H] = 1
        return df_homo

    def x(self):
        return self.df[self.U]

    def y(self):
        return self.df[self.V]
