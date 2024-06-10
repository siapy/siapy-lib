from dataclasses import dataclass
from typing import Annotated, ClassVar, Iterable

import pandas as pd


@dataclass
class Pixels:
    _data: pd.DataFrame

    # Constants:
    U: Annotated[ClassVar[str], "u coordinate on the image"] = "u"
    V: Annotated[ClassVar[str], "v coordinate on the image"] = "v"

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
