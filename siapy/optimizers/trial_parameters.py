from dataclasses import dataclass
from typing import Any, Literal, Sequence

from pydantic import BaseModel


class FloatParameter(BaseModel):
    name: str
    low: float
    high: float
    step: float | None = None
    log: bool = False


class IntParameter(BaseModel):
    name: str
    low: int
    high: int
    step: int = 1
    log: bool = False


class CategoricalParameter(BaseModel):
    name: str
    choices: Sequence[None | bool | int | float | str]


ParametersDictType = dict[
    Literal["float_parameters", "int_parameters", "categorical_parameters"],
    list[dict[str, Any]],
]


@dataclass
class TrialParameters:
    def __init__(
        self,
        float_parameters: list[FloatParameter] | None = None,
        int_parameters: list[IntParameter] | None = None,
        categorical_parameters: list[CategoricalParameter] | None = None,
    ):
        self._float_parameters = float_parameters or []
        self._int_parameters = int_parameters or []
        self._categorical_parameters = categorical_parameters or []

    @classmethod
    def from_dict(cls, parameters: ParametersDictType):
        float_params = [
            FloatParameter(**fp) for fp in parameters.get("float_parameters", [])
        ]
        int_params = [IntParameter(**ip) for ip in parameters.get("int_parameters", [])]
        cat_params = [
            CategoricalParameter(**cp)
            for cp in parameters.get("categorical_parameters", [])
        ]
        return cls(
            float_parameters=float_params,
            int_parameters=int_params,
            categorical_parameters=cat_params,
        )

    @property
    def float_parameters(self):
        return self._float_parameters

    @property
    def int_parameters(self):
        return self._int_parameters

    @property
    def categorical_parameters(self):
        return self._categorical_parameters
