from dataclasses import dataclass
from typing import Annotated, Any, Sequence

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
    Annotated[
        str,
        "Can be one of: 'float_parameters', 'int_parameters', 'categorical_parameters'.",
    ],
    list[
        Annotated[
            dict[str, Any],
            "Dictionary of parameters, belonging to specific type of parameter.",
        ]
    ],
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
    def from_dict(cls, parameters: ParametersDictType) -> "TrialParameters":
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
    def float_parameters(self) -> list[FloatParameter]:
        return self._float_parameters

    @property
    def int_parameters(self) -> list[IntParameter]:
        return self._int_parameters

    @property
    def categorical_parameters(self) -> list[CategoricalParameter]:
        return self._categorical_parameters
