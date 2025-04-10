from functools import partial
from typing import Annotated, Callable, Iterable, Literal, Any

import numpy as np
from numpy.typing import NDArray
from sklearn import model_selection
from sklearn.base import BaseEstimator

from siapy.core.types import ArrayLike1dType, ArrayLike2dType
from siapy.optimizers.evaluators import (
    ScorerFuncType,
    cross_validation,
    hold_out_validation,
)
from siapy.utils.general import initialize_object

__all__ = [
    "Scorer",
]


class Scorer:
    def __init__(self, scorer: Callable[..., float]) -> None:
        self._scorer = scorer

    def __call__(
        self,
        model: BaseEstimator,
        X: ArrayLike2dType,
        y: ArrayLike1dType,
        X_val: ArrayLike2dType | None = None,
        y_val: ArrayLike1dType | None = None,
    ) -> float:
        return self._scorer(model, X, y, X_val, y_val)

    @classmethod
    def init_cross_validator_scorer(
        cls,
        scoring: str | ScorerFuncType | None = None,
        cv: int
        | model_selection.BaseCrossValidator
        | model_selection._split._RepeatedSplits
        | Iterable[int]
        | Literal["RepeatedKFold", "RepeatedStratifiedKFold"]
        | None = None,
        n_jobs: Annotated[
            int | None,
            "Number of jobs to run in parallel. `-1` means using all processors.",
        ] = None,
    ) -> "Scorer":
        if isinstance(cv, str) and cv in [
            "RepeatedKFold",
            "RepeatedStratifiedKFold",
        ]:
            cv = initialize_object(
                module=model_selection,
                module_name=cv,
                n_splits=3,
                n_repeats=5,
                random_state=0,
            )
        scorer = partial(
            cross_validation,
            scoring=scoring,
            cv=cv,  # type: ignore
            groups=None,
            n_jobs=n_jobs,
            verbose=0,
            params=None,
            pre_dispatch=1,
            error_score=0,
        )
        return cls(scorer)

    @classmethod
    def init_hold_out_scorer(
        cls,
        scoring: str | ScorerFuncType | None = None,
        test_size: float | None = 0.2,
        stratify: NDArray[np.floating[Any]] | None = None,
    ) -> "Scorer":
        scorer = partial(
            hold_out_validation,
            scoring=scoring,
            test_size=test_size,
            random_state=0,
            shuffle=True,
            stratify=stratify,
        )
        return cls(scorer)
