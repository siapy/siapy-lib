from functools import partial
from typing import Iterable, Literal

import numpy as np
from sklearn import model_selection
from sklearn.base import BaseEstimator

from siapy.core.types import ArrayLike1dType, ArrayLike2dType
from siapy.utils.evaluators import ScorerFuncType, cross_validation, hold_out_validation
from siapy.utils.general import initialize_object


class Scorer:
    def __init__(self, scorer):
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
        | Iterable
        | Literal["RepeatedKFold", "RepeatedStratifiedKFold"]
        | None = None,
    ) -> "Scorer":
        if isinstance(cv, str) and cv in ["RepeatedKFold", "RepeatedStratifiedKFold"]:
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
            n_jobs=1,
            verbose=0,
            fit_params=None,
            pre_dispatch=1,
            error_score=0,
        )
        return cls(scorer)

    @classmethod
    def init_hold_out_scorer(
        cls,
        scoring: str | ScorerFuncType | None = None,
        test_size: float | None = 0.2,
        stratify: np.ndarray | None = None,
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
