from typing import Annotated, Any, Callable, Iterable, Literal

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator
from sklearn.metrics import get_scorer
from sklearn.model_selection import (
    BaseCrossValidator,
    cross_val_score,
    train_test_split,
)

from siapy.core import logger
from siapy.core.exceptions import (
    InvalidInputError,
    MethodNotImplementedError,
)
from siapy.core.types import ArrayLike1dType, ArrayLike2dType

__all__ = [
    "cross_validation",
    "hold_out_validation",
]

ScorerFuncType = Callable[[BaseEstimator, ArrayLike2dType, ArrayLike1dType], float]


def check_model_prediction_methods(model: BaseEstimator) -> None:
    required_methods = ["fit", "predict", "score"]
    for method in required_methods:
        if not hasattr(model, method):
            raise MethodNotImplementedError(model.__class__.__name__, method)


def cross_validation(
    model: BaseEstimator,
    X: ArrayLike2dType,
    y: ArrayLike1dType,
    X_val: Annotated[ArrayLike2dType | None, "Not used, only for compatibility"] = None,
    y_val: Annotated[ArrayLike1dType | None, "Not used, only for compatibility"] = None,
    *,
    groups: ArrayLike1dType | None = None,
    scoring: str | ScorerFuncType | None = None,
    cv: int | BaseCrossValidator | Iterable[Any] | None = None,
    n_jobs: int | None = 1,
    verbose: int = 0,
    params: dict[str, Any] | None = None,
    pre_dispatch: int | str = 1,
    error_score: Literal["raise"] | int = 0,
) -> float:
    if X_val is not None or y_val is not None:
        logger.info("Specification of X_val and y_val is redundant for cross_validation.These parameters are ignored.")
    check_model_prediction_methods(model)
    score = cross_val_score(
        estimator=model,
        X=X,  # type: ignore
        y=y,
        groups=groups,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        verbose=verbose,
        params=params,
        pre_dispatch=pre_dispatch,
        error_score=error_score,
    )
    return score.mean()


def hold_out_validation(
    model: BaseEstimator,
    X: ArrayLike2dType,
    y: ArrayLike1dType,
    X_val: ArrayLike2dType | None = None,
    y_val: ArrayLike1dType | None = None,
    *,
    scoring: str | ScorerFuncType | None = None,
    test_size: float | None = 0.2,
    random_state: int | None = None,
    shuffle: bool = True,
    stratify: NDArray[np.floating[Any]] | None = None,
) -> float:
    if X_val is not None and y_val is not None:
        x_train, x_test, y_train, y_test = X, X_val, y, y_val
    elif X_val is not None or y_val is not None:
        raise InvalidInputError(
            input_value={"X_val": X_val, "y_val": y_val},
            message="To manually define validation set, both X_val and y_val must be specified.",
        )
    else:
        x_train, x_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            shuffle=shuffle,
            stratify=stratify,
        )
    check_model_prediction_methods(model)
    model.fit(x_train, y_train)  # type: ignore

    if scoring:
        if isinstance(scoring, str):
            scoring_func = get_scorer(scoring)
        else:
            scoring_func = scoring
        score = scoring_func(model, x_test, y_test)
    else:
        score = model.score(x_test, y_test)  # type: ignore
    return score
