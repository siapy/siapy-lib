from sklearn.base import BaseEstimator

from siapy.core.types import ArrayLike1dType, ArrayLike2dType


class Scorer:
    def __init__(self):
        pass

    def __call__(
        self,
        model: BaseEstimator,
        X: ArrayLike2dType,
        y: ArrayLike1dType,
        X_val: ArrayLike2dType | None = None,
        y_val: ArrayLike1dType | None = None,
    ) -> float:
        pass
