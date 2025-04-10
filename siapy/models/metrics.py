# cSpell:disable
from typing import Any, Literal, NamedTuple

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    max_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    root_mean_squared_error,
)

from siapy.core.exceptions import InvalidInputError

__all__ = [
    "calculate_classification_metrics",
    "calculate_regression_metrics",
]


def normalized_rmse(
    y_true: NDArray[np.floating[Any]],
    y_pred: NDArray[np.floating[Any]],
    normalize_by: Literal["range", "mean"] = "range",
) -> float:
    rmse = root_mean_squared_error(y_true, y_pred)
    if normalize_by == "range":
        normalizer = np.max(y_true) - np.min(y_true)
    elif normalize_by == "mean":
        normalizer = np.mean(y_true)
    else:
        raise InvalidInputError(
            input_value=normalize_by,
            message="Unknown normalizer. Possible values are: 'range' or 'mean'.",
        )
    return float(rmse / normalizer)


class ClassificationMetrics(NamedTuple):
    accuracy: float
    precision: float
    recall: float
    f1: float

    def __str__(self) -> str:
        return (
            f"Accuracy: {self.accuracy:.2f}\n"
            f"Precision: {self.precision:.2f}\n"
            f"Recall: {self.recall:.2f}\n"
            f"F1: {self.f1:.2f}\n"
        )

    def to_dict(self) -> dict[str, float]:
        return self._asdict()


class RegressionMetrics(NamedTuple):
    mae: float
    mse: float
    rmse: float
    r2: float
    pe: float
    maxe: float
    nrmse_mean: float
    nrmse_range: float

    def __str__(self) -> str:
        return (
            f"Mean absolute error: {self.mae:.2f}\n"
            f"Mean squared error: {self.mse:.2f}\n"
            f"Root mean squared error: {self.rmse:.2f}\n"
            f"R2 score: {self.r2:.2f}\n"
            f"Mean absolute percentage error: {self.pe:.2f}\n"
            f"Max error: {self.maxe:.2f}\n"
            f"Normalized root mean squared error (by mean): {self.nrmse_mean:.2f}\n"
            f"Normalized root mean squared error (by range): {self.nrmse_range:.2f}\n"
        )

    def to_dict(self) -> dict[str, float]:
        return self._asdict()


def calculate_classification_metrics(
    y_true: NDArray[np.floating[Any]],
    y_pred: NDArray[np.floating[Any]],
    average: Literal["micro", "macro", "samples", "weighted", "binary"] | None = "weighted",
) -> ClassificationMetrics:
    accuracy = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, average=average))
    recall = float(recall_score(y_true, y_pred, average=average))
    f1 = float(f1_score(y_true, y_pred, average=average))
    return ClassificationMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
    )


def calculate_regression_metrics(
    y_true: NDArray[np.floating[Any]],
    y_pred: NDArray[np.floating[Any]],
) -> RegressionMetrics:
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(root_mean_squared_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    pe = float(mean_absolute_percentage_error(y_true, y_pred))
    maxe = float(max_error(y_true, y_pred))
    nrmse_mean = float(normalized_rmse(y_true, y_pred, normalize_by="mean"))
    nrmse_range = float(normalized_rmse(y_true, y_pred, normalize_by="range"))
    return RegressionMetrics(
        mae=mae,
        mse=mse,
        rmse=rmse,
        r2=r2,
        pe=pe,
        maxe=maxe,
        nrmse_mean=nrmse_mean,
        nrmse_range=nrmse_range,
    )
