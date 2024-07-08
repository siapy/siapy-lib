# cSpell:disable
from typing import Literal, NamedTuple

import numpy as np
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
)


def normalized_RMSE(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    normalize_by: Literal["range", "mean"] = "range",
):
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    if normalize_by == "range":
        normalizer = np.max(y_true) - np.min(y_true)
    elif normalize_by == "mean":
        normalizer = np.mean(y_true)
    else:
        raise ValueError(f"Unknown normalizer {normalize_by}")
    return rmse / normalizer


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

    def to_dict(self) -> dict:
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

    def to_dict(self) -> dict:
        return self._asdict()


def calculate_classification_metrics(
    y_true,
    y_pred,
    average: Literal["micro", "macro", "samples", "weighted", "binary"]
    | None = "weighted",
):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=average)
    recall = recall_score(y_true, y_pred, average=average)
    f1 = f1_score(y_true, y_pred, average=average)
    return ClassificationMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
    )


def calculate_regression_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred, squared=True)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    pe = mean_absolute_percentage_error(y_true, y_pred)
    maxe = max_error(y_true, y_pred)
    nrmse_mean = normalized_RMSE(y_true, y_pred, normalize_by="mean")
    nrmse_range = normalized_RMSE(y_true, y_pred, normalize_by="range")
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
