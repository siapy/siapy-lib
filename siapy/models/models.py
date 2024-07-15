from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from siapy.datasets import DatasetDataFrame


class ClassificationAlgorithm(Protocol):
    def fit(self, X, y): ...

    def predict(self, X): ...


def data_to_numpy(data: pd.DataFrame | pd.Series | np.ndarray):
    if isinstance(data, DatasetDataFrame):
        return data.signals().to_numpy()
    elif isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
        return data.to_numpy()
    else:
        return data


class ClassificationModel(BaseEstimator, TransformerMixin):
    def __init__(self, model: ClassificationAlgorithm):
        self._model = model

    @property
    def model(self) -> ClassificationAlgorithm:
        return self._model

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray):
        _X = data_to_numpy(X)
        _y = data_to_numpy(y)
        self.model.fit(_X, _y)

    def predict(self, X: pd.DataFrame | np.ndarray):
        _X = data_to_numpy(X)
        return self.model.predict(_X)
