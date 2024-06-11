from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn import preprocessing


class BaseDecisionAlgo(ABC):
    def __init__(self, cfg):
        self.cfg = cfg
        self.encoder = preprocessing.LabelEncoder()

    @abstractmethod
    def _fit(self, X, y):
        pass

    @abstractmethod
    def _predict(self, X):
        pass

    ################################

    def _transform_data(self, data):
        """Transform data to numpy array and encode labels"""
        data_trans = []
        for class_label, df in data.items():
            df["target"] = class_label
            data_trans.append(df)

        data_trans = pd.concat(data_trans, ignore_index=True)
        X = np.array(data_trans.signature.to_list())
        y = self.encoder.fit_transform(data_trans.target)
        return X, y

    def fit(self, data):
        X, y = self._transform_data(data)
        self._fit(X, y)

    def predict(self, X):
        return self.encoder.inverse_transform(self._predict(X))
