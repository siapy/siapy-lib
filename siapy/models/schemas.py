import pickle

import numpy as np
import pandas as pd
from pydantic import BaseModel


class BaseModelConfig(BaseModel):
    class Config:
        arbitrary_types_allowed = True


class ClassificationTarget(BaseModelConfig):
    label: pd.Series
    value: pd.Series
    encoding: pd.Series

    def __getitem__(self, indices):
        label = self.label.iloc[indices]
        value = self.value.iloc[indices]
        return ClassificationTarget(label=label, value=value, encoding=self.encoding)

    def __len__(self):
        return len(self.label)

    @classmethod
    def from_dict(cls, data):
        label = pd.Series(data["label"], name="label")
        value = pd.Series(data["value"], name="value")
        encoding = pd.Series(data["encoding"], name="encoding")
        return cls(label=label, value=value, encoding=encoding)

    def to_dict(self):
        return {
            "label": self.label.to_list(),
            "value": self.value.to_list(),
            "encoding": self.encoding.to_list(),
        }

    def reset_index(self):
        return ClassificationTarget(
            label=self.label.reset_index(drop=True),
            value=self.value.reset_index(drop=True),
            encoding=self.encoding,
        )


class RegressionTarget(BaseModelConfig):
    value: pd.Series
    name: str

    def __getitem__(self, indices):
        value = self.value.iloc[indices]
        return RegressionTarget(value=value, name=self.name)

    def __len__(self):
        return len(self.value)

    @classmethod
    def from_dict(cls, data):
        value = pd.Series(data["value"], name="value")
        name = data["name"]
        return cls(value=value, name=name)

    def to_dict(self):
        return {
            "value": self.value.to_list(),
            "name": self.name,
        }

    def reset_index(self):
        return RegressionTarget(value=self.value.reset_index(drop=True), name=self.name)


class StructuredData(BaseModelConfig):
    data: pd.DataFrame
    meta: pd.DataFrame
    target: ClassificationTarget | RegressionTarget | None = None

    def __getitem__(self, indices):
        data = self.data.iloc[indices]
        meta = self.meta.iloc[indices]
        target = None if self.target is None else self.target[indices]
        return StructuredData(data=data, meta=meta, target=target)

    @classmethod
    def from_dict(cls, data):
        data_ = pd.DataFrame(data["data"])
        meta = pd.DataFrame(data["meta"])
        target = StructuredData.target_from_dict(data["target"])
        return cls(data=data_, meta=meta, target=target)

    @staticmethod
    def target_from_dict(data):
        if data is None:
            return None

        elif "name" in data:
            return RegressionTarget.from_dict(data)

        elif "encoding" in data:
            return ClassificationTarget.from_dict(data)

        else:
            raise ValueError("Invalid target dict.")

    @classmethod
    def from_bytes(cls, data):
        return cls.from_dict(pickle.loads(data))

    def to_dict(self):
        return {
            "data": self.data.to_dict(),
            "meta": self.meta.to_dict(),
            "target": self.target.to_dict() if self.target is not None else None,
        }

    def to_bytes(self):
        return pickle.dumps(self.to_dict())

    def reset_index(self):
        return StructuredData(
            data=self.data.reset_index(drop=True),
            meta=self.meta.reset_index(drop=True),
            target=self.target.reset_index() if self.target is not None else None,
        )


class Prediction(BaseModelConfig):
    predictions: np.ndarray
    name: str = ""

    @classmethod
    def from_dict(cls, data):
        predictions = np.array(data["predictions"])
        name = data["name"]
        return cls(predictions=predictions, name=name)

    @classmethod
    def from_bytes(cls, data):
        return cls.from_dict(pickle.loads(data))

    def to_dict(self):
        return {
            "predictions": self.predictions,
            "name": self.name,
        }

    def to_bytes(self):
        return pickle.dumps(self.to_dict())
