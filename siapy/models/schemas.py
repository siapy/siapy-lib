import pandas as pd
from pydantic import BaseModel, ConfigDict


class ClassificationTarget(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
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


class RegressionTarget(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
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
