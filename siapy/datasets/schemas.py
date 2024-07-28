from abc import ABC, abstractmethod
from typing import Any, Iterable

import pandas as pd
from pydantic import BaseModel, ConfigDict

from .helpers import generate_classification_target, generate_regression_target


class Target(BaseModel, ABC):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    value: pd.Series

    @abstractmethod
    def __getitem__(self, indices: Any) -> "Target": ...

    @abstractmethod
    def __len__(self) -> int: ...

    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict[str, Any]) -> "Target": ...

    @classmethod
    @abstractmethod
    def from_iterable(cls, data: Iterable) -> "Target": ...

    @abstractmethod
    def to_dict(self) -> dict[str, Any]: ...

    @abstractmethod
    def to_dataframe(self) -> pd.DataFrame: ...

    @abstractmethod
    def reset_index(self) -> "Target": ...


class ClassificationTarget(Target):
    label: pd.Series
    value: pd.Series
    encoding: pd.Series

    def __getitem__(self, indices: Any) -> "ClassificationTarget":
        label = self.label.iloc[indices]
        value = self.value.iloc[indices]
        return ClassificationTarget(label=label, value=value, encoding=self.encoding)

    def __len__(self) -> int:
        return len(self.label)

    @classmethod
    def from_iterable(cls, data: Iterable) -> "ClassificationTarget":
        label = pd.DataFrame(data, columns=["label"])
        return generate_classification_target(label, "label")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ClassificationTarget":
        label = pd.Series(data["label"], name="label")
        value = pd.Series(data["value"], name="value")
        encoding = pd.Series(data["encoding"], name="encoding")
        return cls(label=label, value=value, encoding=encoding)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label.to_list(),
            "value": self.value.to_list(),
            "encoding": self.encoding.to_list(),
        }

    def to_dataframe(self) -> pd.DataFrame:
        return pd.concat([self.value, self.label], axis=1)

    def reset_index(self) -> "ClassificationTarget":
        return ClassificationTarget(
            label=self.label.reset_index(drop=True),
            value=self.value.reset_index(drop=True),
            encoding=self.encoding,
        )


class RegressionTarget(Target):
    value: pd.Series
    name: str

    def __getitem__(self, indices: Any) -> "RegressionTarget":
        value = self.value.iloc[indices]
        return RegressionTarget(value=value, name=self.name)

    def __len__(self) -> int:
        return len(self.value)

    @classmethod
    def from_iterable(cls, data: Iterable) -> "RegressionTarget":
        value = pd.DataFrame(data, columns=["value"])
        return generate_regression_target(value, "value")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RegressionTarget":
        value = pd.Series(data["value"], name="value")
        name = data["name"]
        return cls(value=value, name=name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value.to_list(),
            "name": self.name,
        }

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.value)

    def reset_index(self) -> "RegressionTarget":
        return RegressionTarget(value=self.value.reset_index(drop=True), name=self.name)


class TabularDatasetData(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    pixels: pd.DataFrame
    signals: pd.DataFrame
    metadata: pd.DataFrame
    target: Target | None = None

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._validate_lengths()

    def __setattr__(self, name: str, value: Any) -> None:
        super().__setattr__(name, value)
        if name in self.model_fields.keys():
            self._validate_lengths()

    def __getitem__(self, indices: Any) -> "TabularDatasetData":
        pixels = self.pixels.iloc[indices]
        signals = self.signals.iloc[indices]
        metadata = self.metadata.iloc[indices]
        target = None if self.target is None else self.target.__getitem__(indices)
        return TabularDatasetData(
            pixels=pixels, signals=signals, metadata=metadata, target=target
        )

    def __len__(self) -> int:
        return len(self.pixels)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TabularDatasetData":
        pixels = pd.DataFrame(data["pixels"])
        signals = pd.DataFrame(data["signals"])
        metadata = pd.DataFrame(data["metadata"])
        target = TabularDatasetData.target_from_dict(data.get("target", None))
        return cls(pixels=pixels, signals=signals, metadata=metadata, target=target)

    @staticmethod
    def target_from_dict(data: dict[str, Any] | None) -> Target | None:
        if data is None:
            return None

        regression_keys = set(RegressionTarget.model_fields.keys())
        classification_keys = set(ClassificationTarget.model_fields.keys())
        data_keys = set(data.keys())

        if data_keys.issubset(regression_keys):
            return RegressionTarget.from_dict(data)
        elif data_keys.issubset(classification_keys):
            return ClassificationTarget.from_dict(data)
        else:
            raise ValueError("Invalid target dict.")

    def _validate_lengths(self) -> None:
        if not (len(self.pixels) == len(self.signals) == len(self.metadata)):
            raise ValueError("Lengths of pixels, signals, and metadata must be equal")
        if self.target is not None and len(self.target) != len(self):
            raise ValueError(
                "Target length must be equal to the length of the dataset."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "pixels": self.pixels.to_dict(),
            "signals": self.signals.to_dict(),
            "metadata": self.metadata.to_dict(),
            "target": self.target.to_dict() if self.target is not None else None,
        }

    def to_dataframe(self) -> pd.DataFrame:
        combined_df = pd.concat([self.pixels, self.signals, self.metadata], axis=1)
        if self.target is not None:
            target_series = self.target.to_dataframe()
            combined_df = pd.concat([combined_df, target_series], axis=1)
        return combined_df

    def reset_index(self) -> "TabularDatasetData":
        return TabularDatasetData(
            pixels=self.pixels.reset_index(drop=True),
            signals=self.signals.reset_index(drop=True),
            metadata=self.metadata.reset_index(drop=True),
            target=self.target.reset_index() if self.target is not None else None,
        )
