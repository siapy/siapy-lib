from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Iterable, Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict

from siapy.core.exceptions import InvalidInputError
from siapy.entities import Signatures

from .helpers import generate_classification_target, generate_regression_target

__all__ = [
    "ClassificationTarget",
    "RegressionTarget",
    "TabularDatasetData",
]


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
    def from_iterable(cls, data: Iterable[Any]) -> "Target": ...

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
    def from_iterable(cls, data: Iterable[Any]) -> "ClassificationTarget":
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
    name: str = "value"

    def __getitem__(self, indices: Any) -> "RegressionTarget":
        value = self.value.iloc[indices]
        return RegressionTarget(value=value, name=self.name)

    def __len__(self) -> int:
        return len(self.value)

    @classmethod
    def from_iterable(cls, data: Iterable[Any]) -> "RegressionTarget":
        value = pd.DataFrame(data, columns=["value"])
        return generate_regression_target(value, "value")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RegressionTarget":
        value = pd.Series(data["value"], name="value")
        name = data["name"] if "name" in data else "value"
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


@dataclass
class TabularDatasetData:
    signatures: Signatures
    metadata: pd.DataFrame
    target: Target | None = None

    def __len__(self) -> int:
        return len(self.signatures)

    def __repr__(self) -> str:
        return f"TabularDatasetData(signatures={self.signatures}, metadata={self.metadata}, target={self.target})"

    def __getitem__(self, indices: Any) -> "TabularDatasetData":
        signatures = self.signatures[indices]
        metadata = self.metadata.iloc[indices]
        if isinstance(metadata, pd.Series):
            metadata = pd.DataFrame(metadata).T
        target = None if self.target is None else self.target.__getitem__(indices)
        return TabularDatasetData(signatures=signatures, metadata=metadata, target=target)

    def __post_init__(self) -> None:
        self._validate_lengths()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TabularDatasetData":
        signatures = Signatures.from_dict({"pixels": data["pixels"], "signals": data["signals"]})
        metadata = pd.DataFrame(data["metadata"])
        target = TabularDatasetData.target_from_dict(data.get("target", None))
        return cls(signatures=signatures, metadata=metadata, target=target)

    @staticmethod
    def target_from_dict(data: dict[str, Any] | None = None) -> Optional[Target]:
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
            raise InvalidInputError(data, "Invalid target dict.")

    def _validate_lengths(self) -> None:
        if len(self.signatures) != len(self.metadata):
            raise InvalidInputError(
                {
                    "signatures_length": len(self.signatures),
                    "metadata_length": len(self.metadata),
                },
                "Lengths of signatures and metadata must be equal",
            )
        if self.target is not None and len(self.target) != len(self):
            raise InvalidInputError(
                {
                    "target_length": len(self.target),
                    "dataset_length": len(self),
                },
                "Target length must be equal to the length of the dataset.",
            )

    def set_attributes(
        self,
        *,
        signatures: Signatures | None = None,
        metadata: pd.DataFrame | None = None,
        target: Target | None = None,
    ) -> "TabularDatasetData":
        current_data = self.copy()
        signatures = signatures if signatures is not None else current_data.signatures
        metadata = metadata if metadata is not None else current_data.metadata
        target = target if target is not None else current_data.target
        return TabularDatasetData(signatures=signatures, metadata=metadata, target=target)

    def to_dict(self) -> dict[str, Any]:
        signatures_dict = self.signatures.to_dict()
        return {
            "pixels": signatures_dict["pixels"],
            "signals": signatures_dict["signals"],
            "metadata": self.metadata.to_dict(),
            "target": self.target.to_dict() if self.target is not None else None,
        }

    def to_dataframe(self) -> pd.DataFrame:
        combined_df = pd.concat([self.signatures.to_dataframe(), self.metadata], axis=1)
        if self.target is not None:
            target_series = self.target.to_dataframe()
            combined_df = pd.concat([combined_df, target_series], axis=1)
        return combined_df

    def to_dataframe_multiindex(self) -> pd.DataFrame:
        signatures_df = self.signatures.to_dataframe_multiindex()

        metadata_columns = pd.MultiIndex.from_tuples(
            [("metadata", col) for col in self.metadata.columns], names=["category", "field"]
        )
        metadata_df = pd.DataFrame(self.metadata.values, columns=metadata_columns)

        combined_df = pd.concat([signatures_df, metadata_df], axis=1)

        if self.target is not None:
            target_df = self.target.to_dataframe()
            if isinstance(self.target, ClassificationTarget):
                target_columns = pd.MultiIndex.from_tuples(
                    [("target", col) for col in target_df.columns],
                    names=["category", "field"],
                )
            elif isinstance(self.target, RegressionTarget):
                target_columns = pd.MultiIndex.from_tuples(
                    [("target", self.target.name)],
                    names=["category", "field"],
                )
            else:
                raise InvalidInputError(
                    self.target,
                    "Invalid target type. Expected ClassificationTarget or RegressionTarget.",
                )
            target_df = pd.DataFrame(target_df.values, columns=target_columns)
            combined_df = pd.concat([combined_df, target_df], axis=1)

        return combined_df

    def reset_index(self) -> "TabularDatasetData":
        return TabularDatasetData(
            signatures=self.signatures.reset_index(),
            metadata=self.metadata.reset_index(drop=True),
            target=self.target.reset_index() if self.target is not None else None,
        )

    def copy(self) -> "TabularDatasetData":
        return TabularDatasetData(
            signatures=self.signatures.copy(),
            metadata=self.metadata.copy(),
            target=self.target.model_copy() if self.target is not None else None,
        )
