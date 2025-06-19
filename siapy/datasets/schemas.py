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
    """Abstract base class for machine learning target variables.

    This class defines the interface for target variables used in machine learning
    datasets, supporting both classification and regression targets.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    value: pd.Series

    @abstractmethod
    def __getitem__(self, indices: Any) -> "Target":
        """Get a subset of the target data using indexing.

        Args:
            indices: Index or slice to select subset of target data.

        Returns:
            New Target instance containing the selected subset.
        """
        ...

    @abstractmethod
    def __len__(self) -> int:
        """Get the number of target values.

        Returns:
            Number of target values in the dataset.
        """
        ...

    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict[str, Any]) -> "Target":
        """Create a Target instance from a dictionary.

        Args:
            data: Dictionary containing target data with appropriate keys.

        Returns:
            New Target instance created from the dictionary data.
        """
        ...

    @classmethod
    @abstractmethod
    def from_iterable(cls, data: Iterable[Any]) -> "Target":
        """Create a Target instance from an iterable of values.

        Args:
            data: Iterable containing target values.

        Returns:
            New Target instance created from the iterable data.
        """
        ...

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Convert the target to a dictionary representation.

        Returns:
            Dictionary containing the target data.
        """
        ...

    @abstractmethod
    def to_dataframe(self) -> pd.DataFrame:
        """Convert the target to a pandas DataFrame.

        Returns:
            DataFrame representation of the target data.
        """
        ...

    @abstractmethod
    def reset_index(self) -> "Target":
        """Reset the index of all internal pandas objects to a default integer index.

        Returns:
            New Target instance with reset indices.
        """
        ...


class ClassificationTarget(Target):
    """Target variable for classification tasks.

    Represents categorical target variables with string labels, numerical values,
    and optional encoding information for machine learning classification tasks.
    """

    label: pd.Series
    value: pd.Series
    encoding: pd.Series

    def __getitem__(self, indices: Any) -> "ClassificationTarget":
        """Get a subset of the classification target using indexing.

        Args:
            indices: Index or slice to select subset of target data.

        Returns:
            New ClassificationTarget instance containing the selected subset.
            The encoding is preserved across all instances.
        """
        label = self.label.iloc[indices]
        value = self.value.iloc[indices]
        return ClassificationTarget(label=label, value=value, encoding=self.encoding)

    def __len__(self) -> int:
        """Get the number of classification target values.

        Returns:
            Number of target values in the classification dataset.
        """
        return len(self.label)

    @classmethod
    def from_iterable(cls, data: Iterable[Any]) -> "ClassificationTarget":
        """Create a ClassificationTarget from an iterable of labels.

        Automatically generates numerical values and encoding for the provided labels.

        Args:
            data: Iterable containing classification labels.

        Returns:
            New ClassificationTarget instance with generated values and encoding.
        """
        label = pd.DataFrame(data, columns=["label"])
        return generate_classification_target(label, "label")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ClassificationTarget":
        """Create a ClassificationTarget from a dictionary.

        Args:
            data: Dictionary with keys 'label', 'value', and 'encoding'.<br>
                - label: List of string labels <br>
                - value: List of numerical values corresponding to labels <br>
                - encoding: List of encoding information for labels

        Returns:
            New ClassificationTarget instance created from the dictionary.
        """
        label = pd.Series(data["label"], name="label")
        value = pd.Series(data["value"], name="value")
        encoding = pd.Series(data["encoding"], name="encoding")
        return cls(label=label, value=value, encoding=encoding)

    def to_dict(self) -> dict[str, Any]:
        """Convert the classification target to a dictionary.

        Returns:
            Dictionary with keys 'label', 'value', and 'encoding' containing list representations of the respective pandas Series.
        """
        return {
            "label": self.label.to_list(),
            "value": self.value.to_list(),
            "encoding": self.encoding.to_list(),
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Convert the classification target to a pandas DataFrame.

        Returns:
            DataFrame containing 'value' and 'label' columns. The encoding information is not included in the DataFrame representation.
        """
        return pd.concat([self.value, self.label], axis=1)

    def reset_index(self) -> "ClassificationTarget":
        """Reset indices of label and value Series to default integer index.

        Returns:
            New ClassificationTarget instance with reset indices for label and value. The encoding Series is preserved as-is since it represents the overall encoding scheme rather than instance-specific data.
        """
        return ClassificationTarget(
            label=self.label.reset_index(drop=True),
            value=self.value.reset_index(drop=True),
            encoding=self.encoding,
        )


class RegressionTarget(Target):
    """Target variable for regression tasks.

    Represents continuous numerical target variables for machine learning
    regression tasks with an optional descriptive name.
    """

    value: pd.Series
    name: str = "value"

    def __getitem__(self, indices: Any) -> "RegressionTarget":
        """Get a subset of the regression target using indexing.

        Args:
            indices: Index or slice to select subset of target data.

        Returns:
            New RegressionTarget instance containing the selected subset. The name is preserved across instances.
        """
        value = self.value.iloc[indices]
        return RegressionTarget(value=value, name=self.name)

    def __len__(self) -> int:
        """Get the number of regression target values.

        Returns:
            Number of target values in the regression dataset.
        """
        return len(self.value)

    @classmethod
    def from_iterable(cls, data: Iterable[Any]) -> "RegressionTarget":
        """Create a RegressionTarget from an iterable of numerical values.

        Args:
            data: Iterable containing numerical regression target values.

        Returns:
            New RegressionTarget instance with default name "value".
        """
        value = pd.DataFrame(data, columns=["value"])
        return generate_regression_target(value, "value")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RegressionTarget":
        """Create a RegressionTarget from a dictionary.

        Args:
            data: Dictionary with required key 'value' and optional key 'name'.<br>
                - value: List of numerical target values <br>
                - name: Optional descriptive name for the target variable

        Returns:
            New RegressionTarget instance. Uses "value" as default name if not provided in the dictionary.
        """
        value = pd.Series(data["value"], name="value")
        name = data["name"] if "name" in data else "value"
        return cls(value=value, name=name)

    def to_dict(self) -> dict[str, Any]:
        """Convert the regression target to a dictionary.

        Returns:
            Dictionary with keys 'value' and 'name' containing the list representation of values and the descriptive name.
        """
        return {
            "value": self.value.to_list(),
            "name": self.name,
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Convert the regression target to a pandas DataFrame.

        Returns:
            DataFrame containing a single column with the target values. The column name corresponds to the Series name, not the target name.
        """
        return pd.DataFrame(self.value)

    def reset_index(self) -> "RegressionTarget":
        """Reset the index of the value Series to a default integer index.

        Returns:
            New RegressionTarget instance with reset index for the value Series. The name is preserved.
        """
        return RegressionTarget(value=self.value.reset_index(drop=True), name=self.name)


@dataclass
class TabularDatasetData:
    """Container for tabular machine learning dataset components.

    Combines spectral signatures, metadata, and optional target variables into
    a unified dataset structure for machine learning workflows. Ensures data
    consistency through length validation and provides various data access patterns.
    """

    signatures: Signatures
    metadata: pd.DataFrame
    target: Target | None = None

    def __len__(self) -> int:
        """Get the number of samples in the dataset.

        Returns:
            Number of samples, determined by the length of signatures.
        """
        return len(self.signatures)

    def __repr__(self) -> str:
        """Get string representation of the dataset.

        Returns:
            String representation showing signatures, metadata, and target.
        """
        return f"TabularDatasetData(signatures={self.signatures}, metadata={self.metadata}, target={self.target})"

    def __getitem__(self, indices: Any) -> "TabularDatasetData":
        """Get a subset of the dataset using indexing.

        Args:
            indices: Index or slice to select subset of dataset samples.

        Returns:
            New TabularDatasetData instance containing the selected subset. All components (signatures, metadata, target) are consistently sliced.

        Note:
            If metadata selection results in a pandas Series (single row),
            it is converted to a DataFrame for consistency.
        """
        signatures = self.signatures[indices]
        metadata = self.metadata.iloc[indices]
        if isinstance(metadata, pd.Series):
            metadata = pd.DataFrame(metadata).T
        target = None if self.target is None else self.target.__getitem__(indices)
        return TabularDatasetData(signatures=signatures, metadata=metadata, target=target)

    def __post_init__(self) -> None:
        """Validate dataset consistency after initialization.

        Called automatically after dataclass initialization to ensure
        all components have consistent lengths.

        Raises:
            InvalidInputError: If signatures and metadata lengths don't match, or if target length doesn't match dataset length.
        """
        self._validate_lengths()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TabularDatasetData":
        """Create a TabularDatasetData instance from a dictionary.

        Args:
            data: Dictionary containing dataset components with keys: <br>
                - pixels: Dictionary for pixel data <br>
                - signals: Dictionary for signal data <br>
                - metadata: Dictionary for metadata <br>
                - target: Optional dictionary for target data

        Returns:
            New TabularDatasetData instance created from the dictionary data.
        """
        signatures = Signatures.from_dict({"pixels": data["pixels"], "signals": data["signals"]})
        metadata = pd.DataFrame(data["metadata"])
        target = TabularDatasetData.target_from_dict(data.get("target", None))
        return cls(signatures=signatures, metadata=metadata, target=target)

    @staticmethod
    def target_from_dict(data: dict[str, Any] | None = None) -> Optional[Target]:
        """Create an appropriate Target instance from a dictionary.

        Automatically determines whether to create a ClassificationTarget or
        RegressionTarget based on the keys present in the data dictionary.

        Args:
            data: Optional dictionary containing target data. If None, returns None. <br>
                Keys determine the target type: <br>
                - Classification: Requires keys compatible with ClassificationTarget <br>
                - Regression: Requires keys compatible with RegressionTarget <br>

        Returns:
            ClassificationTarget, RegressionTarget, or None based on input data.

        Raises:
            InvalidInputError: If the dictionary keys don't match any known target type.
        """
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
        """Validate that all dataset components have consistent lengths.

        Ensures data integrity by checking that signatures, metadata, and
        target (if present) all have the same number of samples.

        Raises:
            InvalidInputError: If signatures and metadata lengths don't match, or if target length doesn't match the dataset length.
        """
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
        """Create a new dataset with updated attributes.

        Creates a copy of the current dataset with specified attributes replaced.
        Unspecified attributes are copied from the current dataset.

        Args:
            signatures: Optional new Signatures to replace current signatures.
            metadata: Optional new DataFrame to replace current metadata.
            target: Optional new Target to replace current target.

        Returns:
            New TabularDatasetData instance with updated attributes.

        Note:
            The returned dataset will be validated for length consistency.
        """
        current_data = self.copy()
        signatures = signatures if signatures is not None else current_data.signatures
        metadata = metadata if metadata is not None else current_data.metadata
        target = target if target is not None else current_data.target
        return TabularDatasetData(signatures=signatures, metadata=metadata, target=target)

    def to_dict(self) -> dict[str, Any]:
        """Convert the dataset to a dictionary representation.

        Returns:
            Dictionary with keys 'pixels', 'signals', 'metadata', and 'target'. The target key contains None if no target is present.
        """
        signatures_dict = self.signatures.to_dict()
        return {
            "pixels": signatures_dict["pixels"],
            "signals": signatures_dict["signals"],
            "metadata": self.metadata.to_dict(),
            "target": self.target.to_dict() if self.target is not None else None,
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Convert the dataset to a single pandas DataFrame.

        Combines signatures, metadata, and target (if present) into a single
        DataFrame with all columns at the same level.

        Returns:
            DataFrame containing all dataset components as columns.
        """
        combined_df = pd.concat([self.signatures.to_dataframe(), self.metadata], axis=1)
        if self.target is not None:
            target_series = self.target.to_dataframe()
            combined_df = pd.concat([combined_df, target_series], axis=1)
        return combined_df

    def to_dataframe_multiindex(self) -> pd.DataFrame:
        """Convert the dataset to a pandas DataFrame with MultiIndex columns.

        Creates a DataFrame where columns are organized hierarchically by category
        (pixel, signal, metadata, target) and field names within each category.

        Returns:
            DataFrame with MultiIndex columns having levels ['category', 'field'].

        Raises:
            InvalidInputError: If target type is not ClassificationTarget or RegressionTarget.
        """
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
        """Reset indices of all dataset components to default integer indices.

        Creates a new dataset with all pandas objects having their indices
        reset to consecutive integers starting from 0.

        Returns:
            New TabularDatasetData instance with reset indices for all components.
        """
        return TabularDatasetData(
            signatures=self.signatures.reset_index(),
            metadata=self.metadata.reset_index(drop=True),
            target=self.target.reset_index() if self.target is not None else None,
        )

    def copy(self) -> "TabularDatasetData":
        """Create a deep copy of the dataset.

        Creates a new TabularDatasetData instance with copied versions of all
        components, ensuring that modifications to the copy don't affect the original.

        Returns:
            New TabularDatasetData instance that is a deep copy of the current dataset.
        """
        return TabularDatasetData(
            signatures=self.signatures.copy(),
            metadata=self.metadata.copy(),
            target=self.target.model_copy() if self.target is not None else None,
        )
