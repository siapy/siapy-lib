from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from .schemas import ClassificationTarget, RegressionTarget, TabularDatasetData


def generate_classification_target(
    dataframe: pd.DataFrame,
    column_names: str | list[str],
) -> "ClassificationTarget":
    from .schemas import (
        ClassificationTarget,  # Local import to avoid circular dependency
    )

    if isinstance(column_names, str):
        column_names = [column_names]
    # create one column labels from multiple columns
    label = dataframe[column_names].apply(tuple, axis=1)
    # Convert tuples to strings with '__' delimiter
    label = label.apply(lambda x: "__".join(x))
    # encode to numbers
    encoded_np, encoding_np = pd.factorize(label)
    encoded = pd.Series(encoded_np, name="encoded")
    encoding = pd.Series(encoding_np, name="encoding")
    return ClassificationTarget(label=label, value=encoded, encoding=encoding)


def generate_regression_target(
    dataframe: pd.DataFrame,
    column_name: str,
) -> "RegressionTarget":
    from .schemas import RegressionTarget  # Local import to avoid circular dependency

    return RegressionTarget(name=column_name, value=dataframe[column_name])


def merge_signals_from_multiple_cameras(data: "TabularDatasetData"):
    data.signals.copy()
