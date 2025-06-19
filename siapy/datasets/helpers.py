from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from .schemas import (
        ClassificationTarget,
        RegressionTarget,
        TabularDatasetData,
    )

__all__ = [
    "generate_classification_target",
    "generate_regression_target",
    "merge_signals_from_multiple_cameras",
]


def generate_classification_target(
    dataframe: pd.DataFrame,
    column_names: str | list[str],
) -> "ClassificationTarget":
    """Generate a classification target from DataFrame columns.

    Creates a classification target by combining one or more DataFrame columns into
    encoded labels suitable for machine learning classification tasks. Multiple columns
    are combined using a '__' delimiter and then factorized into numeric values.

    Args:
        dataframe: The input DataFrame containing the target data.
        column_names: Name(s) of the column(s) to use for generating the classification target.
            Can be a single column name as string or multiple column names as list.

    Returns:
        A ClassificationTarget object containing the original labels, encoded numeric values, and the encoding mapping.

    Example:
        ```python
        import pandas as pd
        from siapy.datasets.helpers import generate_classification_target

        df = pd.DataFrame({
            'category': ['A', 'B', 'A', 'C'],
            'subcategory': ['X', 'Y', 'X', 'Z']
        })

        # Single column
        target = generate_classification_target(df, 'category')

        # Multiple columns
        target = generate_classification_target(df, ['category', 'subcategory'])
        ```
    """
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
    """Generate a regression target from a DataFrame column.

    Creates a regression target from a single DataFrame column for use in
    machine learning regression tasks.

    Args:
        dataframe: The input DataFrame containing the target data.
        column_name: Name of the column to use for generating the regression target.

    Returns:
        A RegressionTarget object containing the column name and values.

    Example:
        ```python
        import pandas as pd
        from siapy.datasets.helpers import generate_regression_target

        df = pd.DataFrame({
            'temperature': [20.1, 25.3, 18.7, 22.9],
            'humidity': [45.2, 60.8, 38.1, 52.3]
        })

        target = generate_regression_target(df, 'temperature')
        ```
    """
    from .schemas import (
        RegressionTarget,
    )  # Local import to avoid circular dependency

    return RegressionTarget(name=column_name, value=dataframe[column_name])


def merge_signals_from_multiple_cameras(data: "TabularDatasetData") -> None:
    """Merge signals from multiple cameras into a unified dataset.

    This function combines spectral or imaging data collected from multiple camera
    sources into a single coherent dataset structure. The implementation details
    depend on the specific camera configuration and data format requirements.

    Args:
        data: The tabular dataset data containing signals from multiple cameras
            that need to be merged.

    Returns:
        None: The function modifies the input data in-place.

    Note:
        This function is currently not implemented and serves as a placeholder
        for future development of multi-camera signal merging capabilities.

    Todo:
        Implement the actual merging logic based on camera specifications
        and data alignment requirements.
    """
    pass
