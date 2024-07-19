import pandas as pd

from siapy.datasets.schemas import ClassificationTarget, RegressionTarget


def generate_classification_target(
    dataframe: pd.DataFrame,
    column_names: str | list[str],
) -> ClassificationTarget:
    if isinstance(column_names, str):
        column_names = [column_names]
    # create one column labels from multiple columns
    label = dataframe[column_names].apply(tuple, axis=1)
    # encode to numbers
    encoded_np, encoding_np = pd.factorize(label)
    encoded = pd.Series(encoded_np, name="encoded")
    encoding = pd.Series(encoding_np, name="encoding")
    return ClassificationTarget(label=label, value=encoded, encoding=encoding)


def generate_regression_target(
    dataframe: pd.DataFrame,
    column_name: str,
) -> RegressionTarget:
    return RegressionTarget(name=column_name, value=dataframe[column_name])
