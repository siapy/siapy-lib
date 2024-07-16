import pandas as pd

from siapy.models.schemas import ClassificationTarget, RegressionTarget


def test_classification_target_from_dict():
    data = {"label": ["a", "b", "c"], "value": [1, 2, 3], "encoding": ["x", "y", "z"]}
    classification_target = ClassificationTarget.from_dict(data)
    assert isinstance(classification_target, ClassificationTarget)
    assert all(classification_target.label == pd.Series(data["label"], name="label"))
    assert all(classification_target.value == pd.Series(data["value"], name="value"))
    assert all(
        classification_target.encoding == pd.Series(data["encoding"], name="encoding")
    )


def test_classification_target_to_dict():
    data = {"label": ["a", "b", "c"], "value": [1, 2, 3], "encoding": ["x", "y", "z"]}
    classification_target = ClassificationTarget.from_dict(data)
    to_dict_data = classification_target.to_dict()
    assert to_dict_data["label"] == data["label"]
    assert to_dict_data["value"] == data["value"]
    assert to_dict_data["encoding"] == data["encoding"]


def test_classification_target_getitem():
    data = {"label": ["a", "b", "c"], "value": [1, 2, 3], "encoding": ["x", "y", "z"]}
    classification_target = ClassificationTarget.from_dict(data)
    sliced_target = classification_target[1:3]
    assert all(sliced_target.label == pd.Series(["b", "c"], index=[1, 2], name="label"))
    assert all(sliced_target.value == pd.Series([2, 3], index=[1, 2], name="value"))


def test_classification_target_len():
    data = {"label": ["a", "b", "c"], "value": [1, 2, 3], "encoding": ["x", "y", "z"]}
    classification_target = ClassificationTarget.from_dict(data)
    assert len(classification_target) == 3


def test_classification_target_reset_index():
    data = {"label": ["a", "b", "c"], "value": [1, 2, 3], "encoding": ["x", "y", "z"]}
    classification_target = ClassificationTarget.from_dict(data)
    reset_target = classification_target.reset_index()
    assert all(reset_target.label.index == pd.RangeIndex(start=0, stop=3))
    assert all(reset_target.value.index == pd.RangeIndex(start=0, stop=3))
    assert all(reset_target.label == classification_target.label)
    assert all(reset_target.value == classification_target.value)
    assert all(reset_target.encoding == classification_target.encoding)


def test_regression_target_from_dict():
    data = {"value": [1.0, 2.5, 3.3], "name": "test_series"}
    regression_target = RegressionTarget.from_dict(data)
    assert isinstance(regression_target, RegressionTarget)
    assert all(regression_target.value == pd.Series(data["value"], name="value"))
    assert regression_target.name == data["name"]


def test_regression_target_to_dict():
    data = {"value": [1.0, 2.5, 3.3], "name": "test_series"}
    regression_target = RegressionTarget.from_dict(data)
    to_dict_data = regression_target.to_dict()
    assert to_dict_data["value"] == data["value"]
    assert to_dict_data["name"] == data["name"]


def test_regression_target_getitem():
    data = {"value": [1.0, 2.5, 3.3], "name": "test_series"}
    regression_target = RegressionTarget.from_dict(data)
    sliced_target = regression_target[1:3]
    assert all(sliced_target.value == pd.Series([2.5, 3.3], index=[1, 2], name="value"))


def test_regression_target_len():
    data = {"value": [1.0, 2.5, 3.3], "name": "test_series"}
    regression_target = RegressionTarget.from_dict(data)
    assert len(regression_target) == 3


def test_regression_target_reset_index():
    data = {"value": [1.0, 2.5, 3.3], "name": "test_series"}
    regression_target = RegressionTarget.from_dict(data)
    reset_target = regression_target.reset_index()
    assert all(reset_target.value.index == pd.RangeIndex(start=0, stop=3))
    assert all(reset_target.value == regression_target.value)
    assert reset_target.name == regression_target.name
