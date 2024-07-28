import pandas as pd
import pytest

from siapy.datasets.schemas import (
    ClassificationTarget,
    RegressionTarget,
    TabularDatasetData,
)


def test_classification_target_from_iterable():
    data = ["a", "b", "c"]
    classification_target = ClassificationTarget.from_iterable(data)
    assert isinstance(classification_target, ClassificationTarget)
    pd.testing.assert_series_equal(
        classification_target.label, pd.Series(["a", "b", "c"])
    )


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


def test_classification_target_to_dataframe():
    data = {"label": ["a", "b", "c"], "value": [1, 2, 3], "encoding": ["x", "y", "z"]}
    classification_target = ClassificationTarget.from_dict(data)
    df = classification_target.to_dataframe()
    expected_df = pd.DataFrame({"value": [1, 2, 3], "label": ["a", "b", "c"]})
    pd.testing.assert_frame_equal(df, expected_df)


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


def test_regression_target_from_iterable():
    data = [1.0, 2.5, 3.3]
    regression_target = RegressionTarget.from_iterable(data)
    assert isinstance(regression_target, RegressionTarget)
    pd.testing.assert_series_equal(
        regression_target.value, pd.Series([1.0, 2.5, 3.3], name="value")
    )


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


def test_regression_target_to_dataframe():
    data = {"value": [1.0, 2.5, 3.3], "name": "test_series"}
    regression_target = RegressionTarget.from_dict(data)
    df = regression_target.to_dataframe()
    expected_df = pd.DataFrame({"value": [1.0, 2.5, 3.3]})
    pd.testing.assert_frame_equal(df, expected_df)


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


def test_tabular_dataset_data_from_dict():
    data = {
        "pixels": {"0": [255, 255], "1": [0, 0]},
        "signals": {"0": [1.0, 2.0], "1": [3.0, 4.0]},
        "metadata": {"0": ["meta1", "meta1"], "1": ["meta2", "meta2"]},
    }
    tabular_dataset_data = TabularDatasetData.from_dict(data)
    assert isinstance(tabular_dataset_data, TabularDatasetData)
    pd.testing.assert_frame_equal(
        tabular_dataset_data.pixels, pd.DataFrame(data["pixels"]).astype(int)
    )
    pd.testing.assert_frame_equal(
        tabular_dataset_data.signals, pd.DataFrame(data["signals"])
    )
    pd.testing.assert_frame_equal(
        tabular_dataset_data.metadata, pd.DataFrame(data["metadata"])
    )


def test_tabular_dataset_data_to_dict():
    data = {
        "pixels": {"0": [255, 255], "1": [0, 0]},
        "signals": {"0": [1.0, 2.0], "1": [3.0, 4.0]},
        "metadata": {"0": ["meta1", "meta1"], "1": ["meta2", "meta2"]},
        "target": {"label": ["a", "b"], "value": [1, 2], "encoding": ["x", "y"]},
    }
    tabular_dataset_data = TabularDatasetData.from_dict(data)
    to_dict_data = tabular_dataset_data.to_dict()
    assert to_dict_data["pixels"] == pd.DataFrame(data["pixels"]).to_dict()
    assert to_dict_data["signals"] == pd.DataFrame(data["signals"]).to_dict()
    assert to_dict_data["metadata"] == pd.DataFrame(data["metadata"]).to_dict()
    assert to_dict_data["target"]["label"] == data["target"]["label"]
    assert to_dict_data["target"]["value"] == data["target"]["value"]
    assert to_dict_data["target"]["encoding"] == data["target"]["encoding"]


def test_tabular_dataset_data_to_dataframe():
    data = {
        "pixels": {"0": [255, 255], "1": [0, 0]},
        "signals": {"0": [1.0, 2.0], "1": [3.0, 4.0]},
        "metadata": {"0": ["meta1", "meta1"], "1": ["meta2", "meta2"]},
        "target": {"label": ["a", "b"], "value": [1, 2], "encoding": ["x", "y"]},
    }
    tabular_dataset_data = TabularDatasetData.from_dict(data)
    df = tabular_dataset_data.to_dataframe()
    expected_df = pd.concat(
        [
            pd.DataFrame(data["pixels"]).astype(int),
            pd.DataFrame(data["signals"]),
            pd.DataFrame(data["metadata"]),
            pd.DataFrame({"value": [1, 2], "label": ["a", "b"]}),
        ],
        axis=1,
    )
    pd.testing.assert_frame_equal(df, expected_df)


def test_tabular_dataset_data_getitem():
    data = {
        "pixels": {"0": [255, 255], "1": [0, 0], "2": [128, 128]},
        "signals": {"0": [1.0, 2.0], "1": [3.0, 4.0], "2": [5.0, 6.0]},
        "metadata": {
            "0": ["meta1", "meta1"],
            "1": ["meta2", "meta2"],
            "2": ["meta3", "meta3"],
        },
        "target": {
            "label": ["a", "b"],
            "value": [1, 2],
            "encoding": ["x", "y"],
        },
    }
    tabular_dataset_data = TabularDatasetData.from_dict(data)
    sliced_data = tabular_dataset_data[1:3]
    assert isinstance(sliced_data, TabularDatasetData)
    pd.testing.assert_frame_equal(
        sliced_data.pixels, pd.DataFrame(data["pixels"]).iloc[1:3].astype(int)
    )
    pd.testing.assert_frame_equal(
        sliced_data.signals, pd.DataFrame(data["signals"]).iloc[1:3]
    )
    pd.testing.assert_frame_equal(
        sliced_data.metadata, pd.DataFrame(data["metadata"]).iloc[1:3]
    )
    assert all(sliced_data.target.label == pd.Series(["b"], index=[1], name="label"))


def test_tabular_dataset_data_reset_index():
    data = {
        "pixels": {"0": [255, 255], "1": [0, 0]},
        "signals": {"0": [1.0, 2.0], "1": [3.0, 4.0]},
        "metadata": {"meta1": ["a", "b"], "meta2": ["c", "d"]},
        "target": {"label": ["a", "b"], "value": [1, 2], "encoding": ["x", "y"]},
    }
    tabular_dataset_data = TabularDatasetData.from_dict(data)
    sliced_data = tabular_dataset_data[1:2]
    reset_data = sliced_data.reset_index()

    assert isinstance(reset_data, TabularDatasetData)
    pd.testing.assert_frame_equal(
        reset_data.pixels,
        pd.DataFrame(data["pixels"]).iloc[1:2].astype(int).reset_index(drop=True),
    )
    pd.testing.assert_frame_equal(
        reset_data.signals,
        pd.DataFrame(data["signals"]).iloc[1:2].reset_index(drop=True),
    )
    pd.testing.assert_frame_equal(
        reset_data.metadata,
        pd.DataFrame(data["metadata"]).iloc[1:2].reset_index(drop=True),
    )
    expected_target_df = pd.DataFrame({"label": ["b"], "value": [2]}).reset_index(
        drop=True
    )
    assert all(reset_data.target.label == expected_target_df["label"])
    assert all(reset_data.target.value == expected_target_df["value"])
    pd.testing.assert_series_equal(
        reset_data.target.encoding,
        pd.Series(data["target"]["encoding"], name="encoding"),
    )


def test_tabular_dataset_data_target_from_dict_none():
    assert TabularDatasetData.target_from_dict(None) is None


def test_tabular_dataset_data_target_from_dict_regression():
    data = {"value": [1.0, 2.5, 3.3], "name": "test_series"}
    target = TabularDatasetData.target_from_dict(data)
    assert isinstance(target, RegressionTarget)
    assert all(target.value == pd.Series(data["value"], name="value"))
    assert target.name == data["name"]


def test_tabular_dataset_data_target_from_dict_classification():
    data = {"label": ["a", "b", "c"], "value": [1, 2, 3], "encoding": ["x", "y", "z"]}
    target = TabularDatasetData.target_from_dict(data)
    assert isinstance(target, ClassificationTarget)
    assert all(target.label == pd.Series(data["label"], name="label"))
    assert all(target.value == pd.Series(data["value"], name="value"))
    assert all(target.encoding == pd.Series(data["encoding"], name="encoding"))


def test_tabular_dataset_data_target_from_dict_invalid():
    data = {"unknown_key": "some_value"}
    with pytest.raises(ValueError):
        TabularDatasetData.target_from_dict(data)


def test_tabular_dataset_data_init():
    data = {
        "pixels": pd.DataFrame({"a": [1, 2, 3]}),
        "signals": pd.DataFrame({"b": [4, 5, 6]}),
        "metadata": pd.DataFrame({"c": [7, 8, 9]}),
        "target": {
            "label": ["a", "b", "c"],
            "value": [1, 2, 3],
            "encoding": ["x", "y", "z"],
        },
    }
    target = ClassificationTarget.from_dict(data["target"])

    # Valid setattr
    dataset = TabularDatasetData(
        pixels=data["pixels"],
        signals=data["signals"],
        metadata=data["metadata"],
        target=target,
    )
    assert isinstance(dataset, TabularDatasetData)

    # Invalid setattr -> signals
    data = {
        "pixels": pd.DataFrame({"a": [1, 2, 3]}),
        "signals": pd.DataFrame({"b": [4, 5]}),
        "metadata": pd.DataFrame({"c": [7, 8, 9]}),
    }

    with pytest.raises(
        ValueError, match="Lengths of pixels, signals, and metadata must be equal"
    ):
        TabularDatasetData(
            pixels=data["pixels"],
            signals=data["signals"],
            metadata=data["metadata"],
        )


def test_tabular_dataset_data_setattr():
    data = {
        "pixels": pd.DataFrame({"a": [1, 2, 3]}),
        "signals": pd.DataFrame({"b": [4, 5, 6]}),
        "metadata": pd.DataFrame({"c": [7, 8, 9]}),
        "target": {
            "label": ["a", "b", "c"],
            "value": [1, 2, 3],
            "encoding": ["x", "y", "z"],
        },
    }
    target = ClassificationTarget.from_dict(data["target"])
    dataset_ = TabularDatasetData(
        pixels=data["pixels"],
        signals=data["signals"],
        metadata=data["metadata"],
        target=target,
    )

    # Valid setattr
    new_pixels = pd.DataFrame({"a": [10, 11, 12]})
    dataset = dataset_.model_copy()
    dataset.pixels = new_pixels
    assert dataset.pixels.equals(new_pixels)

    # Invalid setattr -> signals
    invalid_signals = pd.DataFrame({"b": [4, 5]})
    with pytest.raises(
        ValueError, match="Lengths of pixels, signals, and metadata must be equal"
    ):
        dataset.signals = invalid_signals

    # Invalid setattr -> target
    dataset = dataset_.model_copy()
    invalid_target = target[:2]
    with pytest.raises(
        ValueError, match="Target length must be equal to the length of the dataset."
    ):
        dataset.target = invalid_target


def test_tabular_dataset_data_len():
    data = {
        "pixels": pd.DataFrame({"a": [1, 2, 3]}),
        "signals": pd.DataFrame({"b": [4, 5, 6]}),
        "metadata": pd.DataFrame({"c": [7, 8, 9]}),
        "target": {
            "label": ["a", "b", "c"],
            "value": [1, 2, 3],
            "encoding": ["x", "y", "z"],
        },
    }
    target = ClassificationTarget.from_dict(data["target"])
    dataset = TabularDatasetData(
        pixels=data["pixels"],
        signals=data["signals"],
        metadata=data["metadata"],
        target=target,
    )
    assert len(dataset) == len(data["pixels"])
