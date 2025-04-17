import pandas as pd
import pytest

from siapy.core.exceptions import InvalidInputError
from siapy.datasets.schemas import (
    ClassificationTarget,
    RegressionTarget,
    TabularDatasetData,
)
from siapy.entities import Pixels, Signatures
from siapy.entities.signatures import Signals


def test_classification_target_from_iterable():
    data = ["a", "b", "c"]
    classification_target = ClassificationTarget.from_iterable(data)
    assert isinstance(classification_target, ClassificationTarget)
    pd.testing.assert_series_equal(classification_target.label, pd.Series(["a", "b", "c"]))


def test_classification_target_from_dict():
    data = {
        "label": ["a", "b", "c"],
        "value": [1, 2, 3],
        "encoding": ["x", "y", "z"],
    }
    classification_target = ClassificationTarget.from_dict(data)
    assert isinstance(classification_target, ClassificationTarget)
    assert all(classification_target.label == pd.Series(data["label"], name="label"))
    assert all(classification_target.value == pd.Series(data["value"], name="value"))
    assert all(classification_target.encoding == pd.Series(data["encoding"], name="encoding"))


def test_classification_target_to_dict():
    data = {
        "label": ["a", "b", "c"],
        "value": [1, 2, 3],
        "encoding": ["x", "y", "z"],
    }
    classification_target = ClassificationTarget.from_dict(data)
    to_dict_data = classification_target.to_dict()
    assert to_dict_data["label"] == data["label"]
    assert to_dict_data["value"] == data["value"]
    assert to_dict_data["encoding"] == data["encoding"]


def test_classification_target_to_dataframe():
    data = {
        "label": ["a", "b", "c"],
        "value": [1, 2, 3],
        "encoding": ["x", "y", "z"],
    }
    classification_target = ClassificationTarget.from_dict(data)
    df = classification_target.to_dataframe()
    expected_df = pd.DataFrame({"value": [1, 2, 3], "label": ["a", "b", "c"]})
    pd.testing.assert_frame_equal(df, expected_df)


def test_classification_target_getitem():
    data = {
        "label": ["a", "b", "c"],
        "value": [1, 2, 3],
        "encoding": ["x", "y", "z"],
    }
    classification_target = ClassificationTarget.from_dict(data)
    sliced_target = classification_target[1:3]
    assert all(sliced_target.label == pd.Series(["b", "c"], index=[1, 2], name="label"))
    assert all(sliced_target.value == pd.Series([2, 3], index=[1, 2], name="value"))


def test_classification_target_len():
    data = {
        "label": ["a", "b", "c"],
        "value": [1, 2, 3],
        "encoding": ["x", "y", "z"],
    }
    classification_target = ClassificationTarget.from_dict(data)
    assert len(classification_target) == 3


def test_classification_target_reset_index():
    data = {
        "label": ["a", "b", "c"],
        "value": [1, 2, 3],
        "encoding": ["x", "y", "z"],
    }
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
    pd.testing.assert_series_equal(regression_target.value, pd.Series([1.0, 2.5, 3.3], name="value"))


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


def test_tabular_dataset_data_init():
    # Create test data
    pixels = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    signals = pd.DataFrame({"band1": [7, 8, 9], "band2": [10, 11, 12]})
    signatures = Signatures(Pixels(pixels), Signals(signals))
    metadata = pd.DataFrame({"meta1": [13, 14, 15], "meta2": [16, 17, 18]})

    target_data = {
        "label": ["a", "b", "c"],
        "value": [1, 2, 3],
        "encoding": ["x", "y", "z"],
    }
    target = ClassificationTarget.from_dict(target_data)

    # Valid initialization
    dataset = TabularDatasetData(
        signatures=signatures,
        metadata=metadata,
        target=target,
    )
    assert isinstance(dataset, TabularDatasetData)
    assert dataset.signatures is signatures
    assert dataset.metadata.equals(metadata)
    assert dataset.target is target

    # Invalid initialization - metadata length mismatch
    invalid_metadata = pd.DataFrame({"meta": [1, 2]})  # Only 2 rows
    with pytest.raises(
        InvalidInputError,
        match="Lengths of signatures and metadata must be equal",
    ):
        TabularDatasetData(
            signatures=signatures,
            metadata=invalid_metadata,
        )

    # Invalid initialization - target length mismatch
    invalid_target = target[:2]  # Only 2 rows
    with pytest.raises(
        InvalidInputError,
        match="Target length must be equal to the length of the dataset.",
    ):
        TabularDatasetData(
            signatures=signatures,
            metadata=metadata,
            target=invalid_target,
        )


def test_tabular_dataset_data_len():
    # Create test data
    pixels = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    signals = pd.DataFrame({"band1": [7, 8, 9], "band2": [10, 11, 12]})
    signatures = Signatures(Pixels(pixels), Signals(signals))
    metadata = pd.DataFrame({"meta1": [13, 14, 15], "meta2": [16, 17, 18]})

    target_data = {
        "label": ["a", "b", "c"],
        "value": [1, 2, 3],
        "encoding": ["x", "y", "z"],
    }
    target = ClassificationTarget.from_dict(target_data)

    # Create the dataset
    dataset = TabularDatasetData(
        signatures=signatures,
        metadata=metadata,
        target=target,
    )

    # Test length
    assert len(dataset) == 3
    assert len(dataset) == len(signatures)
    assert len(dataset) == len(metadata)


def test_tabular_dataset_data_getitem():
    data = {
        "pixels": {"x": [255, 255, 255], "y": [0, 0, 0]},
        "signals": {"0": [1.0, 2.0, 3.0], "1": [3.0, 4.0, 5.0], "2": [5.0, 6.0, 7.0]},
        "metadata": {
            "0": ["meta1", "meta1", "meta1"],
            "1": ["meta2", "meta2", "meta1"],
            "2": ["meta3", "meta3", "meta1"],
        },
        "target": {
            "label": ["a", "b", "c"],
            "value": [1, 2, 3],
            "encoding": ["x", "y", "z"],
        },
    }
    tabular_dataset_data = TabularDatasetData.from_dict(data)
    sliced_data = tabular_dataset_data[1:3]

    assert isinstance(sliced_data, TabularDatasetData)

    # Check signatures (pixels and signals)
    expected_pixels_df = pd.DataFrame(data["pixels"]).iloc[1:3]
    expected_signals_df = pd.DataFrame(data["signals"]).iloc[1:3]
    expected_metadata_df = pd.DataFrame(data["metadata"]).iloc[1:3]

    pd.testing.assert_frame_equal(sliced_data.signatures.pixels.df, expected_pixels_df)
    pd.testing.assert_frame_equal(sliced_data.signatures.signals.df, expected_signals_df)
    pd.testing.assert_frame_equal(sliced_data.metadata, expected_metadata_df)

    # Check target
    assert all(sliced_data.target.label == pd.Series(["b", "c"], index=[1, 2], name="label"))
    assert all(sliced_data.target.value == pd.Series([2, 3], index=[1, 2], name="value"))


def test_tabular_dataset_data_from_dict():
    data = {
        "pixels": {"x": [255, 255], "y": [0, 0]},
        "signals": {"0": [1.0, 2.0], "1": [3.0, 4.0]},
        "metadata": {"0": ["meta1", "meta1"], "1": ["meta2", "meta2"]},
    }
    tabular_dataset_data = TabularDatasetData.from_dict(data)
    assert isinstance(tabular_dataset_data, TabularDatasetData)

    # Create expected Signatures object
    expected_pixels = Pixels(pd.DataFrame(data["pixels"]).astype(int))
    expected_signals = Signals(pd.DataFrame(data["signals"]))
    expected_signatures = Signatures(expected_pixels, expected_signals)

    # Test signatures
    assert isinstance(tabular_dataset_data.signatures, Signatures)
    pd.testing.assert_frame_equal(tabular_dataset_data.signatures.pixels.df, expected_signatures.pixels.df)
    pd.testing.assert_frame_equal(tabular_dataset_data.signatures.signals.df, expected_signatures.signals.df)

    # Test metadata
    pd.testing.assert_frame_equal(tabular_dataset_data.metadata, pd.DataFrame(data["metadata"]))


def test_tabular_dataset_data_target_from_dict_none():
    assert TabularDatasetData.target_from_dict(None) is None


def test_tabular_dataset_data_target_from_dict_invalid():
    data = {"unknown_key": "some_value"}
    with pytest.raises(InvalidInputError):
        TabularDatasetData.target_from_dict(data)


def test_tabular_dataset_data_target_from_dict_regression():
    data = {"value": [1.0, 2.5, 3.3], "name": "test_series"}
    target = TabularDatasetData.target_from_dict(data)
    assert isinstance(target, RegressionTarget)
    assert all(target.value == pd.Series(data["value"], name="value"))
    assert target.name == data["name"]


def test_tabular_dataset_data_target_from_dict_classification():
    data = {
        "label": ["a", "b", "c"],
        "value": [1, 2, 3],
        "encoding": ["x", "y", "z"],
    }
    target = TabularDatasetData.target_from_dict(data)
    assert isinstance(target, ClassificationTarget)
    assert all(target.label == pd.Series(data["label"], name="label"))
    assert all(target.value == pd.Series(data["value"], name="value"))
    assert all(target.encoding == pd.Series(data["encoding"], name="encoding"))


def test_tabular_dataset_data_with_no_target():
    """Test that TabularDatasetData works correctly without a target."""
    pixels = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
    signals = pd.DataFrame({"band1": [5, 6], "band2": [7, 8]})
    signatures = Signatures(Pixels(pixels), Signals(signals))
    metadata = pd.DataFrame({"meta": [9, 10]})

    dataset = TabularDatasetData(signatures=signatures, metadata=metadata, target=None)
    assert dataset.target is None

    # Test to_dict with no target
    data_dict = dataset.to_dict()
    assert data_dict["target"] is None

    # Test to_dataframe with no target
    df = dataset.to_dataframe()
    assert len(df.columns) == len(signatures.to_dataframe().columns) + len(metadata.columns)

    # Test reset_index with no target
    reset_dataset = dataset.reset_index()
    assert reset_dataset.target is None


def test_tabular_dataset_data_set_attributes():
    data = {
        "pixels": {"x": [255, 255, 255], "y": [0, 0, 0]},
        "signals": {"0": [1.0, 2.0, 3.0], "1": [3.0, 4.0, 5.0], "2": [5.0, 6.0, 7.0]},
        "metadata": {
            "0": ["meta1", "meta1", "meta1"],
            "1": ["meta2", "meta2", "meta1"],
            "2": ["meta3", "meta3", "meta1"],
        },
        "target": {
            "label": ["a", "b", "c"],
            "value": [1, 2, 3],
            "encoding": ["x", "y", "z"],
        },
    }
    initial_dataset = TabularDatasetData.from_dict(data)

    # test signatures
    new_signatures = Signatures(
        Pixels(pd.DataFrame({"x": [1, 2, 3], "y": [3, 4, 5]})),
        Signals(pd.DataFrame({"band1": [5, 6, 7], "band2": [7, 8, 9]})),
    )
    new_dataset = initial_dataset.set_attributes(signatures=new_signatures)
    assert isinstance(new_dataset, TabularDatasetData)
    assert new_dataset.signatures.pixels.df.equals(new_signatures.pixels.df)
    assert new_dataset.signatures.signals.df.equals(new_signatures.signals.df)
    assert new_dataset.metadata.equals(initial_dataset.metadata)
    assert new_dataset.target == initial_dataset.target

    # test metadata
    new_metadata = pd.DataFrame({"new_meta": ["a", "b", "c"]})
    new_dataset = initial_dataset.set_attributes(metadata=new_metadata)
    assert isinstance(new_dataset, TabularDatasetData)
    assert new_dataset.signatures.pixels.df.equals(initial_dataset.signatures.pixels.df)
    assert new_dataset.signatures.signals.df.equals(initial_dataset.signatures.signals.df)
    assert new_dataset.metadata.equals(new_metadata)
    assert new_dataset.target == initial_dataset.target

    # test target
    new_target_data = {
        "label": ["x", "y", "z"],
        "value": [10, 20, 30],
        "encoding": ["a", "b", "c"],
    }
    new_target = ClassificationTarget.from_dict(new_target_data)
    new_dataset = initial_dataset.set_attributes(target=new_target)
    assert isinstance(new_dataset, TabularDatasetData)
    assert new_dataset.signatures.pixels.df.equals(initial_dataset.signatures.pixels.df)
    assert new_dataset.signatures.signals.df.equals(initial_dataset.signatures.signals.df)
    assert new_dataset.metadata.equals(initial_dataset.metadata)


def test_tabular_dataset_data_to_dict():
    data = {
        "pixels": {"x": [255, 255, 255], "y": [0, 0, 0]},
        "signals": {"0": [1.0, 2.0, 3.0], "1": [3.0, 4.0, 5.0], "2": [5.0, 6.0, 7.0]},
        "metadata": {
            "0": ["meta1", "meta1", "meta1"],
            "1": ["meta2", "meta2", "meta1"],
            "2": ["meta3", "meta3", "meta1"],
        },
        "target": {
            "label": ["a", "b", "c"],
            "value": [1, 2, 3],
            "encoding": ["x", "y", "z"],
        },
    }
    tabular_dataset_data = TabularDatasetData.from_dict(data)
    to_dict_data = tabular_dataset_data.to_dict()

    assert isinstance(to_dict_data, dict)
    assert to_dict_data == TabularDatasetData.from_dict(to_dict_data).to_dict()

    assert to_dict_data["pixels"] == pd.DataFrame(data["pixels"]).to_dict()
    assert to_dict_data["signals"] == pd.DataFrame(data["signals"]).to_dict()
    assert to_dict_data["metadata"] == pd.DataFrame(data["metadata"]).to_dict()

    assert to_dict_data["target"]["label"] == data["target"]["label"]
    assert to_dict_data["target"]["value"] == data["target"]["value"]
    assert to_dict_data["target"]["encoding"] == data["target"]["encoding"]


def test_tabular_dataset_data_to_dataframe():
    data = {
        "pixels": {"x": [255, 255, 255], "y": [0, 0, 0]},
        "signals": {"0": [1.0, 2.0, 3.0], "1": [3.0, 4.0, 5.0], "2": [5.0, 6.0, 7.0]},
        "metadata": {
            "0": ["meta1", "meta1", "meta1"],
            "1": ["meta2", "meta2", "meta1"],
            "2": ["meta3", "meta3", "meta1"],
        },
        "target": {
            "label": ["a", "b", "c"],
            "value": [1, 2, 3],
            "encoding": ["x", "y", "z"],
        },
    }
    tabular_dataset_data = TabularDatasetData.from_dict(data)
    df = tabular_dataset_data.to_dataframe()
    expected_df = pd.concat(
        [
            pd.DataFrame(data["pixels"]).astype(int),
            pd.DataFrame(data["signals"]),
            pd.DataFrame(data["metadata"]),
            pd.DataFrame({"value": [1, 2, 3], "label": ["a", "b", "c"]}),
        ],
        axis=1,
    )
    pd.testing.assert_frame_equal(df, expected_df)


def test_tabular_dataset_data_to_dataframe_multiindex():
    data = {
        "pixels": {"x": [255, 255, 255], "y": [0, 0, 0]},
        "signals": {"0": [1.0, 2.0, 3.0], "1": [3.0, 4.0, 5.0], "2": [5.0, 6.0, 7.0]},
        "metadata": {
            "meta1": ["a", "b", "c"],
            "meta2": ["d", "e", "f"],
        },
        "target": {
            "label": ["a", "b", "c"],
            "value": [1, 2, 3],
            "encoding": ["x", "y", "z"],
        },
    }
    tabular_dataset_data = TabularDatasetData.from_dict(data)
    df = tabular_dataset_data.to_dataframe_multiindex()

    pd.testing.assert_frame_equal(pd.DataFrame(df.pixel), tabular_dataset_data.signatures.pixels.df)
    pd.testing.assert_frame_equal(pd.DataFrame(df.signal), tabular_dataset_data.signatures.signals.df)
    pd.testing.assert_frame_equal(pd.DataFrame(df.metadata), pd.DataFrame(data["metadata"]))
    pd.testing.assert_frame_equal(
        pd.DataFrame(df.target), pd.DataFrame({"value": [1, 2, 3], "label": ["a", "b", "c"]}), check_dtype=False
    )

    data["target"] = {}
    data["target"]["value"] = [1, 2, 3]

    tabular_dataset_data = TabularDatasetData.from_dict(data)
    df = tabular_dataset_data.to_dataframe_multiindex()

    pd.testing.assert_frame_equal(pd.DataFrame(df.pixel), tabular_dataset_data.signatures.pixels.df)
    pd.testing.assert_frame_equal(pd.DataFrame(df.signal), tabular_dataset_data.signatures.signals.df)
    pd.testing.assert_frame_equal(pd.DataFrame(df.metadata), pd.DataFrame(data["metadata"]))
    pd.testing.assert_frame_equal(pd.DataFrame(df.target), pd.DataFrame({"value": [1, 2, 3]}), check_dtype=False)


def test_tabular_dataset_data_reset_index():
    data = {
        "pixels": {"x": [255, 255], "y": [0, 0]},
        "signals": {"0": [1.0, 2.0], "1": [3.0, 4.0]},
        "metadata": {"meta1": ["a", "b"], "meta2": ["c", "d"]},
        "target": {
            "label": ["a", "b"],
            "value": [1, 2],
            "encoding": ["x", "y"],
        },
    }
    tabular_dataset_data = TabularDatasetData.from_dict(data)
    sliced_data = tabular_dataset_data[1:2]
    reset_data = sliced_data.reset_index()

    assert isinstance(reset_data, TabularDatasetData)

    # Create expected Signatures for comparison
    expected_pixels_df = pd.DataFrame(data["pixels"]).iloc[1:2].astype(int).reset_index(drop=True)
    expected_signals_df = pd.DataFrame(data["signals"]).iloc[1:2].reset_index(drop=True)

    # Test signatures
    pd.testing.assert_frame_equal(reset_data.signatures.pixels.df, expected_pixels_df)
    pd.testing.assert_frame_equal(reset_data.signatures.signals.df, expected_signals_df)

    # Test metadata
    pd.testing.assert_frame_equal(
        reset_data.metadata,
        pd.DataFrame(data["metadata"]).iloc[1:2].reset_index(drop=True),
    )

    # Test target
    expected_target_df = pd.DataFrame({"label": ["b"], "value": [2]}).reset_index(drop=True)
    assert all(reset_data.target.label == expected_target_df["label"])
    assert all(reset_data.target.value == expected_target_df["value"])
    pd.testing.assert_series_equal(
        reset_data.target.encoding,
        pd.Series(data["target"]["encoding"], name="encoding"),
    )


def test_tabular_dataset_data_copy():
    # Create test data
    pixels = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    signals = pd.DataFrame({"band1": [7, 8, 9], "band2": [10, 11, 12]})
    signatures = Signatures(Pixels(pixels), Signals(signals))
    metadata = pd.DataFrame({"meta1": [13, 14, 15], "meta2": [16, 17, 18]})

    target_data = {
        "label": ["a", "b", "c"],
        "value": [1, 2, 3],
        "encoding": ["x", "y", "z"],
    }
    target = ClassificationTarget.from_dict(target_data)

    # Create original dataset
    original = TabularDatasetData(signatures=signatures, metadata=metadata, target=target)

    # Create copy
    copied = original.copy()

    # Verify the copy is a new object but with equal data
    assert copied is not original
    assert copied.signatures is not original.signatures
    assert copied.metadata is not original.metadata
    assert copied.target is not original.target

    # Verify the data equality
    pd.testing.assert_frame_equal(copied.signatures.pixels.df, original.signatures.pixels.df)
    pd.testing.assert_frame_equal(copied.signatures.signals.df, original.signatures.signals.df)
    pd.testing.assert_frame_equal(copied.metadata, original.metadata)
    assert copied.target.to_dict() == original.target.to_dict()

    # Modify the copy and verify the original is unchanged
    copied.metadata.loc[0, "meta1"] = 999
    copied.signatures.pixels.df.loc[0, "x"] = 888

    assert original.metadata.loc[0, "meta1"] == 13  # Original should remain unchanged
    assert original.signatures.pixels.df.loc[0, "x"] == 1  # Original should remain unchanged
    assert copied.metadata.loc[0, "meta1"] == 999  # Copy should be changed
    assert copied.signatures.pixels.df.loc[0, "x"] == 888  # Copy should be changed

    # Test with no target
    original_no_target = TabularDatasetData(signatures=signatures, metadata=metadata, target=None)
    copied_no_target = original_no_target.copy()

    assert copied_no_target.target is None
    assert copied_no_target is not original_no_target
    assert copied_no_target.signatures is not original_no_target.signatures
    assert copied_no_target.metadata is not original_no_target.metadata
