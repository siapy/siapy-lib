from pathlib import Path

import pandas as pd
import pytest

from siapy.datasets.helpers import (
    generate_classification_target,
    generate_regression_target,
    merge_signals_from_multiple_cameras,
)
from siapy.datasets.schemas import ClassificationTarget, RegressionTarget
from siapy.entities import Pixels


@pytest.fixture(scope="module")
def sample_dataframe() -> pd.DataFrame:
    data = {
        Pixels.coords.U: [1, 2],
        Pixels.coords.V: [3, 4],
        "0": [5, 6],
        "1": [7, 8],
        "image_idx": [0, 1],
        "shape_idx": [3, 4],
        "image_filepath": [
            Path("/path/to/image_a.tif"),
            Path("/path/to/image_b.tif"),
        ],
        "camera_id": ["camera_a", "camera_b"],
        "shape_type": ["rectangle", "circle"],
        "shape_label": ["c", "d"],
    }
    return pd.DataFrame(data)


def test_dataframe_generate_classification_target_single_column(sample_dataframe):
    classification_target = generate_classification_target(
        sample_dataframe, "shape_type"
    )
    assert isinstance(classification_target, ClassificationTarget)
    assert all(
        classification_target.label == pd.Series(["rectangle", "circle"], name="label")
    )
    assert classification_target.value.name == "encoded"
    assert classification_target.encoding.name == "encoding"
    assert list(classification_target.value) == [0, 1]
    assert classification_target.encoding.to_dict() == {
        0: "rectangle",
        1: "circle",
    }


def test_dataframe_generate_classification_target_multiple_columns(sample_dataframe):
    classification_target = generate_classification_target(
        sample_dataframe, ["shape_type", "shape_label"]
    )
    assert isinstance(classification_target, ClassificationTarget)
    assert all(
        classification_target.label
        == pd.Series(["rectangle__c", "circle__d"], name="label")
    )
    assert list(classification_target.value) == [0, 1]
    assert classification_target.encoding.to_dict() == {
        0: "rectangle__c",
        1: "circle__d",
    }


def test_dataframe_generate_regression_target(sample_dataframe):
    regression_target = generate_regression_target(sample_dataframe, "0")
    assert isinstance(regression_target, RegressionTarget)
    assert regression_target.name == "0"
    assert all(regression_target.value == sample_dataframe["0"])


def test_merge_signals_from_multiple_cameras(spectral_tabular_dataset):
    merge_signals_from_multiple_cameras(spectral_tabular_dataset.dataset_data)
