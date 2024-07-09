from pathlib import Path

import pandas as pd
import pytest

from siapy.datasets import TabularDataset
from siapy.datasets.tabular import DatasetDataFrame, TabularDataEntity
from siapy.entities import Pixels, Shape, SpectralImageSet


@pytest.fixture(scope="module")
def sample_dataframe() -> DatasetDataFrame:
    data = {
        Pixels.coords.U: [1, 2],
        Pixels.coords.V: [3, 4],
        "0": [5, 6],
        "1": [7, 8],
        "image_idx": ["a", "b"],
        "image_filepath": [
            Path("/path/to/image_a.tif"),
            Path("/path/to/image_b.tif"),
        ],
        "camera_id": ["camera_a", "camera_b"],
        "shape_type": ["rectangle", "circle"],
        "shape_label": ["c", "d"],
    }
    df = pd.DataFrame(data)
    return DatasetDataFrame(df)


def test_constructor(sample_dataframe):
    assert isinstance(sample_dataframe._constructor(), type(sample_dataframe))


def test_signals(sample_dataframe):
    signals_df = sample_dataframe.signals()
    assert list(signals_df.columns) == ["0", "1"]
    assert signals_df.equals(pd.DataFrame({"0": [5, 6], "1": [7, 8]}))


def test_metadata(sample_dataframe):
    metadata_df = sample_dataframe.metadata()
    assert list(metadata_df.columns) == [
        "image_idx",
        "image_filepath",
        "camera_id",
        "shape_type",
        "shape_label",
    ]
    assert metadata_df.equals(
        pd.DataFrame(
            {
                "image_idx": ["a", "b"],
                "image_filepath": [
                    Path("/path/to/image_a.tif"),
                    Path("/path/to/image_b.tif"),
                ],
                "camera_id": ["camera_a", "camera_b"],
                "shape_type": ["rectangle", "circle"],
                "shape_label": ["c", "d"],
            }
        )
    )


def test_pixels(sample_dataframe):
    pixels_df = sample_dataframe.pixels()
    assert list(pixels_df.columns) == [Pixels.coords.U, Pixels.coords.V]
    assert pixels_df.equals(
        pd.DataFrame({Pixels.coords.U: [1, 2], Pixels.coords.V: [3, 4]})
    )


@pytest.fixture(scope="module")
def spectral_images_set(spectral_images):
    pixels_input = [(10, 15), (60, 66)]
    pixels = Pixels.from_iterable(pixels_input)
    rectangle = Shape.from_shape_type(shape_type="rectangle", pixels=pixels)

    spectral_images.vnir.geometric_shapes.append(rectangle)
    spectral_images.swir.geometric_shapes.append(rectangle)

    images = [
        spectral_images.vnir,
        spectral_images.swir,
        spectral_images.vnir,
    ]

    return SpectralImageSet(images)


def test_tabular(spectral_images_set):
    dataset = TabularDataset(spectral_images_set)
    dataset.process_image_data()
    dataset.generate_dataset()


def test_tabular_len(spectral_images_set):
    dataset = TabularDataset(spectral_images_set)
    dataset.process_image_data()
    assert len(dataset) == 3


def test_tabular_str(spectral_images_set):
    dataset = TabularDataset(spectral_images_set)
    dataset.process_image_data()
    expected_str = "<TabularDataset object with 3 data entities>"
    assert str(dataset) == expected_str


def test_tabular_iter(spectral_images_set):
    dataset = TabularDataset(spectral_images_set)
    dataset.process_image_data()
    for entity in dataset:
        assert isinstance(entity, TabularDataEntity)


def test_tabular_getitem(spectral_images_set):
    dataset = TabularDataset(spectral_images_set)
    dataset.process_image_data()
    first_entity = dataset[0]
    assert isinstance(first_entity, TabularDataEntity)


def test_tabular_image_set(spectral_images_set):
    dataset = TabularDataset(spectral_images_set)
    assert isinstance(dataset.image_set, SpectralImageSet)


def test_tabular_data_entities(spectral_images_set):
    dataset = TabularDataset(spectral_images_set)
    dataset.process_image_data()
    data_entities = dataset.data_entities
    assert all(isinstance(entity, TabularDataEntity) for entity in data_entities)


def test_tabular_process_image_data(spectral_images_set):
    dataset = TabularDataset(spectral_images_set)
    dataset.process_image_data()
    assert len(dataset.data_entities) > 0


def test_tabular_generate_dataset(spectral_images_set):
    dataset = TabularDataset(spectral_images_set)
    dataset.process_image_data()
    df = dataset.generate_dataset()
    assert isinstance(df, DatasetDataFrame)
    assert not df.empty
