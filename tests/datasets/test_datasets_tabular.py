from siapy.datasets.schemas import TabularDatasetData
from siapy.datasets.tabular import TabularDataEntity
from siapy.entities import SpectralImageSet


def test_tabular_len(spectral_tabular_dataset):
    dataset = spectral_tabular_dataset.dataset
    assert len(dataset) == 3


def test_tabular_str(spectral_tabular_dataset):
    dataset = spectral_tabular_dataset.dataset
    expected_str = "<TabularDataset object with 3 data entities>"
    assert str(dataset) == expected_str


def test_tabular_iter(spectral_tabular_dataset):
    dataset = spectral_tabular_dataset.dataset
    for entity in dataset:
        assert isinstance(entity, TabularDataEntity)


def test_tabular_getitem(spectral_tabular_dataset):
    dataset = spectral_tabular_dataset.dataset
    first_entity = dataset[0]
    assert isinstance(first_entity, TabularDataEntity)


def test_tabular_image_set(spectral_tabular_dataset):
    dataset = spectral_tabular_dataset.dataset
    assert isinstance(dataset.image_set, SpectralImageSet)


def test_tabular_data_entities(spectral_tabular_dataset):
    dataset = spectral_tabular_dataset.dataset
    data_entities = dataset.data_entities
    assert all(isinstance(entity, TabularDataEntity) for entity in data_entities)


def test_tabular_process_image_data(spectral_tabular_dataset):
    dataset = spectral_tabular_dataset.dataset
    assert len(dataset.data_entities) > 0


def test_tabular_generate_dataset(spectral_tabular_dataset):
    data = spectral_tabular_dataset.dataset_data
    assert isinstance(data, TabularDatasetData)
    assert not data.pixels.empty
    assert not data.signals.empty
    assert not data.metadata.empty
    assert data.target is None
