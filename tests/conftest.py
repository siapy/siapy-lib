from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.datasets import make_classification

from siapy.core.configs import TEST_DATA_DIR
from siapy.datasets.tabular import TabularDataset, TabularDatasetData
from siapy.entities import Pixels, Shape, SpectralImage, SpectralImageSet


class PytestConfigs(SimpleNamespace):
    image_vnir_hdr_path = TEST_DATA_DIR / "vnir.hdr"
    image_vnir_img_path = TEST_DATA_DIR / "vnir.hyspex"
    image_swir_hdr_path = TEST_DATA_DIR / "swir.hdr"
    image_swir_img_path = TEST_DATA_DIR / "swir.hyspex"
    image_vnir_name = "VNIR_1600_SN0034"
    image_swir_name = "SWIR_384me_SN3109"


@pytest.fixture(scope="session")
def configs():
    return PytestConfigs()


class SpectralImages(SimpleNamespace):
    vnir: SpectralImage
    swir: SpectralImage
    vnir_np: np.ndarray
    swir_np: np.ndarray


@pytest.fixture(scope="module")
def spectral_images(configs) -> SpectralImages:
    spectral_image_vnir = SpectralImage.envi_open(
        header_path=configs.image_vnir_hdr_path,
        image_path=configs.image_vnir_img_path,
    )
    spectral_image_swir = SpectralImage.envi_open(
        header_path=configs.image_swir_hdr_path,
        image_path=configs.image_swir_img_path,
    )
    spectral_image_vnir_np = spectral_image_vnir.to_numpy()
    spectral_image_swir_np = spectral_image_swir.to_numpy()
    return SpectralImages(
        vnir=spectral_image_vnir,
        swir=spectral_image_swir,
        vnir_np=spectral_image_vnir_np,
        swir_np=spectral_image_swir_np,
    )


class CorrespondingPixels(SimpleNamespace):
    vnir: Pixels
    swir: Pixels


@pytest.fixture(scope="module")
def corresponding_pixels() -> CorrespondingPixels:
    pixels_vnir = np.array(
        [
            [1007, 620],
            [417, 1052],
            [439, 1582],
            [1100, 1866],
            [832, 1090],
            [1133, 1079],
            [854, 1407],
            [1138, 1413],
        ]
    )
    pixels_swir = np.array(
        [
            [252, 110],
            [99, 219],
            [107, 354],
            [268, 422],
            [207, 230],
            [279, 225],
            [210, 309],
            [283, 309],
        ]
    )
    return CorrespondingPixels(
        vnir=Pixels.from_iterable(pixels_vnir),
        swir=Pixels.from_iterable(pixels_swir),
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


class TabularDatasetReturn(SimpleNamespace):
    dataset: TabularDataset
    dataset_data: TabularDatasetData


@pytest.fixture(scope="module")
def spectral_tabular_dataset(spectral_images_set):
    dataset = TabularDataset(spectral_images_set)
    dataset.process_image_data()
    dataset_data = dataset.generate_dataset_data()
    return TabularDatasetReturn(dataset=dataset, dataset_data=dataset_data)


@pytest.fixture(scope="module")
def mock_sklearn_dataset():
    X, y = make_classification(
        n_samples=100, n_features=10, n_classes=2, random_state=0
    )
    return X, y
