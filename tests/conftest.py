from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.datasets import make_classification

from siapy.core.configs import TEST_DATA_DIR
from siapy.datasets.tabular import TabularDataset, TabularDatasetData
from siapy.entities import Pixels, SpectralImage, SpectralImageSet
from siapy.entities.shapes import Shape
from tests.data_manager import verify_testdata_integrity


class PytestConfigs(SimpleNamespace):
    image_vnir_hdr_path = TEST_DATA_DIR / "hyspex" / "vnir.hdr"
    image_vnir_img_path = TEST_DATA_DIR / "hyspex" / "vnir.hyspex"
    image_swir_hdr_path = TEST_DATA_DIR / "hyspex" / "swir.hdr"
    image_swir_img_path = TEST_DATA_DIR / "hyspex" / "swir.hyspex"
    image_vnir_name = "VNIR_1600_SN0034"
    image_swir_name = "SWIR_384me_SN3109"

    image_micasense_blue = TEST_DATA_DIR / "micasense" / "blue.tif"
    image_micasense_green = TEST_DATA_DIR / "micasense" / "green.tif"
    image_micasense_red = TEST_DATA_DIR / "micasense" / "red.tif"
    image_micasense_nir = TEST_DATA_DIR / "micasense" / "nir.tif"
    image_micasense_rededge = TEST_DATA_DIR / "micasense" / "rededge.tif"
    image_micasense_merged = TEST_DATA_DIR / "micasense" / "merged.tif"

    shapefile_point = TEST_DATA_DIR / "micasense" / "point.shp"
    shapefile_buffer = TEST_DATA_DIR / "micasense" / "buffer.shp"


@pytest.fixture(scope="session")
def configs():
    verify_testdata_integrity()
    return PytestConfigs()


class SpectralImages(SimpleNamespace):
    vnir: SpectralImage
    swir: SpectralImage
    vnir_np: np.ndarray
    swir_np: np.ndarray


@pytest.fixture(scope="module")
def spectral_images(configs) -> SpectralImages:
    spectral_image_vnir = SpectralImage.spy_open(
        header_path=configs.image_vnir_hdr_path,
        image_path=configs.image_vnir_img_path,
    )
    spectral_image_swir = SpectralImage.spy_open(
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
    x_min = 10
    y_min = 15
    x_max = 60
    y_max = 66

    rectangle = Shape.from_rectangle(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)

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
    X, y = make_classification(n_samples=100, n_features=10, n_classes=2, random_state=0)
    return X, y
