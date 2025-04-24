# mypy: ignore-errors
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest
import rioxarray  # noqa
import spectral as sp

from siapy.core.exceptions import InvalidInputError
from siapy.entities import SpectralImage
from siapy.entities.images import RasterioLibImage, SpectralLibImage
from siapy.entities.shapes import Shape
from siapy.utils.images import (
    blockfy_image,
    calculate_correction_factor,
    calculate_correction_factor_from_panel,
    calculate_image_background_percentage,
    convert_radiance_image_to_reflectance,
    rasterio_create_image,
    rasterio_save_image,
    spy_create_image,
    spy_merge_images_by_specter,
    spy_save_image,
)
from siapy.utils.signatures import get_signatures_within_convex_hull

# Spectral


@pytest.mark.manual
def test_save_image_manual(spectral_images):
    image = spectral_images.vnir_np
    with TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir, "test_image.hdr")
        spy_save_image(image, save_path)
        assert save_path.exists()


@pytest.mark.manual
def test_merge_images_by_specter_manual(spectral_images):
    vnir = spectral_images.vnir
    swir = spectral_images.swir
    with TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir, "test_image.hdr")
        spy_merge_images_by_specter(
            image_original=vnir,
            image_to_merge=swir,
            save_path=save_path,
        )
        assert save_path.exists()


def test_save_image():
    with TemporaryDirectory() as tmpdir:
        image = np.random.default_rng().random((100, 100))
        save_path = Path(tmpdir, "test_image.hdr")
        spy_save_image(image, save_path)
        assert save_path.exists()


def test_save_image_overwrite_argument():
    with TemporaryDirectory() as tmpdir:
        image = np.random.default_rng().random((100, 100))
        save_path = Path(tmpdir, "test_image.hdr")
        spy_save_image(image, save_path)
        with pytest.raises(sp.io.envi.EnviException):
            spy_save_image(image, save_path, overwrite=False)


def test_save_image_metadata_argument():
    with TemporaryDirectory() as tmpdir:
        image = np.random.default_rng().random((100, 100))
        metadata = {"description": "test"}
        save_path = Path(tmpdir, "test_image.hdr")
        spy_save_image(image, save_path, metadata=metadata)
        image_disc = SpectralImage.spy_open(header_path=save_path)
        assert image_disc.metadata["description"] == "test"


def test_save_image_dtype_argument():
    with TemporaryDirectory() as tmpdir:
        image = np.random.default_rng().random((100, 100))
        save_path = Path(tmpdir, "test_image.hdr")
        dtype = np.uint16
        spy_save_image(image, save_path, dtype=dtype)
        image_disc = SpectralImage.spy_open(header_path=save_path)
        assert image_disc.image.file.dtype == np.dtype(dtype)


def test_save_image_path_argument():
    with TemporaryDirectory() as tmpdir:
        image = np.random.default_rng().random((100, 100))
        save_path_str = os.path.join(tmpdir, "test_image.hdr")
        spy_save_image(image, save_path_str)
        assert Path(save_path_str).exists()


def test_create_image():
    with TemporaryDirectory() as tmpdir:
        image = np.random.default_rng().random((100, 100, 3))
        save_path = Path(tmpdir, "test_image.hdr")
        result = spy_create_image(image, save_path)
        assert isinstance(result, SpectralImage)
        assert save_path.exists()
        assert np.array_equal(result.to_numpy(), image.astype("float32"))


def test_create_image_overwrite_argument():
    with TemporaryDirectory() as tmpdir:
        image = np.random.default_rng().random((100, 100, 3))
        save_path = Path(tmpdir, "test_image.hdr")
        spy_create_image(image, save_path)
        with pytest.raises(Exception):
            spy_create_image(image, save_path, overwrite=False)


def test_create_image_dtype_argument():
    with TemporaryDirectory() as tmpdir:
        image = np.random.default_rng().random((100, 100, 3))
        save_path = Path(tmpdir, "test_image.hdr")
        dtype = np.uint8
        result = spy_create_image(image, save_path, dtype=dtype)
        assert result.image.file.dtype == np.dtype(dtype)


def test_merge_images_by_specter():
    class MockSpectralImage(SpectralImage[SpectralLibImage]):
        def __init__(self, array: np.ndarray):
            self.array = array

        def to_numpy(self) -> np.ndarray:
            return self.array

        @property
        def shape(self) -> tuple[int, int, int]:
            return self.array.shape

    mock_vnir = MockSpectralImage(np.random.default_rng(seed=0).random((100, 100, 10)))
    mock_swir = MockSpectralImage(np.random.default_rng(seed=0).random((200, 100, 20)))

    with TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir, "test_image_merged.hdr")
        spy_merge_images_by_specter(
            image_original=mock_vnir,
            image_to_merge=mock_swir,
            save_path=save_path,
            auto_metadata_extraction=False,
        )
        assert save_path.exists()


# Rasterio


def test_rasterio_save_image():
    with TemporaryDirectory() as tmpdir:
        image = np.random.default_rng().random((100, 100, 3))
        save_path = Path(tmpdir, "test_image.tif")
        rasterio_save_image(image, save_path)
        assert save_path.exists()
        loaded = SpectralImage.rasterio_open(filepath=save_path)
        np.testing.assert_array_almost_equal(loaded.to_numpy(), image.astype("float32"))


def test_rasterio_save_image_overwrite_argument():
    with TemporaryDirectory() as tmpdir:
        image = np.random.default_rng().random((100, 100, 3))
        save_path = Path(tmpdir, "test_image.tif")
        rasterio_save_image(image, save_path)
        with pytest.raises(InvalidInputError):
            rasterio_save_image(image, save_path, overwrite=False)
        rasterio_save_image(image, save_path, overwrite=True)


def test_rasterio_save_image_metadata_argument():
    with TemporaryDirectory() as tmpdir:
        image = np.random.default_rng().random((100, 100, 3))
        metadata = {"description": "test image", "wavelength": [1.0, 2.0, 3.0]}
        save_path = Path(tmpdir, "test_image.tif")
        rasterio_save_image(image, save_path, metadata=metadata)
        loaded = rioxarray.open_rasterio(save_path)
        assert "description" in loaded.attrs
        assert loaded.attrs["description"] == "test image"
        assert "wavelength" in loaded.attrs


def test_rasterio_save_image_dtype_argument():
    with TemporaryDirectory() as tmpdir:
        image = np.random.default_rng().random((100, 100, 3))
        save_path = Path(tmpdir, "test_image.tif")
        dtype = np.uint8
        rasterio_save_image(image, save_path, dtype=dtype)
        loaded = rioxarray.open_rasterio(save_path)
        assert loaded.dtype == np.dtype(dtype)


def test_rasterio_save_image_with_kwargs():
    with TemporaryDirectory() as tmpdir:
        image = np.random.default_rng().random((100, 100, 3))
        save_path = Path(tmpdir, "test_image.tif")
        rasterio_save_image(image, save_path, compress="lzw")
        assert save_path.exists()


def test_rasterio_create_image():
    with TemporaryDirectory() as tmpdir:
        image = np.random.default_rng().random((100, 100, 3))
        save_path = Path(tmpdir, "test_image.tif")

        result = rasterio_create_image(image, save_path)

        assert isinstance(result, SpectralImage)
        assert save_path.exists()

        # Check the image data
        image_data = result.to_numpy()
        np.testing.assert_array_almost_equal(image_data, image.astype("float32"))

        # Verify it's using RasterioLibImage backend
        assert isinstance(result.image, RasterioLibImage)


def test_rasterio_create_image_overwrite_argument():
    with TemporaryDirectory() as tmpdir:
        image = np.random.default_rng().random((100, 100, 3))
        save_path = Path(tmpdir, "test_image.tif")
        rasterio_create_image(image, save_path)
        with pytest.raises(InvalidInputError):
            rasterio_create_image(image, save_path, overwrite=False)


def test_rasterio_create_image_metadata_argument():
    with TemporaryDirectory() as tmpdir:
        image = np.random.default_rng().random((100, 100, 3))
        metadata = {"description": "test image", "wavelength": [11.0, 12.0, 13.0]}
        save_path = Path(tmpdir, "test_image.tif")
        result = rasterio_create_image(image, save_path, metadata=metadata)
        assert "description" in result.metadata
        assert result.metadata["description"] == "test image"
        # assert result.to_xarray().band.values == [11.0, 12.0, 13.0]


def test_rasterio_create_image_with_kwargs():
    with TemporaryDirectory() as tmpdir:
        image = np.random.default_rng().random((100, 100, 3))
        save_path = Path(tmpdir, "test_image.tif")
        result = rasterio_create_image(image, save_path, compress="lzw")
        assert isinstance(result, SpectralImage)
        assert save_path.exists()


# Other


def test_calculate_correction_factor():
    # Test with valid values
    panel_radiance_mean = np.array([0.2, 0.3, 0.4])
    panel_reference_reflectance = 0.5
    correction_factor = calculate_correction_factor(panel_radiance_mean, panel_reference_reflectance)
    expected = np.array([2.5, 1.6666667, 1.25])
    np.testing.assert_array_almost_equal(correction_factor, expected)

    # Test with invalid values
    with pytest.raises(InvalidInputError):
        calculate_correction_factor(panel_radiance_mean, -0.1)
    with pytest.raises(InvalidInputError):
        calculate_correction_factor(panel_radiance_mean, 1.1)


def test_calculate_correction_factor_from_panel_with_label(spectral_images):
    rect = Shape.from_rectangle(x_min=200, y_min=350, x_max=300, y_max=400, label="reference_panel")
    image_vnir = spectral_images.vnir
    image_vnir.geometric_shapes.append(rect)

    panel_correction = calculate_correction_factor_from_panel(
        image=image_vnir,
        panel_reference_reflectance=0.2,
        panel_shape_label="reference_panel",
    )

    assert isinstance(panel_correction, np.ndarray)
    assert panel_correction.shape == (image_vnir.bands,)

    a = image_vnir.to_numpy()
    pixels = get_signatures_within_convex_hull(image_vnir, rect)[0].pixels
    c = a[pixels.y(), pixels.x(), :]
    assert np.array_equal(
        np.full(image_vnir.bands, 0.2),
        np.round(c.mean(axis=0) * panel_correction, 2),
    )


def test_calculate_correction_factor_from_panel_without_label(spectral_images):
    image_vnir = spectral_images.vnir
    panel_correction = calculate_correction_factor_from_panel(
        image=image_vnir,
        panel_reference_reflectance=0.3,
    )
    direct_panel_calculation = np.full(image_vnir.bands, 0.3) / image_vnir.average_intensity(axis=(0, 1))
    assert np.array_equal(direct_panel_calculation, panel_correction)


def test_convert_radiance_image_to_reflectance(spectral_images):
    image_vnir = spectral_images.vnir
    panel_correction = np.random.default_rng().random(image_vnir.bands)
    result = convert_radiance_image_to_reflectance(image=image_vnir, panel_correction=panel_correction)
    assert isinstance(result, np.ndarray)
    assert np.array_equal(result, image_vnir.to_numpy() * panel_correction)


def test_calculate_image_background_percentage_mixed_background():
    image = np.random.default_rng(0).random((100, 100, 3))
    image[0:25, 0:25, :] = np.nan
    image[75:100, 75:100, :] = np.nan
    image[55, 50, 1] = np.nan
    percentage = calculate_image_background_percentage(image)
    assert percentage == pytest.approx(12.5099999)

    image = np.full((100, 100, 3), np.nan)
    percentage = calculate_image_background_percentage(image)
    assert percentage == 100


def test_blockfy_image():
    image = np.random.default_rng(0).random((100, 100, 3))

    p, q = 25, 25

    blocks = blockfy_image(image, p, q)

    expected_blocks_per_row = (image.shape[0] - 1) // p + 1
    expected_blocks_per_column = (image.shape[1] - 1) // q + 1
    expected_num_blocks = expected_blocks_per_row * expected_blocks_per_column

    assert len(blocks) == expected_num_blocks

    for block in blocks:
        assert block.shape == (p, q, image.shape[2])

    # Concatenate image blocks back to original image
    reconstructed_image = np.block(
        [
            [
                # col_idx * i + j -> position in a list
                # , where i is the row index and j is the column index of the block, sliced from original image
                [blocks[expected_blocks_per_column * i + j]]
                for j in range(expected_blocks_per_column)
            ]
            for i in range(expected_blocks_per_row)
        ]
    )
    np.testing.assert_array_almost_equal(reconstructed_image[: image.shape[0], : image.shape[1]], image)
