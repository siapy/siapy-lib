# mypy: ignore-errors
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest
import spectral as sp

from siapy.entities import SpectralImage
from siapy.entities.pixels import Pixels
from siapy.entities.shapes import Shape
from siapy.utils.images import (
    blockfy_image,
    calculate_correction_factor_from_panel,
    calculate_image_background_percentage,
    convert_radiance_image_to_reflectance,
    create_image,
    merge_images_by_specter,
    save_image,
)


@pytest.mark.manual
def test_save_image_manual(spectral_images):
    image = spectral_images.vnir_np
    with TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir, "test_image.hdr")
        save_image(image, save_path)
        assert save_path.exists()


@pytest.mark.manual
def test_merge_images_by_specter_manual(spectral_images):
    vnir = spectral_images.vnir
    swir = spectral_images.swir
    with TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir, "test_image.hdr")
        merge_images_by_specter(
            image_original=vnir,
            image_to_merge=swir,
            save_path=save_path,
        )
        assert save_path.exists()


def test_save_image():
    with TemporaryDirectory() as tmpdir:
        image = np.random.default_rng().random((100, 100))
        save_path = Path(tmpdir, "test_image.hdr")
        save_image(image, save_path)
        assert save_path.exists()


def test_save_image_overwrite_argument():
    with TemporaryDirectory() as tmpdir:
        image = np.random.default_rng().random((100, 100))
        save_path = Path(tmpdir, "test_image.hdr")
        save_image(image, save_path)
        with pytest.raises(sp.io.envi.EnviException):
            save_image(image, save_path, overwrite=False)


def test_save_image_metadata_argument():
    with TemporaryDirectory() as tmpdir:
        image = np.random.default_rng().random((100, 100))
        metadata = {"description": "test"}
        save_path = Path(tmpdir, "test_image.hdr")
        save_image(image, save_path, metadata=metadata)
        image_disc = SpectralImage.envi_open(header_path=save_path)
        assert image_disc.metadata["description"] == "test"


def test_save_image_dtype_argument():
    with TemporaryDirectory() as tmpdir:
        image = np.random.default_rng().random((100, 100))
        save_path = Path(tmpdir, "test_image.hdr")
        dtype = np.uint16
        save_image(image, save_path, dtype=dtype)
        image_disc = SpectralImage.envi_open(header_path=save_path)
        assert image_disc.file.dtype == np.dtype(dtype)


def test_save_image_path_argument():
    with TemporaryDirectory() as tmpdir:
        image = np.random.default_rng().random((100, 100))
        save_path_str = os.path.join(tmpdir, "test_image.hdr")
        save_image(image, save_path_str)
        assert Path(save_path_str).exists()


def test_create_image():
    with TemporaryDirectory() as tmpdir:
        image = np.random.default_rng().random((100, 100, 3))
        save_path = Path(tmpdir, "test_image.hdr")
        result = create_image(image, save_path)
        assert isinstance(result, SpectralImage)
        assert save_path.exists()


def test_create_image_overwrite_argument():
    with TemporaryDirectory() as tmpdir:
        image = np.random.default_rng().random((100, 100, 3))
        save_path = Path(tmpdir, "test_image.hdr")
        create_image(image, save_path)
        with pytest.raises(Exception):
            create_image(image, save_path, overwrite=False)


def test_create_image_dtype_argument():
    with TemporaryDirectory() as tmpdir:
        image = np.random.default_rng().random((100, 100, 3))
        save_path = Path(tmpdir, "test_image.hdr")
        dtype = np.uint8
        result = create_image(image, save_path, dtype=dtype)
        assert result.file.dtype == np.dtype(dtype)


def test_merge_images_by_specter():
    class MockSpectralImage(SpectralImage):
        def __init__(self, image: np.ndarray):
            self.image = image

        def to_numpy(self) -> np.ndarray:  # type: ignore
            return self.image

        @property
        def shape(self) -> tuple[int, int, int]:
            return self.image.shape  # type: ignore

    mock_vnir = MockSpectralImage(np.random.default_rng().random((100, 100, 10)))
    mock_swir = MockSpectralImage(np.random.default_rng().random((100, 100, 20)))

    with TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir, "test_image_merged.hdr")
        merge_images_by_specter(
            image_original=mock_vnir,
            image_to_merge=mock_swir,
            save_path=save_path,
            auto_metadata_extraction=False,
        )
        assert save_path.exists()


def test_calculate_correction_factor_from_panel(spectral_images):
    pixels = Pixels.from_iterable([(900, 1150), (1050, 1300)])
    rect = Shape.from_shape_type(
        shape_type="rectangle", pixels=pixels, label="reference_panel"
    )
    image_vnir = spectral_images.vnir
    image_vnir.geometric_shapes.append(rect)

    panel_correction = calculate_correction_factor_from_panel(
        image=image_vnir,
        panel_reference_reflectance=0.2,
        panel_shape_label="reference_panel",
    )

    assert panel_correction is not None
    assert isinstance(panel_correction, np.ndarray)
    assert panel_correction.shape == (image_vnir.bands,)

    a = image_vnir.to_numpy()
    b = rect.convex_hull()
    c = a[b.v(), b.u(), :]
    assert np.array_equal(
        np.full(image_vnir.bands, 0.2), np.round(c.mean(axis=0) * panel_correction, 2)
    )


def test_convert_radiance_image_to_reflectance_without_saving(spectral_images):
    image_vnir = spectral_images.vnir
    panel_correction = np.random.default_rng().random(image_vnir.bands)

    result = convert_radiance_image_to_reflectance(
        image=image_vnir, panel_correction=panel_correction, save_path=None
    )
    assert isinstance(result, np.ndarray)
    assert np.array_equal(result, image_vnir.to_numpy() * panel_correction)


def test_convert_radiance_image_to_reflectance_with_saving(spectral_images, tmp_path):
    image_vnir = spectral_images.vnir
    panel_correction = np.random.default_rng().random(image_vnir.bands)

    with TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir, "test_image.hdr")
        result = convert_radiance_image_to_reflectance(
            image=image_vnir, panel_correction=panel_correction, save_path=save_path
        )
        assert save_path.exists()
        assert isinstance(result, SpectralImage)
        assert np.array_equal(
            result.to_numpy(),
            np.array(image_vnir.to_numpy() * panel_correction).astype("float32"),
        )


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
    np.testing.assert_array_almost_equal(
        reconstructed_image[: image.shape[0], : image.shape[1]], image
    )
