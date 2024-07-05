import numpy as np
import pytest

from siapy.entities.pixels import Pixels
from siapy.transformations import corregistrator
from siapy.utils.plots import pixels_select_click  # noqa: F401


@pytest.mark.manual
def test_pixels_select_click_manual(spectral_images, corresponding_pixels):
    # image_vnir = spectral_images.vnir
    # image_swir = spectral_images.swir
    # pixels_vnir = pixels_select_click(image_vnir)
    # pixels_swir = pixels_select_click(image_swir)
    pixels_vnir = corresponding_pixels.vnir
    pixels_swir = corresponding_pixels.swir

    matx, _ = corregistrator.align(pixels_swir, pixels_vnir, plot_progress=False)
    pixels_transformed = corregistrator.transform(pixels_vnir, matx)
    assert (
        np.sqrt(np.sum((pixels_swir.to_numpy() - pixels_transformed.to_numpy()) ** 2))
        < 10
    )


def test_transform():
    pixels_ref = Pixels.from_iterable(np.array([[1, 2], [3, 4], [5, 6]]))
    transformation_matx = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # Identity matrix
    transformed_pixels = corregistrator.transform(pixels_ref, transformation_matx)
    np.testing.assert_array_equal(pixels_ref.to_numpy(), transformed_pixels.to_numpy())


def test_affine_matx_2d_identity():
    expected_identity = np.eye(3)
    identity_matx = corregistrator.affine_matx_2d()
    np.testing.assert_array_equal(identity_matx, expected_identity)


def test_affine_matx_2d_transformations():
    scale = (2, 3)
    trans = (1, -1)
    rot = 45  # degrees
    shear = (0.1, 0.2)
    matx_2d = corregistrator.affine_matx_2d(scale, trans, rot, shear)
    # This test checks if the matrix is created but does not validate its correctness
    assert matx_2d.shape == (3, 3)
