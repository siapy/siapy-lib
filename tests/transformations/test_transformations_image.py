import numpy as np

from siapy.transformations import image


def test_add_gaussian_noise(spectral_images):
    image_vnir = spectral_images.vnir
    noisy_image = image.add_gaussian_noise(
        image_vnir,
        mean=0.0,
        std=1.0,
        clip_to_max=True,
    )

    assert noisy_image.shape == image_vnir.shape
    assert np.any(noisy_image != image_vnir.to_numpy())


def test_random_crop(spectral_images):
    image_vnir = spectral_images.vnir
    cropped_image = image.random_crop(image_vnir, (50, 50))
    assert cropped_image.shape == (50, 50, image_vnir.bands)


def test_random_mirror(spectral_images):
    image_vnir = spectral_images.vnir
    mirrored_image = image.random_mirror(image_vnir)
    assert mirrored_image.shape == image_vnir.shape


def test_random_rotation(spectral_images):
    image_vnir = spectral_images.vnir
    rotated_image = image.random_rotation(image_vnir, 45)
    assert rotated_image.shape == image_vnir.shape


def test_rescale(spectral_images):
    image_vnir = spectral_images.vnir
    rescaled_image = image.rescale(image_vnir, (50, 50))
    assert rescaled_image.shape == (50, 50, image_vnir.bands)


def test_area_normalization(spectral_images):
    image_vnir = spectral_images.vnir
    normalized_image = image.area_normalization(image_vnir)
    assert normalized_image.shape == image_vnir.shape
