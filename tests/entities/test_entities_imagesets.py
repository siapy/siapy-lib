from pathlib import Path

import pytest

from siapy.entities import SpectralImage, SpectralImageSet
from tests.configs import (
    image_swir_hdr_path,
    image_swir_img_path,
    image_vnir_hdr_path,
    image_vnir_img_path,
)


def test_from_paths_valid():
    header_paths: list[str | Path] = [image_vnir_hdr_path, image_swir_hdr_path]
    image_paths: list[str | Path] = [image_vnir_img_path, image_swir_img_path]
    image_set = SpectralImageSet.from_paths(
        header_paths=header_paths, image_paths=image_paths
    )
    assert len(image_set) == 2


def test_from_paths_invalid():
    header_paths: list[str | Path] = [image_vnir_hdr_path, image_swir_hdr_path]
    image_paths: list[str | Path] = [image_vnir_img_path]
    with pytest.raises(ValueError):
        SpectralImageSet.from_paths(header_paths=header_paths, image_paths=image_paths)


def create_spectral_image(hdr_path, img_path):
    return SpectralImage.envi_open(header_path=hdr_path, image_path=img_path)


def test_len():
    image_set = SpectralImageSet()
    assert len(image_set) == 0

    vnir_image = create_spectral_image(image_vnir_hdr_path, image_vnir_img_path)
    swir_image = create_spectral_image(image_swir_hdr_path, image_swir_img_path)
    image_set = SpectralImageSet(spectral_images=[vnir_image, swir_image])
    assert len(image_set) == 2


def test_str():
    image_set = SpectralImageSet()
    assert str(image_set) == "<SpectralImageSet object with 0 spectral images>"

    vnir_image = create_spectral_image(image_vnir_hdr_path, image_vnir_img_path)
    swir_image = create_spectral_image(image_swir_hdr_path, image_swir_img_path)
    image_set = SpectralImageSet(spectral_images=[vnir_image, swir_image])
    assert str(image_set) == "<SpectralImageSet object with 2 spectral images>"


def test_iter():
    vnir_image = create_spectral_image(image_vnir_hdr_path, image_vnir_img_path)
    swir_image = create_spectral_image(image_swir_hdr_path, image_swir_img_path)
    image_set = SpectralImageSet(spectral_images=[vnir_image, swir_image])
    assert list(image_set) == [
        vnir_image,
        swir_image,
    ]


def test_getitem():
    vnir_image = create_spectral_image(image_vnir_hdr_path, image_vnir_img_path)
    image_set = SpectralImageSet(spectral_images=[vnir_image])
    assert image_set[0] == vnir_image
