from pathlib import Path

import pytest

from siapy.entities import SpectralImage, SpectralImageSet


def test_from_paths_valid(configs):
    header_paths: list[str | Path] = [
        configs.image_vnir_hdr_path,
        configs.image_swir_hdr_path,
    ]
    image_paths: list[str | Path] = [
        configs.image_vnir_img_path,
        configs.image_swir_img_path,
    ]
    image_set = SpectralImageSet.from_paths(
        header_paths=header_paths, image_paths=image_paths
    )
    assert len(image_set) == 2


def test_from_paths_invalid(configs):
    header_paths: list[str | Path] = [
        configs.image_vnir_hdr_path,
        configs.image_swir_hdr_path,
    ]
    image_paths: list[str | Path] = [configs.image_vnir_img_path]
    with pytest.raises(ValueError):
        SpectralImageSet.from_paths(header_paths=header_paths, image_paths=image_paths)


def create_spectral_image(hdr_path, img_path):
    return SpectralImage.envi_open(header_path=hdr_path, image_path=img_path)


def test_len(configs):
    image_set = SpectralImageSet()
    assert len(image_set) == 0

    vnir_image = create_spectral_image(
        configs.image_vnir_hdr_path, configs.image_vnir_img_path
    )
    swir_image = create_spectral_image(
        configs.image_swir_hdr_path, configs.image_swir_img_path
    )
    image_set = SpectralImageSet(spectral_images=[vnir_image, swir_image])
    assert len(image_set) == 2


def test_str(configs):
    image_set = SpectralImageSet()
    assert str(image_set) == "<SpectralImageSet object with 0 spectral images>"

    vnir_image = create_spectral_image(
        configs.image_vnir_hdr_path, configs.image_vnir_img_path
    )
    swir_image = create_spectral_image(
        configs.image_swir_hdr_path, configs.image_swir_img_path
    )
    image_set = SpectralImageSet(spectral_images=[vnir_image, swir_image])
    assert str(image_set) == "<SpectralImageSet object with 2 spectral images>"


def test_iter(configs):
    vnir_image = create_spectral_image(
        configs.image_vnir_hdr_path, configs.image_vnir_img_path
    )
    swir_image = create_spectral_image(
        configs.image_swir_hdr_path, configs.image_swir_img_path
    )
    image_set = SpectralImageSet(spectral_images=[vnir_image, swir_image])
    assert list(image_set) == [
        vnir_image,
        swir_image,
    ]


def test_getitem(configs):
    vnir_image = create_spectral_image(
        configs.image_vnir_hdr_path, configs.image_vnir_img_path
    )
    image_set = SpectralImageSet(spectral_images=[vnir_image])
    assert image_set[0] == vnir_image


def test_images_by_camera_id(configs):
    vnir_image1 = create_spectral_image(
        configs.image_vnir_hdr_path, configs.image_vnir_img_path
    )
    vnir_image2 = create_spectral_image(
        configs.image_vnir_hdr_path, configs.image_vnir_img_path
    )
    swir_image = create_spectral_image(
        configs.image_swir_hdr_path, configs.image_swir_img_path
    )
    image_set = SpectralImageSet(spectral_images=[vnir_image1, vnir_image2, swir_image])
    assert [swir_image] == image_set.images_by_camera_id(configs.image_swir_name)
    assert [vnir_image1, vnir_image2] == image_set.images_by_camera_id(
        configs.image_vnir_name
    )


def test_sort(configs):
    vnir_image1 = create_spectral_image(
        configs.image_vnir_hdr_path, configs.image_vnir_img_path
    )
    vnir_image2 = create_spectral_image(
        configs.image_vnir_hdr_path, configs.image_vnir_img_path
    )
    swir_image1 = create_spectral_image(
        configs.image_swir_hdr_path, configs.image_swir_img_path
    )
    swir_image2 = create_spectral_image(
        configs.image_swir_hdr_path, configs.image_swir_img_path
    )
    swir_image3 = create_spectral_image(
        configs.image_swir_hdr_path, configs.image_swir_img_path
    )

    unordered_set = [vnir_image1, swir_image1, vnir_image2, swir_image2, swir_image3]
    ordered_set = [swir_image1, swir_image2, swir_image3, vnir_image1, vnir_image2]

    image_set = SpectralImageSet(unordered_set.copy())

    assert image_set.images == unordered_set != ordered_set
    assert sorted(image_set.images) == ordered_set != unordered_set
    assert image_set.images == unordered_set != ordered_set
    image_set.sort()
    assert image_set.images == ordered_set != unordered_set
    image_set = SpectralImageSet(unordered_set.copy())
    image_set.images.sort()
    assert image_set.images == ordered_set != unordered_set
