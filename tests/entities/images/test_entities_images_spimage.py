from pathlib import Path

import numpy as np
import pytest
import spectral as sp
from PIL import Image

from siapy.core.exceptions import InvalidFilepathError
from siapy.entities import Pixels, SpectralImage
from siapy.utils.plots import pixels_select_lasso


def test_spy_open(configs):
    spectral_image_vnir = SpectralImage.spy_open(
        header_path=configs.image_vnir_hdr_path,
        image_path=configs.image_vnir_img_path,
    )
    assert isinstance(spectral_image_vnir, SpectralImage)

    spectral_image_swir = SpectralImage.spy_open(
        header_path=configs.image_swir_hdr_path,
        image_path=configs.image_swir_img_path,
    )
    assert isinstance(spectral_image_swir, SpectralImage)


def test_spy_open_invalid():
    with pytest.raises(InvalidFilepathError):
        SpectralImage.spy_open(header_path="invalid_header_path")


def test_fixture_spectral_images(spectral_images):
    assert isinstance(spectral_images.vnir, SpectralImage)
    assert isinstance(spectral_images.swir, SpectralImage)


def test_repr(spectral_images):
    assert isinstance(repr(spectral_images.vnir), str)
    assert isinstance(repr(spectral_images.swir), str)


def test_str(spectral_images):
    assert isinstance(str(spectral_images.vnir), str)
    assert isinstance(str(spectral_images.swir), str)


def test_lt(spectral_images):
    assert spectral_images.vnir > spectral_images.swir


def test_eq(spectral_images):
    assert spectral_images.vnir == spectral_images.vnir
    assert spectral_images.swir != spectral_images.vnir


# Spectral Lib Image


def test_image_file(spectral_images):
    assert isinstance(
        spectral_images.vnir.image.file,
        (sp.io.envi.BilFile, sp.io.envi.BipFile, sp.io.envi.BsqFile),
    )
    assert isinstance(
        spectral_images.swir.image.file,
        (sp.io.envi.BilFile, sp.io.envi.BipFile, sp.io.envi.BsqFile),
    )


def test_metadata(spectral_images):
    vnir_meta = spectral_images.vnir.metadata
    swir_meta = spectral_images.swir.metadata
    assert isinstance(vnir_meta, dict)
    assert isinstance(swir_meta, dict)
    required_keys = ["default bands", "wavelength", "description"]
    assert all(key in vnir_meta.keys() for key in required_keys)
    assert all(key in swir_meta.keys() for key in required_keys)


def test_shape(spectral_images):
    assert isinstance(spectral_images.vnir.shape, tuple)
    assert len(spectral_images.vnir.shape) == 3
    assert isinstance(spectral_images.swir.shape, tuple)
    assert len(spectral_images.swir.shape) == 3


def test_bands(spectral_images):
    assert isinstance(spectral_images.vnir.bands, int)
    assert isinstance(spectral_images.swir.bands, int)


def test_default_bands(spectral_images):
    vnir_db = spectral_images.vnir.default_bands
    swir_db = spectral_images.swir.default_bands
    assert np.array_equal(vnir_db, [55, 41, 12])
    assert np.array_equal(swir_db, [20, 117, 57])


def test_filename(spectral_images, configs):
    assert isinstance(spectral_images.vnir.filepath, Path)
    assert isinstance(spectral_images.swir.filepath, Path)
    assert spectral_images.vnir.filepath.name == configs.image_vnir_img_path.name
    assert spectral_images.swir.filepath.name == configs.image_swir_img_path.name


def test_wavelengths(spectral_images):
    vnir_wave = spectral_images.vnir.wavelengths
    swir_wave = spectral_images.swir.wavelengths
    assert isinstance(vnir_wave, list)
    assert all(isinstance(w, float) for w in vnir_wave)
    assert len(vnir_wave) == 160
    assert isinstance(swir_wave, list)
    assert all(isinstance(w, float) for w in swir_wave)
    assert len(swir_wave) == 288


def test_camera_id(spectral_images, configs):
    vnir_cam_id = spectral_images.vnir.camera_id
    swir_cam_id = spectral_images.swir.camera_id
    assert vnir_cam_id == configs.image_vnir_name
    assert swir_cam_id == configs.image_swir_name


def test_to_numpy(spectral_images):
    spectral_image_vnir = spectral_images.vnir.to_numpy()
    spectral_image_swir = spectral_images.swir.to_numpy()
    assert isinstance(spectral_image_vnir, np.ndarray)
    assert isinstance(spectral_image_swir, np.ndarray)


def test_to_signatures(spectral_images):
    spectral_image_vnir = spectral_images.vnir
    iterable = [(1, 2), (3, 4), (5, 6)]

    pixels = Pixels.from_iterable(iterable)
    signatures = spectral_image_vnir.to_signatures(pixels)

    assert np.array_equal(
        signatures.signals.df.iloc[0].to_numpy(),
        spectral_images.vnir.to_numpy()[2, 1, :],
    )
    assert np.array_equal(
        signatures.signals.df.iloc[1].to_numpy(),
        spectral_images.vnir.to_numpy()[4, 3, :],
    )
    assert np.array_equal(
        signatures.signals.df.iloc[2].to_numpy(),
        spectral_images.vnir.to_numpy()[6, 5, :],
    )

    assert np.array_equal(signatures.pixels.df.iloc[0].to_numpy(), iterable[0])
    assert np.array_equal(signatures.pixels.df.iloc[1].to_numpy(), iterable[1])
    assert np.array_equal(signatures.pixels.df.iloc[2].to_numpy(), iterable[2])


@pytest.mark.manual
def test_to_signatures_perf(spectral_images):
    spectral_image_vnir = spectral_images.vnir
    selected_areas_vnir = pixels_select_lasso(spectral_image_vnir)
    out = spectral_image_vnir.to_signatures(selected_areas_vnir[0]).signals.to_numpy()
    assert isinstance(out, np.ndarray)


def test_to_subarray(spectral_images):
    spectral_image_vnir = spectral_images.vnir
    iterable = [(1, 2), (3, 4), (2, 4)]
    pixels = Pixels.from_iterable(iterable)
    subarray = spectral_image_vnir.to_subarray(pixels)
    expected_subarray = np.full((3, 3, spectral_image_vnir.bands), np.nan)
    image_array = spectral_image_vnir.to_numpy()
    expected_subarray[0, 0, :] = image_array[2, 1, :]
    expected_subarray[2, 2, :] = image_array[4, 3, :]
    expected_subarray[2, 1, :] = image_array[4, 2, :]

    assert expected_subarray.shape == (3, 3, spectral_image_vnir.bands)
    assert np.array_equal(subarray, expected_subarray, equal_nan=True)


def test_mean(spectral_images):
    spectral_image_vnir = spectral_images.vnir

    mean_all = spectral_image_vnir.mean()
    assert isinstance(mean_all, (float, np.floating))
    assert np.isclose(mean_all, np.nanmean(spectral_image_vnir.to_numpy()))

    mean_axis0 = spectral_image_vnir.mean(axis=0)
    assert isinstance(mean_axis0, np.ndarray)
    assert mean_axis0.shape == spectral_image_vnir.to_numpy().shape[1:]
    assert np.allclose(mean_axis0, np.nanmean(spectral_image_vnir.to_numpy(), axis=0))

    mean_axis1 = spectral_image_vnir.mean(axis=1)
    assert isinstance(mean_axis1, np.ndarray)
    assert mean_axis1.shape == (
        spectral_image_vnir.to_numpy().shape[0],
        spectral_image_vnir.to_numpy().shape[2],
    )
    assert np.allclose(mean_axis1, np.nanmean(spectral_image_vnir.to_numpy(), axis=1))

    mean_axis2 = spectral_image_vnir.mean(axis=2)
    assert isinstance(mean_axis2, np.ndarray)
    assert mean_axis2.shape == spectral_image_vnir.to_numpy().shape[:2]
    assert np.allclose(mean_axis2, np.nanmean(spectral_image_vnir.to_numpy(), axis=2))

    mean_axis_tuple = spectral_image_vnir.mean(axis=(0, 1))
    assert isinstance(mean_axis_tuple, np.ndarray)
    assert mean_axis_tuple.shape == (spectral_image_vnir.to_numpy().shape[2],)
    assert np.allclose(mean_axis_tuple, np.nanmean(spectral_image_vnir.to_numpy(), axis=(0, 1)))


def test_to_display(spectral_images):
    spectral_image_vnir = spectral_images.vnir

    image = spectral_image_vnir.to_display(equalize=True)
    assert isinstance(image, Image.Image)
    assert image.mode == "RGB"

    expected_size = (spectral_image_vnir.image.cols, spectral_image_vnir.image.rows)
    assert image.size == expected_size

    pixel_data = np.array(image)
    assert (pixel_data >= 0).all() and (pixel_data <= 255).all()
