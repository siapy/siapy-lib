from pathlib import Path

import numpy as np
import pytest
import xarray as xr
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


def test_rasterio_open(configs):
    spectral_image = SpectralImage.rasterio_open(configs.image_micasense_merged)
    assert isinstance(spectral_image, SpectralImage)
    spectral_image = SpectralImage.rasterio_open(configs.image_micasense_blue)
    assert isinstance(spectral_image, SpectralImage)


def test_from_numpy():
    rng = np.random.default_rng(1)
    array = rng.random((100, 100, 3), dtype=np.float32)
    spectral_image = SpectralImage.from_numpy(array)
    assert isinstance(spectral_image, SpectralImage)


def test_spy_open_invalid():
    with pytest.raises(InvalidFilepathError):
        SpectralImage.spy_open(header_path="invalid_header_path")


def test_fixture_spectral_images(spectral_images):
    assert isinstance(spectral_images.vnir, SpectralImage)
    assert isinstance(spectral_images.swir, SpectralImage)


def test_array_interface(spectral_images):
    """Test the NumPy array interface (__array__ method)."""
    spectral_image_vnir = spectral_images.vnir

    # Test implicit conversion to numpy array
    array = np.asarray(spectral_image_vnir)
    assert isinstance(array, np.ndarray)
    assert array.shape == spectral_image_vnir.shape
    assert np.array_equal(array, spectral_image_vnir.to_numpy(), equal_nan=True)

    # Test implicit conversion with dtype specified
    float32_array = np.asarray(spectral_image_vnir, dtype=np.float32)
    assert float32_array.dtype == np.float32

    # Test numpy operations work directly on the spectral image
    mean_value = np.nanmean(spectral_image_vnir)
    expected_mean = np.nanmean(spectral_image_vnir.to_numpy())
    assert np.isclose(mean_value, expected_mean)

    # Test other numpy functions
    max_value = np.nanmax(spectral_image_vnir)
    expected_max = np.nanmax(spectral_image_vnir.to_numpy())
    assert np.isclose(max_value, expected_max)


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


def test_image_property(spectral_images):
    assert spectral_images.vnir.image is not None
    assert spectral_images.swir.image is not None


def test_geometric_shapes_property(spectral_images):
    assert spectral_images.vnir.geometric_shapes is not None
    assert spectral_images.swir.geometric_shapes is not None


def test_filepath_property(spectral_images):
    assert isinstance(spectral_images.vnir.filepath, Path)
    assert isinstance(spectral_images.swir.filepath, Path)


def test_metadata_property(spectral_images):
    assert isinstance(spectral_images.vnir.metadata, dict)
    assert isinstance(spectral_images.swir.metadata, dict)


def test_shape_property(spectral_images):
    assert isinstance(spectral_images.vnir.shape, tuple)
    assert len(spectral_images.vnir.shape) == 3


def test_bands_property(spectral_images):
    assert isinstance(spectral_images.vnir.bands, int)
    assert spectral_images.vnir.bands > 0


def test_default_bands_property(spectral_images):
    assert isinstance(spectral_images.vnir.default_bands, list)
    assert all(isinstance(x, int) for x in spectral_images.vnir.default_bands)


def test_wavelengths_property(spectral_images):
    assert isinstance(spectral_images.vnir.wavelengths, list)
    assert all(isinstance(x, float) for x in spectral_images.vnir.wavelengths)


def test_camera_id_property(spectral_images):
    assert isinstance(spectral_images.vnir.camera_id, str)


def test_to_display(spectral_images):
    assert isinstance(spectral_images.vnir.to_display(), Image.Image)


def test_to_numpy(spectral_images):
    assert isinstance(spectral_images.vnir.to_numpy(), np.ndarray)
    assert spectral_images.vnir.to_numpy().shape == (
        spectral_images.vnir.image.rows,
        spectral_images.vnir.image.cols,
        spectral_images.vnir.image.bands,
    )


def test_to_xarray(spectral_images):
    assert isinstance(spectral_images.vnir.to_xarray(), xr.DataArray)


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


def test_average_intensity(spectral_images):
    spectral_image_vnir = spectral_images.vnir

    mean_all = spectral_image_vnir.average_intensity()
    assert isinstance(mean_all, (float, np.floating))
    assert np.isclose(mean_all, np.nanmean(spectral_image_vnir.to_numpy()))

    mean_axis0 = spectral_image_vnir.average_intensity(axis=0)
    assert isinstance(mean_axis0, np.ndarray)
    assert mean_axis0.shape == spectral_image_vnir.to_numpy().shape[1:]
    assert np.allclose(mean_axis0, np.nanmean(spectral_image_vnir.to_numpy(), axis=0))

    mean_axis1 = spectral_image_vnir.average_intensity(axis=1)
    assert isinstance(mean_axis1, np.ndarray)
    assert mean_axis1.shape == (
        spectral_image_vnir.to_numpy().shape[0],
        spectral_image_vnir.to_numpy().shape[2],
    )
    assert np.allclose(mean_axis1, np.nanmean(spectral_image_vnir.to_numpy(), axis=1))

    mean_axis2 = spectral_image_vnir.average_intensity(axis=2)
    assert isinstance(mean_axis2, np.ndarray)
    assert mean_axis2.shape == spectral_image_vnir.to_numpy().shape[:2]
    assert np.allclose(mean_axis2, np.nanmean(spectral_image_vnir.to_numpy(), axis=2))

    mean_axis_tuple = spectral_image_vnir.average_intensity(axis=(0, 1))
    assert isinstance(mean_axis_tuple, np.ndarray)
    assert mean_axis_tuple.shape == (spectral_image_vnir.to_numpy().shape[2],)
    assert np.allclose(mean_axis_tuple, np.nanmean(spectral_image_vnir.to_numpy(), axis=(0, 1)))
