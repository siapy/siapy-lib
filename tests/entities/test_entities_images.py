from pathlib import Path

import numpy as np
import pytest
import spectral as sp
from PIL import Image

from siapy.entities import Pixels, Shape, SpectralImage
from siapy.entities.images import GeometricShapes, _parse_description


def test_envi_open(configs):
    spectral_image_vnir = SpectralImage.envi_open(
        header_path=configs.image_vnir_hdr_path,
        image_path=configs.image_vnir_img_path,
    )
    assert isinstance(spectral_image_vnir, SpectralImage)

    spectral_image_swir = SpectralImage.envi_open(
        header_path=configs.image_swir_hdr_path,
        image_path=configs.image_swir_img_path,
    )
    assert isinstance(spectral_image_swir, SpectralImage)


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


def test_file(spectral_images):
    assert isinstance(
        spectral_images.vnir.file,
        (sp.io.envi.BilFile, sp.io.envi.BipFile, sp.io.envi.BsqFile),
    )
    assert isinstance(
        spectral_images.swir.file,
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


def test_rows(spectral_images):
    assert isinstance(spectral_images.vnir.rows, int)
    assert isinstance(spectral_images.swir.rows, int)


def test_cols(spectral_images):
    assert isinstance(spectral_images.vnir.cols, int)
    assert isinstance(spectral_images.swir.cols, int)


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


def test_description(spectral_images):
    vnir_desc = spectral_images.vnir.description
    swir_desc = spectral_images.swir.description
    assert isinstance(vnir_desc, dict)
    assert isinstance(swir_desc, dict)
    required_keys = ["ID"]
    assert all(key in vnir_desc.keys() for key in required_keys)
    assert all(key in swir_desc.keys() for key in required_keys)


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


def test_remove_nan(spectral_images):
    image = np.array([[[1, 2, np.nan], [4, 2, 6]], [[np.nan, 8, 9], [10, 11, 12]]])
    result = spectral_images.vnir._remove_nan(image.copy())

    assert np.array_equal(result[0, 0], np.array([0, 0, 0]))
    assert np.array_equal(result[0, 1], np.array([4, 2, 6]))
    assert np.array_equal(result[1, 0], np.array([0, 0, 0]))
    assert np.array_equal(result[1, 1], np.array([10, 11, 12]))

    # Call the _remove_nan method with a non-default nan_value
    result = spectral_images.vnir._remove_nan(image.copy(), nan_value=99)

    # Check that all nan values have been replaced with 99
    assert (result == 99).sum() == 6


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


def test_mean(spectral_images):
    spectral_image_vnir = spectral_images.vnir

    mean_all = spectral_image_vnir.mean()
    assert isinstance(mean_all, float)
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
    assert np.allclose(
        mean_axis_tuple, np.nanmean(spectral_image_vnir.to_numpy(), axis=(0, 1))
    )


def test_to_display(spectral_images):
    spectral_image_vnir = spectral_images.vnir

    image = spectral_image_vnir.to_display(equalize=True)
    assert isinstance(image, Image.Image)
    assert image.mode == "RGB"

    expected_size = (spectral_image_vnir.cols, spectral_image_vnir.rows)
    assert image.size == expected_size

    pixel_data = np.array(image)
    assert (pixel_data >= 0).all() and (pixel_data <= 255).all()


def test_geometric_shapes_getter(spectral_images):
    assert isinstance(spectral_images.vnir.geometric_shapes, GeometricShapes)
    assert len(spectral_images.vnir.geometric_shapes) == 0


def test_geometric_shapes_setter_valid(spectral_images, corresponding_pixels):
    pixels_vnir = corresponding_pixels.vnir
    rect = Shape.from_shape_type(shape_type="rectangle", pixels=pixels_vnir)
    shapes = [rect, rect]
    spectral_images.vnir.geometric_shapes.shapes = shapes
    assert spectral_images.vnir.geometric_shapes.shapes == shapes


def test_geometric_shapes_setter_invalid(spectral_images):
    # Assuming invalid_shape is not an instance of Shape
    invalid_shape = "not a shape"
    shapes = [invalid_shape]

    with pytest.raises(ValueError):
        spectral_images.vnir.geometric_shapes.shapes = shapes


def test_geometric_shapes_iteration(spectral_images, corresponding_pixels):
    rect = Shape.from_shape_type(
        shape_type="rectangle", pixels=corresponding_pixels.vnir
    )
    spectral_images.vnir.geometric_shapes.shapes = [rect, rect]
    for shape in spectral_images.vnir.geometric_shapes:
        assert isinstance(shape, Shape)


def test_geometric_shapes_getitem(spectral_images, corresponding_pixels):
    rect = Shape.from_shape_type(
        shape_type="rectangle", pixels=corresponding_pixels.vnir
    )
    spectral_images.vnir.geometric_shapes.shapes = [rect]
    assert spectral_images.vnir.geometric_shapes[0] == rect


def test_geometric_shapes_setitem_valid(spectral_images, corresponding_pixels):
    rect = Shape.from_shape_type(
        shape_type="rectangle", pixels=corresponding_pixels.vnir
    )
    spectral_images.vnir.geometric_shapes.shapes = [rect]
    new_rect = Shape.from_shape_type(
        shape_type="rectangle", pixels=corresponding_pixels.vnir
    )
    spectral_images.vnir.geometric_shapes[0] = new_rect
    assert spectral_images.vnir.geometric_shapes[0] == new_rect


def test_geometric_shapes_setitem_invalid(spectral_images):
    invalid_shape = "not a shape"
    spectral_images.vnir.geometric_shapes.shapes = []
    with pytest.raises(ValueError):
        spectral_images.vnir.geometric_shapes[0] = invalid_shape


def test_geometric_shapes_len(spectral_images, corresponding_pixels):
    rect = Shape.from_shape_type(
        shape_type="rectangle", pixels=corresponding_pixels.vnir
    )
    spectral_images.vnir.geometric_shapes.shapes = [rect, rect]
    assert len(spectral_images.vnir.geometric_shapes) == 2


def test_geometric_shapes_eq(spectral_images, corresponding_pixels):
    rect1 = Shape.from_shape_type(
        shape_type="rectangle", pixels=corresponding_pixels.vnir
    )
    rect2 = Shape.from_shape_type(
        shape_type="rectangle", pixels=corresponding_pixels.vnir
    )
    spectral_images.vnir.geometric_shapes.shapes = [rect1, rect2]
    other_geometric_shapes = GeometricShapes(spectral_images.vnir)
    other_geometric_shapes.shapes = [rect1, rect2]
    assert spectral_images.vnir.geometric_shapes == other_geometric_shapes


def test_geometric_shapes_append_valid(spectral_images, corresponding_pixels):
    rect = Shape.from_shape_type(
        shape_type="rectangle", pixels=corresponding_pixels.vnir
    )
    spectral_images.vnir.geometric_shapes.clear()
    spectral_images.vnir.geometric_shapes.append(rect)
    assert rect in spectral_images.vnir.geometric_shapes
    assert len(spectral_images.vnir.geometric_shapes) == 1


def test_geometric_shapes_append_invalid(spectral_images):
    invalid_shape = "not a shape"
    with pytest.raises(ValueError):
        spectral_images.vnir.geometric_shapes.append(invalid_shape)


def test_geometric_shapes_extend_valid(spectral_images, corresponding_pixels):
    rect = Shape.from_shape_type(
        shape_type="rectangle", pixels=corresponding_pixels.vnir
    )
    spectral_images.vnir.geometric_shapes.clear()
    spectral_images.vnir.geometric_shapes.extend([rect, rect])
    assert len(spectral_images.vnir.geometric_shapes) == 2


def test_geometric_shapes_extend_invalid(spectral_images):
    invalid_shape = "not a shape"
    with pytest.raises(ValueError):
        spectral_images.vnir.geometric_shapes.extend([invalid_shape])


def test_geometric_shapes_insert_valid(spectral_images, corresponding_pixels):
    rect = Shape.from_shape_type(
        shape_type="rectangle", pixels=corresponding_pixels.vnir
    )
    spectral_images.vnir.geometric_shapes.insert(0, rect)
    assert spectral_images.vnir.geometric_shapes[0] == rect


def test_geometric_shapes_insert_invalid(spectral_images):
    invalid_shape = "not a shape"
    with pytest.raises(ValueError):
        spectral_images.vnir.geometric_shapes.insert(0, invalid_shape)


def test_geometric_shapes_remove_valid(spectral_images, corresponding_pixels):
    rect = Shape.from_shape_type(
        shape_type="rectangle", pixels=corresponding_pixels.vnir
    )
    spectral_images.vnir.geometric_shapes.clear()
    spectral_images.vnir.geometric_shapes.append(rect)
    spectral_images.vnir.geometric_shapes.remove(rect)
    assert rect not in spectral_images.vnir.geometric_shapes


def test_geometric_shapes_remove_invalid(spectral_images):
    invalid_shape = "not a shape"
    with pytest.raises(ValueError):
        spectral_images.vnir.geometric_shapes.remove(invalid_shape)


def test_geometric_shapes_pop(spectral_images, corresponding_pixels):
    rect = Shape.from_shape_type(
        shape_type="rectangle", pixels=corresponding_pixels.vnir
    )
    spectral_images.vnir.geometric_shapes.clear()
    spectral_images.vnir.geometric_shapes.append(rect)
    popped_shape = spectral_images.vnir.geometric_shapes.pop()
    assert popped_shape == rect
    assert len(spectral_images.vnir.geometric_shapes) == 0


def test_geometric_shapes_clear(spectral_images, corresponding_pixels):
    rect = Shape.from_shape_type(
        shape_type="rectangle", pixels=corresponding_pixels.vnir
    )
    spectral_images.vnir.geometric_shapes.append(rect)
    spectral_images.vnir.geometric_shapes.clear()
    assert len(spectral_images.vnir.geometric_shapes) == 0


def test_geometric_shapes_index_valid(spectral_images, corresponding_pixels):
    rect = Shape.from_shape_type(
        shape_type="rectangle", pixels=corresponding_pixels.vnir
    )
    spectral_images.vnir.geometric_shapes.append(rect)
    index = spectral_images.vnir.geometric_shapes.index(rect)
    assert index == 0


def test_geometric_shapes_index_invalid(spectral_images, corresponding_pixels):
    rect = Shape.from_shape_type(
        shape_type="rectangle", pixels=corresponding_pixels.vnir
    )
    invalid_shape = "not a shape"
    spectral_images.vnir.geometric_shapes.append(rect)
    with pytest.raises(ValueError):
        spectral_images.vnir.geometric_shapes.index(invalid_shape)


def test_geometric_shapes_count(spectral_images, corresponding_pixels):
    rect = Shape.from_shape_type(
        shape_type="rectangle", pixels=corresponding_pixels.vnir
    )
    spectral_images.vnir.geometric_shapes.clear()
    spectral_images.vnir.geometric_shapes.append(rect)
    spectral_images.vnir.geometric_shapes.append(rect)
    count = spectral_images.vnir.geometric_shapes.count(rect)
    assert count == 2


def test_geometric_shapes_reverse(spectral_images, corresponding_pixels):
    rect1 = Shape.from_shape_type(
        shape_type="rectangle", pixels=corresponding_pixels.vnir
    )
    rect2 = Shape.from_shape_type(
        shape_type="rectangle", pixels=corresponding_pixels.vnir
    )
    spectral_images.vnir.geometric_shapes.append(rect1)
    spectral_images.vnir.geometric_shapes.append(rect2)
    spectral_images.vnir.geometric_shapes.reverse()
    assert spectral_images.vnir.geometric_shapes[0] == rect2
    assert spectral_images.vnir.geometric_shapes[1] == rect1


def test_geometric_shapes_sort(spectral_images, corresponding_pixels):
    rect1 = Shape.from_shape_type(
        shape_type="rectangle", pixels=corresponding_pixels.vnir, label="B"
    )
    rect2 = Shape.from_shape_type(
        shape_type="rectangle", pixels=corresponding_pixels.vnir, label="A"
    )
    spectral_images.vnir.geometric_shapes.clear()
    spectral_images.vnir.geometric_shapes.append(rect1)
    spectral_images.vnir.geometric_shapes.append(rect2)
    spectral_images.vnir.geometric_shapes.sort(key=lambda shape: shape.label)
    assert spectral_images.vnir.geometric_shapes[0].label == "A"
    assert spectral_images.vnir.geometric_shapes[1].label == "B"


def test_geometric_shapes_get_by_name_found(spectral_images, corresponding_pixels):
    rect1 = Shape.from_shape_type(
        shape_type="rectangle", pixels=corresponding_pixels.vnir, label="Rect1"
    )
    rect2 = Shape.from_shape_type(
        shape_type="rectangle", pixels=corresponding_pixels.vnir, label="Rect2"
    )
    spectral_images.vnir.geometric_shapes.shapes = [rect1, rect2]
    found_shape = spectral_images.vnir.geometric_shapes.get_by_name("Rect1")
    assert found_shape == rect1
    assert found_shape.label == "Rect1"


def test_geometric_shapes_get_by_name_not_found(spectral_images, corresponding_pixels):
    # Create a shape
    rect = Shape.from_shape_type(
        shape_type="rectangle", pixels=corresponding_pixels.vnir, label="Rect"
    )
    spectral_images.vnir.geometric_shapes.shapes = [rect]
    found_shape = spectral_images.vnir.geometric_shapes.get_by_name("NonExistent")
    assert found_shape is None


def test_parse_description_simple():
    description = "Frameperiod = 20060\nIntegration time = 20000"
    expected = {"Frameperiod": 20060, "Integration time": 20000}
    assert _parse_description(description) == expected


def test_parse_description_with_floats_and_ints():
    description = "Binning = 2\nPixelsize x = 0.000187"
    expected = {"Binning": 2, "Pixelsize x": 0.000187}
    assert _parse_description(description) == expected


def test_parse_description_with_commas():
    description = "Rotating stage position = 0.000000,15.700000,degrees"
    expected = {"Rotating stage position": [0.000000, 15.700000, "degrees"]}
    assert _parse_description(description) == expected


def test_parse_description_empty_value():
    description = "Comment ="
    expected = {"Comment": ""}
    assert _parse_description(description) == expected


def test_parse_description_invalid_format_raises_value_error():
    description = "This is not a valid format"
    with pytest.raises(ValueError):
        _parse_description(description)
