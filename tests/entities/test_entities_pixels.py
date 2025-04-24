import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import pytest

from siapy.core.exceptions import InvalidInputError, InvalidTypeError
from siapy.entities import Pixels
from siapy.entities.pixels import (
    HomogeneousCoordinate,
    PixelCoordinate,
    validate_pixel_input,
    validate_pixel_input_dimensions,
)

iterable = [(1, 2), (3, 4), (5, 6)]
iterable_homo = [(1, 2, 1), (3, 4, 1), (5, 6, 1)]


def test_len():
    pixels = Pixels.from_iterable(iterable)
    assert len(pixels) == 3


def test_array_interface():
    pixels = Pixels.from_iterable([(1, 2), (3, 4), (5, 6)])

    # Test implicit conversion to numpy array
    array = np.asarray(pixels)
    assert isinstance(array, np.ndarray)
    assert array.shape == (3, 2)
    assert np.array_equal(array, pixels.to_numpy())

    # Test with dtype conversion
    int_array = np.asarray(pixels, dtype=np.int32)
    assert int_array.dtype == np.int32
    assert np.array_equal(int_array, pixels.to_numpy().astype(np.int32))

    # Test numpy operations
    mean_coords = np.mean(pixels, axis=0)
    expected_mean = np.mean(pixels.to_numpy(), axis=0)
    assert np.array_equal(mean_coords, expected_mean)


def test_getitem():
    pixels = Pixels.from_iterable(iterable)

    # Test single index
    pixel = pixels[0]
    assert isinstance(pixel, Pixels)
    assert pixel.df.iloc[0, 0] == 1
    assert pixel.df.iloc[0, 1] == 2

    # Get last item
    pixel = pixels[2]
    assert pixel.df.iloc[0, 0] == 5
    assert pixel.df.iloc[0, 1] == 6

    # Test slicing
    slice_pixels = pixels[0:2]
    assert isinstance(slice_pixels, Pixels)
    assert len(slice_pixels) == 2
    assert slice_pixels.df.iloc[0, 0] == 1
    assert slice_pixels.df.iloc[1, 0] == 3

    # Test fancy indexing
    fancy_pixels = pixels[[0, 2]]
    assert isinstance(fancy_pixels, Pixels)
    assert len(fancy_pixels) == 2
    assert fancy_pixels.df.iloc[0, 0] == 1
    assert fancy_pixels.df.iloc[1, 0] == 5


def test_equality():
    pixels1 = Pixels.from_iterable(iterable)
    pixels2 = Pixels.from_iterable(iterable)
    pixels3 = Pixels.from_iterable([(1, 2), (3, 4)])

    assert pixels1 == pixels2
    assert pixels1 != pixels3
    assert pixels1 != "not a pixels object"


def test_from_iterable():
    pixels = Pixels.from_iterable(iterable)
    assert isinstance(pixels, Pixels)
    assert pixels.df.equals(pd.DataFrame(iterable, columns=[Pixels.coords.X, Pixels.coords.Y]))


def test_df():
    df = pd.DataFrame(iterable, columns=["x", "y"])
    pixels = Pixels(df)
    assert pixels.df.equals(df)


def test_df_homogenious():
    df = pd.DataFrame(iterable, columns=[Pixels.coords.X, Pixels.coords.Y])
    pixels = Pixels(df)
    df_homogenious = pixels.df_homogenious()
    assert df_homogenious.equals(
        pd.DataFrame(
            iterable_homo,
            columns=[Pixels.coords.X, Pixels.coords.Y, Pixels.coords.H],
        )
    )


def test_u():
    df = pd.DataFrame(iterable, columns=[Pixels.coords.X, Pixels.coords.Y])
    pixels = Pixels(df)
    expected_x = pd.Series([1, 3, 5], name=Pixels.coords.X)
    assert pixels.x().equals(expected_x)


def test_v():
    df = pd.DataFrame(iterable, columns=[Pixels.coords.X, Pixels.coords.Y])
    pixels = Pixels(df)
    expected_y = pd.Series([2, 4, 6], name=Pixels.coords.Y)
    assert pixels.y().equals(expected_y)


def test_to_numpy():
    df = pd.DataFrame(iterable, columns=[Pixels.coords.X, Pixels.coords.Y])
    pixels = Pixels(df)
    expected_array = df.to_numpy()
    assert np.array_equal(pixels.to_numpy(), expected_array)


def test_to_list():
    df = pd.DataFrame(iterable, columns=[Pixels.coords.X, Pixels.coords.Y])
    pixels = Pixels(df)
    expected_list = pixels.to_list()
    assert expected_list == [[1, 2], [3, 4], [5, 6]]


def test_save_and_load_to_parquet():
    pixels = Pixels.from_iterable(iterable)
    with TemporaryDirectory() as tmpdir:
        parquet_file = Path(tmpdir, "test_pixels.parquet")
        pixels.save_to_parquet(parquet_file)
        assert os.path.exists(parquet_file)
        loaded_pixels = Pixels.load_from_parquet(parquet_file)
        assert isinstance(loaded_pixels, Pixels)
        assert loaded_pixels.df.equals(pixels.df)


def test_invalid_input_dimensions():
    # Test with wrong number of columns
    with pytest.raises(InvalidInputError):
        df = pd.DataFrame([(1, 2, 3)], columns=["x", "x", "z"])
        Pixels(df)

    # Test with wrong column names
    with pytest.raises(InvalidInputError):
        df = pd.DataFrame([(1, 2)], columns=["u", "v"])
        Pixels(df)


def test_validate_pixel_input_dimensions():
    # Valid DataFrame
    valid_df = pd.DataFrame([(1, 2)], columns=[HomogeneousCoordinate.X, HomogeneousCoordinate.Y])
    validate_pixel_input_dimensions(valid_df)

    # Test Series rejection
    series = pd.Series([1, 2], index=[HomogeneousCoordinate.X, HomogeneousCoordinate.Y])
    with pytest.raises(InvalidTypeError) as exc_info:
        validate_pixel_input_dimensions(series)
    assert "Expected a DataFrame, but got a Series" in str(exc_info.value)

    # Test empty DataFrame
    empty_df = pd.DataFrame(columns=[HomogeneousCoordinate.X, HomogeneousCoordinate.Y])
    with pytest.raises(InvalidInputError) as exc_info:
        validate_pixel_input_dimensions(empty_df)
    assert "Input DataFrame is empty" in str(exc_info.value)

    # Test wrong number of columns
    wrong_cols_df = pd.DataFrame([(1, 2, 3)], columns=["a", "b", "c"])
    with pytest.raises(InvalidInputError) as exc_info:
        validate_pixel_input_dimensions(wrong_cols_df)
    assert "expected 2 columns" in str(exc_info.value)

    # Test wrong column names
    wrong_names_df = pd.DataFrame([(1, 2)], columns=["a", "b"])
    with pytest.raises(InvalidInputError) as exc_info:
        validate_pixel_input_dimensions(wrong_names_df)
    assert "Invalid column names" in str(exc_info.value)

    # Test reordered column names (should still be valid)
    reordered_df = pd.DataFrame([(1, 2)], columns=[HomogeneousCoordinate.Y, HomogeneousCoordinate.X])
    validate_pixel_input_dimensions(reordered_df)


def test_get_coordinate():
    pixels = Pixels.from_iterable(iterable)
    coord = pixels.get_coordinate(0)
    assert isinstance(coord, PixelCoordinate)
    assert coord.x == 1
    assert coord.y == 2

    # Test out of bounds
    with pytest.raises(IndexError):
        pixels.get_coordinate(3)


def test_as_type():
    float_iterable = [(1.5, 2.7), (3.2, 4.9), (5.1, 6.3)]
    float_pixels = Pixels.from_iterable(float_iterable)

    int_pixels = float_pixels.as_type(int)

    assert isinstance(int_pixels, Pixels)
    assert int_pixels is not float_pixels

    assert int_pixels.get_coordinate(0).x == 1
    assert int_pixels.get_coordinate(0).y == 2
    assert int_pixels.get_coordinate(1).x == 3
    assert int_pixels.get_coordinate(1).y == 4
    assert int_pixels.get_coordinate(2).x == 5
    assert int_pixels.get_coordinate(2).y == 6

    assert float_pixels.get_coordinate(0).x == 1.5
    assert float_pixels.get_coordinate(0).y == 2.7

    # Test converting integers to float
    int_iterable = [(1, 2), (3, 4), (5, 6)]
    int_pixels = Pixels.from_iterable(int_iterable)
    float_pixels = int_pixels.as_type(float)

    # Verify conversion to float
    assert isinstance(float_pixels.get_coordinate(0).x, float)
    assert isinstance(float_pixels.get_coordinate(0).y, float)
    assert float_pixels.get_coordinate(0).x == 1.0
    assert float_pixels.get_coordinate(0).y == 2.0


def test_validate_pixel_input():
    # Test with valid Pixels instance
    original_pixels = Pixels.from_iterable(iterable)
    result_pixels = validate_pixel_input(original_pixels)
    assert result_pixels is original_pixels  # Should return same instance

    # Test with valid DataFrame
    valid_df = pd.DataFrame(iterable, columns=[HomogeneousCoordinate.X, HomogeneousCoordinate.Y])
    result_df_pixels = validate_pixel_input(valid_df)
    assert isinstance(result_df_pixels, Pixels)
    assert result_df_pixels.df.equals(valid_df)

    # Test with valid numpy array
    valid_array = np.array(iterable)
    result_array_pixels = validate_pixel_input(valid_array)
    assert isinstance(result_array_pixels, Pixels)
    assert np.array_equal(result_array_pixels.to_numpy(), valid_array)

    # Test with valid list of tuples
    valid_list = [(1.0, 2.0), (3.0, 4.0)]
    result_list_pixels = validate_pixel_input(valid_list)
    assert isinstance(result_list_pixels, Pixels)
    assert result_list_pixels.to_list() == [list(coord) for coord in valid_list]

    # Test with invalid numpy array dimensions
    invalid_array_3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    with pytest.raises(InvalidInputError) as exc_info:
        validate_pixel_input(invalid_array_3d)
    assert "NumPy array must be 2D" in str(exc_info.value)

    invalid_array_cols = np.array([[1, 2, 3], [4, 5, 6]])
    with pytest.raises(InvalidInputError) as exc_info:
        validate_pixel_input(invalid_array_cols)
    assert "shape (n, 2)" in str(exc_info.value)

    # Test with unsupported type
    with pytest.raises(InvalidTypeError) as exc_info:
        validate_pixel_input(123)
    assert "Unsupported input type" in str(exc_info.value)

    # Test with invalid DataFrame (wrong columns)
    invalid_df = pd.DataFrame(iterable, columns=["a", "b"])
    with pytest.raises(InvalidInputError) as exc_info:
        validate_pixel_input(invalid_df)
    assert "Invalid column names" in str(exc_info.value)

    # TODO: test for iterable incorrect
