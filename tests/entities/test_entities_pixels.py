import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import pytest

from siapy.core.exceptions import InvalidInputError
from siapy.entities import Pixels
from siapy.entities.pixels import HomogeneousCoordinate, validate_pixel_input_dimensions

iterable = [(1, 2), (3, 4), (5, 6)]
iterable_homo = [(1, 2, 1), (3, 4, 1), (5, 6, 1)]


def test_len():
    pixels = Pixels.from_iterable(iterable)
    assert len(pixels) == 3


def test_getitem():
    pixels = Pixels.from_iterable(iterable)
    pixel = pixels[0]
    assert isinstance(pixel, tuple)
    assert pixel.x == 1
    assert pixel.y == 2

    # Get last item
    pixel = pixels[2]
    assert pixel.x == 5
    assert pixel.y == 6


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
    assert pixels.u().equals(expected_x)


def test_v():
    df = pd.DataFrame(iterable, columns=[Pixels.coords.X, Pixels.coords.Y])
    pixels = Pixels(df)
    expected_y = pd.Series([2, 4, 6], name=Pixels.coords.Y)
    assert pixels.v().equals(expected_y)


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
    valid_df = pd.DataFrame([(1, 2)], columns=[HomogeneousCoordinate.X, HomogeneousCoordinate.Y])
    validate_pixel_input_dimensions(valid_df)

    wrong_cols_df = pd.DataFrame([(1, 2, 3)], columns=["a", "b", "c"])
    with pytest.raises(InvalidInputError) as exc_info:
        validate_pixel_input_dimensions(wrong_cols_df)
    assert "expected 2 columns" in str(exc_info.value)

    wrong_names_df = pd.DataFrame([(1, 2)], columns=["a", "b"])
    with pytest.raises(InvalidInputError) as exc_info:
        validate_pixel_input_dimensions(wrong_names_df)
    assert "Invalid column names" in str(exc_info.value)

    reordered_df = pd.DataFrame([(1, 2)], columns=[HomogeneousCoordinate.Y, HomogeneousCoordinate.X])
    validate_pixel_input_dimensions(reordered_df)
